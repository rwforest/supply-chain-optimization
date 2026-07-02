"""Supply chain stress-test resolver.

Reads the normalized ``nodes`` / ``edges`` / ``bom`` tables from a Databricks
SQL warehouse, reconstructs the nested ``dataset`` dict, and runs the pyomo
TTR/TTS optimization. This is the same computation that previously lived inside
``agent/supply_chain_agent.py::optimization_tool`` — extracted here so it can be
served as an MCP tool from a Databricks App.
"""

from __future__ import annotations

import hashlib
import io
import json
import os

import pandas as pd
import pyomo.environ as pyo
from databricks import sql
from databricks.sdk.core import Config

import utils
import dataset_io as dio

CATALOG = os.getenv("SC_CATALOG", "supply_chain_stress_test")
SCHEMA = os.getenv("SC_SCHEMA", "data")

# Maximum number of changed variables to inline in the tool result. The delta
# between the baseline and disrupted solves is what mitigation advice needs;
# capping it keeps the payload bounded regardless of network size. Changes are
# ranked by magnitude so the most significant reroutes always survive the cap.
MAX_CHANGES = int(os.getenv("SC_MAX_CHANGES", "300"))

# Values below this are treated as unchanged (decision vars are integer-domain,
# so this only absorbs solver float noise).
_EPS = 1e-6

# Optional UC volume to persist the full optimized network to when the delta is
# truncated (e.g. "/Volumes/supply_chain_stress_test/data/results"). Unset by
# default so the base deployment needs no extra WRITE_VOLUME grant.
RESULT_VOLUME = os.getenv("SC_RESULT_VOLUME")


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three UC tables from the SQL warehouse into pandas DataFrames."""
    cfg = Config()
    http_path = f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}"
    conn = sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )
    try:
        with conn.cursor() as cur:
            frames = {}
            for name in ("nodes", "edges", "bom"):
                cur.execute(f"SELECT * FROM {CATALOG}.{SCHEMA}.{name}")
                cols = [c[0] for c in cur.description]
                frames[name] = pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()
    return frames["nodes"], frames["edges"], frames["bom"]


def load_dataset() -> dict:
    """Reconstruct the nested dataset dict from the UC tables."""
    nodes_df, edges_df, bom_df = _load_frames()
    return dio.reconstruct_dataset_from_frames(nodes_df, edges_df, bom_df)


def _var_values(model) -> dict[str, float]:
    """Flatten a solved model's decision variables into a ``key -> value`` map.

    Keys are stable strings like ``u[T2_4]`` or ``y[('T2_4', 'T1_1')]`` so two
    solves can be diffed directly.
    """
    values: dict[str, float] = {}
    for v in model.component_data_objects(ctype=pyo.Var, active=True):
        key = f"{v.parent_component().name}[{v.index()}]"
        values[key] = pyo.value(v)
    return values


def _compute_delta(before: dict[str, float], after: dict[str, float]) -> list[dict]:
    """Variables whose optimized value changed between the two solves.

    Returned records are sorted by absolute change (largest first) so callers
    that truncate keep the most significant reroutes.
    """
    changes = []
    for key in before.keys() | after.keys():
        b = before.get(key, 0.0)
        a = after.get(key, 0.0)
        if abs(a - b) > _EPS:
            changes.append({"var": key, "baseline": b, "disrupted": a, "change": a - b})
    changes.sort(key=lambda c: abs(c["change"]), reverse=True)
    return changes


def _persist_full_network(before: dict, after: dict, disrupted: list[str], ttr: float) -> str | None:
    """Write the complete optimized network to a UC volume as CSV; return path.

    Returns ``None`` (never raises) if no volume is configured or the write
    fails, so a persistence problem can't take down the tool call.
    """
    if not RESULT_VOLUME:
        return None
    try:
        rows = []
        for key in before.keys() | after.keys():
            rows.append({
                "var": key,
                "baseline": before.get(key, 0.0),
                "disrupted": after.get(key, 0.0),
            })
        df = pd.DataFrame(rows).sort_values("var")
        tag = hashlib.sha1(
            f"{sorted(disrupted)}|{ttr}".encode()
        ).hexdigest()[:12]
        path = f"{RESULT_VOLUME.rstrip('/')}/stress_test_{tag}.csv"

        from databricks.sdk import WorkspaceClient

        buf = io.BytesIO(df.to_csv(index=False).encode())
        WorkspaceClient().files.upload(path, buf, overwrite=True)
        return path
    except Exception:
        return None


def run_stress_test(disrupted: list[str], ttr: float) -> str:
    """Run the TTR (with/without disruption) and TTS optimization.

    Returns a compact ``key=value`` string with the scenario summary plus only
    the variables whose optimized value *changed* between the baseline and the
    disrupted solve. The delta is what mitigation advice needs, and it stays
    small even on large networks (the full assignment scales with nodes+edges).
    The change list is capped at ``MAX_CHANGES`` and, when a result volume is
    configured, the complete network is persisted and its path returned.
    """
    dataset = load_dataset()

    # TTR without disruption (baseline)
    df_without = utils.build_and_solve_ttr(dataset, [], ttr, True)
    values_without = _var_values(df_without["model"].values[0])

    # TTR with disruption
    df_with = utils.build_and_solve_ttr(dataset, disrupted, ttr, True)
    values_with = _var_values(df_with["model"].values[0])

    # TTS with disruption
    df_tts = utils.build_and_solve_tts(dataset, disrupted, False)

    changes = _compute_delta(values_without, values_with)
    total_changes = len(changes)
    truncated = total_changes > MAX_CHANGES
    shown = changes[:MAX_CHANGES]

    full_network_path = None
    if truncated:
        full_network_path = _persist_full_network(
            values_without, values_with, disrupted, ttr
        )

    num_nodes = len(dataset["tier1"]) + len(dataset["tier2"]) + len(dataset["tier3"])
    summary = {
        "disrupted": disrupted,
        "ttr": ttr,
        "termination_condition": str(df_with["termination_condition"].values[0]),
        "lost_profit": float(df_with["lost_profit"].values[0]),
        "tts": float(df_tts["tts"].values[0]),
        "network_size": {"nodes": num_nodes, "edges": len(dataset["edges"])},
        "num_changed_variables": total_changes,
        "changes_truncated": truncated,
    }
    if full_network_path:
        summary["full_network_path"] = full_network_path

    # Compact, readable change list: "u[T2_4]: 1323.0 -> 0.0"
    changes_str = "; ".join(
        f"{c['var']}: {c['baseline']:g} -> {c['disrupted']:g}" for c in shown
    )

    parts = [f"{k}={v}" for k, v in summary.items()]
    parts.append(f"network_changes_vs_baseline=[{changes_str}]")
    return ",".join(parts)

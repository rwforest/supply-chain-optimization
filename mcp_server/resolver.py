"""Supply chain stress-test resolver.

Reads the normalized ``nodes`` / ``edges`` / ``bom`` tables from a Databricks
SQL warehouse, reconstructs the nested ``dataset`` dict, and runs the pyomo
TTR/TTS optimization. This is the same computation that previously lived inside
``agent/supply_chain_agent.py::optimization_tool`` — extracted here so it can be
served as an MCP tool from a Databricks App.
"""

from __future__ import annotations

import hashlib
import os
import uuid

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

# Lakebase (Postgres) sink for the full optimized network. The complete
# assignment scales with nodes+edges, so it is persisted here every run and the
# tool result carries only a scenario_id pointer plus the delta. When the app
# has a Lakebase resource attached, PGHOST/PGDATABASE/PGUSER are injected and
# PGPASSWORD is minted per-call via an OAuth token (1h TTL). SC_LAKEBASE_INSTANCE
# names the instance for token generation.
LAKEBASE_INSTANCE = os.getenv("SC_LAKEBASE_INSTANCE", "supply-chain-lakebase")
LAKEBASE_SCHEMA = os.getenv("SC_LAKEBASE_SCHEMA", "supply_chain")


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


def _scenario_id(disrupted: list[str], ttr: float) -> str:
    """Stable id for a scenario so re-running upserts rather than duplicates."""
    return hashlib.sha1(f"{sorted(disrupted)}|{ttr}".encode()).hexdigest()[:16]


def _lakebase_conn():
    """Open a psycopg connection to Lakebase.

    Uses the app-injected PG* env vars for host/db/user and mints a fresh OAuth
    token as the password. Returns ``None`` if Lakebase is not configured.
    """
    host = os.getenv("PGHOST")
    if not host:
        return None

    import psycopg
    from databricks.sdk import WorkspaceClient

    user = os.getenv("PGUSER") or Config().client_id
    dbname = os.getenv("PGDATABASE", "databricks_postgres")
    port = os.getenv("PGPORT", "5432")

    token = os.getenv("PGPASSWORD")
    if not token:
        cred = WorkspaceClient().database.generate_database_credential(
            request_id=str(uuid.uuid4()), instance_names=[LAKEBASE_INSTANCE]
        )
        token = cred.token

    return psycopg.connect(
        host=host, dbname=dbname, user=user, password=token,
        port=port, sslmode="require",
    )


def _persist_full_network(
    scenario_id: str,
    summary: dict,
    before: dict,
    after: dict,
) -> bool:
    """Upsert the scenario summary + full optimized network into Lakebase.

    Returns True on success, False (never raises) if Lakebase is unconfigured or
    the write fails, so a persistence problem can't take down the tool call.
    """
    try:
        conn = _lakebase_conn()
        if conn is None:
            return False
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {LAKEBASE_SCHEMA}.stress_test_runs
                    (scenario_id, disrupted, ttr, termination_condition,
                     lost_profit, tts, num_nodes, num_edges, num_changed_variables)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scenario_id) DO UPDATE SET
                    disrupted = EXCLUDED.disrupted,
                    ttr = EXCLUDED.ttr,
                    termination_condition = EXCLUDED.termination_condition,
                    lost_profit = EXCLUDED.lost_profit,
                    tts = EXCLUDED.tts,
                    num_nodes = EXCLUDED.num_nodes,
                    num_edges = EXCLUDED.num_edges,
                    num_changed_variables = EXCLUDED.num_changed_variables,
                    created_at = now()
                """,
                (
                    scenario_id, summary["disrupted"], summary["ttr"],
                    summary["termination_condition"], summary["lost_profit"],
                    summary["tts"], summary["network_size"]["nodes"],
                    summary["network_size"]["edges"], summary["num_changed_variables"],
                ),
            )
            # Replace the detail rows for this scenario, then bulk insert.
            cur.execute(
                f"DELETE FROM {LAKEBASE_SCHEMA}.stress_test_network WHERE scenario_id = %s",
                (scenario_id,),
            )
            rows = [
                (scenario_id, key, before.get(key, 0.0), after.get(key, 0.0))
                for key in before.keys() | after.keys()
            ]
            cur.executemany(
                f"INSERT INTO {LAKEBASE_SCHEMA}.stress_test_network "
                f"(scenario_id, var, baseline, disrupted_v) VALUES (%s, %s, %s, %s)",
                rows,
            )
        conn.close()
        return True
    except Exception:
        return False


def run_stress_test(disrupted: list[str], ttr: float) -> str:
    """Run the TTR (with/without disruption) and TTS optimization.

    Returns a compact ``key=value`` string with the scenario summary plus only
    the variables whose optimized value *changed* between the baseline and the
    disrupted solve. The delta is what mitigation advice needs, and it stays
    small even on large networks (the full assignment scales with nodes+edges).
    The change list is capped at ``MAX_CHANGES``; the complete optimized network
    is always persisted to Lakebase under ``scenario_id`` (queryable in the
    ``stress_test_runs`` / ``stress_test_network`` tables).
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

    num_nodes = len(dataset["tier1"]) + len(dataset["tier2"]) + len(dataset["tier3"])
    scenario_id = _scenario_id(disrupted, ttr)
    summary = {
        "scenario_id": scenario_id,
        "disrupted": disrupted,
        "ttr": ttr,
        "termination_condition": str(df_with["termination_condition"].values[0]),
        "lost_profit": float(df_with["lost_profit"].values[0]),
        "tts": float(df_tts["tts"].values[0]),
        "network_size": {"nodes": num_nodes, "edges": len(dataset["edges"])},
        "num_changed_variables": total_changes,
        "changes_truncated": truncated,
    }

    # Always persist the full network to Lakebase; report whether it landed so
    # the agent knows if the scenario_id is queryable.
    summary["full_network_persisted"] = _persist_full_network(
        scenario_id, summary, values_without, values_with
    )

    # Compact, readable change list: "u[T2_4]: 1323.0 -> 0.0"
    changes_str = "; ".join(
        f"{c['var']}: {c['baseline']:g} -> {c['disrupted']:g}" for c in shown
    )

    parts = [f"{k}={v}" for k, v in summary.items()]
    parts.append(f"network_changes_vs_baseline=[{changes_str}]")
    return ",".join(parts)

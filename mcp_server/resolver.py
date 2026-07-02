"""Supply chain stress-test resolver.

Reads the normalized ``nodes`` / ``edges`` / ``bom`` tables from a Databricks
SQL warehouse, reconstructs the nested ``dataset`` dict, and runs the pyomo
TTR/TTS optimization. This is the same computation that previously lived inside
``agent/supply_chain_agent.py::optimization_tool`` — extracted here so it can be
served as an MCP tool from a Databricks App.
"""

from __future__ import annotations

import os

import pandas as pd
import pyomo.environ as pyo
from databricks import sql
from databricks.sdk.core import Config

import utils
import dataset_io as dio

CATALOG = os.getenv("SC_CATALOG", "supply_chain_stress_test")
SCHEMA = os.getenv("SC_SCHEMA", "data")


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


def _var_records(model) -> list[dict]:
    records = []
    for v in model.component_data_objects(ctype=pyo.Var, active=True):
        records.append(
            {
                "var_name": v.parent_component().name,
                "index": v.index(),
                "value": pyo.value(v),
            }
        )
    return records


def run_stress_test(disrupted: list[str], ttr: float) -> str:
    """Run the TTR (with/without disruption) and TTS optimization.

    Mirrors ``optimization_tool`` in ``agent/supply_chain_agent.py``: solves the
    TTR model without disruption, then with the disruption, then the TTS model,
    and returns a single comma-joined ``key=value`` string.
    """
    dataset = load_dataset()

    # TTR without disruption (baseline)
    df_without = utils.build_and_solve_ttr(dataset, [], ttr, True)
    records_without = _var_records(df_without["model"].values[0])

    # TTR with disruption
    df_with = utils.build_and_solve_ttr(dataset, disrupted, ttr, True)
    records_with = _var_records(df_with["model"].values[0])

    df = df_with.drop(["model"], axis=1)
    df["optimized_network_without_disruption"] = str(records_without)
    df["optimized_network_with_disruption"] = str(records_with)

    # TTS with disruption
    df_tts = utils.build_and_solve_tts(dataset, disrupted, False)
    df["tts"] = df_tts["tts"].values[0]

    return ",".join(f"{k}={v}" for k, v in df.iloc[0].astype(str).items())

"""Serialization helpers between the nested ``generate_data`` dataset and the
normalized Unity Catalog tables that back the Genie space and the MCP resolver.

The optimization functions in ``scripts.utils`` (``build_and_solve_ttr`` /
``build_and_solve_tts``) consume a single nested ``dataset`` dict. To make the
data (a) queryable in natural language through a Genie space and (b) a single
source of truth shared with the optimization MCP server, we decompose that dict
into three relational tables and provide the inverse reconstruction.

Three tables:

- ``nodes`` : one row per node (all tiers) with its operational parameters.
- ``edges`` : one row per directed supply link ``source -> target``.
- ``bom``   : bill-of-materials, one row per (product node, material type).

``explode_dataset_to_tables`` and ``reconstruct_dataset_from_frames`` are pure
pandas functions (no Spark / notebook dependency) so the Databricks App hosting
the MCP server can import and use ``reconstruct_dataset_from_frames`` directly.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tier_of(node_id: str) -> int:
    """T1_3 -> 1, T2_7 -> 2, T3_12 -> 3."""
    return int(node_id.split("_")[0][1:])


def _natural_key(node_id: str) -> tuple[int, int]:
    """Sort key so T2_2 precedes T2_10 (numeric, not lexicographic)."""
    tier, num = node_id.split("_")
    return (int(tier[1:]), int(num))


# ---------------------------------------------------------------------------
# dataset dict  ->  relational tables
# ---------------------------------------------------------------------------
def explode_dataset_to_tables(dataset: dict) -> dict[str, pd.DataFrame]:
    """Decompose a ``generate_data`` dataset dict into normalized DataFrames.

    Parameters
    ----------
    dataset : dict
        The nested dict produced by ``scripts.utils.generate_data``.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys ``"nodes"``, ``"edges"`` and ``"bom"``.
    """
    tier1 = dataset["tier1"]
    tier2 = dataset["tier2"]
    tier3 = dataset["tier3"]
    edges = dataset["edges"]
    smt = dataset["supplier_material_type"]
    f = dataset["f"]
    s = dataset["s"]
    d = dataset["d"]
    c = dataset["c"]
    r = dataset["r"]

    # --- nodes -------------------------------------------------------------
    node_rows = []
    for node in tier1 + tier2 + tier3:
        tier = _tier_of(node)
        node_rows.append(
            {
                "node_id": node,
                "tier": tier,
                "material_type": smt.get(node),          # null for tier 1
                "profit_margin": f.get(node),            # tier 1 only
                "inventory": int(s[node]),               # all nodes
                "demand": d.get(node),                   # tier 1 only
                "capacity": int(c[node]),                # all nodes
            }
        )
    nodes_df = pd.DataFrame(
        node_rows,
        columns=[
            "node_id",
            "tier",
            "material_type",
            "profit_margin",
            "inventory",
            "demand",
            "capacity",
        ],
    )
    # demand is tier-1 only; keep as nullable integer
    nodes_df["demand"] = nodes_df["demand"].astype("Int64")

    # --- edges -------------------------------------------------------------
    edge_rows = []
    for src, tgt in edges:
        edge_rows.append(
            {
                "source_node": src,
                "target_node": tgt,
                "source_tier": _tier_of(src),
                "target_tier": _tier_of(tgt),
                "material_type": smt.get(src),  # material the source supplies
            }
        )
    edges_df = pd.DataFrame(
        edge_rows,
        columns=[
            "source_node",
            "target_node",
            "source_tier",
            "target_tier",
            "material_type",
        ],
    )

    # --- bom (bill of materials) ------------------------------------------
    bom_rows = []
    for product_node, inner in r.items():
        for material_type, qty in inner.items():
            bom_rows.append(
                {
                    "product_node": product_node,
                    "material_type": material_type,
                    "quantity_required": float(qty),
                }
            )
    bom_df = pd.DataFrame(
        bom_rows,
        columns=["product_node", "material_type", "quantity_required"],
    )

    return {"nodes": nodes_df, "edges": edges_df, "bom": bom_df}


# ---------------------------------------------------------------------------
# relational tables  ->  dataset dict
# ---------------------------------------------------------------------------
def reconstruct_dataset_from_frames(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    bom_df: pd.DataFrame,
) -> dict:
    """Rebuild the nested ``dataset`` dict from the normalized tables.

    The returned dict has the exact key set consumed by
    ``scripts.utils.build_and_solve_ttr`` / ``build_and_solve_tts``.
    ``N_minus``, ``N_plus`` and ``P`` are derived from the edges exactly as in
    ``scripts.utils.generate_data`` so the optimization sees identical input.
    """
    nodes_df = nodes_df.copy()
    nodes_df["_key"] = nodes_df["node_id"].map(_natural_key)
    nodes_df = nodes_df.sort_values("_key")

    tier1 = nodes_df.loc[nodes_df["tier"] == 1, "node_id"].tolist()
    tier2 = nodes_df.loc[nodes_df["tier"] == 2, "node_id"].tolist()
    tier3 = nodes_df.loc[nodes_df["tier"] == 3, "node_id"].tolist()

    # supplier_material_type: nodes that carry a material type (tiers 2 & 3)
    supplier_material_type = {
        row.node_id: row.material_type
        for row in nodes_df.itertuples()
        if pd.notna(row.material_type)
    }

    # material_types: sorted unique material codes (matches the lexicographic
    # generation order of generate_data, e.g. aaa, aab, aac, ...)
    material_types = sorted(set(supplier_material_type.values()))

    # scalar / tabular params
    f = {
        row.node_id: float(row.profit_margin)
        for row in nodes_df.itertuples()
        if row.tier == 1 and pd.notna(row.profit_margin)
    }
    s = {row.node_id: int(row.inventory) for row in nodes_df.itertuples()}
    d = {
        row.node_id: int(row.demand)
        for row in nodes_df.itertuples()
        if row.tier == 1 and pd.notna(row.demand)
    }
    c = {row.node_id: int(row.capacity) for row in nodes_df.itertuples()}

    # edges as list of (source, target) tuples
    edges = [
        (row.source_node, row.target_node) for row in edges_df.itertuples()
    ]

    # --- derive N_minus / N_plus / P exactly as generate_data does ---------
    N_minus = {j: [] for j in tier1 + tier2}
    for i, j in edges:
        if j in N_minus:
            k = supplier_material_type[i]
            if k not in N_minus[j]:
                N_minus[j].append(k)

    N_plus = {i: [] for i in tier2 + tier3}
    for i, j in edges:
        if i in N_plus and j not in N_plus[i]:
            N_plus[i].append(j)

    P = {j: {} for j in tier1 + tier2}
    for i, j in edges:
        if j in P:
            k = supplier_material_type[i]
            P[j].setdefault(k, []).append(i)

    # r (bill of materials) straight from the bom table
    r = {j: {} for j in tier1 + tier2}
    for row in bom_df.itertuples():
        val = row.quantity_required
        # keep integer coefficients as int (matches generate_data), else float
        r[row.product_node][row.material_type] = (
            int(val) if float(val).is_integer() else float(val)
        )

    return {
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "edges": edges,
        "material_types": material_types,
        "supplier_material_type": supplier_material_type,
        "f": f,
        "s": s,
        "d": d,
        "c": c,
        "r": r,
        "N_minus": N_minus,
        "N_plus": N_plus,
        "P": P,
    }


# ---------------------------------------------------------------------------
# Parity check
# ---------------------------------------------------------------------------
def datasets_equivalent(a: dict, b: dict) -> bool:
    """Order-insensitive semantic equality of two dataset dicts.

    Row/edge ordering is not preserved by relational tables, and list-valued
    derived fields (``N_minus``, ``N_plus``, ``P``, ``edges``) only need to
    match as *sets* for the optimization to produce identical results. This
    helper normalizes that ordering before comparing.
    """
    if set(a.keys()) != set(b.keys()):
        return False

    def norm(key, val):
        if key in ("tier1", "tier2", "tier3", "material_types"):
            return sorted(val)
        if key == "edges":
            return sorted(tuple(e) for e in val)
        if key in ("N_minus", "N_plus"):
            return {k: sorted(v) for k, v in val.items()}
        if key == "P":
            return {j: {k: sorted(i) for k, i in inner.items()} for j, inner in val.items()}
        if key in ("f", "s", "d", "c"):
            return {k: float(v) for k, v in val.items()}
        if key == "r":
            return {j: {k: float(v) for k, v in inner.items()} for j, inner in val.items()}
        return val

    return all(norm(k, a[k]) == norm(k, b[k]) for k in a)

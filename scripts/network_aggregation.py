"""Aggregate/disaggregate network decomposition — modeled after Adexa's
Strategic Network Optimizer (SNO): reduce the network before solving a
what-if, instead of always re-solving the *entire* network at full fidelity
for every scenario the way ``scripts.utils.build_and_solve_ttr``/``tts`` (and
this repo's own time-phased engine in ``scripts/multi_period_planning.py``)
do today.

The idea: pool many similar, low-individual-impact tier-3 nodes (same
material, same criticality tag) into one synthetic "group" node, solve the
smaller aggregate network, then restore only the part of the network that
actually needs full fidelity for a given disruption — the disrupted node's
own group (an aggregate node can't be "35% disrupted"), its downstream
consumer cone, and its upstream alternate suppliers — and re-solve just that
mixed-fidelity network.

**Aggregation error is one-directional and optimistic, not a safe bound.**
Pooling capacity/inventory (``c_agg = sum(c_i)``, ``s_agg = sum(s_i)``) and
taking the union of members' outgoing edges only ever *loosens* constraints
relative to the true, disaggregated network — it can never make the
aggregate model pessimistic relative to reality. A result computed purely
on the aggregate network should be read as "at least this good," never as a
conservative worst case. This is exactly why disaggregation restores full
fidelity around any node that's actually disrupted, rather than trusting the
aggregate solve there.

The disaggregation scope here is a **fixed-radius** rule (``expansion_hops``
hops of downstream consumers, plus direct upstream alternates) rather than a
principled dual-based selection (e.g. "restore anything with material MIP
dual value above some threshold"). That's a deliberate simplification: MIP
duals aren't well-defined without relaxing integrality on the aggregate
solve, and doing that rigorously is future work, not implemented here.
"""

from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from dataclasses import dataclass, field

from scripts.realistic_topologies import derive_indexes

_SUM_FIELDS = ("s", "c")
_MAX_FIELDS = ("production_delay",)
_MEAN_FIELDS = ("unit_cost", "holding_cost", "emissions_factor")
_FIRST_FIELDS = ("supplier_material_type", "criticality", "region")


@dataclass
class AggregationResult:
    aggregate_dataset: dict
    node_to_group: dict[str, str] = field(default_factory=dict)
    group_to_members: dict[str, list[str]] = field(default_factory=dict)


def _aggregate_field(field_name: str, group_to_members: dict[str, list[str]], dataset: dict, agg_type: str) -> dict:
    """Build the field dict for the aggregate dataset: pass original values
    through unchanged for nodes that weren't grouped, and compute one
    aggregated value per synthetic group node."""
    grouped_away = {m for members in group_to_members.values() for m in members}
    orig = dataset.get(field_name, {})
    result = {node: val for node, val in orig.items() if node not in grouped_away}

    for group_node, members in group_to_members.items():
        member_vals = [orig[m] for m in members if m in orig]
        if not member_vals:
            continue
        if agg_type == "sum":
            result[group_node] = sum(member_vals)
        elif agg_type == "max":
            result[group_node] = max(member_vals)
        elif agg_type == "mean":
            result[group_node] = sum(member_vals) / len(member_vals)
        elif agg_type == "first":
            result[group_node] = member_vals[0]
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown agg_type {agg_type!r}")
    return result


def _rebuild_edges(dataset: dict, node_to_group: dict[str, str]) -> list[tuple[str, str]]:
    """Remap every edge endpoint through ``node_to_group`` (identity for
    ungrouped nodes) and dedup — this is what turns each grouped node's
    individual outgoing edges into the group node's *union* of edges."""
    seen: set[tuple[str, str]] = set()
    edges: list[tuple[str, str]] = []
    for i, j in dataset["edges"]:
        new_edge = (node_to_group.get(i, i), node_to_group.get(j, j))
        if new_edge not in seen:
            seen.add(new_edge)
            edges.append(new_edge)
    return edges


def build_aggregate_dataset(
    dataset: dict,
    aggregate_tiers: tuple[int, ...] = (3,),
    group_keys: tuple[str, ...] = ("supplier_material_type", "criticality"),
    exclude_named_anchors: bool = True,
) -> AggregationResult:
    """Roll up nodes in ``aggregate_tiers`` that share the same values for
    every key in ``group_keys`` into one synthetic group node each. Sole,
    hard-to-replace anchors (``criticality == "monopoly_bottleneck"``) are
    never grouped by default — a monopoly supplier can't be pooled away
    into an anonymous aggregate, since doing so would hide the very
    single-point-of-failure risk this whole accelerator exists to surface.

    Note this deliberately does NOT key off ``company_name``: every
    generator in ``scripts/realistic_topologies.py`` gives the *large
    majority* of tier2/tier3 nodes a company name — real, cited anchors
    (TSMC, ASML, ...) in the complex tier, purely fictional flavor names in
    the simple/medium tiers, and even most complex-tier "rest of market"
    peers. Treating any non-empty ``company_name`` as un-poolable would
    exclude nearly every node in the simple/medium tiers, defeating
    aggregation entirely there; ``criticality`` is the field that actually
    distinguishes "irreplaceable" from "one of many similar suppliers"
    across every generator."""
    tier_lists = {1: list(dataset["tier1"]), 2: list(dataset["tier2"]), 3: list(dataset["tier3"])}
    criticality = dataset.get("criticality", {})

    def is_named_anchor(n: str) -> bool:
        return criticality.get(n) == "monopoly_bottleneck"

    groups_by_key: dict[tuple, list[str]] = defaultdict(list)
    for tier in aggregate_tiers:
        for n in tier_lists.get(tier, []):
            if exclude_named_anchors and is_named_anchor(n):
                continue
            key = (tier,) + tuple(dataset.get(gk, {}).get(n) for gk in group_keys)
            groups_by_key[key].append(n)

    node_to_group: dict[str, str] = {}
    group_to_members: dict[str, list[str]] = {}
    group_to_tier: dict[str, int] = {}
    group_idx = 0
    for key in sorted(groups_by_key, key=lambda k: [str(x) for x in k]):
        members = groups_by_key[key]
        if len(members) < 2:
            continue  # a singleton "group" gains nothing from pooling
        group_idx += 1
        tier = key[0]
        group_node = f"AGG{tier}_{group_idx:04d}"
        group_to_members[group_node] = sorted(members)
        group_to_tier[group_node] = tier
        for m in members:
            node_to_group[m] = group_node

    new_tiers = dict(tier_lists)
    for tier in aggregate_tiers:
        grouped_this_tier = {m for m in tier_lists[tier] if m in node_to_group}
        kept = [n for n in tier_lists[tier] if n not in grouped_this_tier]
        new_group_nodes = sorted(g for g, t in group_to_tier.items() if t == tier)
        new_tiers[tier] = kept + new_group_nodes

    edges = _rebuild_edges(dataset, node_to_group)

    supplier_material_type = _aggregate_field("supplier_material_type", group_to_members, dataset, "first")
    s = _aggregate_field("s", group_to_members, dataset, "sum")
    c = _aggregate_field("c", group_to_members, dataset, "sum")

    optional_fields: dict[str, dict] = {}
    for f_name in _MAX_FIELDS:
        if f_name in dataset:
            optional_fields[f_name] = _aggregate_field(f_name, group_to_members, dataset, "max")
    for f_name in _MEAN_FIELDS:
        if f_name in dataset:
            optional_fields[f_name] = _aggregate_field(f_name, group_to_members, dataset, "mean")
    for f_name in _FIRST_FIELDS:
        if f_name in dataset and f_name != "supplier_material_type":
            optional_fields[f_name] = _aggregate_field(f_name, group_to_members, dataset, "first")

    if "company_name" in dataset:
        grouped_away = {m for members in group_to_members.values() for m in members}
        company = {n: v for n, v in dataset["company_name"].items() if n not in grouped_away}
        for g in group_to_members:
            company[g] = ""
        optional_fields["company_name"] = company

    if "transport_emissions" in dataset:
        base_vals = list(dataset["transport_emissions"].values())
        default_te = sum(base_vals) / len(base_vals) if base_vals else 0.0
        optional_fields["transport_emissions"] = {
            e: dataset["transport_emissions"].get(e, default_te) for e in edges
        }

    N_minus, N_plus, P, r = derive_indexes(
        new_tiers[1], new_tiers[2], new_tiers[3], edges, supplier_material_type
    )

    aggregate_dataset = {
        "tier1": new_tiers[1],
        "tier2": new_tiers[2],
        "tier3": new_tiers[3],
        "edges": edges,
        "material_types": list(dataset["material_types"]),
        "supplier_material_type": supplier_material_type,
        "f": dict(dataset["f"]),
        "s": s,
        "d": dict(dataset["d"]),
        "c": c,
        "r": r,
        "N_minus": N_minus,
        "N_plus": N_plus,
        "P": P,
        **optional_fields,
    }

    return AggregationResult(
        aggregate_dataset=aggregate_dataset,
        node_to_group=node_to_group,
        group_to_members=group_to_members,
    )


def disaggregation_scope(
    dataset: dict,
    node_to_group: dict[str, str],
    disrupted_nodes: list[str],
    expansion_hops: int | None = 1,
) -> set[str]:
    """Real node IDs (from the original, full-fidelity ``dataset``) that must
    be restored to full fidelity for a disruption at ``disrupted_nodes``:

    1. The disrupted node's entire group, always — an aggregate node can't be
       "35% disrupted."
    2. Its downstream consumer cone, ``expansion_hops`` deep via the
       original ``N_plus`` (``None`` = full downstream closure).
    3. Its upstream alternate suppliers — the other members of every
       ``P[j][k]`` the disrupted node belongs to as a supplier.

    Any node pulled in by (2) or (3) that itself belongs to a group also
    pulls in that entire group, for the same reason as (1).
    """
    group_to_members: dict[str, list[str]] = defaultdict(list)
    for node, grp in node_to_group.items():
        group_to_members[grp].append(node)

    def expand_groups(nodes: set[str]) -> set[str]:
        expanded = set(nodes)
        frontier = set(nodes)
        while frontier:
            new = set()
            for n in frontier:
                grp = node_to_group.get(n)
                if grp is not None:
                    for m in group_to_members[grp]:
                        if m not in expanded:
                            expanded.add(m)
                            new.add(m)
            frontier = new
        return expanded

    scope = expand_groups(set(disrupted_nodes))

    N_plus = dataset.get("N_plus", {})
    frontier = set(scope)
    hops = 0
    while frontier and (expansion_hops is None or hops < expansion_hops):
        next_frontier: set[str] = set()
        for n in frontier:
            for j in N_plus.get(n, []):
                if j not in scope:
                    next_frontier.add(j)
        next_frontier = expand_groups(next_frontier) - scope
        scope |= next_frontier
        frontier = next_frontier
        hops += 1

    supplier_to_jk: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for j, k_map in dataset.get("P", {}).items():
        for k, suppliers in k_map.items():
            for supplier in suppliers:
                supplier_to_jk[supplier].append((j, k))

    alt_suppliers: set[str] = set()
    for i in disrupted_nodes:
        for j, k in supplier_to_jk.get(i, []):
            alt_suppliers.update(dataset["P"][j][k])
    scope |= expand_groups(alt_suppliers)

    return scope


def build_partial_disaggregation(dataset: dict, agg_result: AggregationResult, expand_real_nodes: set[str]) -> dict:
    """Rebuild a mixed-fidelity dataset: every node whose group intersects
    ``expand_real_nodes`` is restored to its real, individual identity (with
    the original ``dataset``'s values); everything else stays rolled up as
    in ``agg_result.aggregate_dataset``."""
    expand_real_nodes = set(expand_real_nodes)
    agg = agg_result.aggregate_dataset
    node_to_group = agg_result.node_to_group
    group_to_members = agg_result.group_to_members

    groups_to_expand = {node_to_group[n] for n in expand_real_nodes if n in node_to_group}
    expanded_members = {m for g in groups_to_expand for m in group_to_members[g]}

    def rebuild_tier(agg_tier_nodes: list[str], dataset_tier_nodes: set[str]) -> list[str]:
        kept = [n for n in agg_tier_nodes if n not in groups_to_expand]
        this_tier_members = sorted(m for m in expanded_members if m in dataset_tier_nodes)
        return kept + this_tier_members

    tier1 = rebuild_tier(agg["tier1"], set(dataset["tier1"]))
    tier2 = rebuild_tier(agg["tier2"], set(dataset["tier2"]))
    tier3 = rebuild_tier(agg["tier3"], set(dataset["tier3"]))

    def merge_field(field_name: str) -> dict:
        merged = {k: v for k, v in agg.get(field_name, {}).items() if k not in groups_to_expand}
        for m in expanded_members:
            if m in dataset.get(field_name, {}):
                merged[m] = dataset[field_name][m]
        return merged

    fields: dict[str, dict] = {}
    for field_name in (
        "s", "c", "supplier_material_type", "production_delay",
        "unit_cost", "holding_cost", "emissions_factor",
        "criticality", "region", "company_name",
    ):
        if field_name in agg or field_name in dataset:
            fields[field_name] = merge_field(field_name)

    edges = [e for e in agg["edges"] if e[0] not in groups_to_expand and e[1] not in groups_to_expand]
    edges += [e for e in dataset["edges"] if e[0] in expanded_members or e[1] in expanded_members]
    edges = sorted(set(edges))

    if "transport_emissions" in agg or "transport_emissions" in dataset:
        te = {
            k: v for k, v in agg.get("transport_emissions", {}).items()
            if k[0] not in groups_to_expand and k[1] not in groups_to_expand
        }
        default_te = sum(dataset.get("transport_emissions", {}).values()) / max(
            len(dataset.get("transport_emissions", {})), 1
        )
        for e in edges:
            if e not in te:
                te[e] = dataset.get("transport_emissions", {}).get(e, default_te)
        fields["transport_emissions"] = te

    N_minus, N_plus, P, r = derive_indexes(tier1, tier2, tier3, edges, fields["supplier_material_type"])

    partial_dataset = {
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "edges": edges,
        "material_types": list(dataset["material_types"]),
        "supplier_material_type": fields["supplier_material_type"],
        "f": dict(agg["f"]),
        "s": fields["s"],
        "d": dict(agg["d"]),
        "c": fields["c"],
        "r": r,
        "N_minus": N_minus,
        "N_plus": N_plus,
        "P": P,
    }
    for field_name in (
        "production_delay", "unit_cost", "holding_cost", "emissions_factor",
        "criticality", "region", "company_name", "transport_emissions",
    ):
        if field_name in fields:
            partial_dataset[field_name] = fields[field_name]

    return partial_dataset


def run_aggregate_then_disaggregate(
    dataset: dict,
    disruptions: list,
    n_periods: int,
    period_length_days: float = 7.0,
    expansion_hops: int | None = 1,
    aggregate_tiers: tuple[int, ...] = (3,),
    d_t: dict | None = None,
    return_model: bool = False,
) -> dict:
    """Orchestrate the full aggregate-then-disaggregate workflow: build the
    aggregate network, solve it (translating each disruption's node IDs to
    their group IDs, since the aggregate network no longer has the original
    node), compute the disaggregation scope, rebuild a mixed-fidelity
    dataset, and re-solve that at full fidelity for the affected subtree.
    Returns both result DataFrames, the ``AggregationResult``, the computed
    scope, wall-clock timings for each stage, and tier-3 node counts for
    each of the three network variants (full / aggregate / partial)."""
    from scripts.multi_period_planning import build_and_solve_time_phased_ttr

    t0 = time.perf_counter()
    agg_result = build_aggregate_dataset(dataset, aggregate_tiers=aggregate_tiers)
    t1 = time.perf_counter()

    translated_disruptions = [
        dataclasses.replace(
            d, disrupted_nodes=sorted({agg_result.node_to_group.get(n, n) for n in d.disrupted_nodes})
        )
        for d in disruptions
    ]
    aggregate_result = build_and_solve_time_phased_ttr(
        agg_result.aggregate_dataset, translated_disruptions, n_periods,
        period_length_days, d_t=d_t, return_model=return_model,
    )
    t2 = time.perf_counter()

    scope: set[str] = set()
    for d in disruptions:
        scope |= disaggregation_scope(
            dataset, agg_result.node_to_group, d.disrupted_nodes, expansion_hops=expansion_hops
        )
    partial_dataset = build_partial_disaggregation(dataset, agg_result, scope)
    t3 = time.perf_counter()

    partial_result = build_and_solve_time_phased_ttr(
        partial_dataset, disruptions, n_periods, period_length_days, d_t=d_t, return_model=return_model,
    )
    t4 = time.perf_counter()

    return {
        "aggregate_result": aggregate_result,
        "partial_result": partial_result,
        "agg_result": agg_result,
        "partial_dataset": partial_dataset,
        "disaggregation_scope": scope,
        "timings": {
            "build_aggregate_s": t1 - t0,
            "solve_aggregate_s": t2 - t1,
            "build_partial_s": t3 - t2,
            "solve_partial_s": t4 - t3,
            "total_s": t4 - t0,
        },
        "node_counts": {
            "full_tier3": len(dataset["tier3"]),
            "aggregate_tier3": len(agg_result.aggregate_dataset["tier3"]),
            "partial_tier3": len(partial_dataset["tier3"]),
        },
    }

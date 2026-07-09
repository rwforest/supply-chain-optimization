"""Translate a generated MPS supply-chain dataset (the tier1/tier2/tier3
network produced by ``scripts.realistic_topologies.generate_complex_network("mps")``)
into the concrete inputs for the three operations-research models in the
`09_mps_cuopt_pipeline` notebook:

* **FJSP** (Flexible Job-Shop Scheduling) — back-end assembly/test scheduling
  across MPS's Chengdu wafer-sort/final-test facility and its OSAT partners.
* **CVRPTW** (Capacitated Vehicle Routing with Time Windows) — distribution
  of finished power ICs from Chengdu/Penang hubs to customer destinations.
* **MEIO** (Multi-Echelon Inventory Optimization) — safety-stock placement
  across the whole tier1/tier2/tier3 network.

This module is the single source of truth for *deriving* those problems from
the network, per the design decision that FJSP/CVRPTW/MEIO should be layered
on top of the generated dataset rather than invented from scratch. Every
function here takes and returns plain Python dicts/lists and imports NO
``cuopt``/``cudf``/``pyomo`` — so it is fully unit-testable without a GPU or a
solver installed. The actual model construction/solve lives in
``scripts.fjsp_cuopt``, ``scripts.cvrptw_cuopt``, and ``scripts.meio_pyomo``.

All numeric figures produced here are synthetic/illustrative, consistent with
the rest of the accelerator (see the disclaimer in
``scripts/company_profiles.py``). Nothing here represents real MPS operational
data.
"""

from __future__ import annotations

import math
import random

from scripts.company_profiles import COMPANY_PROFILES

# The real, named MPS tier-2 back-end anchors from company_profiles. Synthetic
# "rest of market" peer nodes ALSO carry a (fictional) ``company_name``, so a
# plain "has a company_name" test does not exclude them — at planet scale the
# OSAT anchor's synthetic peers (all tagged ``advanced_packaging_osat``) would
# otherwise balloon the FJSP machine set from ~11 to ~600, exploding the MILP.
# Keying off this exact anchor-name set keeps the derived machine/depot set
# scale-invariant (the FJSP models MPS's OWN back-end facilities, not the
# entire synthetic packaging market). ``company_profiles`` is pure data — no
# solver/GPU import is pulled in here.
_MPS_TIER2_ANCHOR_NAMES = {a.company_name for a in COMPANY_PROFILES["mps"]["tier2"]}

# Material-type tags (set on MPS's tier-2 anchors in
# ``scripts/company_profiles.py``) that identify back-end assembly/test nodes.
WAFER_SORT_FINAL_TEST = "wafer_sort_final_test"
ADVANCED_PACKAGING_OSAT = "advanced_packaging_osat"
ENGINEERING_OPS = "engineering_ops_support"

# The fixed back-end operation sequence every lot flows through. "flexible" in
# FJSP means each operation may run on any machine in an eligible subset; here
# packaging steps run on OSAT (or Chengdu) machines and electrical_test is
# restricted to wafer-sort/final-test-capable machines (Chengdu + a flagged
# OSAT subset), mirroring MPS owning its own test step.
FJSP_OPERATIONS: list[str] = [
    "die_attach",
    "wire_bond",
    "encapsulation",
    "electrical_test",
]

# Which machine groups are eligible for each operation. Packaging operations
# run on OSAT/Chengdu packaging machines; electrical_test runs only on
# test-capable machines.
_PACKAGING_GROUPS = {ADVANCED_PACKAGING_OSAT, WAFER_SORT_FINAL_TEST}
_TEST_GROUPS = {WAFER_SORT_FINAL_TEST}


def _anchor_tier2_by_material(dataset: dict, material_type: str) -> list[str]:
    """Return the tier-2 node IDs whose material_type matches, restricted to
    the *real named MPS anchors* (``company_name`` in
    ``_MPS_TIER2_ANCHOR_NAMES``) so we key off MPS's own back-end facilities
    rather than the synthetic "rest of market" fan-out — which also carries a
    fictional ``company_name`` and would otherwise inflate the set at scale."""
    smt = dataset["supplier_material_type"]
    names = dataset.get("company_name", {})
    return [
        node
        for node in dataset["tier2"]
        if smt.get(node) == material_type
        and names.get(node) in _MPS_TIER2_ANCHOR_NAMES
    ]


def select_backend_machines(dataset: dict) -> list[dict]:
    """Build the FJSP machine list from MPS's back-end tier-2 anchors.

    Each anchor facility (Chengdu, the OSAT partners) is split into a small
    number of parallel sub-machines sized by that node's capacity (``c``) so
    the schedule has real parallelism to exploit. Chengdu and OSAT machines
    are ``test_capable`` per the network's material tags; packaging-only
    machines are not.

    Returns a list of machine dicts:
        {"machine_id", "facility", "group", "region", "test_capable",
         "throughput"}
    """
    c = dataset["c"]
    region = dataset.get("region", {})
    machines: list[dict] = []

    def _emit(node: str, group: str, n_sub: int) -> None:
        node_cap = max(1, int(c.get(node, 1)))
        per_machine = max(1, node_cap // n_sub)
        for k in range(n_sub):
            machines.append(
                {
                    "machine_id": f"{node}__M{k}",
                    "facility": node,
                    "group": group,
                    "region": region.get(node, "Unknown"),
                    "test_capable": group in _TEST_GROUPS,
                    "throughput": per_machine,
                }
            )

    # Chengdu wafer-sort/final-test: fewer, high-value test cells.
    for node in _anchor_tier2_by_material(dataset, WAFER_SORT_FINAL_TEST):
        _emit(node, WAFER_SORT_FINAL_TEST, n_sub=3)
    # OSAT packaging lines: more parallel packaging machines.
    for node in _anchor_tier2_by_material(dataset, ADVANCED_PACKAGING_OSAT):
        _emit(node, ADVANCED_PACKAGING_OSAT, n_sub=4)

    return machines


def eligible_machines(operation: str, machines: list[dict]) -> list[str]:
    """Machine IDs physically capable of processing ``operation``."""
    if operation == "electrical_test":
        return [m["machine_id"] for m in machines if m["test_capable"]]
    # die_attach / wire_bond / encapsulation run on any packaging-capable machine
    return [m["machine_id"] for m in machines if m["group"] in _PACKAGING_GROUPS]


def derive_fjsp_jobs(
    dataset: dict, machines: list[dict], lot_size: int = 500, max_jobs: int = 40
) -> list[dict]:
    """Convert tier-1 product-line demand (``d``) into a set of production
    lots (jobs), each with the fixed 4-operation sequence.

    The number of lots per product line is ``ceil(demand / lot_size)``, capped
    across all lines at ``max_jobs`` so the FJSP stays a tractable, notebook-
    scale demo (real cuOpt instances go far larger). Each job records its
    source tier-1 node and product line for interpretation.
    """
    d = dataset["d"]
    product_line = dataset.get("product_line", {})
    tier1 = dataset["tier1"]

    # Lots per product line, proportional to demand.
    raw_counts = {
        node: max(1, math.ceil(d.get(node, 0) / lot_size)) for node in tier1
    }
    total = sum(raw_counts.values())
    if total > max_jobs:
        # Scale down proportionally but keep at least 1 lot per line.
        scale = max_jobs / total
        raw_counts = {
            node: max(1, int(round(cnt * scale))) for node, cnt in raw_counts.items()
        }

    jobs: list[dict] = []
    job_idx = 0
    for node in tier1:
        for _lot in range(raw_counts[node]):
            if job_idx >= max_jobs:
                break
            jobs.append(
                {
                    "job_id": f"J{job_idx}",
                    "source_tier1": node,
                    "product_line": product_line.get(node, node),
                    "operations": list(FJSP_OPERATIONS),
                }
            )
            job_idx += 1
    return jobs


def derive_fjsp_processing_times(
    jobs: list[dict], machines: list[dict], dataset: dict, seed: int = 11
) -> dict[tuple[str, str, str], float]:
    """Per-(job, operation, machine) processing times.

    The base time for an operation on a machine is anchored on the machine's
    facility ``production_delay`` (from ``calibrate_cost_fields``, if present;
    otherwise a small default) and jittered per job so lots differ. Only
    eligible (job, operation, machine) triples get an entry.
    """
    rng = random.Random(seed)
    production_delay = dataset.get("production_delay", {})
    machine_by_id = {m["machine_id"]: m for m in machines}

    # Per-operation multipliers so the sequence has realistic relative cost:
    # test is the slowest/most valuable step, die_attach the quickest.
    op_multiplier = {
        "die_attach": 0.6,
        "wire_bond": 1.0,
        "encapsulation": 0.8,
        "electrical_test": 1.4,
    }

    times: dict[tuple[str, str, str], float] = {}
    for job in jobs:
        for op in job["operations"]:
            for mid in eligible_machines(op, machines):
                facility = machine_by_id[mid]["facility"]
                base = production_delay.get(facility, 8.0)
                mult = op_multiplier.get(op, 1.0)
                jitter = rng.uniform(0.85, 1.15)
                times[(job["job_id"], op, mid)] = round(base * mult * jitter, 3)
    return times


def select_cvrptw_depots(dataset: dict) -> list[dict]:
    """Depots = MPS's Chengdu and Penang facilities (the finished-goods
    dispatch hubs). Restricted to the real named MPS anchors so a synthetic
    peer (e.g. one of Penang's small fan-out) can't be mistaken for a depot —
    keeping the depot set scale-invariant like the FJSP machine set."""
    smt = dataset["supplier_material_type"]
    names = dataset.get("company_name", {})
    region = dataset.get("region", {})
    depots: list[dict] = []
    for node in dataset["tier2"]:
        if names.get(node) not in _MPS_TIER2_ANCHOR_NAMES:
            continue
        if smt.get(node) in (WAFER_SORT_FINAL_TEST, ENGINEERING_OPS):
            depots.append(
                {
                    "depot_id": node,
                    "name": names[node],
                    "region": region.get(node, "Unknown"),
                }
            )
    return depots


def derive_cvrptw_customers(dataset: dict, seed: int = 23) -> list[dict]:
    """One synthetic customer per tier-1 product line, tagged with an
    illustrative OEM-destination region and a demand volume derived from that
    line's ``d``. Regions are drawn from the network's own region set so the
    distance matrix has entries for them."""
    rng = random.Random(seed)
    d = dataset["d"]
    product_line = dataset.get("product_line", {})
    regions = sorted(set(dataset.get("region", {}).values())) or ["USA"]

    customers: list[dict] = []
    for i, node in enumerate(dataset["tier1"]):
        customers.append(
            {
                "customer_id": f"C{i}",
                "source_tier1": node,
                "product_line": product_line.get(node, node),
                "region": rng.choice(regions),
                "demand": max(1, int(d.get(node, 1))),
            }
        )
    return customers


def _dominant_criticality_for_tier1(dataset: dict, tier1_node: str) -> str:
    """The 'worst' criticality among the suppliers that feed a tier-1 node,
    used to tighten/loosen that customer's delivery window."""
    order = {"monopoly_bottleneck": 3, "oligopoly": 2, "diversified_commodity": 1, "generic": 0}
    criticality = dataset.get("criticality", {})
    p_index = dataset.get("P", {})
    worst = "generic"
    for _material, suppliers in p_index.get(tier1_node, {}).items():
        for s in suppliers:
            crit = criticality.get(s, "generic")
            if order.get(crit, 0) > order.get(worst, 0):
                worst = crit
    return worst


def derive_cvrptw_time_windows(
    customers: list[dict], dataset: dict, horizon: int = 480
) -> dict[str, tuple[int, int]]:
    """Delivery time windows (in minutes, over a ``horizon``-minute shift).

    Customers fed predominantly by ``monopoly_bottleneck`` suppliers get
    tighter windows (their supply is fragile, so delivery timing is critical);
    ``diversified_commodity``-fed customers get looser windows.
    """
    width_by_crit = {
        "monopoly_bottleneck": 60,
        "oligopoly": 120,
        "diversified_commodity": 240,
        "generic": 180,
    }
    windows: dict[str, tuple[int, int]] = {}
    n = max(1, len(customers))
    for i, cust in enumerate(customers):
        crit = _dominant_criticality_for_tier1(dataset, cust["source_tier1"])
        width = width_by_crit.get(crit, 180)
        # Stagger window starts across the shift so they don't all overlap.
        start = int((i / n) * max(1, horizon - width))
        windows[cust["customer_id"]] = (start, min(horizon, start + width))
    return windows


# Illustrative inter-region travel times (minutes) for the synthetic distance/
# time matrix. Not real geography — a coarse "same region is fast, cross-
# continent is slow" lookup so the CVRPTW has structure.
_REGION_TRAVEL_MINUTES = {
    ("China", "China"): 30,
    ("Malaysia", "Malaysia"): 30,
    ("Taiwan", "Taiwan"): 30,
    ("China", "Malaysia"): 90,
    ("China", "Taiwan"): 75,
    ("China", "South Korea"): 90,
    ("China", "USA"): 240,
    ("China", "Japan"): 100,
    ("Malaysia", "Taiwan"): 100,
    ("Malaysia", "USA"): 260,
    ("Taiwan", "USA"): 230,
    ("Taiwan", "Japan"): 80,
    ("South Korea", "USA"): 220,
    ("USA", "USA"): 40,
}


def _region_travel_minutes(a: str, b: str) -> int:
    if a == b:
        return 30
    return _REGION_TRAVEL_MINUTES.get((a, b)) or _REGION_TRAVEL_MINUTES.get((b, a)) or 180


def derive_cvrptw_distance_time_matrix(
    depots: list[dict], customers: list[dict]
) -> dict:
    """Build a synthetic distance/time matrix over depot(s) + customers using
    an inter-region lookup (no real geocoding). Returns:

        {"locations": [ids...], "index": {id: i}, "time": [[...]],
         "distance": [[...]]}

    with location 0 being the (first) depot, the routing convention cuOpt
    expects. ``distance`` is just ``time`` scaled to synthetic km.
    """
    locations = [d["depot_id"] for d in depots] + [c["customer_id"] for c in customers]
    region_of = {d["depot_id"]: d["region"] for d in depots}
    region_of.update({c["customer_id"]: c["region"] for c in customers})
    index = {loc: i for i, loc in enumerate(locations)}

    n = len(locations)
    time = [[0] * n for _ in range(n)]
    distance = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(locations):
        for j, b in enumerate(locations):
            if i == j:
                continue
            t = _region_travel_minutes(region_of[a], region_of[b])
            time[i][j] = t
            distance[i][j] = round(t * 1.5, 1)  # synthetic km ~ 1.5 km/min
    return {"locations": locations, "index": index, "time": time, "distance": distance}


# Service-level z-factor (inverse-normal CDF) keyed off criticality: a fragile,
# sole-sourced node must hold safety stock to a higher service level.
_SERVICE_Z = {
    "monopoly_bottleneck": 2.326,  # ~99%
    "oligopoly": 1.645,  # ~95%
    "diversified_commodity": 1.282,  # ~90%
    "generic": 1.645,
}

# Fraction of a node's mean demand used as its demand standard deviation. A
# documented constant (real MEIO would estimate this from history); kept
# modest so safety-stock terms stay sane.
DEMAND_STD_FRACTION = 0.30


def derive_meio_network(
    dataset: dict,
    max_nodes: int | None = None,
    sample_fraction: float = 1.0,
    seed: int = 41,
    always_include_criticality: tuple[str, ...] = ("monopoly_bottleneck", "oligopoly"),
) -> dict:
    """Assemble the MEIO (Guaranteed Service Model) network from the
    tier1∪tier2∪tier3 dataset.

    Returns a dict with:
        nodes:        list of node ids
        edges:        list of (upstream, downstream) supplier->consumer pairs
        holding_cost: {node: unit holding cost}    (from calibrate_cost_fields)
        lead_time:    {node: processing/production lead time}  (production_delay)
        demand_mean:  {node: mean demand}
        demand_std:   {node: demand std}
        service_z:    {node: safety factor z}
        max_service_time: {tier1_node: max quoted service time}

    Demand is propagated the same direction the LP uses: tier-1 nodes carry
    the finished-goods demand ``d``; upstream nodes inherit demand from the
    tier-1 lines they (transitively) feed via a simple sum over edges.

    **Scaling.** At the ``"planet"`` preset (~76k nodes) the piecewise-linear
    GSM MILP has an INC block (binaries + constraints) per node, so an
    unbounded solve is intractable in HiGHS. ``max_nodes`` / ``sample_fraction``
    subsample the network before building the model, mirroring
    ``scripts.disruption_scenarios.single_supplier_failure_scenarios``:

    * All tier-1 nodes, all *named real anchors* (``company_name`` in
      ``_MPS_TIER2_ANCHOR_NAMES``), and all nodes whose ``criticality`` is in
      ``always_include_criticality`` are ALWAYS kept — these are the fragile,
      capital-intensive nodes where safety-stock placement actually matters.
    * The remaining (commodity) nodes are randomly sampled via a seeded rng.
    * ``max_nodes`` (if set) caps the total kept-node count; it takes
      precedence over ``sample_fraction`` by deriving the fraction needed to
      hit the cap. ``sample_fraction=1.0`` with ``max_nodes=None`` (the
      defaults) reproduces today's full-network behavior exactly.

    Demand is propagated over the FULL network first, then the kept nodes
    retain their true propagated demand; only edges with both endpoints kept
    are retained (a node's upstream/downstream that got sampled out simply
    drops from the GSM service-time coupling — an accepted approximation at
    planet scale, flagged in the summary print).
    """
    tier1 = dataset["tier1"]
    tier2 = dataset["tier2"]
    tier3 = dataset["tier3"]
    nodes = list(tier1) + list(tier2) + list(tier3)
    node_set = set(nodes)

    holding_cost = dataset.get("holding_cost", {})
    production_delay = dataset.get("production_delay", {})
    criticality = dataset.get("criticality", {})
    names = dataset.get("company_name", {})
    d = dataset["d"]

    # Edges as (upstream supplier, downstream consumer). The dataset's
    # ``edges`` are already (src=supplier, tgt=consumer).
    edges = [
        (src, tgt)
        for src, tgt in dataset["edges"]
        if src in node_set and tgt in node_set
    ]

    # Demand mean per node, propagated over the FULL network so kept nodes keep
    # accurate demand: tier-1 uses d directly; upstream nodes get the sum of
    # the demand of the immediate downstream nodes they supply (a coarse,
    # documented propagation — not the LP's exact BOM ratios).
    demand_mean: dict[str, float] = dict.fromkeys(nodes, 0.0)
    for n in tier1:
        demand_mean[n] = float(d.get(n, 0.0))
    downstream_of: dict[str, list[str]] = {n: [] for n in nodes}
    for src, tgt in edges:
        downstream_of[src].append(tgt)
    for n in tier2:
        demand_mean[n] = sum(demand_mean.get(c, 0.0) for c in downstream_of[n]) or 1.0
    for n in tier3:
        demand_mean[n] = sum(demand_mean.get(c, 0.0) for c in downstream_of[n]) or 1.0

    # --- Optional subsampling for tractability at planet scale ---------------
    def _always_keep(n: str) -> bool:
        return (
            n in dataset["tier1"]  # never drop finished-goods nodes
            or names.get(n) in _MPS_TIER2_ANCHOR_NAMES  # real named anchors
            or criticality.get(n) in always_include_criticality
        )

    if max_nodes is not None or sample_fraction < 1.0:
        rng = random.Random(seed)
        always = [n for n in nodes if _always_keep(n)]
        rest = [n for n in nodes if not _always_keep(n)]

        if max_nodes is not None:
            target_rest = max(0, max_nodes - len(always))
        else:
            target_rest = round(sample_fraction * len(rest))
        sampled = rng.sample(rest, min(target_rest, len(rest)))

        kept = set(always) | set(sampled)
        nodes = [n for n in nodes if n in kept]  # preserve tier order
        edges = [(s, t) for s, t in edges if s in kept and t in kept]
        print(
            f"[derive_meio_network] planet-scale subsample: kept {len(nodes)} nodes "
            f"({len(always)} always-kept: tier-1 + named anchors + "
            f"{'/'.join(always_include_criticality)}; {len(sampled)}/{len(rest)} "
            f"commodity nodes sampled), {len(edges)} edges retained"
        )

    demand_std = {n: round(demand_mean[n] * DEMAND_STD_FRACTION, 3) for n in nodes}
    service_z = {n: _SERVICE_Z.get(criticality.get(n, "generic"), 1.645) for n in nodes}
    lead_time = {n: float(production_delay.get(n, 7.0)) for n in nodes}
    hcost = {n: float(holding_cost.get(n, 1.0)) for n in nodes}
    demand_mean = {n: demand_mean[n] for n in nodes}

    # Max service time quoted to the end customer at each tier-1 node (all
    # tier-1 nodes are always kept, so this is unaffected by sampling).
    max_service_time = dict.fromkeys(tier1, 7.0)

    return {
        "nodes": nodes,
        "edges": edges,
        "holding_cost": hcost,
        "lead_time": lead_time,
        "demand_mean": demand_mean,
        "demand_std": demand_std,
        "service_z": service_z,
        "max_service_time": max_service_time,
    }

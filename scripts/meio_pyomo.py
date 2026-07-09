"""Multi-Echelon Inventory Optimization (MEIO) via the Guaranteed Service
Model (GSM), solved on CPU with Pyomo + HiGHS.

This is Scenario I of the MPS pipeline (see
`MPS Supply Chain Optimization Pipeline.md` §5). Unlike the FJSP and CVRPTW
scenarios — which run on NVIDIA cuOpt's GPU solvers — MEIO stays CPU-only by
design: its exact objective has a square-root nonlinearity

    total safety-stock cost = Σ_i  h_i · z_i · σ_i · sqrt(NLT_i)

(``h`` = unit holding cost, ``z`` = service-level safety factor, ``σ`` =
demand std, ``NLT`` = net lead time at node ``i``) that the repo's existing
HiGHS (LP/MILP-only) solver can't handle natively.

``mode="piecewise_linear"`` (the default) approximates ``sqrt(NLT_i)`` with a
piecewise-linear interpolant via Pyomo's ``Piecewise`` component (SOS2
convex-combination), so the model stays solvable by the existing Pyomo/HiGHS
stack (as a small MILP — HiGHS handles the SOS2 branching) with no new
dependency. ``mode="nonlinear"`` is an opt-in stub that would require a
nonlinear solver (e.g. Ipopt via ``cyipopt``, a future ``nlp`` extras group)
and is intentionally NOT built out here.

Why SOS2 and not a plain LP relaxation: ``sqrt`` is *concave* and the
objective *minimizes* the safety-stock cost, so minimizing ``c·sqrt(NLT)`` is
a non-convex minimization. A plain convex-combination (lambda) relaxation lets
a minimizing solver put weight on the two *extreme* breakpoints — the global
chord, which badly under-estimates a concave curve — while pure tangent-cut
outer approximation over-estimates. The SOS2 restriction (at most two
*adjacent* lambdas nonzero) forces the approximation onto the actual
piecewise-linear interpolant of ``sqrt``, giving an error bounded by the
breakpoint spacing. Breakpoints span ``[0, nlt_ub]`` where ``nlt_ub`` is the
DAG longest-path lead time (a tight per-node net-lead-time bound), so they
stay dense over the range each node can actually occupy. The notebook and
tests sanity-check the gap against the true-sqrt objective.

The network is produced by ``scripts.mps_derivation.derive_meio_network`` —
this module never imports ``mps_derivation`` (it just consumes the dict), so
the two are independently testable.
"""

from __future__ import annotations

import math

# Breakpoints (in net-lead-time units, e.g. days) at which sqrt is sampled for
# the piecewise-linear approximation. Denser near 0 where sqrt curves most.
DEFAULT_NLT_BREAKPOINTS: list[float] = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]


def sqrt_pwl_breakpoints(max_nlt: float, extra: list[float] | None = None) -> list[float]:
    """Return sqrt breakpoints covering ``[0, max_nlt]`` — the defaults plus a
    final point at ``max_nlt`` so the approximation spans the whole feasible
    net-lead-time range."""
    pts = [p for p in (extra or DEFAULT_NLT_BREAKPOINTS) if p <= max_nlt]
    if not pts or pts[0] != 0.0:
        pts = [0.0, *pts]
    if pts[-1] < max_nlt:
        pts.append(round(max_nlt, 6))
    return sorted(set(pts))


def _longest_path_lead(
    nodes: list[str], edges: list[tuple[str, str]], lead: dict
) -> float:
    """Longest cumulative lead time from any source to any node along the
    supplier->consumer DAG. Used as a tight upper bound on net lead time.

    Robust to cycles (should not occur in a well-formed DAG, but the generated
    network isn't guaranteed acyclic): a visited-set guard caps recursion so a
    stray cycle can't loop forever — it just stops extending that path.
    """
    suppliers_of: dict[str, list[str]] = {n: [] for n in nodes}
    for src, tgt in edges:
        if tgt in suppliers_of:
            suppliers_of[tgt].append(src)

    memo: dict[str, float] = {}

    def longest_to(node: str, stack: frozenset) -> float:
        if node in memo:
            return memo[node]
        if node in stack:  # cycle guard — don't extend through it
            return 0.0
        best = 0.0
        child_stack = stack | {node}
        for sup in suppliers_of.get(node, []):
            best = max(best, longest_to(sup, child_stack))
        val = best + float(lead.get(node, 0.0))
        # Only memoize when not inside a cycle-broken branch, so the cache
        # stays correct for the common acyclic case.
        if node not in stack:
            memo[node] = val
        return val

    return max((longest_to(n, frozenset()) for n in nodes), default=0.0)


def build_meio_model(network: dict, mode: str = "piecewise_linear"):
    """Build a Pyomo ``ConcreteModel`` for the GSM MEIO problem.

    ``network`` is the dict returned by
    ``scripts.mps_derivation.derive_meio_network`` with keys ``nodes``,
    ``edges``, ``holding_cost``, ``lead_time``, ``demand_std``, ``service_z``,
    and ``max_service_time``.

    Decision variables (all continuous, ≥ 0):
        S[i]   guaranteed (outbound) service time quoted by node i
        SI[i]  inbound service time promised to node i by its suppliers
        NLT[i] net lead time at node i = SI[i] + lead_time[i] - S[i]

    Objective: minimize Σ_i h_i · z_i · σ_i · sqrt(NLT[i]), with sqrt
    approximated by the piecewise-linear lambda formulation when
    ``mode="piecewise_linear"``.
    """
    if mode == "nonlinear":
        raise NotImplementedError(
            "mode='nonlinear' requires a nonlinear solver (e.g. Ipopt via a "
            "future 'nlp' extras group). Use mode='piecewise_linear' (default), "
            "which stays solvable by the repo's existing HiGHS stack."
        )
    if mode != "piecewise_linear":
        raise ValueError(f"unknown mode {mode!r}; expected 'piecewise_linear' or 'nonlinear'")

    import pyomo.environ as pyo

    nodes = list(network["nodes"])
    edges = list(network["edges"])
    h = network["holding_cost"]
    lead = network["lead_time"]
    z = network["service_z"]
    sigma = network["demand_std"]
    max_service = network.get("max_service_time", {})

    # Upstream suppliers of each node (edge = (supplier, consumer)).
    suppliers_of: dict[str, list[str]] = {n: [] for n in nodes}
    for src, tgt in edges:
        suppliers_of[tgt].append(src)

    # Upper bound on any service/net-lead time: the longest cumulative lead
    # time along any supplier->consumer path in the DAG. No node's inbound
    # service time (hence NLT) can exceed the deepest chain feeding it. This is
    # far tighter than sum-over-all-nodes, keeping the sqrt breakpoints dense
    # over the range NLT can actually occupy.
    nlt_ub = max(1.0, _longest_path_lead(nodes, edges, lead))

    m = pyo.ConcreteModel()
    m.NODES = pyo.Set(initialize=nodes, ordered=True)

    m.h = pyo.Param(m.NODES, initialize={n: float(h.get(n, 1.0)) for n in nodes})
    m.lead = pyo.Param(m.NODES, initialize={n: float(lead.get(n, 0.0)) for n in nodes})
    m.z = pyo.Param(m.NODES, initialize={n: float(z.get(n, 1.645)) for n in nodes})
    m.sigma = pyo.Param(m.NODES, initialize={n: float(sigma.get(n, 0.0)) for n in nodes})

    m.S = pyo.Var(m.NODES, domain=pyo.NonNegativeReals, bounds=(0, nlt_ub))
    m.SI = pyo.Var(m.NODES, domain=pyo.NonNegativeReals, bounds=(0, nlt_ub))
    m.NLT = pyo.Var(m.NODES, domain=pyo.NonNegativeReals, bounds=(0, nlt_ub))

    # Net lead time definition: NLT_i = SI_i + lead_i - S_i.
    def nlt_rule(mdl, i):
        return mdl.NLT[i] == mdl.SI[i] + mdl.lead[i] - mdl.S[i]
    m.NetLeadTime = pyo.Constraint(m.NODES, rule=nlt_rule)

    # Inbound service time: a node can't start until all its suppliers have
    # delivered, so SI_i >= S_j for every supplier j of i.
    def inbound_rule(mdl, i, j):
        return mdl.SI[i] >= mdl.S[j]
    inbound_pairs = [(i, j) for i in nodes for j in suppliers_of[i]]
    m.InboundIndex = pyo.Set(initialize=inbound_pairs, dimen=2)
    m.InboundService = pyo.Constraint(m.InboundIndex, rule=inbound_rule)

    # Downstream service-time cap at demand (tier-1) nodes.
    def max_service_rule(mdl, i):
        if i in max_service:
            return mdl.S[i] <= float(max_service[i])
        return pyo.Constraint.Skip
    m.MaxService = pyo.Constraint(m.NODES, rule=max_service_rule)

    # --- Piecewise-linear (SOS2) approximation of sqrt(NLT_i) ---------------
    # sqrtNLT[i] tracks the piecewise-linear interpolant of sqrt over the
    # breakpoints; Pyomo's Piecewise builds the SOS2 convex-combination that
    # ties sqrtNLT[i] to NLT[i] on the true interpolant (see module docstring
    # for why SOS2 adjacency is required for this concave-under-minimization).
    breakpoints = sqrt_pwl_breakpoints(nlt_ub)
    sqrt_values = [math.sqrt(b) for b in breakpoints]

    m.sqrtNLT = pyo.Var(m.NODES, domain=pyo.NonNegativeReals, bounds=(0, math.sqrt(nlt_ub)))

    # Use the incremental ("INC") representation rather than SOS2: it enforces
    # the same breakpoint adjacency using ordinary binary variables + linear
    # constraints, so it works through HiGHS's MILP interface (which does not
    # support native SOS constraints via Pyomo's persistent solver).
    m.SqrtPWL = pyo.Piecewise(
        m.NODES,
        m.sqrtNLT,
        m.NLT,
        pw_pts=breakpoints,
        f_rule=sqrt_values,
        pw_constr_type="EQ",
        pw_repn="INC",
    )

    def obj_rule(mdl):
        return sum(mdl.h[i] * mdl.z[i] * mdl.sigma[i] * mdl.sqrtNLT[i] for i in mdl.NODES)
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    m._meio_breakpoints = breakpoints  # stash for introspection/tests
    return m


def solve_meio(model, tee: bool = False) -> dict:
    """Solve a model from ``build_meio_model`` with HiGHS and return a summary
    dict: termination condition, total safety-stock cost, and per-node net
    lead time / guaranteed service time / (approx) safety stock."""
    import pyomo.environ as pyo

    solver = pyo.SolverFactory("highs")
    result = solver.solve(model, tee=tee)

    term = str(result.solver.termination_condition)
    per_node = {}
    for i in model.NODES:
        nlt = pyo.value(model.NLT[i])
        per_node[i] = {
            "net_lead_time": round(nlt, 4),
            "guaranteed_service_time": round(pyo.value(model.S[i]), 4),
            "safety_stock_cost": round(
                pyo.value(model.h[i] * model.z[i] * model.sigma[i] * model.sqrtNLT[i]), 4
            ),
        }
    return {
        "termination_condition": term,
        "total_safety_stock_cost": round(pyo.value(model.OBJ), 4),
        "per_node": per_node,
    }


def true_safety_stock_cost(network: dict, per_node: dict) -> float:
    """Recompute the EXACT (true-sqrt) total safety-stock cost from a solved
    solution's net-lead-times, for bounding the piecewise-linear approximation
    error against ``solve_meio``'s reported ``total_safety_stock_cost``."""
    h = network["holding_cost"]
    z = network["service_z"]
    sigma = network["demand_std"]
    total = 0.0
    for n, info in per_node.items():
        total += (
            float(h.get(n, 1.0))
            * float(z.get(n, 1.645))
            * float(sigma.get(n, 0.0))
            * math.sqrt(max(0.0, info["net_lead_time"]))
        )
    return round(total, 4)

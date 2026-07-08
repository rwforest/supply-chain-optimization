"""Time-phased, multi-period supply-chain planning — modeled after Adexa's
biggest structural advantage over a single-snapshot LP: real time buckets,
lead-time-offset material flow, and inventory that carries over from one
period to the next, instead of one fixed-horizon "how much is lost over `t`
time units" snapshot.

``scripts.utils.build_and_solve_ttr``/``build_and_solve_tts`` (and the 8
objective variants alongside them) are a **single-snapshot** LP: one scalar
horizon ``t``, a disruption that either is or isn't active for the whole
horizon, and no notion of a shipment departing now and arriving later. This
module is a **separate, additive** capability that sits alongside them —
``scripts/utils.py`` is never imported for anything but its pure,
already-tested ``_prep_lp_data`` helper, and is not modified.

Only one objective (the TTR-style "minimize lost profit") is implemented in
time-phased form here — enough to prove the temporal mechanics (lead-time
offset, inventory carryover, per-period disruption windows) are correct. The
same Sets/Params/Vars/Objective/Constraints skeleton generalizes to the other
8 objectives in ``scripts/utils.py`` if ever needed; that generalization is
out of scope for this module.

Known limitations (see README for the full discussion):

- **Cold-start pipeline**: material in transit *before* the horizon starts is
  assumed zero unless ``initial_pipeline`` is supplied, which likely
  understates early-period throughput for any node with ``production_delay >
  0``. A real deployment would seed this from actual open purchase orders.
- **End-of-horizon effect**: a shipment that departs in the last
  ``offset[i]`` periods of the horizon never arrives in time to be counted —
  a standard rolling-horizon artifact. A full rolling re-solve (re-optimizing
  a sliding window and only committing the first period's decisions) is out
  of scope here.
- **MIP scale growth**: every variable is now indexed by period, so model
  size grows linearly with ``n_periods``. Keep demonstrations at simple/
  medium/complex network scale, not planet-scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

import scripts.utils as utils
from scripts.disruption_scenarios import DisruptionScenario, ScenarioType


@dataclass(frozen=True)
class TimePhasedDisruption:
    """Like ``scripts.disruption_scenarios.DisruptionScenario``, but with a
    start and duration on a discrete period axis instead of a single scalar
    ``ttr`` — a disruption is now something that begins at a specific period
    and lasts a specific number of periods, which is what actually happens in
    a real supply chain (a plant fire doesn't retroactively affect last
    week's shipments)."""

    scenario_id: str
    name: str
    description: str
    real_world_basis: str
    scenario_type: ScenarioType
    disrupted_nodes: list[str]
    period_start: int
    duration_periods: int

    def periods(self) -> set[int]:
        return set(range(self.period_start, self.period_start + self.duration_periods))


def from_legacy_scenario(
    scenario: DisruptionScenario, period_length_days: float, period_start: int = 0
) -> TimePhasedDisruption:
    """Convert an existing scalar-``ttr`` ``DisruptionScenario`` into a
    ``TimePhasedDisruption`` — ``duration_periods`` is the number of periods
    needed to cover ``scenario.ttr`` days, rounded up so the disruption's
    time-phased footprint is never shorter than the original TTR implies."""
    duration_periods = max(1, math.ceil(scenario.ttr / period_length_days))
    return TimePhasedDisruption(
        scenario_id=scenario.scenario_id,
        name=scenario.name,
        description=scenario.description,
        real_world_basis=scenario.real_world_basis,
        scenario_type=scenario.scenario_type,
        disrupted_nodes=list(scenario.disrupted_nodes),
        period_start=period_start,
        duration_periods=duration_periods,
    )


def default_periodic_demand(
    dataset: dict,
    n_periods: int,
    period_length_days: float = 7.0,
    seasonality: dict[str, list[float]] | None = None,
) -> dict[str, dict[int, float]]:
    """Flat-repeat ``dataset["d"][j] * period_length_days`` (tier-1 demand per
    time unit, scaled to one period's worth of days) across every period —
    by default, total demand over the horizon matches what a single-snapshot
    ``build_and_solve_ttr`` run with ``t = n_periods * period_length_days``
    would see, so the two engines are comparable at baseline.

    ``seasonality``, if given (e.g. from
    ``scripts.scenario_calibration.sample_seasonality_index``), is a
    per-node list of period multipliers (mean ~1.0) applied on top of the
    flat baseline, cycling if shorter than ``n_periods``.

    Values are rounded to the nearest integer: ``l``/``s`` are integer-domain
    variables in ``build_and_solve_time_phased_ttr``, so a fractional
    ``d_t[j,t]`` (which a non-integer seasonality multiplier would otherwise
    produce) makes the per-period flow-balance equation infeasible.
    """
    d = dataset["d"]
    d_t: dict[str, dict[int, float]] = {}
    for j in dataset["tier1"]:
        base = d[j] * period_length_days
        multipliers = seasonality.get(j) if seasonality else None
        if multipliers:
            d_t[j] = {t: round(base * multipliers[t % len(multipliers)]) for t in range(n_periods)}
        else:
            d_t[j] = {t: base for t in range(n_periods)}
    return d_t


def _prep_time_phased_data(
    dataset: dict,
    disruptions: list[TimePhasedDisruption],
    n_periods: int,
    period_length_days: float,
    d_t: dict[str, dict[int, float]] | None = None,
) -> dict:
    """New prep helper for the time-phased engine — does NOT replace
    ``scripts.utils._prep_lp_data``. Reuses it (read-only) for the static,
    non-temporal parts of the schema (``V/D/U/K/N_minus/N_plus/P/f/r``), and
    adds everything period-related on top: the period index ``T``, each
    upstream node's lead-time ``offset`` in periods, the per-node/period
    demand table ``d_t``, and the per-period disrupted-node sets ``S_t``."""
    base = utils._prep_lp_data(dataset, disrupted=[])
    T = list(range(n_periods))

    production_delay = dataset.get("production_delay", {})
    offset = {
        i: math.ceil(production_delay.get(i, 0.0) / period_length_days)
        if production_delay.get(i, 0.0) > 0
        else 0
        for i in base["U"]
    }

    if d_t is None:
        d_t = default_periodic_demand(dataset, n_periods, period_length_days)
    d_t_flat = {(j, t): d_t[j][t] for j in base["V"] for t in T}

    S_t: dict[int, set[str]] = {t: set() for t in T}
    for disruption in disruptions:
        for t in disruption.periods():
            if t in S_t:
                S_t[t].update(disruption.disrupted_nodes)

    nodes = base["V"] + base["U"]
    s0 = {i: dataset["s"][i] for i in nodes}
    c = {i: dataset["c"][i] for i in nodes}

    return {
        "V": base["V"],
        "D": base["D"],
        "U": base["U"],
        "K": base["K"],
        "N_minus": base["N_minus"],
        "N_plus": base["N_plus"],
        "P": base["P"],
        "f": base["f"],
        "r": base["r"],
        "c": c,
        "s0": s0,
        "T": T,
        "offset": offset,
        "d_t": d_t,
        "d_t_flat": d_t_flat,
        "S_t": S_t,
    }


def build_and_solve_time_phased_ttr(
    dataset: dict,
    disruptions: list[TimePhasedDisruption],
    n_periods: int,
    period_length_days: float = 7.0,
    d_t: dict[str, dict[int, float]] | None = None,
    initial_pipeline: dict[tuple[str, str], float] | None = None,
    return_model: bool = False,
) -> pd.DataFrame:
    """Time-phased analogue of ``scripts.utils.build_and_solve_ttr``: same
    Sets -> Params -> Vars -> Objective -> Constraints -> solve skeleton,
    with a period index ``t`` threaded through every variable.

    ``initial_pipeline`` supplies the quantity assumed already in transit
    from ``i`` to ``j`` for any shipment whose departure period would fall
    before period 0 (i.e. it departed before the horizon started); it
    defaults to 0 for every ``(i, j)`` pair not given explicitly, which is a
    cold-start simplification documented in this module's docstring.
    """
    import pyomo.environ as pyo

    initial_pipeline = initial_pipeline or {}
    data = _prep_time_phased_data(dataset, disruptions, n_periods, period_length_days, d_t)

    m = pyo.ConcreteModel()

    # Sets
    m.V = pyo.Set(initialize=data["V"])
    m.D = pyo.Set(initialize=data["D"])
    m.U = pyo.Set(initialize=data["U"])
    m.K = pyo.Set(initialize=data["K"])
    m.T = pyo.Set(initialize=data["T"])

    m.N_minus = pyo.Set(m.D, initialize=lambda mdl, j: data["N_minus"][j])
    m.N_plus = pyo.Set(m.U, initialize=lambda mdl, i: data["N_plus"][i])

    m.NODES = pyo.Set(initialize=list(set(data["V"]) | set(data["U"])))

    m.P = pyo.Set(dimen=3, initialize=[
        (i, j, k)
        for (j, k), I in data["P"].items()
        for i in I
    ])

    # Parameters
    m.f = pyo.Param(m.V, initialize=data["f"], within=pyo.NonNegativeReals)
    m.s0 = pyo.Param(m.NODES, initialize=data["s0"], within=pyo.NonNegativeIntegers)
    m.c = pyo.Param(m.NODES, initialize=data["c"], within=pyo.NonNegativeIntegers)
    m.r = pyo.Param(m.K, m.NODES, initialize=data["r"], within=pyo.NonNegativeReals)
    m.d_t = pyo.Param(m.V, m.T, initialize=data["d_t_flat"], within=pyo.NonNegativeReals)

    # Decision variables (all period-indexed)
    m.u = pyo.Var(m.NODES, m.T, domain=pyo.NonNegativeIntegers)  # production of i in period t
    m.l = pyo.Var(m.V, m.T, domain=pyo.NonNegativeIntegers)  # lost volume of product j in period t
    m.s = pyo.Var(m.NODES, m.T, domain=pyo.NonNegativeIntegers)  # end-of-period inventory of i

    m.y_index = pyo.Set(within=m.U * m.NODES, initialize=lambda mdl: [
        (i, j) for i in mdl.U for j in mdl.N_plus[i]
    ])
    m.y = pyo.Var(m.y_index, m.T, domain=pyo.NonNegativeIntegers)  # shipment i->j departing in period t

    offset = data["offset"]  # plain dict: lead-time offset in periods, only meaningful for i in U

    # Objective: min sum over all periods and products of lost profit
    def obj_rule(mdl):
        return sum(mdl.f[j] * mdl.l[j, t] for j in mdl.V for t in mdl.T)
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # BOM production, lead-time-offset on the shipment: a unit of material k
    # used by j in period t must have departed its supplier i in period
    # t - offset[i]; departures before period 0 are covered by
    # initial_pipeline instead of a (nonexistent) negative-period variable.
    def bom_production_rule(mdl, j, k, t):
        total = 0
        for i in data["P"][(j, k)]:
            t_dep = t - offset.get(i, 0)
            if t_dep >= 0:
                total += mdl.y[i, j, t_dep]
            else:
                total += initial_pipeline.get((i, j), 0.0)
        return mdl.u[j, t] - total / mdl.r[k, j] <= 0
    m.BomProduction = pyo.Constraint(
        [(j, k, t) for j in m.D for k in m.N_minus[j] for t in m.T],
        rule=bom_production_rule,
    )

    # Supplier (tier2/tier3) inventory carryover: this period's ending
    # inventory = last period's ending inventory (or the opening balance, for
    # t=0) + production - everything shipped out this period.
    def supplier_inventory_rule(mdl, i, t):
        prev = mdl.s0[i] if t == 0 else mdl.s[i, t - 1]
        shipped = sum(mdl.y[i, j, t] for j in mdl.N_plus[i])
        return mdl.s[i, t] == prev + mdl.u[i, t] - shipped
    m.SupplierInventory = pyo.Constraint(m.U, m.T, rule=supplier_inventory_rule)

    # Finished-goods inventory carryover and demand fulfillment: this
    # period's ending inventory = last period's (or opening balance) +
    # production - demand actually satisfied (demand minus lost volume).
    def product_inventory_rule(mdl, j, t):
        prev = mdl.s0[j] if t == 0 else mdl.s[j, t - 1]
        return mdl.s[j, t] == prev + mdl.u[j, t] - (mdl.d_t[j, t] - mdl.l[j, t])
    m.ProductInventory = pyo.Constraint(m.V, m.T, rule=product_inventory_rule)

    def lost_le_demand_rule(mdl, j, t):
        return mdl.l[j, t] <= mdl.d_t[j, t]
    m.LostLeDemand = pyo.Constraint(m.V, m.T, rule=lost_le_demand_rule)

    # Disruption: production forced to 0 for exactly the periods each
    # TimePhasedDisruption is active, not the whole horizon.
    disrupted_pairs = [(i, t) for t, nodes in data["S_t"].items() for i in nodes]
    m.Disrupted = pyo.Constraint(disrupted_pairs, rule=lambda mdl, i, t: mdl.u[i, t] == 0)

    # Capacity: per-period cap, scaled from the dataset's per-time-unit rate.
    def capacity_rule(mdl, i, t):
        return mdl.u[i, t] <= mdl.c[i] * period_length_days
    m.Capacity = pyo.Constraint(m.NODES, m.T, rule=capacity_rule)

    # Solve
    solver = pyo.SolverFactory("highs")
    result = solver.solve(m, tee=False)

    disrupted_nodes = sorted({n for d in disruptions for n in d.disrupted_nodes})
    columns = ["disrupted", "n_periods", "period_length_days", "termination_condition", "lost_profit"]
    row = [disrupted_nodes, n_periods, period_length_days, result.solver.termination_condition, pyo.value(m.OBJ)]
    if return_model:
        columns.append("model")
        row.append(m)
    return pd.DataFrame([row], columns=columns)


def extract_time_phased_solution(model, dataset: dict) -> pd.DataFrame:
    """Long-format per-(node, period) table of production, inventory, and
    lost volume from a solved ``build_and_solve_time_phased_ttr`` model
    (call with ``return_model=True`` to get ``model``), for plotting or
    notebook use. ``lost`` is ``None`` for non-product (tier2/tier3) nodes."""
    import pyomo.environ as pyo

    v_nodes = set(model.V)
    rows = []
    for i in model.NODES:
        for t in model.T:
            rows.append({
                "node": i,
                "period": t,
                "produced": pyo.value(model.u[i, t]),
                "inventory": pyo.value(model.s[i, t]),
                "lost": pyo.value(model.l[i, t]) if i in v_nodes else None,
            })
    return pd.DataFrame(rows)

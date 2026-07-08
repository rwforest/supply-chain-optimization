"""Tests for the additive time-phased multi-period planning engine
(``scripts/multi_period_planning.py``) and the aggregate/disaggregate
network-decomposition engine (``scripts/network_aggregation.py``).

Neither module touches ``scripts/utils.py``, ``scripts/disruption_scenarios.py``,
``scripts/realistic_topologies.py``, ``scripts/dataset_io.py``, or
``scripts/company_profiles.py`` — see the README's "Time-Phased Multi-Period
Planning & Network Decomposition" section for the full design.

Run with: python -m pytest tests/ -v   (must be run from the repo root so
``scripts`` resolves as a package).
"""

from __future__ import annotations

import random

import pytest

import scripts.disruption_scenarios as ds_lib
import scripts.multi_period_planning as mpp
import scripts.network_aggregation as agg_lib
import scripts.realistic_topologies as rt
import scripts.scenario_calibration as sc

GENERATORS = {
    "simple": lambda: rt.generate_simple_network(),
    "medium": lambda: rt.generate_medium_network(),
}


@pytest.fixture(scope="module")
def datasets():
    return {name: gen() for name, gen in GENERATORS.items()}


# A minimal hand-built two-node dataset (S1 -> V1) for exact-arithmetic unit
# tests on lead-time offset and inventory carryover, where the expected
# numbers can be computed by hand rather than merely "solved optimally."
def _tiny_dataset(lead_time_days: float = 14.0) -> dict:
    return {
        "tier1": ["V1"],
        "tier2": ["S1"],
        "tier3": [],
        "edges": [("S1", "V1")],
        "material_types": ["mat_a"],
        "supplier_material_type": {"S1": "mat_a"},
        "f": {"V1": 0.5},
        "s": {"V1": 0, "S1": 100},
        "d": {"V1": 10},
        "c": {"V1": 1000, "S1": 1000},
        "r": {"V1": {"mat_a": 1.0}},
        "N_minus": {"V1": ["mat_a"], "S1": []},
        "N_plus": {"S1": ["V1"]},
        "P": {"V1": {"mat_a": ["S1"]}},
        "production_delay": {"S1": lead_time_days},
    }


# ---------------------------------------------------------------------------
# TimePhasedDisruption / from_legacy_scenario
# ---------------------------------------------------------------------------


def test_time_phased_disruption_periods():
    d = mpp.TimePhasedDisruption(
        scenario_id="d1", name="d1", description="", real_world_basis="",
        scenario_type="single_node", disrupted_nodes=["S1"],
        period_start=2, duration_periods=3,
    )
    assert d.periods() == {2, 3, 4}


def test_from_legacy_scenario_rounds_duration_up():
    scenario = ds_lib.DisruptionScenario(
        scenario_id="s1", name="s1", description="", real_world_basis="",
        scenario_type="single_node", disrupted_nodes=["S1"], ttr=15,
    )
    # ttr=15 days at 7-day periods -> ceil(15/7) = 3 periods, never fewer.
    tp = mpp.from_legacy_scenario(scenario, period_length_days=7.0)
    assert tp.duration_periods == 3
    assert tp.period_start == 0
    assert tp.disrupted_nodes == ["S1"]


# ---------------------------------------------------------------------------
# default_periodic_demand
# ---------------------------------------------------------------------------


def test_default_periodic_demand_flat_matches_single_snapshot_total():
    dataset = _tiny_dataset()
    n_periods = 4
    period_length_days = 7.0
    d_t = mpp.default_periodic_demand(dataset, n_periods, period_length_days)
    assert d_t["V1"] == {t: 10 * 7.0 for t in range(n_periods)}
    assert sum(d_t["V1"].values()) == dataset["d"]["V1"] * n_periods * period_length_days


def test_default_periodic_demand_seasonality_preserves_mean_and_is_integer():
    dataset = _tiny_dataset()
    n_periods = 12
    seasonality = {"V1": [1.0 + 0.2 * ((t % 4) - 1.5) for t in range(n_periods)]}
    d_t = mpp.default_periodic_demand(dataset, n_periods, period_length_days=7.0, seasonality=seasonality)
    for val in d_t["V1"].values():
        assert val == int(val), "seasonal demand must be rounded to an integer to stay LP-feasible"
    flat_total = dataset["d"]["V1"] * 7.0 * n_periods
    seasonal_total = sum(d_t["V1"].values())
    assert abs(seasonal_total - flat_total) / flat_total < 0.05


# ---------------------------------------------------------------------------
# build_and_solve_time_phased_ttr: lead-time offset + inventory carryover
# ---------------------------------------------------------------------------


def test_lead_time_offset_delays_first_arrival():
    """S1 has a 14-day lead time at 7-day periods -> offset=2. A shipment
    departing S1 in period t only arrives (and can satisfy V1's demand) in
    period t+2. V1 starts with zero inventory, so periods 0 and 1 must show
    100% lost demand, and period 2 onward should recover."""
    dataset = _tiny_dataset(lead_time_days=14.0)
    n_periods = 5
    result = mpp.build_and_solve_time_phased_ttr(dataset, [], n_periods, period_length_days=7.0, return_model=True)
    assert str(result.iloc[0]["termination_condition"]).lower() == "optimal"
    model = result.iloc[0]["model"]
    sol = mpp.extract_time_phased_solution(model, dataset)

    lost_by_period = sol[sol["node"] == "V1"].set_index("period")["lost"]
    assert lost_by_period[0] == pytest.approx(70.0)  # full period-0 demand lost
    assert lost_by_period[1] == pytest.approx(70.0)  # full period-1 demand lost
    assert lost_by_period[2] == pytest.approx(0.0)  # first shipment has arrived


def test_zero_lead_time_has_no_arrival_delay():
    dataset = _tiny_dataset(lead_time_days=0.0)
    n_periods = 3
    result = mpp.build_and_solve_time_phased_ttr(dataset, [], n_periods, period_length_days=7.0, return_model=True)
    assert str(result.iloc[0]["termination_condition"]).lower() == "optimal"
    model = result.iloc[0]["model"]
    sol = mpp.extract_time_phased_solution(model, dataset)
    lost_by_period = sol[sol["node"] == "V1"].set_index("period")["lost"]
    assert lost_by_period[0] == pytest.approx(0.0)


def test_supplier_inventory_carryover_ties_to_opening_balance():
    """s[i, -1] must equal dataset['s'][i] exactly: with disruption at S1 in
    period 0 only, S1's period-0 ending inventory equals its opening balance
    (100) minus whatever it actually ships out in period 0 (read directly off
    the model's y variable — V1's `produced` is not a reliable proxy for
    shipped quantity, since this degenerate cost-free LP is free to ship more
    raw material than V1 strictly needs)."""
    import pyomo.environ as pyo

    dataset = _tiny_dataset(lead_time_days=0.0)
    disruption = mpp.TimePhasedDisruption(
        scenario_id="d1", name="d1", description="", real_world_basis="",
        scenario_type="single_node", disrupted_nodes=["S1"],
        period_start=0, duration_periods=1,
    )
    n_periods = 3
    result = mpp.build_and_solve_time_phased_ttr(
        dataset, [disruption], n_periods, period_length_days=7.0, return_model=True
    )
    model = result.iloc[0]["model"]
    sol = mpp.extract_time_phased_solution(model, dataset)
    s1_period0 = sol[(sol["node"] == "S1") & (sol["period"] == 0)].iloc[0]
    assert s1_period0["produced"] == pytest.approx(0.0)  # disrupted
    shipped = pyo.value(model.y["S1", "V1", 0])
    assert s1_period0["inventory"] == pytest.approx(100.0 - shipped)


def test_disruption_forces_zero_production_only_in_active_periods():
    dataset = _tiny_dataset(lead_time_days=0.0)
    disruption = mpp.TimePhasedDisruption(
        scenario_id="d1", name="d1", description="", real_world_basis="",
        scenario_type="single_node", disrupted_nodes=["S1"],
        period_start=1, duration_periods=1,
    )
    n_periods = 3
    result = mpp.build_and_solve_time_phased_ttr(
        dataset, [disruption], n_periods, period_length_days=7.0, return_model=True
    )
    assert str(result.iloc[0]["termination_condition"]).lower() == "optimal"
    model = result.iloc[0]["model"]
    sol = mpp.extract_time_phased_solution(model, dataset)
    s1 = sol[sol["node"] == "S1"].set_index("period")
    assert s1.loc[1, "produced"] == pytest.approx(0.0)
    # Periods 0 and 2 are not disrupted; nothing in the model forces them to 0.
    assert s1.loc[0, "produced"] >= 0.0
    assert s1.loc[2, "produced"] >= 0.0


@pytest.mark.parametrize("name", ["simple", "medium"])
def test_time_phased_ttr_solves_optimally_on_generator_output(datasets, name):
    dataset = datasets[name]
    n_periods = 6
    baseline = mpp.build_and_solve_time_phased_ttr(dataset, [], n_periods, period_length_days=7.0)
    assert str(baseline.iloc[0]["termination_condition"]).lower() == "optimal"

    rng = random.Random(1)
    scenario = ds_lib.single_supplier_failure_scenarios(dataset, rng)[0]
    disruption = mpp.from_legacy_scenario(scenario, period_length_days=7.0)
    disrupted = mpp.build_and_solve_time_phased_ttr(dataset, [disruption], n_periods, period_length_days=7.0)
    assert str(disrupted.iloc[0]["termination_condition"]).lower() == "optimal"
    assert disrupted.iloc[0]["lost_profit"] >= baseline.iloc[0]["lost_profit"]


def test_seasonality_index_integrates_with_time_phased_solve(datasets):
    dataset = datasets["medium"]
    n_periods = 6
    rng = random.Random(3)
    seasonality = sc.sample_seasonality_index(rng, dataset["tier1"], n_periods, "medium")
    d_t = mpp.default_periodic_demand(dataset, n_periods, period_length_days=7.0, seasonality=seasonality)
    result = mpp.build_and_solve_time_phased_ttr(dataset, [], n_periods, period_length_days=7.0, d_t=d_t)
    assert str(result.iloc[0]["termination_condition"]).lower() == "optimal"


# ---------------------------------------------------------------------------
# network_aggregation: build_aggregate_dataset conservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["simple", "medium"])
def test_aggregate_dataset_conserves_capacity_and_inventory(datasets, name):
    dataset = datasets[name]
    result = agg_lib.build_aggregate_dataset(dataset)
    assert result.group_to_members, f"expected at least one group on the {name} dataset"
    for group_node, members in result.group_to_members.items():
        assert result.aggregate_dataset["s"][group_node] == sum(dataset["s"][m] for m in members)
        assert result.aggregate_dataset["c"][group_node] == sum(dataset["c"][m] for m in members)


@pytest.mark.parametrize("name", ["simple", "medium"])
def test_aggregate_dataset_reduces_tier3_node_count(datasets, name):
    dataset = datasets[name]
    result = agg_lib.build_aggregate_dataset(dataset)
    assert len(result.aggregate_dataset["tier3"]) < len(dataset["tier3"])


@pytest.mark.parametrize("name", ["simple", "medium"])
def test_aggregate_dataset_indexes_match_independent_recomputation(datasets, name):
    dataset = datasets[name]
    result = agg_lib.build_aggregate_dataset(dataset)
    agg = result.aggregate_dataset
    N_minus, N_plus, P, r = rt.derive_indexes(
        agg["tier1"], agg["tier2"], agg["tier3"], agg["edges"], agg["supplier_material_type"],
    )
    assert N_minus == agg["N_minus"]
    assert N_plus == agg["N_plus"]
    assert P == agg["P"]
    assert r == agg["r"]


def test_monopoly_bottleneck_nodes_are_never_grouped(datasets):
    dataset = rt.generate_complex_network("nvidia")
    monopoly_nodes = {
        n for n in dataset["tier3"]
        if dataset["criticality"].get(n) == "monopoly_bottleneck"
    }
    assert monopoly_nodes, "expected at least one monopoly_bottleneck tier3 node in the nvidia dataset"
    result = agg_lib.build_aggregate_dataset(dataset)
    assert monopoly_nodes.isdisjoint(result.node_to_group.keys())
    assert monopoly_nodes.issubset(set(result.aggregate_dataset["tier3"]))


# ---------------------------------------------------------------------------
# disaggregation_scope
# ---------------------------------------------------------------------------


def test_disaggregation_scope_includes_disrupted_nodes_entire_group(datasets):
    dataset = datasets["medium"]
    result = agg_lib.build_aggregate_dataset(dataset)
    group_node, members = next(iter(result.group_to_members.items()))
    scope = agg_lib.disaggregation_scope(dataset, result.node_to_group, [members[0]], expansion_hops=1)
    assert set(members).issubset(scope)


def test_disaggregation_scope_includes_upstream_alternate_suppliers(datasets):
    dataset = datasets["medium"]
    result = agg_lib.build_aggregate_dataset(dataset)
    group_node, members = next(iter(result.group_to_members.items()))
    disrupted_node = members[0]

    alt_suppliers = set()
    for j, k_map in dataset["P"].items():
        for k, suppliers in k_map.items():
            if disrupted_node in suppliers:
                alt_suppliers.update(suppliers)
    alt_suppliers.discard(disrupted_node)

    scope = agg_lib.disaggregation_scope(dataset, result.node_to_group, [disrupted_node], expansion_hops=0)
    if alt_suppliers:
        assert alt_suppliers.issubset(scope)


def test_disaggregation_scope_expansion_hops_zero_excludes_downstream(datasets):
    dataset = datasets["medium"]
    result = agg_lib.build_aggregate_dataset(dataset)
    group_node, members = next(iter(result.group_to_members.items()))
    disrupted_node = members[0]

    downstream = set(dataset["N_plus"].get(disrupted_node, []))
    downstream_outside_group = downstream - set(members)

    scope_zero_hops = agg_lib.disaggregation_scope(
        dataset, result.node_to_group, [disrupted_node], expansion_hops=0
    )
    unexplained_downstream = downstream_outside_group - scope_zero_hops
    # Any downstream node not covered by the alt-supplier rule should be
    # absent when expansion_hops=0 (no consumer-cone expansion at all).
    for n in downstream_outside_group:
        is_alt_supplier_pull = any(
            n in suppliers
            for j, k_map in dataset["P"].items()
            for k, suppliers in k_map.items()
            if disrupted_node in suppliers
        )
        if not is_alt_supplier_pull:
            assert n not in scope_zero_hops or n in unexplained_downstream


# ---------------------------------------------------------------------------
# build_partial_disaggregation + run_aggregate_then_disaggregate
# ---------------------------------------------------------------------------


def test_partial_disaggregation_is_schema_valid(datasets):
    dataset = datasets["medium"]
    result = agg_lib.build_aggregate_dataset(dataset)
    group_node, members = next(iter(result.group_to_members.items()))
    scope = agg_lib.disaggregation_scope(dataset, result.node_to_group, [members[0]], expansion_hops=1)
    partial = agg_lib.build_partial_disaggregation(dataset, result, scope)

    N_minus, N_plus, P, r = rt.derive_indexes(
        partial["tier1"], partial["tier2"], partial["tier3"],
        partial["edges"], partial["supplier_material_type"],
    )
    assert N_minus == partial["N_minus"]
    assert N_plus == partial["N_plus"]
    assert P == partial["P"]
    assert r == partial["r"]
    # Every real, expanded member must appear as an individual node.
    assert set(members).issubset(set(partial["tier3"]) | set(partial["tier2"]) | set(partial["tier1"]))


def test_run_aggregate_then_disaggregate_solves_and_shrinks_the_network(datasets):
    dataset = datasets["medium"]
    rng = random.Random(1)
    scenario = ds_lib.single_supplier_failure_scenarios(dataset, rng)[0]
    disruption = mpp.from_legacy_scenario(scenario, period_length_days=7.0)

    outcome = agg_lib.run_aggregate_then_disaggregate(dataset, [disruption], n_periods=4, period_length_days=7.0)

    assert str(outcome["aggregate_result"].iloc[0]["termination_condition"]).lower() == "optimal"
    assert str(outcome["partial_result"].iloc[0]["termination_condition"]).lower() == "optimal"

    counts = outcome["node_counts"]
    # The whole point of decomposition: aggregate and partial networks must
    # be structurally smaller than the full network's tier-3 node count.
    assert counts["aggregate_tier3"] < counts["full_tier3"]
    assert counts["partial_tier3"] < counts["full_tier3"]
    assert counts["partial_tier3"] >= counts["aggregate_tier3"]

    timings = outcome["timings"]
    assert timings["total_s"] >= 0
    print(f"\n[run_aggregate_then_disaggregate timings] {timings}")

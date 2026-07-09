"""Tests for the realistic scenario ladder: schema correctness of the four
network generators, solvability of the LP for each disruption-scenario
type, and the validate_scenario guard against bad node IDs.

Run with: python -m pytest tests/ -v   (must be run from the repo root so
``scripts`` resolves as a package; a plain ``pytest`` invocation may not put
the repo root on sys.path).
"""

from __future__ import annotations

import random

import pytest
from pyomo.environ import value as pyo_value

import scripts.disruption_scenarios as ds_lib
import scripts.realistic_topologies as rt
import scripts.scenario_calibration as sc
import scripts.utils as utils

REQUIRED_KEYS = {
    "tier1", "tier2", "tier3", "edges", "material_types",
    "supplier_material_type", "f", "s", "d", "c", "r",
    "N_minus", "N_plus", "P",
}

GENERATORS = {
    "simple": lambda: rt.generate_simple_network(),
    "medium": lambda: rt.generate_medium_network(),
    "nvidia": lambda: rt.generate_complex_network("nvidia"),
    "apple": lambda: rt.generate_complex_network("apple"),
    "mps": lambda: rt.generate_complex_network("mps"),
    "apple_real": lambda: rt.generate_complex_network("apple_real"),
}


@pytest.fixture(scope="module")
def datasets():
    return {name: gen() for name, gen in GENERATORS.items()}


def _with_cost_fields(dataset: dict, seed: int = 1, industry_profile: str = "simple") -> dict:
    """Merge calibrated cost/holding/delay/emissions fields into a copy of
    ``dataset``, for exercising the new objective functions in tests."""
    rng = random.Random(seed)
    fields = sc.calibrate_cost_fields(
        rng,
        dataset["tier1"], dataset["tier2"], dataset["tier3"],
        dataset["material_types"], dataset["supplier_material_type"],
        dataset["edges"], dataset.get("criticality"), industry_profile,
        region=dataset.get("region"),
    )
    return {**dataset, **fields}


@pytest.fixture(scope="module")
def cost_dataset(datasets):
    return _with_cost_fields(datasets["simple"])


@pytest.mark.parametrize("name", GENERATORS)
def test_schema_has_required_keys(datasets, name):
    dataset = datasets[name]
    assert REQUIRED_KEYS.issubset(dataset.keys())


@pytest.mark.parametrize("name", GENERATORS)
def test_edges_reference_known_nodes(datasets, name):
    dataset = datasets[name]
    known = set(dataset["tier1"]) | set(dataset["tier2"]) | set(dataset["tier3"])
    for src, tgt in dataset["edges"]:
        assert src in known, f"edge source {src!r} not a known node"
        assert tgt in known, f"edge target {tgt!r} not a known node"


@pytest.mark.parametrize("name", GENERATORS)
def test_indexes_match_independent_recomputation(datasets, name):
    dataset = datasets[name]
    N_minus, N_plus, P, r = rt.derive_indexes(
        dataset["tier1"], dataset["tier2"], dataset["tier3"],
        dataset["edges"], dataset["supplier_material_type"],
    )
    assert N_minus == dataset["N_minus"]
    assert N_plus == dataset["N_plus"]
    assert P == dataset["P"]
    assert r == dataset["r"]


@pytest.mark.parametrize("name", GENERATORS)
def test_parameters_are_well_formed(datasets, name):
    dataset = datasets[name]
    for node, margin in dataset["f"].items():
        assert 0 <= margin <= 1, f"{node} margin {margin} out of [0,1]"
    for node in dataset["tier1"] + dataset["tier2"] + dataset["tier3"]:
        assert dataset["s"][node] > 0
        assert dataset["c"][node] > 0
    for node in dataset["tier1"]:
        assert dataset["d"][node] > 0


@pytest.mark.parametrize("name", ["simple", "medium"])
def test_ttr_and_tts_solve_optimally_for_baseline_and_single_failure(datasets, name):
    dataset = datasets[name]
    baseline = utils.build_and_solve_ttr(dataset, [], 10)
    assert str(baseline.iloc[0]["termination_condition"]).lower() == "optimal"

    rng = random.Random(1)
    scenarios = ds_lib.single_supplier_failure_scenarios(dataset, rng)
    for scenario in scenarios[:5]:
        df = ds_lib.run_scenario_ttr(dataset, scenario)
        assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"
        dft = ds_lib.run_scenario_tts(dataset, scenario)
        # run_scenario_tts always normalizes to a finite or +inf tts, never NaN.
        assert dft.iloc[0]["tts"] == dft.iloc[0]["tts"]  # not NaN


def test_regional_and_material_shortage_scenarios_solve(datasets):
    dataset = datasets["medium"]
    named = ds_lib.named_real_world_scenarios(dataset)
    assert named, "expected at least one real-world scenario to match the medium dataset"
    for scenario in named:
        df = ds_lib.run_scenario_ttr(dataset, scenario)
        assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"
        dft = ds_lib.run_scenario_tts(dataset, scenario)
        assert dft.iloc[0]["tts"] == dft.iloc[0]["tts"]  # not NaN


def test_material_shortage_scenario_disrupting_all_suppliers_still_feasible(datasets):
    """Regression check for the always-feasible proof: disrupting *every*
    supplier of a material type at once must never crash the solver."""
    dataset = datasets["simple"]
    material = dataset["material_types"][0]
    scenario = ds_lib.material_shortage_scenario(dataset, material, ttr=20)
    df = ds_lib.run_scenario_ttr(dataset, scenario)
    assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"


@pytest.mark.parametrize("company", ["nvidia", "apple", "mps", "apple_real"])
def test_complex_network_solves_a_bounded_scenario_subset(datasets, company):
    """Full LP solve at 1500-3000 node scale is sub-second per scenario;
    only a small subset is exercised here to keep test runtime bounded."""
    dataset = datasets[company]
    named = ds_lib.named_real_world_scenarios(dataset)
    assert named, f"expected at least one real-world scenario to match the {company} dataset"
    for scenario in named[:3]:
        df = ds_lib.run_scenario_ttr(dataset, scenario)
        assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"


def test_validate_scenario_rejects_unknown_node(datasets):
    dataset = datasets["simple"]
    bogus = ds_lib.DisruptionScenario(
        scenario_id="bogus",
        name="bogus",
        description="",
        real_world_basis="",
        scenario_type="single_node",
        disrupted_nodes=["T9_999"],
        ttr=5,
    )
    with pytest.raises(ValueError):
        ds_lib.validate_scenario(dataset, bogus)
    with pytest.raises(ValueError):
        ds_lib.run_scenario_ttr(dataset, bogus)


def test_regional_disruption_requires_region_tags():
    dataset = utils.generate_data(N1=5, N2=10, N3=20)
    with pytest.raises(ValueError):
        ds_lib.regional_disruption_scenario(dataset, "Taiwan", ttr=10)


def test_scale_factor_default_matches_current_behavior(datasets):
    """scale_factor=1.0 (the default) must reproduce today's output exactly
    for a fixed seed — regression guard for the planet-scale change."""
    dataset = rt.generate_complex_network("nvidia")
    reference = datasets["nvidia"]
    assert len(dataset["tier2"]) == len(reference["tier2"])
    assert len(dataset["tier3"]) == len(reference["tier3"])
    assert sorted(dataset["edges"]) == sorted(reference["edges"])


def test_scale_factor_grows_anchor_fanout_proportionally():
    """Scaling up n2_target/n3_target without scale_factor would dilute the
    fraction of nodes traceable to a real, named anchor toward generic
    padding; scale_factor should keep that fraction roughly constant."""

    def named_fraction(dataset):
        names = dataset["company_name"]
        total = len(dataset["tier2"]) + len(dataset["tier3"])
        named = sum(1 for v in names.values() if v)
        return named / total

    base = rt.generate_complex_network("nvidia", n2_target=300, n3_target=1000, scale_factor=1.0)
    scaled = rt.generate_complex_network("nvidia", n2_target=900, n3_target=3000, scale_factor=3.0)

    base_fraction = named_fraction(base)
    scaled_fraction = named_fraction(scaled)
    assert scaled_fraction >= base_fraction * 0.5, (
        f"named-anchor fraction dropped too much: base={base_fraction:.3f} "
        f"scaled={scaled_fraction:.3f}"
    )


def test_generate_complex_network_at_scale_presets():
    complex_default = rt.generate_complex_network_at_scale("apple", scale="complex")
    reference = rt.generate_complex_network("apple")
    assert len(complex_default["tier2"]) == len(reference["tier2"])
    assert len(complex_default["tier3"]) == len(reference["tier3"])

    with pytest.raises(ValueError):
        rt.generate_complex_network_at_scale("apple", scale="not_a_real_preset")


def test_generate_complex_network_at_scale_mps_planet():
    """The MPS anchor supports the planet preset just like nvidia/apple — the
    dataset stays schema-valid and is meaningfully larger than complex scale."""
    planet = rt.generate_complex_network_at_scale("mps", scale="planet")
    assert REQUIRED_KEYS.issubset(planet.keys())
    complex_mps = rt.generate_complex_network("mps")
    assert len(planet["tier2"]) > len(complex_mps["tier2"])
    assert len(planet["tier3"]) > len(complex_mps["tier3"])


def test_single_supplier_failure_scenarios_exhaustive_by_default(datasets):
    """sample_fraction=1.0 (the default) produces the exact same scenario
    set as today — one per tier2+tier3 node — regression guard."""
    dataset = datasets["medium"]
    rng = random.Random(1)
    scenarios = ds_lib.single_supplier_failure_scenarios(dataset, rng)
    assert len(scenarios) == len(dataset["tier2"]) + len(dataset["tier3"])
    assert {s.disrupted_nodes[0] for s in scenarios} == set(dataset["tier2"] + dataset["tier3"])


def test_padding_attachment_is_hub_heavy_not_uniform():
    """Generic padding tier-3 nodes should attach via preferential
    attachment, producing a hub-heavy (high-variance) degree distribution
    across tier-2 parents rather than the near-equal counts uniform random
    choice would produce over many draws."""
    dataset = rt.generate_complex_network("nvidia", n2_target=50, n3_target=5000)
    degree: dict[str, int] = {t2: 0 for t2 in dataset["tier2"]}
    for src, tgt in dataset["edges"]:
        if tgt in degree:
            degree[tgt] += 1

    counts = sorted(degree.values(), reverse=True)
    top_5_share = sum(counts[:5]) / sum(counts)
    # Under uniform random attachment across 50 parents, the top 5 would
    # hold roughly 5/50 = 10% of edges; preferential attachment concentrates
    # far more than that onto a handful of hubs.
    assert top_5_share > 0.25, f"expected hub concentration, got top_5_share={top_5_share:.3f}"


def test_single_supplier_failure_scenarios_sample_fraction(datasets):
    dataset = datasets["nvidia"]
    nodes = dataset["tier2"] + dataset["tier3"]
    criticality = dataset["criticality"]
    always_nodes = {n for n in nodes if criticality.get(n) in ("monopoly_bottleneck", "oligopoly")}

    rng = random.Random(1)
    sampled = ds_lib.single_supplier_failure_scenarios(dataset, rng, sample_fraction=0.3)
    sampled_ids = {s.disrupted_nodes[0] for s in sampled}

    assert always_nodes.issubset(sampled_ids)
    assert len(sampled_ids) < len(nodes)
    assert len(sampled_ids) >= len(always_nodes)

    # deterministic for a fixed seeded rng
    rng2 = random.Random(1)
    sampled2 = ds_lib.single_supplier_failure_scenarios(dataset, rng2, sample_fraction=0.3)
    assert [s.scenario_id for s in sampled] == [s.scenario_id for s in sampled2]


# Inventory buffers in the "simple" dataset comfortably cover roughly two
# weeks of demand, so a short horizon (like the ttr=10 used by the TTR/TTS
# tests) lets these hard-demand objectives satisfy everything from existing
# stock with zero production. Use a longer horizon so u > 0 is actually
# required, exercising the new cost coefficients.
COST_TTR = 30


def test_cost_min_solves_optimally_with_positive_cost(cost_dataset):
    df = utils.build_and_solve_cost_min(cost_dataset, [], COST_TTR)
    assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"
    assert df.iloc[0]["total_cost"] > 0


def test_revenue_max_solves_optimally_with_positive_revenue(cost_dataset):
    df = utils.build_and_solve_revenue_max(cost_dataset, [], COST_TTR)
    assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"
    assert df.iloc[0]["revenue"] > 0


def test_inventory_opt_solves_optimally(cost_dataset):
    df = utils.build_and_solve_inventory_opt(cost_dataset, [], COST_TTR)
    assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"
    assert df.iloc[0]["holding_cost_total"] >= 0


def test_lead_time_min_solves_optimally_with_positive_delay(cost_dataset):
    df = utils.build_and_solve_lead_time_min(cost_dataset, [], COST_TTR)
    assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"
    assert df.iloc[0]["total_delay"] > 0


def test_fulfillment_rate_solves_optimally_at_baseline(cost_dataset):
    df = utils.build_and_solve_fulfillment_rate(cost_dataset, [], COST_TTR)
    assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"
    assert df.iloc[0]["lost_fraction"] >= 0


def test_carbon_min_solves_optimally_with_positive_emissions(cost_dataset):
    df = utils.build_and_solve_carbon_min(cost_dataset, [], COST_TTR)
    assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"
    assert df.iloc[0]["total_emissions"] > 0


def test_fixed_u_scenario_planning(cost_dataset):
    node = cost_dataset["tier1"][0]
    baseline = utils.build_and_solve_cost_min(cost_dataset, [], COST_TTR, return_model=True)
    baseline_u = baseline.iloc[0]["model"].u[node].value
    # Pin production above the unconstrained optimum (rather than to 0,
    # which can make hard tier-1 demand infeasible if s+0 < d*t) so the
    # scenario stays feasible while still forcing the objective to change.
    forced = int(baseline_u) + 50
    df = utils.build_and_solve_cost_min(
        cost_dataset, [], COST_TTR, fixed_u={node: forced}, return_model=True
    )
    assert str(df.iloc[0]["termination_condition"]).lower() == "optimal"
    model = df.iloc[0]["model"]
    assert pytest.approx(pyo_value(model.u[node])) == forced
    assert df.iloc[0]["total_cost"] != baseline.iloc[0]["total_cost"]


def test_inventory_opt_respects_s_max(cost_dataset):
    df = utils.build_and_solve_inventory_opt(cost_dataset, [], COST_TTR, return_model=True)
    model = df.iloc[0]["model"]
    for i in model.NODES:
        assert pyo_value(model.s[i]) <= cost_dataset["s"][i] + 1e-6


def test_compute_network_resilience_metrics(datasets):
    dataset = datasets["medium"]
    metrics = utils.compute_network_resilience_metrics(dataset)
    assert metrics["single_point_of_failure_pairs"] >= 0
    assert metrics["avg_suppliers_per_material"] >= 0

    single_sourced = {
        "tier1": ["V1"],
        "tier2": ["S1"],
        "tier3": [],
        "P": {"V1": {"mat_a": ["S1"]}},
    }
    metrics2 = utils.compute_network_resilience_metrics(single_sourced)
    assert metrics2["single_point_of_failure_pairs"] >= 1
    assert metrics2["resilience_score"]["V1"] == 1


def test_calibrate_cost_fields_correlates_with_criticality(datasets):
    # "medium" derives criticality purely from supplier-count-per-material
    # (target_suppliers_per_material=3), so every node lands in
    # diversified_commodity; "nvidia" has hand-authored monopoly_bottleneck/
    # oligopoly anchors (e.g. TSMC, ASML), so it actually exercises the
    # criticality-multiplier spread this test is checking.
    dataset = datasets["nvidia"]
    fields = _with_cost_fields(dataset, seed=7, industry_profile="nvidia")
    criticality = dataset["criticality"]

    monopoly_nodes = [n for n in dataset["tier2"] + dataset["tier3"] if criticality.get(n) == "monopoly_bottleneck"]
    commodity_nodes = [n for n in dataset["tier2"] + dataset["tier3"] if criticality.get(n) == "diversified_commodity"]
    assert monopoly_nodes and commodity_nodes

    def avg(field, nodes):
        return sum(fields[field][n] for n in nodes) / len(nodes)

    assert avg("unit_cost", monopoly_nodes) > avg("unit_cost", commodity_nodes)
    assert avg("emissions_factor", monopoly_nodes) > avg("emissions_factor", commodity_nodes)
    assert avg("production_delay", monopoly_nodes) > avg("production_delay", commodity_nodes)

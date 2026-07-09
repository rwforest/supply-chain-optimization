"""Tests for the MPS cuOpt pipeline (`09_mps_cuopt_pipeline`): the derivation
layer (``scripts/mps_derivation.py``), the solver-agnostic MILP/routing model
specs (``scripts/fjsp_cuopt.py``, ``scripts/cvrptw_cuopt.py``), and the CPU-only
MEIO model (``scripts/meio_pyomo.py``).

The derivation + spec tests import NO cuOpt/cudf and run anywhere. The MEIO
tests use the repo's existing HiGHS/Pyomo stack. The actual GPU solves
(``build_fjsp_problem``/``solve_fjsp``, ``build_cvrptw_datamodel``/
``solve_cvrptw``) are guarded with ``pytest.importorskip`` so they SKIP (not
fail) on a machine without the GPU wheels installed.

Run with: python -m pytest tests/ -v   (from the repo root so ``scripts``
resolves as a package).
"""

from __future__ import annotations

import math
import random

import pytest

import scripts.cvrptw_cuopt as cv
import scripts.fjsp_cuopt as fj
import scripts.meio_pyomo as meio
import scripts.mps_derivation as md
import scripts.realistic_topologies as rt
import scripts.scenario_calibration as sc


@pytest.fixture(scope="module")
def mps_dataset():
    """A generated MPS network with cost/holding/delay/emissions fields merged
    in, exactly as the `09` notebook assembles it before deriving models."""
    d = rt.generate_complex_network("mps")
    rng = random.Random(7)
    fields = sc.calibrate_cost_fields(
        rng,
        d["tier1"], d["tier2"], d["tier3"],
        d["material_types"], d["supplier_material_type"],
        d["edges"], d.get("criticality"), "mps",
        region=d.get("region"),
    )
    return {**d, **fields}


# --------------------------------------------------------------------------
# Derivation layer (no GPU / solver)
# --------------------------------------------------------------------------
def test_backend_machines_have_test_and_packaging_capacity(mps_dataset):
    machines = md.select_backend_machines(mps_dataset)
    assert machines, "expected at least one back-end machine"
    groups = {m["group"] for m in machines}
    assert md.WAFER_SORT_FINAL_TEST in groups, "no Chengdu test machines derived"
    assert md.ADVANCED_PACKAGING_OSAT in groups, "no OSAT packaging machines derived"
    # Only test-capable machines can run electrical_test.
    test_capable = [m for m in machines if m["test_capable"]]
    assert test_capable
    assert all(m["group"] == md.WAFER_SORT_FINAL_TEST for m in test_capable)


def test_fjsp_jobs_each_have_full_operation_sequence(mps_dataset):
    machines = md.select_backend_machines(mps_dataset)
    jobs = md.derive_fjsp_jobs(mps_dataset, machines)
    assert jobs, "expected at least one job"
    assert len(jobs) <= 40  # default max_jobs cap
    for job in jobs:
        assert job["operations"] == md.FJSP_OPERATIONS
        assert job["source_tier1"] in mps_dataset["tier1"]


def test_every_job_operation_has_an_eligible_machine(mps_dataset):
    machines = md.select_backend_machines(mps_dataset)
    jobs = md.derive_fjsp_jobs(mps_dataset, machines)
    pt = md.derive_fjsp_processing_times(jobs, machines, mps_dataset)
    for job in jobs:
        for op in job["operations"]:
            elig = md.eligible_machines(op, machines)
            assert elig, f"no eligible machine for op {op}"
            # every eligible machine must have a processing time entry
            for m in elig:
                assert (job["job_id"], op, m) in pt


def test_electrical_test_restricted_to_test_capable_machines(mps_dataset):
    machines = md.select_backend_machines(mps_dataset)
    test_ids = {m["machine_id"] for m in machines if m["test_capable"]}
    elig = set(md.eligible_machines("electrical_test", machines))
    assert elig
    assert elig.issubset(test_ids)


def test_cvrptw_depots_are_chengdu_and_penang(mps_dataset):
    depots = md.select_cvrptw_depots(mps_dataset)
    names = {dep["name"] for dep in depots}
    assert any("Chengdu" in n for n in names)
    assert any("Penang" in n for n in names)


def test_cvrptw_customers_and_windows_consistent(mps_dataset):
    custs = md.derive_cvrptw_customers(mps_dataset)
    assert len(custs) == len(mps_dataset["tier1"])
    tw = md.derive_cvrptw_time_windows(custs, mps_dataset)
    assert set(tw) == {c["customer_id"] for c in custs}
    for _cid, (lo, hi) in tw.items():
        assert 0 <= lo < hi


def test_cvrptw_distance_matrix_is_square_and_zero_diagonal(mps_dataset):
    depots = md.select_cvrptw_depots(mps_dataset)
    custs = md.derive_cvrptw_customers(mps_dataset)
    mat = md.derive_cvrptw_distance_time_matrix(depots, custs)
    n = len(mat["locations"])
    assert n == len(depots) + len(custs)
    assert len(mat["time"]) == n and all(len(r) == n for r in mat["time"])
    for i in range(n):
        assert mat["time"][i][i] == 0
        assert mat["distance"][i][i] == 0


def test_meio_network_node_set_matches_dataset(mps_dataset):
    net = md.derive_meio_network(mps_dataset)
    expected = set(mps_dataset["tier1"] + mps_dataset["tier2"] + mps_dataset["tier3"])
    assert set(net["nodes"]) == expected
    # every field is defined for every node
    for field in ("holding_cost", "lead_time", "demand_mean", "demand_std", "service_z"):
        assert set(net[field]) >= expected
    # monopoly bottlenecks get the highest service factor
    crit = mps_dataset["criticality"]
    monopoly = [n for n in net["nodes"] if crit.get(n) == "monopoly_bottleneck"]
    commodity = [n for n in net["nodes"] if crit.get(n) == "diversified_commodity"]
    if monopoly and commodity:
        assert net["service_z"][monopoly[0]] > net["service_z"][commodity[0]]


# --------------------------------------------------------------------------
# Solver-agnostic model specs (no GPU / cuopt / cudf)
# --------------------------------------------------------------------------
def test_fjsp_milp_spec_shapes_are_consistent(mps_dataset):
    machines = md.select_backend_machines(mps_dataset)
    jobs = md.derive_fjsp_jobs(mps_dataset, machines)
    pt = md.derive_fjsp_processing_times(jobs, machines, mps_dataset)
    spec = fj._fjsp_milp_spec(jobs, machines, pt)

    n_ops = sum(len(j["operations"]) for j in jobs)
    assert spec.objective == {"Cmax": 1.0}
    assert spec.n_vars > 0 and spec.n_binary > 0
    assert spec.big_m > 0
    # every constraint references only declared variables
    known = set(spec.var_names)
    for coeffs, sense, _rhs in spec.constraints:
        assert set(coeffs).issubset(known)
        assert sense in ("<=", ">=", "==")
    # exactly one assignment (==1) + one completion (==0) equality per op
    eq_constraints = [c for c in spec.constraints if c[1] == "=="]
    assert len(eq_constraints) == 2 * n_ops


def test_fjsp_spec_assignment_constraints_cover_each_operation(mps_dataset):
    machines = md.select_backend_machines(mps_dataset)
    jobs = md.derive_fjsp_jobs(mps_dataset, machines)
    pt = md.derive_fjsp_processing_times(jobs, machines, mps_dataset)
    spec = fj._fjsp_milp_spec(jobs, machines, pt)
    # An assignment constraint is "== 1" over only x[...] binaries.
    assign = [
        c for c in spec.constraints
        if c[1] == "==" and c[2] == 1.0 and all(v.startswith("x[") for v in c[0])
    ]
    assert len(assign) == sum(len(j["operations"]) for j in jobs)


def test_cvrptw_spec_shapes_and_validation(mps_dataset):
    depots = md.select_cvrptw_depots(mps_dataset)
    custs = md.derive_cvrptw_customers(mps_dataset)
    tw = md.derive_cvrptw_time_windows(custs, mps_dataset)
    mat = md.derive_cvrptw_distance_time_matrix(depots, custs)
    spec = cv._cvrptw_datamodel_spec(
        depots, custs, {"n_vehicles": 4, "capacity": 10_000}, mat, tw
    )
    spec.validate()  # raises on any inconsistency
    assert spec.n_locations == len(depots) + len(custs)
    assert spec.demand[spec.depot_index] == 0
    # customer demand entries are positive
    assert sum(spec.demand) > 0
    assert len(spec.vehicle_capacities) == spec.n_vehicles


def test_cvrptw_spec_validate_rejects_bad_matrix(mps_dataset):
    depots = md.select_cvrptw_depots(mps_dataset)
    custs = md.derive_cvrptw_customers(mps_dataset)
    tw = md.derive_cvrptw_time_windows(custs, mps_dataset)
    mat = md.derive_cvrptw_distance_time_matrix(depots, custs)
    spec = cv._cvrptw_datamodel_spec(
        depots, custs, {"n_vehicles": 2, "capacity": 10_000}, mat, tw
    )
    # Corrupt the matrix and confirm validate() catches it.
    spec.cost_matrix = spec.cost_matrix[:-1]
    with pytest.raises(ValueError):
        spec.validate()


# --------------------------------------------------------------------------
# MEIO — CPU-only, uses the existing HiGHS/Pyomo stack
# --------------------------------------------------------------------------
def test_meio_solves_optimally_on_mps_network(mps_dataset):
    net = md.derive_meio_network(mps_dataset)
    model = meio.build_meio_model(net)
    result = meio.solve_meio(model)
    assert result["termination_condition"].lower() == "optimal"
    assert result["total_safety_stock_cost"] >= 0
    # per-node net lead times are non-negative
    for info in result["per_node"].values():
        assert info["net_lead_time"] >= -1e-6


def test_meio_piecewise_linear_approximation_error_is_bounded(mps_dataset):
    net = md.derive_meio_network(mps_dataset)
    model = meio.build_meio_model(net)
    result = meio.solve_meio(model)
    approx = result["total_safety_stock_cost"]
    true = meio.true_safety_stock_cost(net, result["per_node"])
    # The piecewise-linear (INC) approximation should track the true-sqrt
    # objective within a few percent given the default breakpoints.
    if true > 0:
        rel_error = abs(approx - true) / true
        assert rel_error < 0.05, f"pwl approx error {rel_error:.3%} exceeds 5%"


def test_meio_sqrt_breakpoints_span_range():
    pts = meio.sqrt_pwl_breakpoints(50.0)
    assert pts[0] == 0.0
    assert pts[-1] >= 50.0
    assert pts == sorted(pts)


def test_meio_nonlinear_mode_is_a_documented_stub(mps_dataset):
    net = md.derive_meio_network(mps_dataset)
    with pytest.raises(NotImplementedError):
        meio.build_meio_model(net, mode="nonlinear")
    with pytest.raises(ValueError):
        meio.build_meio_model(net, mode="not_a_mode")


def test_meio_tangent_free_true_cost_matches_manual():
    # Two-node toy: verify true_safety_stock_cost computes h*z*sigma*sqrt(NLT).
    net = {
        "holding_cost": {"a": 2.0, "b": 3.0},
        "service_z": {"a": 1.645, "b": 2.326},
        "demand_std": {"a": 10.0, "b": 20.0},
    }
    per_node = {"a": {"net_lead_time": 4.0}, "b": {"net_lead_time": 9.0}}
    expected = 2.0 * 1.645 * 10.0 * math.sqrt(4.0) + 3.0 * 2.326 * 20.0 * math.sqrt(9.0)
    assert meio.true_safety_stock_cost(net, per_node) == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------
# GPU solves — SKIP (not fail) without the cuOpt/cudf wheels installed
# --------------------------------------------------------------------------
def test_fjsp_gpu_build_and_solve(mps_dataset):
    pytest.importorskip("cuopt", reason="cuOpt GPU wheel not installed")
    machines = md.select_backend_machines(mps_dataset)
    jobs = md.derive_fjsp_jobs(mps_dataset, machines)
    pt = md.derive_fjsp_processing_times(jobs, machines, mps_dataset)
    problem, spec = fj.build_fjsp_problem(jobs, machines, pt)
    raw = fj.solve_fjsp(problem, time_limit=30.0)
    schedule = fj.decode_fjsp_result(raw, jobs, machines, pt)
    assert schedule["makespan"] >= 0
    # every job/op is assigned to exactly one machine
    assert len(schedule["assignments"]) == sum(len(j["operations"]) for j in jobs)


def test_cvrptw_gpu_build_and_solve(mps_dataset):
    pytest.importorskip("cuopt", reason="cuOpt GPU wheel not installed")
    pytest.importorskip("cudf", reason="cudf GPU wheel not installed")
    depots = md.select_cvrptw_depots(mps_dataset)
    custs = md.derive_cvrptw_customers(mps_dataset)
    tw = md.derive_cvrptw_time_windows(custs, mps_dataset)
    mat = md.derive_cvrptw_distance_time_matrix(depots, custs)
    data_model, spec = cv.build_cvrptw_datamodel(
        depots, custs, {"n_vehicles": 4, "capacity": 10_000}, mat, tw
    )
    raw = cv.solve_cvrptw(data_model, time_limit=10.0)
    routes = cv.decode_cvrptw_result(raw, spec)
    assert "routes" in routes

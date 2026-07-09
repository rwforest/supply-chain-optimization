"""Flexible Job-Shop Scheduling (FJSP) for MPS back-end assembly/test,
formulated as a sequence-based MILP and solved on NVIDIA cuOpt's GPU MILP
solver (Scenario II of `MPS Supply Chain Optimization Pipeline.md` §6).

Design: the MILP is built in two layers so the model is buildable and testable
WITHOUT a GPU or the ``cuopt`` package installed:

* ``_fjsp_milp_spec(jobs, machines, processing_times)`` — a pure-Python
  function that returns the full MILP as plain data: the variable list (with
  kinds/bounds), the objective coefficients, and the constraint rows (sparse
  coefficient dicts + sense + rhs). It imports nothing from cuOpt, so its
  variable/constraint counts and shapes are unit-testable anywhere.
* ``build_fjsp_problem(...)`` wraps that spec into a cuOpt in-process
  ``Problem`` (``from cuopt.linear_programming.problem import Problem``) — no
  server required, a single GPU-attached notebook cell.
* ``solve_fjsp(problem)`` runs the GPU solve; ``decode_fjsp_result(...)``
  turns the raw variable values back into a per-machine schedule.

Inputs come from ``scripts.mps_derivation`` (``derive_fjsp_jobs``,
``select_backend_machines``, ``derive_fjsp_processing_times``,
``eligible_machines``). This module does not import ``mps_derivation`` — it
re-derives eligibility from the ``processing_times`` keys, so it stays
decoupled and testable in isolation.

MILP formulation (makespan minimization):
    Variables:
        Cmax                          continuous >= 0   (makespan)
        s[j,o]                        continuous >= 0   (start time of op o of job j)
        c[j,o]                        continuous >= 0   (completion time)
        x[j,o,m]                      binary            (op assigned to machine m)
        y[j1,o1,j2,o2,m]              binary            (op1 precedes op2 on m)
    Objective:  minimize Cmax
    Constraints:
        (assign)   sum_m x[j,o,m] = 1                              for each (j,o)
        (compl)    c[j,o] = s[j,o] + sum_m p[j,o,m]*x[j,o,m]       for each (j,o)
        (prec)     s[j,o] >= c[j,o-1]                              for consecutive ops
        (makespan) Cmax >= c[j, last_op]                           for each job
        (disj1)    s[j2,o2] >= c[j1,o1] - M*(3 - y - x1 - x2)      pairwise on shared m
        (disj2)    s[j1,o1] >= c[j2,o2] - M*(2 + y - x1 - x2)      pairwise on shared m
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FjspMilpSpec:
    """Solver-agnostic MILP description produced by ``_fjsp_milp_spec``."""

    var_names: list[str]
    var_kinds: dict[str, str]  # "C" continuous | "B" binary
    var_lb: dict[str, float]
    var_ub: dict[str, float]
    objective: dict[str, float]  # var_name -> coefficient (minimize)
    # Each constraint: (coeffs: {var: coef}, sense: "<="|">="|"==", rhs: float)
    constraints: list[tuple[dict[str, float], str, float]] = field(default_factory=list)
    big_m: float = 0.0

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    @property
    def n_binary(self) -> int:
        return sum(1 for k in self.var_kinds.values() if k == "B")

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)


def _eligible_from_times(
    job_id: str, op: str, processing_times: dict
) -> list[str]:
    """Machines with a processing-time entry for (job, op) — the eligible set."""
    return sorted(
        m for (j, o, m) in processing_times if j == job_id and o == op
    )


def _fjsp_milp_spec(
    jobs: list[dict],
    machines: list[dict],
    processing_times: dict,
) -> FjspMilpSpec:
    """Build the sequence-based FJSP MILP as plain data (no solver import).

    ``jobs``: list of {"job_id", "operations": [...]} (extra keys ignored).
    ``machines``: list of {"machine_id", ...}.
    ``processing_times``: {(job_id, op, machine_id): time}.
    """
    var_names: list[str] = []
    var_kinds: dict[str, str] = {}
    var_lb: dict[str, float] = {}
    var_ub: dict[str, float] = {}

    def add_var(name: str, kind: str, lb: float = 0.0, ub: float = 1e9) -> None:
        var_names.append(name)
        var_kinds[name] = kind
        var_lb[name] = lb
        var_ub[name] = ub

    # Big-M: an upper bound on any completion time = sum of the max processing
    # time of every operation (all ops run back to back on their slowest
    # eligible machine). Loose but valid.
    big_m = 0.0
    op_max_time: dict[tuple[str, str], float] = {}
    for job in jobs:
        for op in job["operations"]:
            elig = _eligible_from_times(job["job_id"], op, processing_times)
            times = [processing_times[(job["job_id"], op, m)] for m in elig]
            mx = max(times) if times else 0.0
            op_max_time[(job["job_id"], op)] = mx
            big_m += mx
    big_m = max(big_m, 1.0)

    add_var("Cmax", "C", 0.0, big_m)

    # Continuous start/completion vars and binary assignment vars.
    for job in jobs:
        jid = job["job_id"]
        for op in job["operations"]:
            add_var(f"s[{jid},{op}]", "C", 0.0, big_m)
            add_var(f"c[{jid},{op}]", "C", 0.0, big_m)
            for m in _eligible_from_times(jid, op, processing_times):
                add_var(f"x[{jid},{op},{m}]", "B", 0.0, 1.0)

    objective = {"Cmax": 1.0}
    constraints: list[tuple[dict[str, float], str, float]] = []

    # (assign) exactly one eligible machine per operation.
    for job in jobs:
        jid = job["job_id"]
        for op in job["operations"]:
            elig = _eligible_from_times(jid, op, processing_times)
            row = {f"x[{jid},{op},{m}]": 1.0 for m in elig}
            constraints.append((row, "==", 1.0))

    # (compl) c = s + sum_m p*x  ->  c - s - sum_m p*x = 0
    for job in jobs:
        jid = job["job_id"]
        for op in job["operations"]:
            elig = _eligible_from_times(jid, op, processing_times)
            row = {f"c[{jid},{op}]": 1.0, f"s[{jid},{op}]": -1.0}
            for m in elig:
                row[f"x[{jid},{op},{m}]"] = -processing_times[(jid, op, m)]
            constraints.append((row, "==", 0.0))

    # (prec) s[j,o] >= c[j,o-1]  ->  s[j,o] - c[j,o-1] >= 0
    for job in jobs:
        jid = job["job_id"]
        ops = job["operations"]
        for idx in range(1, len(ops)):
            row = {f"s[{jid},{ops[idx]}]": 1.0, f"c[{jid},{ops[idx-1]}]": -1.0}
            constraints.append((row, ">=", 0.0))

    # (makespan) Cmax >= c[j, last]  ->  Cmax - c[j,last] >= 0
    for job in jobs:
        jid = job["job_id"]
        last = job["operations"][-1]
        constraints.append(({"Cmax": 1.0, f"c[{jid},{last}]": -1.0}, ">=", 0.0))

    # (disjunctive) for every pair of ops that CAN share a machine m, order
    # them if both are assigned to m. Introduce y[pair,m] in {0,1}.
    #   s2 >= c1 - M*(3 - y - x1 - x2)   ->  s2 - c1 + M*y + M*x1 + M*x2 >= 3M-...
    # We write both disjuncts in >= normal form below.
    op_refs: list[tuple[str, str]] = []
    for job in jobs:
        for op in job["operations"]:
            op_refs.append((job["job_id"], op))

    for i in range(len(op_refs)):
        for k in range(i + 1, len(op_refs)):
            j1, o1 = op_refs[i]
            j2, o2 = op_refs[k]
            elig1 = set(_eligible_from_times(j1, o1, processing_times))
            elig2 = set(_eligible_from_times(j2, o2, processing_times))
            shared = sorted(elig1 & elig2)
            for m in shared:
                yname = f"y[{j1},{o1},{j2},{o2},{m}]"
                add_var(yname, "B", 0.0, 1.0)
                x1 = f"x[{j1},{o1},{m}]"
                x2 = f"x[{j2},{o2},{m}]"
                # disj1: s2 - c1 + M*y + M*x1 + M*x2 >= 3M - M  ==>  >= 2M?
                # Derivation: s2 >= c1 - M*(3 - y - x1 - x2)
                #   s2 - c1 + M*(3 - y - x1 - x2) >= 0
                #   s2 - c1 - M*y - M*x1 - M*x2 >= -3M
                row1 = {
                    f"s[{j2},{o2}]": 1.0,
                    f"c[{j1},{o1}]": -1.0,
                    yname: -big_m,
                    x1: -big_m,
                    x2: -big_m,
                }
                constraints.append((row1, ">=", -3.0 * big_m))
                # disj2: s1 >= c2 - M*(2 + y - x1 - x2)
                #   s1 - c2 + M*(2 + y - x1 - x2) >= 0
                #   s1 - c2 + M*y - M*x1 - M*x2 >= -2M
                row2 = {
                    f"s[{j1},{o1}]": 1.0,
                    f"c[{j2},{o2}]": -1.0,
                    yname: big_m,
                    x1: -big_m,
                    x2: -big_m,
                }
                constraints.append((row2, ">=", -2.0 * big_m))

    return FjspMilpSpec(
        var_names=var_names,
        var_kinds=var_kinds,
        var_lb=var_lb,
        var_ub=var_ub,
        objective=objective,
        constraints=constraints,
        big_m=big_m,
    )


def build_fjsp_problem(jobs: list[dict], machines: list[dict], processing_times: dict):
    """Wrap ``_fjsp_milp_spec`` into a cuOpt in-process ``Problem``.

    Requires the ``cuopt`` package and a CUDA GPU (see the ``gpu`` optional-
    dependency group and ``requirements-gpu.txt``). Returns a tuple
    ``(problem, spec)`` so the caller keeps the variable ordering for decoding.
    """
    from cuopt.linear_programming.problem import Problem  # noqa: F401

    spec = _fjsp_milp_spec(jobs, machines, processing_times)

    problem = Problem()
    var_handles = {}
    for name in spec.var_names:
        kind = spec.var_kinds[name]
        if kind == "B":
            var_handles[name] = problem.add_variable(
                lb=0, ub=1, vtype="I", name=name
            )
        else:
            var_handles[name] = problem.add_variable(
                lb=spec.var_lb[name], ub=spec.var_ub[name], vtype="C", name=name
            )

    # Objective (minimize).
    problem.set_objective(
        sense="minimize",
        expression=sum(
            coef * var_handles[name] for name, coef in spec.objective.items()
        ),
    )

    sense_map = {"<=": "L", ">=": "G", "==": "E"}
    for coeffs, sense, rhs in spec.constraints:
        expr = sum(coef * var_handles[name] for name, coef in coeffs.items())
        problem.add_constraint(expr, sense_map[sense], rhs)

    return problem, spec


def solve_fjsp(problem, time_limit: float = 60.0):
    """Solve a built FJSP ``Problem`` on the GPU. Returns cuOpt's raw solution
    object (variable values + objective). ``time_limit`` in seconds."""
    from cuopt.linear_programming import solver_settings

    settings = solver_settings.SolverSettings()
    settings.set_parameter("time_limit", time_limit)
    return problem.solve(settings)


def decode_fjsp_result(raw, jobs: list[dict], machines: list[dict], processing_times: dict) -> dict:
    """Turn a cuOpt raw solution into a readable schedule.

    Returns:
        {"makespan": float,
         "assignments": [{"job", "operation", "machine", "start", "end"}...],
         "by_machine": {machine_id: [assignment, ...] sorted by start}}
    """
    values = _raw_var_values(raw)

    assignments: list[dict] = []
    for job in jobs:
        jid = job["job_id"]
        for op in job["operations"]:
            start = values.get(f"s[{jid},{op}]", 0.0)
            end = values.get(f"c[{jid},{op}]", 0.0)
            chosen = None
            for m in _eligible_from_times(jid, op, processing_times):
                if values.get(f"x[{jid},{op},{m}]", 0.0) > 0.5:
                    chosen = m
                    break
            assignments.append(
                {
                    "job": jid,
                    "operation": op,
                    "machine": chosen,
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )

    by_machine: dict[str, list[dict]] = {m["machine_id"]: [] for m in machines}
    for a in assignments:
        if a["machine"] in by_machine:
            by_machine[a["machine"]].append(a)
    for m in by_machine:
        by_machine[m].sort(key=lambda a: a["start"])

    makespan = values.get("Cmax", max((a["end"] for a in assignments), default=0.0))
    return {
        "makespan": round(makespan, 3),
        "assignments": assignments,
        "by_machine": by_machine,
    }


def _raw_var_values(raw) -> dict[str, float]:
    """Best-effort extraction of {var_name: value} from a cuOpt solution
    object across cuOpt API versions (the exact accessor has shifted between
    releases). Kept isolated so decoding is resilient to that surface."""
    # Common shapes: raw.get_primal_solution() -> dict{name: val}; or
    # raw.get_variable_values(); or a .vars mapping.
    for accessor in ("get_primal_solution", "get_variable_values"):
        fn = getattr(raw, accessor, None)
        if callable(fn):
            out = fn()
            if isinstance(out, dict):
                return {str(k): float(v) for k, v in out.items()}
    vars_attr = getattr(raw, "vars", None)
    if isinstance(vars_attr, dict):
        return {str(k): float(v) for k, v in vars_attr.items()}
    raise RuntimeError(
        "Could not extract variable values from the cuOpt solution object; "
        "check the installed cuOpt version's solution API and update "
        "_raw_var_values accordingly."
    )

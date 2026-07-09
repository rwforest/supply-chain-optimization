"""Capacitated Vehicle Routing with Time Windows (CVRPTW) for MPS finished-
goods distribution, solved on NVIDIA cuOpt's GPU routing solver (Scenario III
of `MPS Supply Chain Optimization Pipeline.md` §7).

Same two-layer design as ``scripts.fjsp_cuopt`` so the model is buildable and
testable WITHOUT a GPU / ``cuopt`` / ``cudf`` installed:

* ``_cvrptw_datamodel_spec(depots, customers, vehicles, distance_matrix,
  time_windows)`` — a pure-Python function returning the routing model as
  plain data: the location count, the (square) cost/time matrices, per-order
  demand, time windows, and per-vehicle capacities. Imports nothing from cuOpt
  or cudf, so matrix shapes / consistency are unit-testable anywhere.
* ``build_cvrptw_datamodel(...)`` wraps that spec into a
  ``cuopt.routing.DataModel`` (with ``cudf`` for the cost/time matrices).
* ``solve_cvrptw(data_model)`` runs the GES metaheuristic on GPU;
  ``decode_cvrptw_result(...)`` turns the raw routes into per-vehicle stop
  lists.

Inputs come from ``scripts.mps_derivation`` (``select_cvrptw_depots``,
``derive_cvrptw_customers``, ``derive_cvrptw_time_windows``,
``derive_cvrptw_distance_time_matrix``). This module does not import
``mps_derivation``.

Modeling notes:
* Location 0 is the depot (cuOpt's convention). The distance/time matrix from
  ``mps_derivation.derive_cvrptw_distance_time_matrix`` already places the
  first depot at index 0.
* A single homogeneous depot is used for the v1 demo (the first depot in the
  list). Multi-depot support is a documented extension.
* Service time at each stop is a fixed constant; time windows are the hard
  windows from ``derive_cvrptw_time_windows``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fixed per-stop service time (minutes) and a default vehicle count/capacity
# for the demo. Real fleets would come from an Adexa/ERP feed.
DEFAULT_SERVICE_TIME = 15
DEFAULT_N_VEHICLES = 4
DEFAULT_VEHICLE_CAPACITY = 10_000


@dataclass
class CvrptwDataModelSpec:
    """Solver-agnostic routing model produced by ``_cvrptw_datamodel_spec``."""

    n_locations: int  # depot(s) + customers
    n_vehicles: int
    location_ids: list[str]
    cost_matrix: list[list[float]]
    time_matrix: list[list[int]]
    demand: list[int]  # per location; depot demand = 0
    earliest: list[int]  # per location time-window start
    latest: list[int]  # per location time-window end
    service_time: list[int]  # per location
    vehicle_capacities: list[int]
    depot_index: int = 0

    def validate(self) -> None:
        """Raise if the matrices/vectors are inconsistent — the shape checks a
        cuOpt DataModel would otherwise fail on at solve time."""
        n = self.n_locations
        if len(self.location_ids) != n:
            raise ValueError("location_ids length != n_locations")
        for name, mat in (("cost", self.cost_matrix), ("time", self.time_matrix)):
            if len(mat) != n or any(len(row) != n for row in mat):
                raise ValueError(f"{name}_matrix must be {n}x{n}")
        for name, vec in (
            ("demand", self.demand),
            ("earliest", self.earliest),
            ("latest", self.latest),
            ("service_time", self.service_time),
        ):
            if len(vec) != n:
                raise ValueError(f"{name} length {len(vec)} != n_locations {n}")
        if len(self.vehicle_capacities) != self.n_vehicles:
            raise ValueError("vehicle_capacities length != n_vehicles")
        if self.demand[self.depot_index] != 0:
            raise ValueError("depot demand must be 0")
        for i in range(n):
            if self.earliest[i] > self.latest[i]:
                raise ValueError(f"location {i}: earliest > latest")


def _cvrptw_datamodel_spec(
    depots: list[dict],
    customers: list[dict],
    vehicles: dict,
    distance_matrix: dict,
    time_windows: dict,
    service_time: int = DEFAULT_SERVICE_TIME,
) -> CvrptwDataModelSpec:
    """Build the CVRPTW routing model as plain data (no cuOpt/cudf import).

    ``depots``/``customers``: dicts from ``mps_derivation`` (depots have
    ``depot_id``; customers have ``customer_id``, ``demand``).
    ``vehicles``: {"n_vehicles": int, "capacity": int}.
    ``distance_matrix``: {"locations", "index", "time", "distance"} from
    ``mps_derivation.derive_cvrptw_distance_time_matrix``.
    ``time_windows``: {customer_id: (earliest, latest)}.
    """
    locations = list(distance_matrix["locations"])
    index = distance_matrix["index"]
    n = len(locations)

    depot_id = depots[0]["depot_id"]
    depot_index = index[depot_id]

    cost = [list(map(float, row)) for row in distance_matrix["distance"]]
    time = [list(map(int, row)) for row in distance_matrix["time"]]

    demand = [0] * n
    earliest = [0] * n
    # Depot / any non-customer location gets a wide window covering the shift.
    horizon = max((tw[1] for tw in time_windows.values()), default=480)
    latest = [horizon] * n
    svc = [0] * n

    cust_demand = {c["customer_id"]: int(c.get("demand", 1)) for c in customers}
    for cid, dem in cust_demand.items():
        i = index[cid]
        demand[i] = dem
        svc[i] = service_time
        if cid in time_windows:
            earliest[i], latest[i] = int(time_windows[cid][0]), int(time_windows[cid][1])

    n_vehicles = int(vehicles.get("n_vehicles", DEFAULT_N_VEHICLES))
    capacity = int(vehicles.get("capacity", DEFAULT_VEHICLE_CAPACITY))
    vehicle_capacities = [capacity] * n_vehicles

    spec = CvrptwDataModelSpec(
        n_locations=n,
        n_vehicles=n_vehicles,
        location_ids=locations,
        cost_matrix=cost,
        time_matrix=time,
        demand=demand,
        earliest=earliest,
        latest=latest,
        service_time=svc,
        vehicle_capacities=vehicle_capacities,
        depot_index=depot_index,
    )
    spec.validate()
    return spec


def build_cvrptw_datamodel(
    depots: list[dict],
    customers: list[dict],
    vehicles: dict,
    distance_matrix: dict,
    time_windows: dict,
    service_time: int = DEFAULT_SERVICE_TIME,
):
    """Wrap ``_cvrptw_datamodel_spec`` into a ``cuopt.routing.DataModel``.

    Requires ``cuopt`` + ``cudf`` and a CUDA GPU. Returns ``(data_model,
    spec)`` so the caller keeps the location ordering for decoding.
    """
    import cudf
    from cuopt import routing

    spec = _cvrptw_datamodel_spec(
        depots, customers, vehicles, distance_matrix, time_windows, service_time
    )

    data_model = routing.DataModel(spec.n_locations, spec.n_vehicles)

    cost_df = cudf.DataFrame(spec.cost_matrix)
    time_df = cudf.DataFrame(spec.time_matrix)
    data_model.add_cost_matrix(cost_df)
    data_model.add_transit_time_matrix(time_df)

    data_model.set_order_locations(cudf.Series(list(range(spec.n_locations))))
    data_model.add_capacity_dimension(
        "demand",
        cudf.Series(spec.demand),
        cudf.Series(spec.vehicle_capacities),
    )
    data_model.set_order_time_windows(
        cudf.Series(spec.earliest),
        cudf.Series(spec.latest),
    )
    data_model.set_order_service_times(cudf.Series(spec.service_time))

    return data_model, spec


def solve_cvrptw(data_model, time_limit: float = 10.0):
    """Solve a built CVRPTW ``DataModel`` on the GPU with the GES metaheuristic.
    Returns cuOpt's raw routing solution. ``time_limit`` in seconds."""
    from cuopt import routing

    settings = routing.SolverSettings()
    settings.set_time_limit(time_limit)
    return routing.Solve(data_model, settings)


def decode_cvrptw_result(raw, spec: CvrptwDataModelSpec) -> dict:
    """Turn a cuOpt routing solution into readable per-vehicle routes.

    Returns:
        {"status", "total_cost", "n_vehicles_used",
         "routes": {vehicle_id: [location_id, ...]}}
    """
    status = getattr(raw, "get_status", lambda: None)()
    total_cost = None
    for accessor in ("get_total_objective", "final_cost", "get_cost"):
        fn = getattr(raw, accessor, None)
        if callable(fn):
            total_cost = fn()
            break

    routes: dict[int, list[str]] = {}
    route_df = None
    getter = getattr(raw, "get_route", None)
    if callable(getter):
        route_df = getter()
    if route_df is not None:
        # cuOpt returns a cudf/pandas DataFrame with 'truck_id' and 'location'.
        pdf = route_df.to_pandas() if hasattr(route_df, "to_pandas") else route_df
        for _, row in pdf.iterrows():
            vid = int(row.get("truck_id", row.get("vehicle_id", 0)))
            loc_idx = int(row["location"])
            loc_id = spec.location_ids[loc_idx] if loc_idx < len(spec.location_ids) else loc_idx
            routes.setdefault(vid, []).append(loc_id)

    return {
        "status": status,
        "total_cost": total_cost,
        "n_vehicles_used": len(routes),
        "routes": routes,
    }

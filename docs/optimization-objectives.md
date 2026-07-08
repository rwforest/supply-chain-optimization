# Optimization Objectives

[← Back to README](../README.md) · Related: [Disruption Scenarios](disruption-scenarios.md) · [Multi-Period Planning & Decomposition](multi-period-and-decomposition.md)

Every stress test in this accelerator is a linear/integer optimization problem: hold the network topology and a disruption fixed, then let a solver ([HiGHS](https://github.com/ERGO-Code/HiGHS) via [Pyomo](https://pyomo.readthedocs.io/)) find the best achievable outcome under that shock. What "best" means is the **objective function**. This doc walks through every objective implemented in `scripts/utils.py`.

## The shared model skeleton

All the LP objectives below share the same three-tier structure, decision variables, and constraints — only the objective (and a handful of objective-specific parameters/bounds) changes. Understanding the skeleton once makes every objective easy to read.

**Node tiers**
- `tier1` (𝒱) — finished-product nodes that carry external demand `d[j]`.
- `tier2` — direct component/module suppliers.
- `tier3` — their equipment/raw-material suppliers (leaf nodes).
- `D = tier1 + tier2` (everything with a bill of materials); `U = tier2 + tier3` (everything that ships to someone).

**Decision variables**
- `u[i]` — production quantity of node `i` over the horizon.
- `y[i,j]` — allocation (shipment) from upstream node `i` to downstream node `j`.
- `l[j]` — lost demand volume for product `j` (TTR family only).

**Core constraints** (identical across the LP objectives unless noted)
- **BOM production** — a node can't produce more than its scarcest required material allows: `u[j] ≤ Σ y[i,j] / r[k,j]` for each material `k`.
- **Flow balance** — a supplier can't ship more than it produces plus its on-hand inventory: `Σ y[i,j] − u[i] ≤ s[i]`.
- **Demand** — tier-1 output plus inventory must meet demand over the horizon (or, in the revenue objective, is *capped* by demand).
- **Capacity** — `u[j] ≤ c[j] · t`.
- **Disruption** — every disrupted node is forced to `u[j] = 0`.

The two original objectives (`build_and_solve_ttr`, `build_and_solve_tts`) implement this skeleton inline. The six newer LP objectives share it via the `_prep_lp_data` helper and are **purely additive** — they do not modify `ttr`/`tts`.

---

## Objective count, stated honestly

There are **eight `build_and_solve_*` LP objectives** in `scripts/utils.py`:

- **2 original**: `build_and_solve_ttr`, `build_and_solve_tts`
- **6 added**: `cost_min`, `revenue_max`, `inventory_opt`, `lead_time_min`, `fulfillment_rate`, `carbon_min`

Plus **one non-LP structural metric**, `compute_network_resilience_metrics`, which is a graph-topology calculation rather than a solve.

So the total is **9 objectives (8 LP + 1 structural)**, of which **7 were added** on top of the original TTR/TTS pair. (If you've seen this described elsewhere as "TTR/TTS + 8 = 10," that count is off — the code has 6 new LP solves plus the 1 structural metric, not 8 new solves.)

---

## The two original objectives

### 1. TTR — Time to Recover (`build_and_solve_ttr`)

> **Given** a fixed recovery time `t` (how long the disrupted nodes stay down), **minimize** the profit-weighted lost demand.

- **Objective:** `minimize Σ_j f[j] · l[j]` — total lost profit, where `f[j]` is the per-unit profit margin and `l[j]` is unmet demand for product `j`.
- **Interprets as:** "A disruption of known duration `t` hits these nodes. Running the network optimally, how much profit do we lose?"
- **Signature:** `build_and_solve_ttr(dataset, disrupted, ttr, return_model=False)`

### 2. TTS — Time to Survive (`build_and_solve_tts`)

> **Find** the longest disruption duration `t` the network can absorb while still meeting *all* demand.

- **Objective:** `maximize t`, with `t` promoted to a decision variable and demand required to be fully met (`u[j] + s[j] ≥ d[j] · t`).
- **Interprets as:** "How long can these nodes stay down before we're forced to miss a single unit of demand?"
- **Unbounded case:** if the network has enough redundancy/headroom to absorb the disruption *indefinitely*, `run_scenario_tts` normalizes the result to `tts = inf`. The solver's raw objective in that case is a meaningless artifact of wherever the simplex method stopped, not a real bound.
- **Signature:** `build_and_solve_tts(dataset, disrupted, return_model=False)`

TTR and TTS are the two methods from the [MIT paper](https://dspace.mit.edu/handle/1721.1/101782) this accelerator implements.

---

## The six added LP objectives

All six share the `_prep_lp_data` skeleton, take the same `(dataset, disrupted, ttr, ...)` shape, and support `fixed_u`/`fixed_s` overrides for composing scenarios. Each requires one or more **optional** calibrated fields (see [calibration](disruption-scenarios.md#calibration-methodology)); `scripts.scenario_calibration.calibrate_cost_fields` populates them.

| # | Function | Objective | Requires | Notes |
|---|---|---|---|---|
| 3 | `build_and_solve_cost_min` | **min** `Σ unit_cost[i] · u[i]` | `unit_cost` | Cheapest way to still meet all demand over horizon `t`. |
| 4 | `build_and_solve_revenue_max` | **max** `Σ f[j] · u[j]` | — (reuses `f`) | Demand-*capped*: `u[j] ≤ d[j] · t`, so you can't "sell" more than the market absorbs. |
| 5 | `build_and_solve_inventory_opt` | **min** `Σ holding_cost[i] · s[i]` | `holding_cost` | Treats inventory `s` as a *decision variable* bounded above by the calibrated ceiling `dataset["s"]`, minimizing tied-up working capital. |
| 6 | `build_and_solve_lead_time_min` | **min** `Σ production_delay[i] · u[i]` | `production_delay` | Delay-weighted production as a **proxy** for cycle-time reduction — not a full critical-path model of elapsed time. |
| 7 | `build_and_solve_fulfillment_rate` | **min** `Σ l[j] / (d[j] · t)` | — | Maximizes average fulfillment rate. Unlike TTR, weights every unit of unmet demand **equally** regardless of product margin. |
| 8 | `build_and_solve_carbon_min` | **min** `Σ emissions_factor[i]·u[i] + Σ transport_emissions[i,j]·y[i,j]` | `emissions_factor`, `transport_emissions` | Production **and** transport emissions; `transport_emissions` is keyed by `(src, tgt)` edge tuples. |

### Design notes worth knowing

- **`revenue_max` flips the demand constraint.** Every other objective requires demand to be *met*; revenue-max instead *caps* production at demand (you optimize the sales mix under a shock, you don't manufacture into a void).
- **`inventory_opt` is the only one that frees `s`.** Everywhere else, `s[i]` (on-hand inventory) is a fixed input. Here it becomes a variable bounded by the calibrated value, so the solver decides how lean it can run.
- **`lead_time_min` and `carbon_min` are explicitly labeled simplifications** in their docstrings — a delay-weighted production sum is a proxy for lead time, not a scheduling model; emissions are a linear per-unit factor, not a full LCA.

---

## The structural resilience metric (non-LP)

### 9. `compute_network_resilience_metrics(dataset)`

This is **not a solve** — it reads the bill-of-materials graph (`P`, `N_minus`) directly, because resilience-to-single-sourcing is a structural property of the topology, not a flow-allocation outcome. Returns:

- `single_point_of_failure_pairs` — count of `(node, material)` pairs with exactly one supplier.
- `avg_suppliers_per_material` — mean supplier count across all `(node, material)` pairs.
- `resilience_score` — per-node dict: the minimum supplier count across that node's `(node, material)` pairs. Any node with a single-sourced material scores `1`. Leaf tier-3 nodes (no BOM) are omitted.

Use it to *find* the nodes worth stress-testing with the LP objectives above, before you spend solve time on them.

---

## Which objective should I use?

| You want to know… | Use |
|---|---|
| Profit lost during a known-length outage | **TTR** |
| How long we can survive an outage with zero demand miss | **TTS** |
| Cheapest feasible response to a shock | `cost_min` |
| Best sales/revenue mix under a shock | `revenue_max` |
| How lean inventory can run and still cope | `inventory_opt` |
| Lead-time/cycle-time-sensitive response | `lead_time_min` |
| Service level treating all products equally | `fulfillment_rate` |
| Lowest-emissions feasible response | `carbon_min` |
| Which nodes are structurally fragile (before solving anything) | `compute_network_resilience_metrics` |

> **Time-phased variants:** only the TTR objective has been re-implemented on a discrete period axis (with lead-time offsets and inventory carryover) so far — see [Multi-Period Planning & Decomposition](multi-period-and-decomposition.md). The other objectives generalize to that skeleton but aren't implemented time-phased yet.

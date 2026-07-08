# Multi-Period Planning & Network Decomposition

[← Back to README](../README.md) · Related: [Optimization Objectives](optimization-objectives.md) · [Disruption Scenarios](disruption-scenarios.md)

The [eight LP objectives](optimization-objectives.md) are all **single-snapshot** models: one scalar horizon `t`, a disruption that's either on or off for the whole horizon, and no notion of a shipment departing now and arriving later. Two newer modules add capabilities modeled after the two biggest structural advantages a commercial planner like **Adexa** has over that kind of model:

1. **`scripts/multi_period_planning.py`** — real time buckets, lead-time-offset flow, and inventory that carries across periods.
2. **`scripts/network_aggregation.py`** — aggregate/disaggregate decomposition (Adexa's Strategic Network Optimizer pattern): shrink the network before solving a what-if, then restore full fidelity only where it matters.

Both are **new and additive**. Neither modifies `scripts/utils.py`, `disruption_scenarios.py`, `realistic_topologies.py`, `dataset_io.py`, or `company_profiles.py` — they reuse them read-only (`_prep_lp_data`, `derive_indexes`). `08_multi_period_planning.ipynb` is a runnable walkthrough; `tests/test_multi_period_planning.py` covers both.

---

## Part 1 — Time-phased multi-period planning

### What changes vs. the single-snapshot LP

| Single-snapshot (`build_and_solve_ttr`) | Time-phased (`build_and_solve_time_phased_ttr`) |
|---|---|
| One scalar horizon `t` | Discrete period index `T = 0..n_periods-1` |
| Disruption on/off for the whole horizon | Disruption is a `period_start` + `duration_periods` **window** |
| Inventory `s[i]` is a fixed input | Inventory `s[i,t]` **carries over** period to period |
| Shipment is instantaneous | Shipment departing period `t` arrives `t + offset[i]` (lead time) |
| Objective: `min Σ f[j]·l[j]` | Objective: `min Σ_t Σ_j f[j]·l[j,t]` |

### Optional schema additions

All optional — omitting them keeps every existing notebook/test unaffected:

| Field | Shape | Used by |
|---|---|---|
| `production_delay` | `{node: lead_time_days}` | Lead-time offset: `offset[i] = ceil(production_delay[i] / period_length_days)` |
| `unit_cost`, `holding_cost`, `emissions_factor` | `{node: value}` | Not consumed by the time-phased TTR objective itself; calibrated by `scenario_calibration.calibrate_cost_fields` for parity with the other LP objectives |
| `transport_emissions` | `{(src, tgt): value}` | Same as above |

### `TimePhasedDisruption`

Replaces a scalar `ttr` with a `period_start`/`duration_periods` window on the period axis — a plant fire doesn't retroactively affect last week's shipments. `from_legacy_scenario` converts an existing `DisruptionScenario`, with `duration_periods = ceil(ttr / period_length_days)` (rounded **up**, never down, so the time-phased footprint is never shorter than the original TTR implies).

### Lead-time-offset worked example

A supplier with `production_delay = 14` days at `period_length_days = 7` has `offset = 2`: a shipment departing that supplier in period `t` can only satisfy downstream demand starting period `t + 2`. With zero pre-horizon in-transit inventory (the default — see limitations), the first 2 periods of a cold-started network show full lost demand for anything sourced exclusively from that supplier, then recovery once the first shipment arrives.

### Baseline comparability

`default_periodic_demand` flat-repeats `d[j] · period_length_days` across every period, so total demand over the horizon matches what a single-snapshot `build_and_solve_ttr` with `t = n_periods · period_length_days` would see — the two engines are directly comparable at baseline. An optional `seasonality` multiplier (from [`sample_seasonality_index`](disruption-scenarios.md#optional-seasonality-band)) layers on top; values are rounded to integers because `l`/`s` are integer-domain variables.

### Key functions

- `build_and_solve_time_phased_ttr(dataset, disruptions, n_periods, period_length_days=7.0, d_t=None, initial_pipeline=None, return_model=False)` — the solve.
- `extract_time_phased_solution(model, dataset)` — long-format per-`(node, period)` table of production/inventory/lost volume for plotting (`lost` is `None` for non-product nodes).

---

## Part 2 — Aggregate/disaggregate decomposition

The single-snapshot and time-phased engines both re-solve the **entire** network at full fidelity for every scenario, even though a single-node disruption only really perturbs its own neighborhood. `scripts/network_aggregation.py` shrinks the model first.

### How it works

1. **Aggregate** (`build_aggregate_dataset`) — pool tier-3 nodes sharing the same `(supplier_material_type, criticality)` into one synthetic group node:
   - capacity `c_agg = Σ c_i`, inventory `s_agg = Σ s_i`, edges = union of members' edges;
   - cost/delay fields aggregated by sum/max/mean as appropriate.
   - **`monopoly_bottleneck` nodes are never pooled** — an aggregate node can't be "35% disrupted," and pooling away the network's actual single points of failure would defeat the whole point of the accelerator.
2. **Compute scope** (`disaggregation_scope`) — which real nodes must be restored to full fidelity for a given disruption:
   1. the disrupted node's entire group, always;
   2. its downstream consumer cone, `expansion_hops` deep (default 1);
   3. its upstream alternate suppliers (the other members of every `P[j][k]` the disrupted node belongs to as a supplier).
3. **Rebuild & re-solve** (`run_aggregate_then_disaggregate`) — solve aggregate → compute scope → rebuild a mixed-fidelity dataset (full detail in scope, aggregated elsewhere) → re-solve.

### Aggregation error is one-directional and optimistic

Pooling capacity/inventory and taking the union of members' edges only ever **loosens** constraints relative to the true network — it can never make the aggregate model pessimistic. A pure-aggregate result should be read as **"at least this good," never as a conservative worst case.** This is exactly why disaggregation restores full fidelity around any actually-disrupted node rather than trusting the aggregate number there.

### Benchmark — read the timings honestly

Medium dataset (~700 nodes), a 4-period disruption at its longest-lead-time node, `n_periods=10`:

| Network variant | Tier-3 nodes | Solve time | `lost_profit` |
|---|---|---|---|
| Full | 480 | ~0.94s (direct) | 16778.56 |
| Aggregate only | 160 | ~0.94s | 16778.56 (matched full *in this instance* — not guaranteed) |
| Partial (disaggregated) | 160 | ~0.92s | 16778.56 (matches full exactly) |

The partial solve reproduces the full network's result exactly while operating on a structurally smaller model — but at this ~700-node scale, combined aggregate+partial wall-clock (~1.9s) was actually **slower** than one direct full-network solve (~1.4s); HiGHS solve time didn't scale down proportionally with node-count reduction here. The benefit is expected to **grow with network size**, not to be a universal speedup at every scale.

---

## Limitations

These apply to the two modules above, in addition to the [base-engine caveats](disruption-scenarios.md#modeling-caveats):

- **Cold-start pipeline** — material in transit *before* the horizon starts is assumed zero unless `initial_pipeline` is supplied, likely understating early-period throughput for any node with `production_delay > 0`. A real deployment would seed this from open purchase orders.
- **End-of-horizon effect** — a shipment departing in the last `offset[i]` periods never arrives in time to be counted, a standard rolling-horizon artifact. A full rolling re-solve (sliding window, commit only the first period) is out of scope.
- **MIP scale growth** — every variable is period-indexed, so model size grows linearly with `n_periods`. Keep demonstrations at simple/medium/complex scale, not planet-scale.
- **Fixed-radius disaggregation scope** — `disaggregation_scope` uses a fixed hop count, not a principled dual-based selection. MIP duals aren't well-defined without relaxing integrality on the aggregate solve, so dual-based scope selection is future work.
- **Optimistic aggregation** — see above; a pure-aggregate result is a lower bound on impact ("at least this good"), not a conservative estimate.
- **Single-objective scope** — only the TTR-style "minimize lost profit" objective is implemented time-phased. The same skeleton generalizes to the [other objectives](optimization-objectives.md) if needed, but that generalization isn't implemented here.

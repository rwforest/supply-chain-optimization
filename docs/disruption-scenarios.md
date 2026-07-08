# Disruption Scenarios & Calibration

[← Back to README](../README.md) · Related: [Optimization Objectives](optimization-objectives.md) · [Multi-Period Planning & Decomposition](multi-period-and-decomposition.md)

An objective function ([see the objectives doc](optimization-objectives.md)) answers *"what's the best outcome under this shock?"* — but only once you've defined **the shock**. This doc covers the disruption-scenario library, how the network parameters are calibrated so the shocks land somewhere realistic, and the modeling caveats that matter when reading results.

## The scenario library

Notebooks `01`–`04` use "one random node, one random recovery time." Notebooks `05`–`08` replace that with named, first-class scenarios defined in `scripts/disruption_scenarios.py`.

There are **three scenario *types*** (not to be confused with the optimization objectives — a scenario is *what breaks*, an objective is *what you optimize* once it's broken):

| Type | What fails | Requires |
|---|---|---|
| **Single-supplier failure** | One node at a time (the original baseline) | Any dataset |
| **Regional disruption** | Every supplier tagged with a given `region`, all at once | A dataset with `region` tags (medium or complex tier) |
| **Material-wide shortage** | Every supplier of one material type, all at once | Any dataset with material types |

`single_supplier_failure_scenarios(dataset)` generates one scenario per tier-2/tier-3 node by default. At large scale, pass `sample_fraction < 1.0` to keep every `monopoly_bottleneck`/`oligopoly` node (the failures that matter most) while randomly sampling the long tail — it prints a summary of what was kept vs. sampled rather than silently dropping coverage.

### Named real-world scenarios

A handful of regional/material scenarios are anchored on real historical events. Each carries a citation in its `real_world_basis` field:

| Scenario | Real-world basis |
|---|---|
| Taiwan drought | 2021 Taiwan water shortage affecting fabs |
| Shanghai lockdown | 2022 Shanghai COVID-19 lockdown |
| Thailand floods | 2011 Thailand floods (hard-disk / component supply) |
| EUV-equipment export halt | Hypothetical, modeled on 2018–2024 US–China semiconductor export controls |
| HBM memory crunch | 2024–2025 high-bandwidth-memory capacity crunch |

`named_real_world_scenarios(dataset)` returns **only** the events whose target region/material is actually present in the given dataset — so running it against the simple tier (which has neither `region` tags nor the relevant materials) returns nothing, and that's expected. These curated scenarios stay exhaustive at any network scale, since there are only a handful of them.

---

## Calibration methodology

The realism of a stress test lives in the numbers attached to each node. Rather than sampling every parameter independently (the `01`–`04` approach), `scripts/scenario_calibration.py` derives them from network structure:

- **Demand propagation** — `compute_required_throughput` propagates tier-1 demand *down* through the bill of materials, so each supplier's load reflects what its downstream actually needs.
- **Criticality-driven buffers** — each supplier's inventory and capacity headroom are sized off its downstream load **and** a criticality tag:
  - `monopoly_bottleneck` — sole, hard-to-replace supplier → thin inventory buffer, tight capacity headroom (naturally fragile).
  - `oligopoly` — a few suppliers → moderate buffers.
  - `diversified_commodity` — many interchangeable suppliers → generous buffers.

  This produces realistic single-points-of-failure **without hand-picking which node ID gets bad numbers.**
- **Skewed demand** — tier-1 demand is drawn from a right-skewed (lognormal) distribution, not uniform.
- **Cost/emissions fields** — `calibrate_cost_fields` populates the optional `unit_cost`, `holding_cost`, `emissions_factor`, `transport_emissions`, and `production_delay` fields that the [added objectives](optimization-objectives.md#the-six-added-lp-objectives) consume.

These patterns are qualitatively inspired by the public [Kaggle "DataCo Smart Supply Chain" dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (skewed order volumes, clustered margin bands). **There is no runtime dependency on Kaggle** — the parameters are hand-calibrated constants documented in the module, so the notebooks stay reproducible without external credentials or internet egress.

### Optional seasonality band

`sample_seasonality_index` samples a per-tier-1-node sinusoidal demand multiplier (mean ~1.0, random per-node amplitude and phase) for the [time-phased engine's](multi-period-and-decomposition.md) `default_periodic_demand(..., seasonality=...)`. Each `IndustryProfile` carries a `seasonality_amplitude_lo/hi` band (defaulted, so no existing profile needed editing). This is **opt-in** — omitting `seasonality` keeps flat-repeat demand.

---

## Modeling caveats

Read results with these in mind:

- **Fixed inventory can mask short disruptions.** The flow-balance constraint lets a disrupted node keep shipping from its pre-existing on-hand inventory even though its own production is halted — and that inventory is a *fixed* quantity, not scaled by disruption duration. A short disruption at a well-stocked node can show **zero** measurable impact while a longer one at the same node shows real impact. This is expected behavior of the single-snapshot model, not a bug. (The [time-phased engine](multi-period-and-decomposition.md) addresses this with real inventory carryover.)
- **TTS "unbounded" → `inf`.** As noted in the [objectives doc](optimization-objectives.md#2-tts--time-to-survive-build_and_solve_tts), an unbounded TTS result is normalized to `inf` rather than reported as the solver's raw stopping value.
- **Fixed 3-tier depth.** The LP models tier1/tier2/tier3; real Nvidia/Apple chains have 4–6 tiers. The complex generators compress this by treating tier 2 as "direct component/module suppliers" and tier 3 as "their equipment/raw-material suppliers."

---

## Real company names — disclaimer

The complex-tier networks anchor tier 2 and tier 3 on **real, publicly-known companies** (TSMC, SK Hynix, ASML, Foxconn, Pegatron, Corning, …), drawn from public secondary reporting about their supplier relationships (see `scripts/company_profiles.py` for the full list and per-anchor citations, current as of 2026-07).

> **Every numeric figure attached to a node — profit margin, inventory, demand, capacity — is synthetic and illustrative.** None represent actual disclosed financial or operational data from Nvidia, Apple, or any named supplier, and must not be used to draw real inferences about those companies. The simple and medium tiers use entirely fictional company names for the same reason.

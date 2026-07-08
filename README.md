<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![DBR](https://img.shields.io/badge/DBR-17.3_LTS_ML-green?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/17.3lts-ml.html)

## Business Problem

Global disruptions from pandemics, wars, and climate events have exposed vulnerabilities in supply chains—resulting in shortages, cost spikes, and reputational damage. Building resilient supply chains enables companies to maintain service levels, capture market share when competitors falter and protect revenue, margins, and brand credibility during crises.

Stress testing simulates extreme but plausible shocks—such as supplier failures, port closures, or demand surges—to reveal hidden risks and single points of failure. By quantifying the financial impact, organizations can prioritize mitigation strategies, diversify sourcing, build targeted inventory buffers, and implement agile decision rules—strengthening adaptability in the face of uncertainty.

This solution accelerator implements the methodology proposed in a [paper](https://dspace.mit.edu/handle/1721.1/101782), which applies stress testing to supply chain networks using digital twins—virtual models constructed from real operational data. Simulating a wide range of disruption scenarios allows businesses to assess potential impacts, identify vulnerabilities, and make proactive, informed decisions.

At the core of this approach is a linear optimization problem, where we optimize the network configuration toward a common objective subject to a set of constraints. This accelerator uses [Pyomo](https://pyomo.readthedocs.io/en/stable/index.html) and [HiGHS](https://github.com/ERGO-Code/HiGHS) to model and solve the optimization problem, and leverages [Ray](https://docs.databricks.com/aws/en/machine-learning/ray/) to scale the process across thousands of simulations.

Databricks is the ideal platform for building this solution. Key advantages include:

1. **Delta Sharing** – Access to up-to-date operational data is vital for resilient supply chain solutions. Delta Sharing enables seamless data exchange between retailers and suppliers—even if one party isn't using Databricks.

2. **Scalability** – Running linear optimization across networks with thousands of nodes and simulating thousands of disruption scenarios is computationally demanding. Databricks provides horizontal scalability to handle these workloads efficiently.

3. **Open Standards** – Databricks integrates smoothly with open-source and third-party tools, allowing teams to use familiar libraries with minimal friction. This flexibility supports custom modeling of business problems and ensures transparency for auditability, validation, and ongoing refinement.

## Reference Architecture

<img src='images/cartoon.png' width=650>

## Realistic Stress-Test Scenario Ladder

Notebooks `01`–`04` generate a synthetic network with independently-random parameters (profit margin, inventory, demand, capacity are all uniform coin flips) and stress-test it by disrupting one random node at a time. That's a clean way to learn the time-to-recover (TTR) / time-to-survive (TTS) methodology, but it doesn't say much about how a *real* supply chain behaves under stress.

Notebooks `05`–`07` add three additional tiers of realism on top of the **exact same, unmodified** optimization engine (`scripts/utils.py::build_and_solve_ttr`/`build_and_solve_tts`) — all the new work goes into topology, data calibration, and disruption-scenario design, not the LP itself.

| Tier | Notebook | Scale | What's different |
|---|---|---|---|
| **Simple** | `06` | ~27 nodes, single product line | Fictional (illustrative) company names; inventory/capacity are sized from BOM-propagated demand and a per-node criticality tag instead of independent randomness |
| **Medium** | `06` | ~700 nodes, multi-product-line | Adds `region`-tagged suppliers (concentrated in real-world sourcing geographies), making regional/correlated disruptions meaningful |
| **Complex** | `07` | ~2,300 nodes each (up to ~76,000+ via the `"planet"` scale preset — see below) | Two networks anchored on **real, publicly-known companies** — an Nvidia-like AI/GPU chip supply chain and an Apple-like consumer-electronics supply chain |

Datasets are generated in `05_realistic_operational_data`, using generators in `scripts/realistic_topologies.py`.

### Real company names — disclaimer

The complex-tier networks anchor tier 2 (direct component/module suppliers) and tier 3 (their equipment/raw-material suppliers) on real companies — e.g. TSMC, SK Hynix, ASML, Foxconn, Pegatron, Corning — drawn from public secondary reporting about these companies' real supplier relationships (see `scripts/company_profiles.py` for the full list and per-anchor citations), current as of 2026-07.

> **Every numeric figure attached to a node in this accelerator — profit margin, inventory, demand, capacity — is synthetic and illustrative.** None of these figures represent actual disclosed financial or operational data from Nvidia, Apple, or any named supplier, and must not be used to draw real inferences about those companies. The simple and medium tiers use entirely fictional company names for the same reason.

### Calibration methodology

Rather than sampling every parameter independently, `scripts/scenario_calibration.py` propagates tier-1 demand down through the bill of materials (`compute_required_throughput`) and sizes each supplier's inventory/capacity off of its actual downstream load and a criticality tag (`monopoly_bottleneck`, `oligopoly`, or `diversified_commodity`) — a sole, hard-to-replace supplier naturally ends up with a thin inventory buffer and tight capacity headroom, while a diversified commodity supplier gets generous buffers, without hand-picking which node ID gets bad numbers. Tier-1 demand is drawn from a right-skewed (lognormal) distribution rather than uniform. These patterns are qualitatively inspired by the public [Kaggle "DataCo Smart Supply Chain" dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (skewed order volumes, clustered margin bands) — there is **no runtime dependency on Kaggle credentials or a live download**; the parameters are hand-calibrated constants, documented in that module, so the notebooks stay reproducible without external credentials or guaranteed internet egress.

### Disruption-scenario library

`scripts/disruption_scenarios.py` replaces "one random node, one random recovery time" with named, first-class scenarios:

| Type | Description |
|---|---|
| Single-supplier failure | The original baseline — one node at a time |
| Regional disruption | Every supplier tagged with a region fails at once (requires a dataset with `region` tags — medium or complex tier) |
| Material-wide shortage | Every supplier of one material type fails at once |

A handful of the regional/material scenarios are inspired by real historical events (each carries a citation in `real_world_basis`): the 2021 Taiwan drought, the 2022 Shanghai COVID-19 lockdown, the 2011 Thailand floods, a hypothetical EUV-equipment export halt (modeled on 2018-2024 US-China semiconductor export controls), and the 2024-2025 HBM memory capacity crunch. `named_real_world_scenarios(dataset)` only returns the events whose target region/material is actually present in the given dataset.

### Modeling caveat

The flow-balance constraint in the (unmodified) LP lets a disrupted node keep shipping from its pre-existing on-hand inventory even though its own production is halted — and that inventory is a *fixed* quantity, not scaled by how long the disruption lasts. A short disruption at a well-stocked node can show zero measurable impact while a longer one at the same node shows real impact; this is expected behavior of the underlying model, not a bug. Separately, `run_scenario_tts` normalizes an "unbounded" TTS solver result (the network has enough redundancy/headroom to absorb the disruption indefinitely) to `tts = inf`, since the solver's raw objective value in that case is a meaningless artifact of wherever the simplex method stopped, not a real bound.

### How to run

Run `05_realistic_operational_data` to generate the four datasets, then `06_realistic_stress_testing (simple and medium)` (single-node, direct-loop style, mirrors `02`) and `07_realistic_stress_testing (complex network)` (Ray-distributed, mirrors `03`). Tests for the generators and scenario library live in `tests/test_realistic_scenarios.py` — run with `python -m pytest tests/ -v` from the repo root (requires Python 3.12 for the `pyomo`/`highspy` wheels pinned in `uv.lock`).

`07`'s "Optional: Planet-Scale Sample Sweep" section is controlled by a `run_planet_scale` notebook widget (`"yes"`/`"no"`, default `"yes"`) — set it to `"no"` (via the notebook UI or a job's `base_parameters`) to skip the ~76,000-node sweep and only run the main ~2,300-node complex pipeline.

### Deploying as a Databricks Job

All three realistic notebooks (plus `requirements.txt` and `scripts/`) can be run end-to-end as a multi-task Databricks Job: `05` on a single-node job cluster, fanning out to `06` (same single-node cluster) and `07` (a Ray-capable autoscaling job cluster — `Standard_E4d_v4` workers, `Standard_DS4_v2` driver, matching `07`'s documented cluster spec) in parallel. Set the `stress_test_complex` task's `run_planet_scale` base parameter to `"no"` for a faster run that skips the planet-scale sweep.

### Scaling to planet-scale networks

`generate_complex_network` accepts a `scale_factor` (default `1.0`, today's unmodified behavior) that proportionally scales every real anchor's synthetic peer/child count ranges (`AnchorSpec.n_synthetic_peers`/`n_synthetic_children` in `scripts/company_profiles.py`) before drawing from them. Without this, simply raising `n2_target`/`n3_target` would dilute the network toward generic padding — almost none of the new nodes would trace back to a real, named anchor like TSMC or ASML. `SCALE_PRESETS` in `scripts/realistic_topologies.py` bundles a tuned `"planet"` preset (~76,000 total nodes, `scale_factor=25.0`) alongside today's `"complex"` default; use it via `generate_complex_network_at_scale(company, scale="planet")`.

Node generation itself is cheap and scales well past planet-scale node counts (measured on the repo's dev environment): ~420,000 total nodes generate in ~8.5 seconds. The real ceiling is LP solve time, since `build_and_solve_ttr`/`build_and_solve_tts` re-solve a Pyomo model over the *entire* network for every scenario, regardless of which single node is disrupted:

| Total nodes | Generation time | Solve time per scenario |
|---|---|---|
| 2,356 (today's `"complex"`) | <0.1s | ~0.6s |
| 23,506 | ~0.2s | ~4.7s |
| 95,006 | ~1.4s | ~16.4s |
| 190,006 | ~3.6s | ~42s |

Because `single_supplier_failure_scenarios` generates one scenario per tier-2/tier-3 node by default, an exhaustive sweep at planet scale means tens or hundreds of thousands of whole-network solves. Pass `sample_fraction < 1.0` to keep every `monopoly_bottleneck`/`oligopoly` node (the failures that matter most) while randomly sampling the rest — the function prints a summary of what was kept versus sampled rather than silently dropping coverage. The curated `named_real_world_scenarios` (a handful of regional/material events) stay exhaustive at any scale since there are only a handful of them.

This intentionally does not attempt to make an unbounded number of nodes solve in constant time — that would require re-architecting the LP itself (e.g. decomposition, warm starts, a different solver), which is out of scope here since the LP is deliberately left unmodified across this entire realistic-data effort.

### Scale-free padding attachment

Real supply-chain networks are hub-heavy: a handful of suppliers (TSMC, Foxconn) carry far more downstream connections than the long tail of small ones — this is a well-documented property in supply-chain network research, not an artifact of which companies happen to be famous. `generate_complex_network`'s real, named anchors already reflect this by construction (their `AnchorSpec.n_synthetic_children` ranges are hand-tuned to be hub-like). But the generic padding nodes added to reach `n2_target`/`n3_target` used to attach to a *uniformly random* tier-2 parent, so at planet scale — where padding vastly outnumbers the ~15 real anchors — the bulk of the network ended up flatter than real supply chains actually look.

Padding tier-3 nodes now attach via preferential attachment (`PreferentialAttachmentPool` in `scripts/realistic_topologies.py`): each new node's parent is drawn with probability proportional to `existing_degree + 1`, the same growth rule behind a Barabasi-Albert scale-free graph, using the standard O(1)-amortized "weighted pool" trick so it stays cheap at tens of thousands of draws. This is implemented directly rather than by taking a `networkx` dependency — no new package, no change to the LP or dataset schema, and `scale_factor=1.0`'s node/edge *counts* are unaffected (only *which* tier-2 node each padding node attaches to changes).

### Known limitations

- The `region`, `company_name`, and `criticality` fields are not yet passed through `scripts/dataset_io.py`'s `explode_dataset_to_tables`, so they aren't queryable via the Genie space or `agent/supply_chain_agent.py` today — wiring those through is a natural follow-up.
- The LP remains a fixed 3-tier model (tier1/tier2/tier3); real Nvidia/Apple chains have more like 4-6 tiers. The complex-tier generators compress this by treating tier 2 as "direct component/module suppliers" and tier 3 as "their equipment/raw-material suppliers," which is a simplification, not a literal reproduction of either company's actual supply chain depth.

## 🚀 Getting Started

For detailed step-by-step instructions on setting up and running the frontend UI integration, please see **[RUNME.md](./RUNME.md)** which provides:

- Complete environment setup guide
- Databricks credentials configuration
- Frontend build and deployment instructions
- Database schema setup
- Troubleshooting and maintenance tips

## [Vector Lab](https://www.youtube.com/@VectorLab) - Stress Testing Supply Chains with Digital Twins on Databricks

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/yRR_QAm5npw/0.jpg)](https://www.youtube.com/watch?v=yRR_QAm5npw)


## Authors

<ryuta.yoshimatsu@databricks.com>,  <luis.herrera@databricks.com>, <puneet.jain@databricks.com>

## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 

## License

&copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD-3 | https://pypi.org/project/pyomo/
| highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
| ray | Framework for scaling AI/Python applications | Apache 2.0 | https://github.com/ray-project/ray

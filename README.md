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
| **Complex (MPS)** | `09` | ~2,300 nodes (up to ~76,000 via the `"planet"` preset) | A third real-company anchor — a **Monolithic Power Systems-like** fabless PMIC/power-module supply chain (VIS foundry + captive Chengdu wafer-sort/final-test + OSAT partners + Penang hub). Notebook `09` layers three GPU/CPU operations-research models on top of it (see below); a `scale` widget switches it between complex and planet scale. |

Datasets are generated in `05_realistic_operational_data`, using generators in `scripts/realistic_topologies.py`.

### GPU-accelerated OR pipeline (MPS + NVIDIA cuOpt)

Notebook `09_mps_cuopt_pipeline` goes beyond stress-testing the LP: it derives three classic operations-research problems from the network and solves them — **FJSP** (flexible job-shop scheduling of back-end assembly/test) and **CVRPTW** (capacitated vehicle routing with time windows for distribution) on **NVIDIA cuOpt's GPU solvers**, plus **MEIO** (multi-echelon inventory optimization via the Guaranteed Service Model) on the existing Pyomo/HiGHS **CPU** stack. The problems are derived from the generated network (`scripts/mps_derivation.py`), not invented from scratch. MPS was the first example, but the derivation is **company-parameterized** (a `DerivationConfig` + a `company` widget) — `09` runs the same pipeline on the MPS network **or** on `apple_real`, a graph grounded in Apple's *published* supplier list (`scripts/apple_supplier_list.py`). A `scale` widget runs `09` on either the ~2,300-node complex MPS network or the ~76,000-node planet-scale one: FJSP machines and CVRPTW depots stay scale-invariant (keyed off MPS's real named back-end anchors, not the synthetic fan-out), and MEIO subsamples the network — always keeping tier-1, named anchors, and monopoly/oligopoly nodes — to stay CPU-tractable (the full 76k-node GSM MILP takes ~23 min in HiGHS; a ~2,500-node sample solves in seconds within ~0.6% of the exact objective). See the [MPS cuOpt pipeline guide](docs/mps-cuopt-pipeline.md).

> **cuOpt-on-Databricks support caveat:** running NVIDIA cuOpt on Databricks serverless GPU compute is technically feasible but is **not officially supported or documented by Databricks or NVIDIA** as of this writing. The GPU sections of `09` are gated behind a `run_gpu` widget (default `"no"`) so the derivation layer and the CPU MEIO solve run end-to-end on any cluster; the cuOpt wheels install separately from `requirements-gpu.txt` (NVIDIA's package index), never from `requirements.txt`.

### Real company names — disclaimer

The complex-tier networks anchor tier 2 (direct component/module suppliers) and tier 3 (their equipment/raw-material suppliers) on real companies — e.g. TSMC, SK Hynix, ASML, Foxconn, Pegatron, Corning — drawn from public secondary reporting about these companies' real supplier relationships (see `scripts/company_profiles.py` for the full list and per-anchor citations), current as of 2026-07.

The **`apple_real`** network (a second `09` option) is grounded in a **primary source** — Apple's official [Supplier List](Apple-Supplier-List.md) (FY2023). `scripts/apple_supplier_list.py` parses it: every anchor's company name and manufacturing regions come straight from that document (~183 real suppliers); a curated lookup classifies the ~40 recognizable names (Foxconn, Pegatron, Amkor, ASE, TSMC, BOE, LG Display, SK hynix, Corning, Qualcomm, …) into tier/role, and the rest become generic tier-3 components. Names and regions are real/published; the numbers and tier/role assignment remain synthetic.

The **MPS-like** network (used by `09`) anchors on Monolithic Power Systems' publicly-reported structure: **Vanguard International Semiconductor (VIS)** and other foundry partners, MPS's wholly-owned **Chengdu** wafer-sort/final-test facility, OSAT packaging partners, and the **Penang** engineering hub, plus tier-3 equipment/material vendors (Applied Materials, Tokyo Electron, Shin-Etsu, Resonac, Amphenol). Three modeling choices are deliberate and documented in `scripts/company_profiles.py`: (1) the MPS-owned Chengdu/Penang facilities are **captive**, not third-party suppliers, but are modeled as tier-2 nodes because the LP schema only has tier1/tier2/tier3 slots; (2) the OSAT partners use **generic labels** since the source doc doesn't name them; (3) **ASML is deliberately excluded** — MPS's mature-node BCD process doesn't use EUV lithography.

> **Every numeric figure attached to a node in this accelerator — profit margin, inventory, demand, capacity — is synthetic and illustrative.** None of these figures represent actual disclosed financial or operational data from Nvidia, Apple, MPS, or any named supplier, and must not be used to draw real inferences about those companies. The simple and medium tiers use entirely fictional company names for the same reason.

### Deep-dive documentation

The mechanics of *what* gets optimized, *what* breaks, and the newer Adexa-inspired capabilities are documented in dedicated guides under [`docs/`](docs/):

| Guide | Covers |
|---|---|
| **[Optimization Objectives](docs/optimization-objectives.md)** | All 9 objectives — the original **TTR** & **TTS**, the 6 added LP objectives (`cost_min`, `revenue_max`, `inventory_opt`, `lead_time_min`, `fulfillment_rate`, `carbon_min`), and the structural `network_resilience` metric — with each objective function, its required fields, and caveats. |
| **[Disruption Scenarios & Calibration](docs/disruption-scenarios.md)** | The 3 scenario *types* (single-supplier / regional / material-wide), the named real-world events, the structure-driven calibration methodology, and the modeling caveats. |
| **[Multi-Period Planning & Network Decomposition](docs/multi-period-and-decomposition.md)** | The two Adexa-inspired additive modules: time-phased multi-period planning (lead-time offsets, inventory carryover, per-period disruption windows) and aggregate/disaggregate decomposition. |
| **[MPS cuOpt Pipeline (FJSP / CVRPTW / MEIO)](docs/mps-cuopt-pipeline.md)** | The three operations-research formulations in notebook `09`, the `scripts/mps_derivation.py` translation layer, the unofficial cuOpt-on-Databricks-GPU support caveat, and the MEIO piecewise-linear-vs-nonlinear decision. |

**Objectives, in brief:** `scripts/utils.py` has **8 `build_and_solve_*` LP objectives** (TTR + TTS + 6 added) plus **1 non-LP structural metric** (`compute_network_resilience_metrics`) = **9 total**. All 6 added objectives are purely additive — they reuse the `_prep_lp_data` skeleton and never modify `ttr`/`tts`. See the [objectives guide](docs/optimization-objectives.md) for the full breakdown.

**Scenarios, in brief:** `scripts/disruption_scenarios.py` replaces "one random node, one random recovery time" with named single-supplier, regional, and material-wide failures — several anchored on real historical events (2021 Taiwan drought, 2022 Shanghai lockdown, 2011 Thailand floods, an EUV-export-halt scenario, the 2024-25 HBM crunch). Network parameters are calibrated from BOM structure and per-node criticality tags rather than independent randomness. See the [scenarios guide](docs/disruption-scenarios.md).

**Multi-period & decomposition, in brief:** `scripts/multi_period_planning.py` and `scripts/network_aggregation.py` add time-phased planning and aggregate/disaggregate decomposition, both modeled after Adexa's structural advantages over a single-snapshot LP and both purely additive (walkthrough in `08_multi_period_planning.ipynb`). See the [multi-period guide](docs/multi-period-and-decomposition.md).

### Where to start

New to the repo or unsure where to begin? Open **`00_MASTER_start_here.ipynb`** first. It maps every notebook and module, runs an environment check, executes a CPU-only smoke test that proves the core engines work end-to-end (LP stress test + FJSP/CVRPTW derivation + MEIO solve, for both the `mps` and `apple_real` graphs), runs the test suite, and gives a decision guide pointing you to the right notebook for what you want to do.

### How to run

Run `05_realistic_operational_data` to generate the four datasets, then `06_realistic_stress_testing (simple and medium)` (single-node, direct-loop style, mirrors `02`), `07_realistic_stress_testing (complex network)` (Ray-distributed, mirrors `03`), and `08_multi_period_planning` (single-node, time-phased planning + network decomposition, mirrors `06`'s cluster spec). Tests for the generators and scenario library live in `tests/test_realistic_scenarios.py`; tests for the time-phased engine and network decomposition live in `tests/test_multi_period_planning.py` — run either with `python -m pytest tests/ -v` from the repo root (requires Python 3.12 for the `pyomo`/`highspy` wheels pinned in `uv.lock`).

`05`'s dataset generation is controlled by a `regenerate_data` notebook widget (`"yes"`/`"no"`, default `"no"`) — since every generator is deterministically seeded, re-running `05` always reproduces the same six datasets, so by default it skips generation and loads the existing files from the volume instead of redoing the (redundant) work. Set it to `"yes"` (via the notebook UI or a job's `base_parameters`) to force a fresh regenerate-and-overwrite, e.g. after changing a generator or its seed.

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
- Notebook `09_mps_cuopt_pipeline` is **GPU-optional and outside** the `05`→`08` Databricks multi-task Job graph in v1. Its FJSP/CVRPTW cuOpt solves require a CUDA-12 GPU and install from `requirements-gpu.txt` (NVIDIA's package index) — a combination that is technically feasible but not officially supported/documented by Databricks or NVIDIA. The derivation layer and the CPU MEIO solve run on any cluster; the GPU sections are gated behind `run_gpu` (default `"no"`). The full agentic orchestration pipeline (Adexa → LLM/NIM → LangChain → cuOpt → Adexa S&OE) from the source design doc is a documented future extension, not implemented in v1.

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
| cuopt-cu12 | NVIDIA GPU-accelerated decision-optimization engine (notebook `09`, GPU-only) | NVIDIA proprietary — license terms pending verification; confirm at source before redistribution | https://pypi.org/project/cuopt-cu12/
| cudf-cu12 | NVIDIA RAPIDS GPU DataFrame library (notebook `09`, GPU-only) | Apache 2.0 — confirm at source before redistribution | https://pypi.org/project/cudf-cu12/

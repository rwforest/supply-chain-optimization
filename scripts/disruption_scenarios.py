"""Named disruption-scenario library, replacing the "one random node with a
random time-to-recover" pattern used in the original notebooks with
first-class, citable scenarios: single-supplier failure, whole-region
disruption, and material-wide shortage.

``build_and_solve_ttr``/``build_and_solve_tts`` (in ``scripts/utils.py``,
left unmodified) already accept an arbitrary list of disrupted nodes, so
multi-node disruption works mechanically with zero solver changes. Both
models are always feasible/bounded regardless of which or how many nodes are
disrupted (traced by hand: in the TTR model, ``l`` — lost volume — is
unbounded above and appears only in the minimized objective, so routing all
demand to loss always satisfies every constraint; in the TTS model, ``t``'s
effective lower bound is 0, so ``t=0`` with zero production always satisfies
every constraint). The one real failure mode is a bad node ID in a scenario
causing Pyomo to raise at model-construction time — ``validate_scenario``
below turns that into an early, clear ``ValueError``.

Modeling caveat (a property of the existing, unmodified LP, not a bug here):
the flow-balance constraint lets a disrupted node still ship from its
pre-existing inventory even though its own production is halted, and that
inventory is a fixed quantity, not scaled by the disruption's duration. A
short disruption at a node with a deep inventory buffer can therefore show
*zero* impact while a longer one at the same node shows severe impact — this
is expected, not a sign the model "isn't working."
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import pandas as pd

import scripts.utils as utils

ScenarioType = Literal["single_node", "regional", "material_shortage"]


@dataclass(frozen=True)
class DisruptionScenario:
    scenario_id: str
    name: str
    description: str
    real_world_basis: str  # citation/inspiration text; "" if purely illustrative
    scenario_type: ScenarioType
    disrupted_nodes: list[str]
    ttr: float  # used by run_scenario_ttr; ignored by run_scenario_tts


def validate_scenario(dataset: dict, scenario: DisruptionScenario) -> None:
    """Raise a clear ``ValueError`` if a scenario references a node that
    isn't in the dataset, instead of letting Pyomo fail with an opaque
    ``KeyError`` at model-construction time."""
    known = set(dataset["tier1"]) | set(dataset["tier2"]) | set(dataset["tier3"])
    unknown = [n for n in scenario.disrupted_nodes if n not in known]
    if unknown:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} references unknown node(s): {unknown}"
        )


def single_supplier_failure_scenarios(
    dataset: dict,
    rng: random.Random,
    ttr_lo: float = 1,
    ttr_hi: float = 10,
    sample_fraction: float = 1.0,
    always_include_criticality: tuple[str, ...] = ("monopoly_bottleneck", "oligopoly"),
) -> list[DisruptionScenario]:
    """One scenario per tier-2/tier-3 node — the original notebooks'
    baseline behavior, reproduced as a first-class named-scenario type.

    At planet-scale node counts (see ``scripts.realistic_topologies.
    SCALE_PRESETS["planet"]``), an exhaustive one-scenario-per-node sweep
    means re-solving the whole-network LP tens/hundreds of thousands of
    times. ``sample_fraction < 1.0`` keeps every node whose ``criticality``
    tag is in ``always_include_criticality`` (the failures that actually
    matter most) and randomly samples that fraction of the remaining nodes
    via ``rng``, printing a summary instead of silently dropping coverage.
    Nodes without a ``criticality`` tag (e.g. the original ``generate_data``
    output) are treated as sampling-eligible, not always-included.
    """
    nodes = dataset["tier2"] + dataset["tier3"]

    if sample_fraction >= 1.0:
        selected = nodes
    else:
        criticality = dataset.get("criticality", {})
        always = [n for n in nodes if criticality.get(n) in always_include_criticality]
        rest = [n for n in nodes if criticality.get(n) not in always_include_criticality]
        n_sampled = round(sample_fraction * len(rest))
        sampled = rng.sample(rest, min(n_sampled, len(rest)))
        selected = always + sampled
        print(
            f"[single_supplier_failure_scenarios] sampled {len(sampled)}/{len(rest)} "
            f"non-critical nodes ({len(always)} always-included by criticality), "
            f"{len(selected)}/{len(nodes)} total"
        )

    scenarios = []
    for node in selected:
        ttr = rng.randint(int(ttr_lo), int(ttr_hi))
        scenarios.append(
            DisruptionScenario(
                scenario_id=f"single::{node}",
                name=f"Single-supplier failure: {node}",
                description=f"Node {node} fails completely for {ttr} time units.",
                real_world_basis="",
                scenario_type="single_node",
                disrupted_nodes=[node],
                ttr=ttr,
            )
        )
    return scenarios


def regional_disruption_scenario(
    dataset: dict,
    region: str,
    ttr: float,
    name: str | None = None,
    real_world_basis: str = "",
) -> DisruptionScenario:
    """Disrupt every supplier tagged with ``region`` at once. Requires the
    dataset to carry a ``region`` key (the ``generate_data`` output from
    ``scripts/utils.py`` does not have one)."""
    region_map = dataset.get("region")
    if not region_map:
        raise ValueError(
            "dataset has no region tags; use generate_medium_network or "
            "generate_complex_network from scripts.realistic_topologies"
        )
    nodes = [n for n, r in region_map.items() if r == region]
    if not nodes:
        raise ValueError(f"no nodes tagged with region {region!r} in this dataset")
    return DisruptionScenario(
        scenario_id=f"regional::{region}",
        name=name or f"Regional disruption: {region}",
        description=f"All {len(nodes)} supplier(s) in {region} fail simultaneously for {ttr} time units.",
        real_world_basis=real_world_basis,
        scenario_type="regional",
        disrupted_nodes=nodes,
        ttr=ttr,
    )


def material_shortage_scenario(
    dataset: dict,
    material_type: str,
    ttr: float,
    name: str | None = None,
    real_world_basis: str = "",
) -> DisruptionScenario:
    """Disrupt every supplier of ``material_type`` at once. Works on any
    dataset, including the original ``generate_data`` output, since
    ``supplier_material_type`` is part of the required schema."""
    nodes = [
        n for n, m in dataset["supplier_material_type"].items() if m == material_type
    ]
    if not nodes:
        raise ValueError(f"no suppliers of material {material_type!r} in this dataset")
    return DisruptionScenario(
        scenario_id=f"material::{material_type}",
        name=name or f"Material-wide shortage: {material_type}",
        description=f"All {len(nodes)} supplier(s) of {material_type!r} fail simultaneously for {ttr} time units.",
        real_world_basis=real_world_basis,
        scenario_type="material_shortage",
        disrupted_nodes=nodes,
        ttr=ttr,
    )


# Curated real-world-inspired events. Each is only materialized into a
# DisruptionScenario if its target region/material is actually present in
# the given dataset (named_real_world_scenarios skips the rest with a
# printed note rather than raising).
_REAL_WORLD_EVENTS: list[dict] = [
    {
        "scenario_type": "regional",
        "target": "Taiwan",
        "ttr": 14,
        "name": "2021 Taiwan Drought",
        "real_world_basis": (
            "2021 Taiwan drought — water rationing forced semiconductor fabs "
            "to truck in water and curtail output."
        ),
    },
    {
        "scenario_type": "regional",
        "target": "China",
        "ttr": 30,
        "name": "2022 Shanghai COVID-19 Lockdown",
        "real_world_basis": (
            "2022 COVID-19 Shanghai lockdown — halted electronics assembly "
            "and logistics across the Pudong/Kunshan region."
        ),
    },
    {
        "scenario_type": "regional",
        "target": "Thailand",
        "ttr": 45,
        "name": "2011 Thailand Floods",
        "real_world_basis": (
            "2011 Thailand floods — disrupted component manufacturing in "
            "the Nakhon Pathom/Ayutthaya industrial estates."
        ),
    },
    {
        "scenario_type": "regional",
        "target": "Netherlands",
        "ttr": 60,
        "name": "Hypothetical EUV Equipment Export Halt",
        "real_world_basis": (
            "Illustrative scenario modeled on 2018-2024 US-China "
            "semiconductor export-control tightening."
        ),
    },
    {
        "scenario_type": "material_shortage",
        "target": "hbm_memory",
        "ttr": 21,
        "name": "HBM Capacity Crunch",
        "real_world_basis": (
            "2024-2025 HBM (high-bandwidth memory) capacity crunch reported "
            "across the AI accelerator supply chain."
        ),
    },
]


def named_real_world_scenarios(dataset: dict) -> list[DisruptionScenario]:
    """Build the subset of ``_REAL_WORLD_EVENTS`` whose target region or
    material is actually present in ``dataset``."""
    available_regions = set(dataset.get("region", {}).values())
    available_materials = set(dataset["supplier_material_type"].values())

    scenarios: list[DisruptionScenario] = []
    for event in _REAL_WORLD_EVENTS:
        if event["scenario_type"] == "regional":
            if event["target"] not in available_regions:
                print(
                    f"[named_real_world_scenarios] skipping {event['name']!r}: "
                    f"region {event['target']!r} not present in this dataset"
                )
                continue
            scenario = regional_disruption_scenario(
                dataset, event["target"], event["ttr"],
                name=event["name"], real_world_basis=event["real_world_basis"],
            )
        elif event["scenario_type"] == "material_shortage":
            if event["target"] not in available_materials:
                print(
                    f"[named_real_world_scenarios] skipping {event['name']!r}: "
                    f"material {event['target']!r} not present in this dataset"
                )
                continue
            scenario = material_shortage_scenario(
                dataset, event["target"], event["ttr"],
                name=event["name"], real_world_basis=event["real_world_basis"],
            )
        else:  # pragma: no cover - defensive, no such entries today
            continue
        scenarios.append(scenario)
    return scenarios


def run_scenario_ttr(dataset: dict, scenario: DisruptionScenario) -> pd.DataFrame:
    """Run the (unmodified) TTR solver for a named scenario."""
    validate_scenario(dataset, scenario)
    df = utils.build_and_solve_ttr(dataset, scenario.disrupted_nodes, scenario.ttr)
    df.insert(0, "scenario_id", scenario.scenario_id)
    df.insert(1, "scenario_name", scenario.name)
    df.insert(2, "scenario_type", scenario.scenario_type)
    return df


def run_scenario_tts(dataset: dict, scenario: DisruptionScenario) -> pd.DataFrame:
    """Run the (unmodified) TTS solver for a named scenario.

    When the network has enough redundancy/headroom that it can absorb the
    disruption indefinitely, HiGHS reports an "unbounded" termination
    condition — and in that case the raw objective value it still returns
    is an artifact of wherever the simplex method stopped, not a meaningful
    bound. This wrapper detects that and reports ``tts = inf`` instead (a
    genuinely good, meaningful outcome: the network survives the disruption
    indefinitely), keeping the raw solver value in ``tts_raw`` for
    transparency.
    """
    validate_scenario(dataset, scenario)
    df = utils.build_and_solve_tts(dataset, scenario.disrupted_nodes)
    df["tts_raw"] = df["tts"]
    if str(df.iloc[0]["termination_condition"]).lower() != "optimal":
        df["tts"] = float("inf")
    df.insert(0, "scenario_id", scenario.scenario_id)
    df.insert(1, "scenario_name", scenario.name)
    df.insert(2, "scenario_type", scenario.scenario_type)
    return df

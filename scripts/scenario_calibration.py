"""Correlated, topology-aware parameter calibration for the realistic network
generators in ``scripts/realistic_topologies.py``.

The original ``scripts.utils.generate_data`` draws every parameter
(``f``, ``s``, ``d``, ``c``) independently and uniformly at random, which
produces uncorrelated, toy-looking data. Here, tier-1 demand/margin/inventory
follow a right-skewed (lognormal) distribution, and tier-2/tier-3 inventory
and capacity are sized off of *actual* downstream throughput (propagated
through the bill of materials) combined with a per-node "criticality" tag
(monopoly bottleneck vs. oligopoly vs. diversified commodity) — so a sole,
hard-to-replace supplier naturally ends up with tight capacity headroom and
long safety-stock coverage, without hand-picking which node ID gets bad
numbers.

These patterns are qualitatively inspired by the public Kaggle "DataCo Smart
Supply Chain for Big Data Analysis" dataset (profit ratios cluster in a band
with a low-margin tail; order volumes are right-skewed; leaner inventory
buffers correlate with higher late-delivery risk). This module does not
download or otherwise depend on that dataset at runtime — no Kaggle API
credentials or guaranteed internet egress are required to run these
notebooks reproducibly. The parameters below are hand-calibrated constants,
documented here for review.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class IndustryProfile:
    margin_lo: float
    margin_hi: float
    demand_mu: float  # lognormal mu (log-space mean) for tier-1 daily demand
    demand_sigma: float  # lognormal sigma (log-space stdev)
    dos_lo: float  # tier-1 days-of-supply -> inventory
    dos_hi: float
    capacity_headroom_lo: float  # tier-1 capacity = demand * (1 + headroom)
    capacity_headroom_hi: float
    unit_cost_lo: float  # tier-1 baseline unit cost band
    unit_cost_hi: float
    emissions_lo: float  # tier-1 baseline emissions-per-unit band
    emissions_hi: float
    # Opt-in band for scripts.multi_period_planning's seasonal demand
    # multiplier (see sample_seasonality_index below) — defaulted so none of
    # the four INDUSTRY_PROFILES entries below need to be edited to add it.
    seasonality_amplitude_lo: float = 0.05
    seasonality_amplitude_hi: float = 0.25


INDUSTRY_PROFILES: dict[str, IndustryProfile] = {
    "simple": IndustryProfile(0.08, 0.22, math.log(700), 0.35, 10, 25, 0.15, 0.45, 5, 20, 1, 5),
    "medium": IndustryProfile(0.05, 0.28, math.log(900), 0.50, 7, 20, 0.10, 0.40, 8, 30, 2, 8),
    "nvidia": IndustryProfile(0.35, 0.65, math.log(1500), 0.60, 5, 15, 0.05, 0.25, 50, 200, 5, 20),
    "apple": IndustryProfile(0.20, 0.45, math.log(5000), 0.60, 5, 15, 0.05, 0.25, 30, 150, 3, 15),
    # MPS (fabless PMIC / power-module maker): margin band 0.40-0.62 reflects
    # MPS's real ~55-60% gross margins (near/slightly below Nvidia's); demand
    # mu = log(600) is well below Nvidia (1500)/Apple (5000), reflecting much
    # smaller per-line unit volume; unit_cost (20-80) and emissions (2-10) sit
    # between Nvidia's GPU-scale figures and the generic/medium tiers, since
    # power ICs are simpler/cheaper per unit than GPUs but pricier than
    # commodity auto parts. All values are illustrative/hand-calibrated — the
    # same caveat class as the four entries above, NOT real MPS financial data.
    "mps": IndustryProfile(0.40, 0.62, math.log(600), 0.55, 5, 15, 0.05, 0.25, 20, 80, 2, 10),
    # apple_real (Apple's published supplier list): shares the "apple"
    # consumer-electronics band — moderate margins, very high per-line unit
    # volume. Only the anchor names/regions differ (real/published); the
    # illustrative calibration band is intentionally identical.
    "apple_real": IndustryProfile(0.20, 0.45, math.log(5000), 0.60, 5, 15, 0.05, 0.25, 30, 150, 3, 15),
}


@dataclass(frozen=True)
class CriticalityProfile:
    safety_days_lo: float
    safety_days_hi: float
    capacity_multiplier_lo: float
    capacity_multiplier_hi: float
    cost_multiplier_lo: float  # unit_cost = base_unit_cost(material) * cost_multiplier
    cost_multiplier_hi: float
    holding_cost_rate_lo: float  # holding_cost = unit_cost * holding_cost_rate (per period)
    holding_cost_rate_hi: float
    lead_time_days_lo: float  # production_delay
    lead_time_days_hi: float
    emissions_multiplier_lo: float  # emissions_factor = base_emissions(material) * emissions_multiplier
    emissions_multiplier_hi: float


CRITICALITY_PROFILES: dict[str, CriticalityProfile] = {
    # A monopoly bottleneck (e.g. TSMC, ASML) gets a SMALL safety-stock
    # buffer and TIGHT capacity headroom: there is no alternate supplier to
    # fall back on and no idle capacity to absorb a shock, so once its own
    # (thin) inventory runs out, a disruption bites immediately. It also
    # commands a cost/emissions premium (specialized, hard-to-substitute
    # process technology), a long lead time (no spare capacity to expedite),
    # and a low holding-cost rate (thin buffer means little capital tied up).
    "monopoly_bottleneck": CriticalityProfile(3, 10, 1.02, 1.15, 1.8, 3.0, 0.02, 0.03, 30, 90, 1.5, 2.5),
    "oligopoly": CriticalityProfile(10, 25, 1.15, 1.6, 1.2, 1.8, 0.025, 0.04, 15, 45, 1.1, 1.6),
    # A diversified commodity supplier is cheap/easy to stock and easy to
    # replace: generous buffer, generous spare capacity, low risk, low cost
    # premium, short lead time.
    "diversified_commodity": CriticalityProfile(20, 45, 1.6, 3.0, 0.7, 1.1, 0.03, 0.05, 3, 15, 0.7, 1.1),
    "generic": CriticalityProfile(10, 30, 1.3, 2.2, 1.0, 1.4, 0.025, 0.045, 7, 25, 0.9, 1.3),
}


def sample_tier1_params(
    rng: random.Random, tier1: list[str], profile: IndustryProfile
) -> tuple[dict[str, float], dict[str, int], dict[str, int], dict[str, int]]:
    """Sample profit margin, inventory, demand and capacity for finished
    products (tier 1) from a right-skewed demand distribution."""
    f, s, d, c = {}, {}, {}, {}
    for node in tier1:
        f[node] = round(rng.uniform(profile.margin_lo, profile.margin_hi), 2)
        demand = max(1, round(rng.lognormvariate(profile.demand_mu, profile.demand_sigma)))
        dos = rng.uniform(profile.dos_lo, profile.dos_hi)
        headroom = rng.uniform(profile.capacity_headroom_lo, profile.capacity_headroom_hi)
        d[node] = demand
        s[node] = max(1, round(demand * dos))
        c[node] = max(demand, round(demand * (1 + headroom)))
    return f, s, d, c


def sample_seasonality_index(
    rng: random.Random,
    tier1: list[str],
    n_periods: int,
    industry_profile: str,
    amplitude_lo: float | None = None,
    amplitude_hi: float | None = None,
) -> dict[str, list[float]]:
    """Sample a per-tier1-node sinusoidal demand multiplier over
    ``n_periods``, mean 1.0, for ``scripts.multi_period_planning``'s
    ``default_periodic_demand(..., seasonality=...)``. Each node gets its own
    random amplitude (drawn from the industry profile's
    ``seasonality_amplitude_lo/hi`` band, unless overridden here) and phase,
    so nodes don't all peak/trough in lockstep. Opt-in only — omitting
    ``seasonality`` in ``default_periodic_demand`` keeps today's flat-repeat
    behavior."""
    profile = INDUSTRY_PROFILES[industry_profile]
    lo = profile.seasonality_amplitude_lo if amplitude_lo is None else amplitude_lo
    hi = profile.seasonality_amplitude_hi if amplitude_hi is None else amplitude_hi

    seasonality: dict[str, list[float]] = {}
    for node in tier1:
        amplitude = rng.uniform(lo, hi)
        phase = rng.uniform(0, 2 * math.pi)
        seasonality[node] = [
            1.0 + amplitude * math.sin(2 * math.pi * t / n_periods + phase)
            for t in range(n_periods)
        ]
    return seasonality


def compute_required_throughput(
    tier1: list[str],
    tier2: list[str],
    d: dict[str, int],
    N_minus: dict[str, list[str]],
    P: dict[str, dict[str, list[str]]],
    r: dict[str, dict[str, float]],
    rng: random.Random,
) -> dict[str, float]:
    """Propagate tier-1 demand down through the bill of materials so
    tier-2/tier-3 throughput reflects actual downstream load instead of
    independent randomness. Where a (node, material) pair is multi-sourced,
    demand is split across its parents via a per-pair weight sampled once
    and normalized to sum to 1 across those parents."""

    def _propagate(consumer_nodes, consumer_demand):
        parent_throughput: dict[str, float] = {}
        for j in consumer_nodes:
            need = consumer_demand.get(j, 0.0)
            if need <= 0:
                continue
            for k in N_minus.get(j, []):
                candidates = P.get(j, {}).get(k, [])
                if not candidates:
                    continue
                weights = [rng.uniform(0.5, 1.5) for _ in candidates]
                total_w = sum(weights)
                for i, w in zip(candidates, weights):
                    share = need * r[j][k] * (w / total_w)
                    parent_throughput[i] = parent_throughput.get(i, 0.0) + share
        return parent_throughput

    tier2_throughput = _propagate(tier1, d)
    tier3_throughput = _propagate(tier2, tier2_throughput)

    throughput = {node: 0.0 for node in tier2 + list(tier3_throughput.keys())}
    for node, val in tier2_throughput.items():
        throughput[node] = throughput.get(node, 0.0) + val
    for node, val in tier3_throughput.items():
        throughput[node] = throughput.get(node, 0.0) + val
    return throughput


def sample_supplier_params(
    rng: random.Random, throughput: float, criticality: str
) -> tuple[int, int]:
    """Size a supplier's on-hand inventory and capacity off of its BOM-
    propagated throughput and its criticality tag."""
    profile = CRITICALITY_PROFILES.get(criticality, CRITICALITY_PROFILES["generic"])
    safety_days = rng.uniform(profile.safety_days_lo, profile.safety_days_hi)
    cap_mult = rng.uniform(profile.capacity_multiplier_lo, profile.capacity_multiplier_hi)
    effective_throughput = max(throughput, 1.0)
    s = max(1, round(effective_throughput * safety_days))
    c = max(1, round(effective_throughput * cap_mult))
    return s, c


def calibrate_network(
    rng: random.Random,
    tier1: list[str],
    tier2: list[str],
    tier3: list[str],
    N_minus: dict[str, list[str]],
    P: dict[str, dict[str, list[str]]],
    r: dict[str, dict[str, float]],
    industry_profile: str,
    criticality: dict[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, int], dict[str, int], dict[str, int]]:
    """End-to-end calibration for one network: tier-1 params from the
    industry profile, then tier-2/tier-3 params from BOM-propagated
    throughput and per-node criticality tags (default "generic")."""
    profile = INDUSTRY_PROFILES[industry_profile]
    f, s, d, c = sample_tier1_params(rng, tier1, profile)

    throughput = compute_required_throughput(tier1, tier2, d, N_minus, P, r, rng)

    criticality = criticality or {}
    for node in tier2 + tier3:
        node_s, node_c = sample_supplier_params(
            rng, throughput.get(node, 0.0), criticality.get(node, "generic")
        )
        s[node] = node_s
        c[node] = node_c

    return f, s, d, c


def calibrate_cost_fields(
    rng: random.Random,
    tier1: list[str],
    tier2: list[str],
    tier3: list[str],
    material_types: list[str],
    supplier_material_type: dict[str, str],
    edges: list[tuple[str, str]],
    criticality: dict[str, str] | None,
    industry_profile: str,
    region: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Calibrate cost/holding-cost/lead-time/emissions fields for the new
    LP objectives, using the same criticality/industry-profile-driven
    pattern as ``calibrate_network`` rather than independent random
    sampling. Returns a dict with keys ``unit_cost``, ``holding_cost``,
    ``production_delay``, ``emissions_factor`` (all per-node) and
    ``transport_emissions`` (per-edge, keyed by ``(src, tgt)``)."""
    profile = INDUSTRY_PROFILES[industry_profile]
    criticality = criticality or {}
    region = region or {}

    # One base unit-cost / base emissions figure per material type, so all
    # suppliers of the same material share a band (mirrors how
    # supplier_material_type already groups nodes).
    base_unit_cost = {
        material: rng.uniform(profile.unit_cost_lo, profile.unit_cost_hi)
        for material in material_types
    }
    base_emissions = {
        material: rng.uniform(profile.emissions_lo, profile.emissions_hi)
        for material in material_types
    }

    unit_cost: dict[str, float] = {}
    holding_cost: dict[str, float] = {}
    production_delay: dict[str, float] = {}
    emissions_factor: dict[str, float] = {}

    for node in tier1:
        unit_cost[node] = round(rng.uniform(profile.unit_cost_lo, profile.unit_cost_hi), 2)
        emissions_factor[node] = round(rng.uniform(profile.emissions_lo, profile.emissions_hi), 3)
        holding_cost[node] = round(unit_cost[node] * rng.uniform(0.02, 0.05), 3)
        production_delay[node] = round(rng.uniform(1, 5), 1)

    for node in tier2 + tier3:
        crit_profile = CRITICALITY_PROFILES.get(
            criticality.get(node, "generic"), CRITICALITY_PROFILES["generic"]
        )
        material = supplier_material_type.get(node)
        cost_mult = rng.uniform(crit_profile.cost_multiplier_lo, crit_profile.cost_multiplier_hi)
        holding_rate = rng.uniform(crit_profile.holding_cost_rate_lo, crit_profile.holding_cost_rate_hi)
        lead_time = rng.uniform(crit_profile.lead_time_days_lo, crit_profile.lead_time_days_hi)
        emissions_mult = rng.uniform(crit_profile.emissions_multiplier_lo, crit_profile.emissions_multiplier_hi)

        node_unit_cost = base_unit_cost.get(material, rng.uniform(profile.unit_cost_lo, profile.unit_cost_hi)) * cost_mult
        unit_cost[node] = round(node_unit_cost, 2)
        holding_cost[node] = round(node_unit_cost * holding_rate, 3)
        production_delay[node] = round(lead_time, 1)
        emissions_factor[node] = round(
            base_emissions.get(material, rng.uniform(profile.emissions_lo, profile.emissions_hi)) * emissions_mult, 3
        )

    transport_emissions: dict[tuple[str, str], float] = {}
    for src, tgt in edges:
        base = round(rng.uniform(0.1, 0.5), 3)
        if region and region.get(src) and region.get(tgt) and region.get(src) != region.get(tgt):
            base *= 2
        transport_emissions[(src, tgt)] = base

    return {
        "unit_cost": unit_cost,
        "holding_cost": holding_cost,
        "production_delay": production_delay,
        "emissions_factor": emissions_factor,
        "transport_emissions": transport_emissions,
    }

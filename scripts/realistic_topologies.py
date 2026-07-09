"""Realistic 3-tier network generators: simple, medium, and complex
(Nvidia-like / Apple-like) supply-chain topologies.

These generators produce dataset dicts with the *exact* schema required by
``scripts.utils.build_and_solve_ttr``/``build_and_solve_tts``
(``tier1,tier2,tier3,edges,material_types,supplier_material_type,f,s,d,c,r,
N_minus,N_plus,P``), plus optional metadata keys (``region``,
``company_name``, ``criticality``, ``product_line``) that the solvers never
read (they only ever access the dataset by the specific keys above, never
iterate ``dataset.keys()``) but which the disruption-scenario library and
notebooks use for realism (regional/correlated disruptions, narrative).

``scripts/utils.py``'s own ``generate_data`` (and the notebooks that call it
with ``# DO NOT CHANGE!``) is intentionally left untouched — this module is
fully additive. Topology-derivation logic (``N_minus``/``N_plus``/``P``/``r``)
is duplicated from ``generate_data`` rather than imported, for that reason.

Every generator takes an explicit ``seed`` and only ever uses a local
``random.Random`` instance — never the global ``random`` module — so nothing
here can perturb the ``random.seed(777)`` sequences the existing notebooks
depend on.
"""

from __future__ import annotations

import random
from typing import Literal

from scripts.company_profiles import (
    COMPANY_PROFILES,
    FICTIONAL_NAME_PREFIXES,
    FICTIONAL_NAME_SUFFIXES,
    MATERIAL_CATALOGS,
    PRODUCT_LINE_LABELS,
    RAW_MATERIAL_CATALOG,
    REGION_WEIGHTS,
)
from scripts.scenario_calibration import calibrate_network


class NodeIdAllocator:
    """Allocates globally-unique ``T{tier}_{n}`` node IDs within one
    generator run, keeping IDs compatible with ``scripts/dataset_io.py``'s
    tier-parsing helpers even though the solvers themselves treat node IDs
    as opaque strings."""

    def __init__(self) -> None:
        self._next = {1: 1, 2: 1, 3: 1}

    def next(self, tier: int) -> str:
        n = self._next[tier]
        self._next[tier] += 1
        return f"T{tier}_{n}"


def _fictional_company_name(rng: random.Random, used: set[str]) -> str:
    """A plausible-but-fictional company name, e.g. "Riverside Machining"."""
    for _ in range(50):
        name = f"{rng.choice(FICTIONAL_NAME_PREFIXES)} {rng.choice(FICTIONAL_NAME_SUFFIXES)}"
        if name not in used:
            used.add(name)
            return name
    name = f"{name} #{len(used)}"
    used.add(name)
    return name


def fan_out_synthetic_subsuppliers(
    rng: random.Random,
    ids: NodeIdAllocator,
    parent_id: str,
    parent_tier: int,
    count: int,
    region: str,
    region_jitter: dict[str, float] | None = None,
) -> tuple[list[str], list[tuple[str, str]], dict[str, str], dict[str, str]]:
    """Create ``count`` synthetic nodes one tier below ``parent_id`` that
    each supply it. Children are grouped into up to 8 distinct raw-material
    "buckets" (rather than each getting a unique material) so several
    children end up as alternate suppliers of the same material — modeling
    realistic multi-sourcing instead of every input being a lone supplier.

    Returns ``(new_node_ids, new_edges, material_by_node, region_by_node)``.
    """
    child_tier = parent_tier + 1
    if child_tier > 3:
        raise ValueError(f"cannot fan out below tier 3 (parent_tier={parent_tier})")
    if count <= 0:
        return [], [], {}, {}

    # ~4 alternate suppliers per material: enough for genuine multi-sourcing
    # without making every material trivially over-diversified (which would
    # make every single-supplier disruption a non-event).
    n_materials = max(1, round(count / 4))
    materials = [
        f"{rng.choice(RAW_MATERIAL_CATALOG)}_{parent_id.lower()}_{i}"
        for i in range(n_materials)
    ]

    new_ids: list[str] = []
    new_edges: list[tuple[str, str]] = []
    material_by_node: dict[str, str] = {}
    region_by_node: dict[str, str] = {}
    for idx in range(count):
        child = ids.next(child_tier)
        node_region = region
        if region_jitter:
            node_region = rng.choices(
                list(region_jitter), weights=list(region_jitter.values())
            )[0]
        new_ids.append(child)
        new_edges.append((child, parent_id))
        material_by_node[child] = materials[idx % n_materials]
        region_by_node[child] = node_region
    return new_ids, new_edges, material_by_node, region_by_node


def derive_indexes(
    tier1: list[str],
    tier2: list[str],
    tier3: list[str],
    edges: list[tuple[str, str]],
    supplier_material_type: dict[str, str],
) -> tuple[dict, dict, dict, dict]:
    """Recompute ``N_minus``/``N_plus``/``P``/``r`` from a topology — same
    logic as the index-derivation block in ``scripts.utils.generate_data``,
    duplicated here so that file stays untouched."""
    N_minus: dict[str, list[str]] = {j: [] for j in tier1 + tier2}
    for i, j in edges:
        if j in N_minus:
            k = supplier_material_type[i]
            if k not in N_minus[j]:
                N_minus[j].append(k)

    N_plus: dict[str, list[str]] = {i: [] for i in tier2 + tier3}
    for i, j in edges:
        if i in N_plus and j not in N_plus[i]:
            N_plus[i].append(j)

    P: dict[str, dict[str, list[str]]] = {j: {} for j in tier1 + tier2}
    for i, j in edges:
        if j in P:
            k = supplier_material_type[i]
            P[j].setdefault(k, []).append(i)

    r: dict[str, dict[str, float]] = {j: {} for j in tier1 + tier2}
    for j, ks in N_minus.items():
        for k in ks:
            r[j][k] = 1

    return N_minus, N_plus, P, r


def assemble_dataset(
    tier1: list[str],
    tier2: list[str],
    tier3: list[str],
    edges: list[tuple[str, str]],
    material_types: list[str],
    supplier_material_type: dict[str, str],
    f: dict[str, float],
    s: dict[str, int],
    d: dict[str, int],
    c: dict[str, int],
    **extra,
) -> dict:
    """Build the dataset dict with the exact required schema plus any
    optional metadata (``region``, ``company_name``, ``criticality``, ...)."""
    N_minus, N_plus, P, r = derive_indexes(tier1, tier2, tier3, edges, supplier_material_type)
    dataset = {
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "edges": edges,
        "material_types": material_types,
        "supplier_material_type": supplier_material_type,
        "f": f,
        "s": s,
        "d": d,
        "c": c,
        "r": r,
        "N_minus": N_minus,
        "N_plus": N_plus,
        "P": P,
    }
    dataset.update(extra)
    return dataset


def _expand_material_catalog(catalog: list[str], target_count: int) -> list[str]:
    """Expand a small named catalog into ``target_count`` distinct material
    types (e.g. "steel_stamped_bracket", "steel_stamped_bracket_v1", ...) so
    the number of alternate suppliers per exact material stays realistic
    (a handful, not dozens) even at medium/large node counts."""
    if target_count <= len(catalog):
        return catalog[:target_count]
    materials: list[str] = []
    variant = 0
    while len(materials) < target_count:
        for name in catalog:
            if len(materials) >= target_count:
                break
            materials.append(name if variant == 0 else f"{name}_v{variant}")
        variant += 1
    return materials


def _build_layered_topology(
    rng: random.Random,
    ids: NodeIdAllocator,
    n1: int,
    n2: int,
    n3: int,
    tier2_catalog: list[str],
    tier3_catalog: list[str],
    target_suppliers_per_material: int = 3,
) -> tuple[list[str], list[str], list[str], list[tuple[str, str]], dict[str, str]]:
    """Generic 3-tier random topology, used by the simple/medium generators.
    Every tier-1 node gets >=1 tier-2 supplier, every tier-2 node supplies
    1-3 tier-1 nodes, and every tier-2 node gets >=1 tier-3 supplier.
    ``tier2_catalog``/``tier3_catalog`` are expanded so each exact material
    type ends up with roughly ``target_suppliers_per_material`` suppliers
    (real multi-sourcing, not blanket dozens-deep redundancy)."""
    tier1 = [ids.next(1) for _ in range(n1)]
    tier2 = [ids.next(2) for _ in range(n2)]
    tier3 = [ids.next(3) for _ in range(n3)]

    tier2_materials = _expand_material_catalog(
        tier2_catalog, max(1, round(n2 / target_suppliers_per_material))
    )
    tier3_materials = _expand_material_catalog(
        tier3_catalog, max(1, round(n3 / target_suppliers_per_material))
    )

    shuffled_tier2 = tier2.copy()
    rng.shuffle(shuffled_tier2)
    shuffled_tier3 = tier3.copy()
    rng.shuffle(shuffled_tier3)

    supplier_material_type: dict[str, str] = {}
    for idx, node in enumerate(shuffled_tier2):
        supplier_material_type[node] = tier2_materials[idx % len(tier2_materials)]
    for idx, node in enumerate(shuffled_tier3):
        supplier_material_type[node] = tier3_materials[idx % len(tier3_materials)]

    edges: list[tuple[str, str]] = []

    t2_out: dict[str, set[str]] = {t2: set() for t2 in tier2}
    shuffled_t1 = tier1.copy()
    rng.shuffle(shuffled_t1)
    for idx, t1_node in enumerate(shuffled_t1):
        t2_node = tier2[idx % n2]
        edges.append((t2_node, t1_node))
        t2_out[t2_node].add(t1_node)

    for t2_node in tier2:
        desired = rng.randint(1, min(3, n1))
        attempts = 0
        while len(t2_out[t2_node]) < desired and attempts < 20:
            candidate = rng.choice(tier1)
            if candidate not in t2_out[t2_node]:
                edges.append((t2_node, candidate))
                t2_out[t2_node].add(candidate)
            attempts += 1

    for idx, t3_node in enumerate(tier3):
        t2_node = tier2[idx % n2]
        edges.append((t3_node, t2_node))

    return tier1, tier2, tier3, edges, supplier_material_type


class PreferentialAttachmentPool:
    """Draws nodes with probability proportional to ``initial_degree + 1``,
    growing more skewed toward already-popular nodes as more are drawn
    (Barabasi-Albert-style preferential attachment) — the standard way real,
    hub-heavy networks like supply chains are grown. Real supply chains have
    a handful of suppliers (TSMC, Foxconn) carrying far more downstream
    connections than the long tail; uniform random attachment can't
    reproduce that shape no matter how many nodes you add.

    Implemented via the standard O(1)-amortized "weighted pool" trick (each
    node starts in the pool ``initial_degree + 1`` times; each draw appends
    the chosen node once more) rather than recomputing weights from scratch
    per draw (``random.choices`` is O(n) per call) or taking a ``networkx``
    dependency for one growth rule."""

    def __init__(self, nodes: list[str], initial_degree: dict[str, int] | None = None):
        initial_degree = initial_degree or {}
        self._pool: list[str] = []
        for node in nodes:
            self._pool.extend([node] * (initial_degree.get(node, 0) + 1))

    def pick(self, rng: random.Random) -> str:
        choice = rng.choice(self._pool)
        self._pool.append(choice)
        return choice


def _criticality_from_supplier_counts(supplier_material_type: dict[str, str]) -> dict[str, str]:
    """Derive a criticality tag per node from how many alternate suppliers
    exist for its exact material type: a sole supplier is a monopoly
    bottleneck, two is an oligopoly, three or more is a diversified
    commodity. Used for the simple/medium generators, which (unlike the
    complex generators) don't have hand-authored criticality from real
    company anchors."""
    counts: dict[str, int] = {}
    for material in supplier_material_type.values():
        counts[material] = counts.get(material, 0) + 1
    criticality = {}
    for node, material in supplier_material_type.items():
        n = counts[material]
        if n <= 1:
            criticality[node] = "monopoly_bottleneck"
        elif n == 2:
            criticality[node] = "oligopoly"
        else:
            criticality[node] = "diversified_commodity"
    return criticality


def generate_simple_network(seed: int = 101, industry: str = "auto_parts") -> dict:
    """A small (~27-node), single-product-line, single-region illustrative
    manufacturer, with realistic (fictional) company names and BOM-
    propagated inventory/capacity instead of independent uniform randomness."""
    rng = random.Random(seed)
    ids = NodeIdAllocator()
    n1, n2, n3 = 3, 8, 16

    tier2_catalog = MATERIAL_CATALOGS[industry]
    tier3_catalog = [f"{m}_raw_input" for m in RAW_MATERIAL_CATALOG]

    tier1, tier2, tier3, edges, supplier_material_type = _build_layered_topology(
        rng, ids, n1, n2, n3, tier2_catalog, tier3_catalog, target_suppliers_per_material=2
    )
    material_types = sorted(set(supplier_material_type.values()))

    region = "US-Midwest"
    used_names: set[str] = set()
    regions = {node: region for node in tier2 + tier3}
    company_name = {node: _fictional_company_name(rng, used_names) for node in tier2 + tier3}

    criticality = _criticality_from_supplier_counts(supplier_material_type)
    N_minus, _, P, r = derive_indexes(tier1, tier2, tier3, edges, supplier_material_type)
    f, s, d, c = calibrate_network(
        rng, tier1, tier2, tier3, N_minus, P, r, industry_profile="simple", criticality=criticality
    )

    return assemble_dataset(
        tier1, tier2, tier3, edges, material_types, supplier_material_type,
        f, s, d, c,
        region=regions, company_name=company_name, criticality=criticality, industry=industry,
    )


def generate_medium_network(
    seed: int = 202,
    n1: int = 8,
    n2: int = 220,
    n3: int = 480,
    industry: str = "auto_parts_multi_line",
) -> dict:
    """A ~700-node, multi-product-line manufacturer with region-tagged
    suppliers (concentrated in a few real-world sourcing geographies), so
    regional/correlated disruption scenarios are meaningful."""
    rng = random.Random(seed)
    ids = NodeIdAllocator()

    tier2_catalog = MATERIAL_CATALOGS[industry]
    tier3_catalog = [f"{m}_component" for m in RAW_MATERIAL_CATALOG]

    tier1, tier2, tier3, edges, supplier_material_type = _build_layered_topology(
        rng, ids, n1, n2, n3, tier2_catalog, tier3_catalog, target_suppliers_per_material=3
    )
    material_types = sorted(set(supplier_material_type.values()))

    region_weights = REGION_WEIGHTS["medium"]
    regions = {
        node: rng.choices(list(region_weights), weights=list(region_weights.values()))[0]
        for node in tier2 + tier3
    }
    used_names: set[str] = set()
    company_name = {node: _fictional_company_name(rng, used_names) for node in tier2 + tier3}

    product_labels = PRODUCT_LINE_LABELS["medium"]
    product_line = {node: product_labels[i % len(product_labels)] for i, node in enumerate(tier1)}

    criticality = _criticality_from_supplier_counts(supplier_material_type)
    N_minus, _, P, r = derive_indexes(tier1, tier2, tier3, edges, supplier_material_type)
    f, s, d, c = calibrate_network(
        rng, tier1, tier2, tier3, N_minus, P, r, industry_profile="medium", criticality=criticality
    )

    return assemble_dataset(
        tier1, tier2, tier3, edges, material_types, supplier_material_type,
        f, s, d, c,
        region=regions, company_name=company_name, criticality=criticality,
        product_line=product_line, industry=industry,
    )


def _scaled_bounds(bounds: tuple[int, int], scale_factor: float) -> tuple[int, int]:
    """Scale an ``AnchorSpec`` peer/child ``(lo, hi)`` range by ``scale_factor``
    while preserving "no peers/children" (``(0, 0)``) exactly, so growing the
    network keeps the same proportion of nodes traceable to a real, named
    anchor instead of diluting toward generic padding as node counts rise."""
    lo, hi = bounds
    if lo == 0 and hi == 0:
        return 0, 0
    scaled_lo = max(1, round(lo * scale_factor)) if lo > 0 else 0
    scaled_hi = max(scaled_lo, round(hi * scale_factor))
    return scaled_lo, scaled_hi


SCALE_PRESETS: dict[str, dict] = {
    # Today's default — unchanged, ~2,300 total nodes.
    "complex": {"n2_target": 150, "n3_target": 2200, "scale_factor": 1.0},
    # ~76,000 total nodes; anchor fan-out scaled 25x so real-company anchors
    # (TSMC, ASML, Foxconn, ...) stay proportionally represented rather than
    # diluted by generic padding. Measured LP solve time at this scale is a
    # few seconds per single-node scenario (see README "Scaling to
    # planet-scale networks") — use ``sample_fraction`` in
    # ``scripts.disruption_scenarios.single_supplier_failure_scenarios`` to
    # keep a full sweep tractable at this size.
    "planet": {"n2_target": 6000, "n3_target": 70000, "scale_factor": 25.0},
}


def generate_complex_network_at_scale(
    company: Literal["nvidia", "apple", "mps"], scale: str = "complex", seed: int = 303
) -> dict:
    """Convenience wrapper around ``generate_complex_network`` using a named
    entry from ``SCALE_PRESETS`` instead of hand-tuning ``n2_target``/
    ``n3_target``/``scale_factor`` together."""
    if scale not in SCALE_PRESETS:
        raise ValueError(f"unknown scale {scale!r}; expected one of {list(SCALE_PRESETS)}")
    return generate_complex_network(company, seed=seed, **SCALE_PRESETS[scale])


def generate_complex_network(
    company: Literal["nvidia", "apple", "mps"],
    seed: int = 303,
    n1: int = 6,
    n2_target: int = 150,
    n3_target: int = 2200,
    scale_factor: float = 1.0,
) -> dict:
    """A large (~1500-3000-node), Nvidia-like or Apple-like network. Real,
    publicly-known companies (see ``scripts/company_profiles.py``) anchor
    tier 2 (direct component/module suppliers) and tier 3 (their equipment/
    raw-material suppliers); each anchor is fanned out with synthetic peers
    ("rest of market") and synthetic sub-suppliers to reach realistic scale.
    All attached numeric figures (profit margin, inventory, demand,
    capacity) are synthetic/illustrative — see the module docstring in
    ``scripts/company_profiles.py``.

    ``scale_factor`` multiplies every anchor's ``n_synthetic_peers``/
    ``n_synthetic_children`` range (see ``_scaled_bounds``) so growing
    ``n2_target``/``n3_target`` to reach planet-scale node counts doesn't
    dilute the fraction of nodes descended from a real, named anchor toward
    generic padding. Defaults to ``1.0`` (today's unmodified behavior); see
    ``SCALE_PRESETS``/``generate_complex_network_at_scale`` for a preset
    ``"planet"`` scale.
    """
    if company not in COMPANY_PROFILES:
        raise ValueError(f"unknown company {company!r}; expected one of {list(COMPANY_PROFILES)}")

    rng = random.Random(seed)
    ids = NodeIdAllocator()

    product_labels = PRODUCT_LINE_LABELS[company]
    tier1 = [ids.next(1) for _ in range(n1)]
    product_line = {node: product_labels[i % len(product_labels)] for i, node in enumerate(tier1)}

    tier2_anchors = COMPANY_PROFILES[company]["tier2"]
    tier3_anchors = COMPANY_PROFILES[company]["tier3"]
    region_weights = REGION_WEIGHTS[company]

    tier2: list[str] = []
    tier3: list[str] = []
    edges: list[tuple[str, str]] = []
    supplier_material_type: dict[str, str] = {}
    region: dict[str, str] = {}
    company_name: dict[str, str] = {}
    criticality: dict[str, str] = {}
    company_to_node: dict[str, str] = {}
    used_names: set[str] = set()

    # --- tier-2 real anchors + "rest of market" peers + fanned-out tier-3 children ---
    for spec in tier2_anchors:
        anchor_node = ids.next(2)
        tier2.append(anchor_node)
        supplier_material_type[anchor_node] = spec.material_type
        region[anchor_node] = spec.region
        company_name[anchor_node] = spec.company_name
        criticality[anchor_node] = spec.criticality
        company_to_node[spec.company_name] = anchor_node

        for _ in range(rng.randint(*_scaled_bounds(spec.n_synthetic_peers, scale_factor))):
            peer = ids.next(2)
            tier2.append(peer)
            supplier_material_type[peer] = spec.material_type
            region[peer] = spec.region
            company_name[peer] = _fictional_company_name(rng, used_names)
            criticality[peer] = "diversified_commodity"

        other_regions = [r for r in region_weights if r != spec.region]
        region_jitter = {spec.region: 0.8}
        if other_regions:
            region_jitter.update({r: 0.2 / len(other_regions) for r in other_regions})

        n_children = rng.randint(*_scaled_bounds(spec.n_synthetic_children, scale_factor))
        child_ids, child_edges, child_materials, child_regions = fan_out_synthetic_subsuppliers(
            rng, ids, anchor_node, 2, n_children, spec.region, region_jitter=region_jitter
        )
        tier3.extend(child_ids)
        edges.extend(child_edges)
        supplier_material_type.update(child_materials)
        region.update(child_regions)
        for node in child_ids:
            company_name[node] = ""
            criticality[node] = "diversified_commodity"

    # --- tier-3 real anchors, wired to the specific tier-2 companies they feed ---
    for spec in tier3_anchors:
        anchor_node = ids.next(3)
        tier3.append(anchor_node)
        supplier_material_type[anchor_node] = spec.material_type
        region[anchor_node] = spec.region
        company_name[anchor_node] = spec.company_name
        criticality[anchor_node] = spec.criticality

        targets = (
            [company_to_node[name] for name in spec.feeds if name in company_to_node]
            if spec.feeds
            else list(company_to_node.values())
        )
        for target in targets:
            edges.append((anchor_node, target))

        for _ in range(rng.randint(*_scaled_bounds(spec.n_synthetic_peers, scale_factor))):
            peer = ids.next(3)
            tier3.append(peer)
            supplier_material_type[peer] = spec.material_type
            region[peer] = spec.region
            company_name[peer] = _fictional_company_name(rng, used_names)
            criticality[peer] = "diversified_commodity"
            for target in targets:
                edges.append((peer, target))

    # --- tier-2 -> tier-1 wiring: every tier-2 node supplies 1-3 products ---
    t2_out: dict[str, set[str]] = {t2: set() for t2 in tier2}
    shuffled_t1 = tier1.copy()
    rng.shuffle(shuffled_t1)
    for idx, t1_node in enumerate(shuffled_t1):
        t2_node = tier2[idx % len(tier2)]
        edges.append((t2_node, t1_node))
        t2_out[t2_node].add(t1_node)
    for t2_node in tier2:
        desired = rng.randint(1, min(3, n1))
        attempts = 0
        while len(t2_out[t2_node]) < desired and attempts < 20:
            candidate = rng.choice(tier1)
            if candidate not in t2_out[t2_node]:
                edges.append((t2_node, candidate))
                t2_out[t2_node].add(candidate)
            attempts += 1

    # --- pad out to the requested scale with generic synthetic nodes ---
    # (a distinct "generic_component" material namespace, never reusing a
    # named anchor's material, so monopoly/oligopoly anchors don't get
    # diluted by dozens of unrelated padding peers; materials are expanded
    # to keep ~4 alternate suppliers per material, not dozens)
    n2_padding = max(0, n2_target - len(tier2))
    tier2_padding_materials = _expand_material_catalog(
        [f"{m}_generic_component" for m in RAW_MATERIAL_CATALOG], max(1, round(n2_padding / 4))
    )
    for i in range(n2_padding):
        node = ids.next(2)
        tier2.append(node)
        supplier_material_type[node] = tier2_padding_materials[i % len(tier2_padding_materials)]
        region[node] = rng.choices(list(region_weights), weights=list(region_weights.values()))[0]
        company_name[node] = _fictional_company_name(rng, used_names)
        criticality[node] = "diversified_commodity"
        picked: set[str] = set()
        for _ in range(rng.randint(1, min(3, n1))):
            candidate = rng.choice(tier1)
            if candidate not in picked:
                edges.append((node, candidate))
                picked.add(candidate)

    # Preferential (Barabasi-Albert-style) attachment instead of uniform
    # random choice: without this, the ~15 real anchors are the only hubs in
    # the network and the tens of thousands of generic padding nodes added
    # at planet scale attach uniformly at random, which is not how real
    # supply chains look (see README "Scale-free padding attachment").
    # Seed each tier-2 node's initial pool count from its existing tier-3
    # children (real anchors already look like hubs here) so the hub-heavy
    # shape holds across both real anchors and padding.
    degree_t2_from_t3: dict[str, int] = {t2: 0 for t2 in tier2}
    for src, tgt in edges:
        if tgt in degree_t2_from_t3:
            degree_t2_from_t3[tgt] += 1
    attachment_pool = PreferentialAttachmentPool(tier2, initial_degree=degree_t2_from_t3)

    n3_padding = max(0, n3_target - len(tier3))
    tier3_padding_materials = _expand_material_catalog(
        [f"{m}_generic" for m in RAW_MATERIAL_CATALOG], max(1, round(n3_padding / 4))
    )
    for i in range(n3_padding):
        node = ids.next(3)
        parent = attachment_pool.pick(rng)
        tier3.append(node)
        supplier_material_type[node] = tier3_padding_materials[i % len(tier3_padding_materials)]
        region[node] = rng.choices(list(region_weights), weights=list(region_weights.values()))[0]
        company_name[node] = ""
        criticality[node] = "diversified_commodity"
        edges.append((node, parent))

    material_types = sorted(set(supplier_material_type.values()))

    N_minus, _, P, r = derive_indexes(tier1, tier2, tier3, edges, supplier_material_type)
    f, s, d, c = calibrate_network(
        rng, tier1, tier2, tier3, N_minus, P, r,
        industry_profile=company, criticality=criticality,
    )

    return assemble_dataset(
        tier1, tier2, tier3, edges, material_types, supplier_material_type,
        f, s, d, c,
        region=region, company_name=company_name, criticality=criticality,
        product_line=product_line, company=company,
    )


def visualize_anchor_backbone(dataset: dict) -> None:
    """Lightweight visualization for complex (1500+ node) networks, where
    the existing ``scripts.utils.visualize_network`` scatter plot is too
    dense to read. Draws only the real, named anchor companies and labels
    each with how many synthetic ("rest of market"/sub-supplier) nodes feed
    into it."""
    import matplotlib.pyplot as plt

    company_name = dataset.get("company_name", {})
    region = dataset.get("region", {})
    edges = dataset["edges"]

    anchors = {n: name for n, name in company_name.items() if name}
    if not anchors:
        raise ValueError("dataset has no company_name-tagged anchors to visualize")

    tier2_anchors = [n for n in anchors if n in dataset["tier2"]]
    tier3_anchors = [n for n in anchors if n in dataset["tier3"]]

    fan_out_count = {n: 0 for n in anchors}
    for src, tgt in edges:
        if tgt in fan_out_count and not company_name.get(src):
            fan_out_count[tgt] += 1

    y_positions: dict[str, int] = {}
    y = 0
    for n in tier3_anchors:
        y_positions[n] = y
        y += 1
    y += 1
    for n in tier2_anchors:
        y_positions[n] = y
        y += 1

    fig, ax = plt.subplots(figsize=(14, max(6, 0.4 * len(anchors))))

    for src, tgt in edges:
        if src in y_positions and tgt in y_positions:
            ax.plot([0.05, 0.95], [y_positions[src], y_positions[tgt]],
                     color="grey", lw=0.6, alpha=0.5, zorder=0)

    for n, ypos in y_positions.items():
        xpos = 1 if n in tier2_anchors else 0
        ax.scatter([xpos], [ypos], s=400, c="tab:blue" if xpos == 1 else "tab:orange",
                   edgecolor="k", zorder=1)
        label = f"{anchors[n]}  ({region.get(n, '')})"
        if fan_out_count[n]:
            label += f"   +{fan_out_count[n]} synthetic suppliers"
        ax.text(xpos + 0.05, ypos, label, va="center", fontsize=9, zorder=2)

    ax.set_xlim(-0.5, 3.5)
    ax.axis("off")
    ax.set_title("Named-anchor backbone (real companies only; synthetic fan-out shown as counts)")
    plt.tight_layout()
    plt.show()

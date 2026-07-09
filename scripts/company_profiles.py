"""Real-company anchor data for the "complex" (Nvidia/Apple-scale) realistic
network generators, plus fictional catalogs for the "simple"/"medium" tiers.

DISCLAIMER
----------
Company names and their approximate role/geography in the supply chain
(``NVIDIA_TIER2_ANCHORS``, ``NVIDIA_TIER3_ANCHORS``, ``APPLE_TIER2_ANCHORS``,
``APPLE_TIER3_ANCHORS``) are drawn from public secondary reporting (press
coverage, analyst commentary, published supplier lists) about these
companies' real supplier relationships, current as of 2026-07. Every numeric
figure attached to a node elsewhere in this codebase (profit margin,
inventory, demand, capacity — see ``scripts/scenario_calibration.py``) is
synthetic and illustrative. None of these figures represent actual disclosed
financial or operational data from Nvidia, Apple, or any named supplier, and
must not be used to draw real inferences about those companies.

The ``apple_real`` profile is a step further: its anchor names AND regions come
from Apple's *published* Supplier List (FY2023), parsed by
``scripts/apple_supplier_list.py`` from the committed ``Apple-Supplier-List.md``
— a primary source rather than secondary reporting. The attached numbers there
are still synthetic; only the names/regions are real.

This module is deliberately the only place in the codebase that makes a
real-company claim, so it can be reviewed/updated in isolation.

For the "simple" and "medium" tiers we intentionally use fictional company
names (``MATERIAL_CATALOGS``) so the "real vs. illustrative" line stays
unambiguous — nobody could mistake "Riverside Tier-2 Machining" for a real
company.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorSpec:
    """A single real, named node to seed a complex-tier network with.

    ``n_synthetic_peers``/``n_synthetic_children`` are inclusive (lo, hi)
    ranges: the generator draws a random peer/child count per anchor from
    these bounds so repeated runs (with a fixed seed) still vary sensibly in
    scale while staying within realistic bounds.
    """

    company_name: str
    tier: int  # 2 or 3
    material_type: str
    region: str
    criticality: str  # "monopoly_bottleneck" | "oligopoly" | "diversified_commodity"
    n_synthetic_peers: tuple[int, int]
    n_synthetic_children: tuple[int, int]
    real_world_basis: str = ""
    # Tier-3 anchors only: which tier-2 company_name(s) this node feeds. An
    # empty tuple means "feeds every tier-2 anchor in the network" (used for
    # broadly-applicable materials like connectors). Ignored for tier-2 specs.
    feeds: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Nvidia-like (fabless AI/GPU accelerator) complex network anchors
# ---------------------------------------------------------------------------
# Sources (public reporting, 2024-2026): TSMC is Nvidia's sole leading-edge
# wafer foundry and CoWoS advanced-packaging provider; SK Hynix, Samsung and
# Micron supply HBM stacked memory; Foxconn performs system
# integration/assembly for AI server racks; ASML is the sole supplier of EUV
# lithography systems required for leading-edge nodes; Amphenol supplies
# high-speed interconnects (e.g. NVLink cabling).
NVIDIA_TIER2_ANCHORS: list[AnchorSpec] = [
    AnchorSpec("TSMC", 2, "leading_edge_wafer_foundry", "Taiwan", "monopoly_bottleneck", (0, 1), (150, 300)),
    AnchorSpec("SK Hynix", 2, "hbm_memory", "South Korea", "oligopoly", (2, 3), (60, 120)),
    AnchorSpec("Samsung Semiconductor", 2, "hbm_memory", "South Korea", "oligopoly", (2, 3), (60, 120)),
    AnchorSpec("Micron", 2, "hbm_memory", "USA", "oligopoly", (2, 3), (40, 90)),
    AnchorSpec("Amkor Technology", 2, "advanced_packaging_osat", "South Korea", "oligopoly", (3, 6), (40, 80)),
    AnchorSpec("Foxconn", 2, "system_integration", "Taiwan", "diversified_commodity", (5, 8), (80, 160)),
    AnchorSpec("Ibiden", 2, "ic_substrate", "Japan", "oligopoly", (2, 4), (30, 60)),
    AnchorSpec("Unimicron", 2, "ic_substrate", "Taiwan", "oligopoly", (2, 4), (30, 60)),
]
NVIDIA_TIER3_ANCHORS: list[AnchorSpec] = [
    AnchorSpec("ASML", 3, "euv_lithography_equipment", "Netherlands", "monopoly_bottleneck", (0, 0), (0, 0), feeds=("TSMC",)),
    AnchorSpec("Applied Materials", 3, "deposition_etch_equipment", "USA", "oligopoly", (2, 3), (0, 0), feeds=("TSMC",)),
    AnchorSpec("Lam Research", 3, "etch_equipment", "USA", "oligopoly", (2, 3), (0, 0), feeds=("TSMC",)),
    AnchorSpec("Tokyo Electron", 3, "wafer_process_equipment", "Japan", "oligopoly", (2, 3), (0, 0), feeds=("TSMC",)),
    AnchorSpec("Shin-Etsu Chemical", 3, "silicon_wafer", "Japan", "oligopoly", (2, 3), (0, 0), feeds=("TSMC",)),
    AnchorSpec("SUMCO", 3, "silicon_wafer", "Japan", "oligopoly", (2, 3), (0, 0), feeds=("TSMC",)),
    AnchorSpec("Amphenol", 3, "high_speed_connectors", "USA", "diversified_commodity", (5, 8), (0, 0), feeds=("Foxconn",)),
    AnchorSpec("Resonac Holdings", 3, "packaging_underfill_material", "Japan", "oligopoly", (2, 4), (0, 0), feeds=("TSMC", "Amkor Technology")),
]

# ---------------------------------------------------------------------------
# Apple-like (consumer electronics OEM) complex network anchors
# ---------------------------------------------------------------------------
# Sources: Apple's annual public Supplier List names Foxconn, Pegatron and
# Luxshare as its primary final-assembly contract manufacturers; TSMC
# fabricates Apple's custom SoCs; Samsung Display/LG Display supply OLED
# panels; Sony Semiconductor supplies camera image sensors; Corning supplies
# cover glass; CATL supplies battery cells for some product lines.
APPLE_TIER2_ANCHORS: list[AnchorSpec] = [
    AnchorSpec("Foxconn", 2, "final_assembly", "China", "oligopoly", (3, 5), (150, 300)),
    AnchorSpec("Pegatron", 2, "final_assembly", "China", "oligopoly", (3, 5), (100, 200)),
    AnchorSpec("Luxshare Precision", 2, "final_assembly_connectors", "China", "diversified_commodity", (3, 5), (80, 150)),
    AnchorSpec("TSMC", 2, "soc_foundry", "Taiwan", "monopoly_bottleneck", (0, 1), (150, 300)),
    AnchorSpec("Samsung Display", 2, "oled_display", "South Korea", "oligopoly", (2, 4), (60, 120)),
    AnchorSpec("LG Display", 2, "oled_display", "South Korea", "oligopoly", (2, 4), (60, 120)),
    AnchorSpec("Sony Semiconductor", 2, "camera_sensor", "Japan", "oligopoly", (2, 4), (40, 80)),
]
APPLE_TIER3_ANCHORS: list[AnchorSpec] = [
    AnchorSpec("Corning", 3, "cover_glass", "USA", "oligopoly", (2, 3), (0, 0), feeds=("Foxconn", "Pegatron")),
    AnchorSpec("Largan Precision", 3, "camera_lens", "Taiwan", "oligopoly", (2, 3), (0, 0), feeds=("Foxconn", "Pegatron", "Luxshare Precision")),
    AnchorSpec("CATL", 3, "battery_cell", "China", "oligopoly", (3, 5), (0, 0), feeds=("Foxconn", "Pegatron", "Luxshare Precision")),
    AnchorSpec("ASML", 3, "euv_lithography_equipment", "Netherlands", "monopoly_bottleneck", (0, 0), (0, 0), feeds=("TSMC",)),
    AnchorSpec("Qualcomm", 3, "cellular_modem", "USA", "oligopoly", (2, 3), (0, 0), feeds=("Foxconn", "Pegatron", "Luxshare Precision")),
    AnchorSpec("Skyworks Solutions", 3, "rf_front_end", "USA", "oligopoly", (2, 4), (0, 0), feeds=("Foxconn", "Pegatron", "Luxshare Precision")),
]

# ---------------------------------------------------------------------------
# Monolithic Power Systems-like (fabless PMIC / power-module maker) anchors
# ---------------------------------------------------------------------------
# Sources (public reporting + "MPS Supply Chain Optimization Pipeline.md" in
# this repo, 2022-2026): MPS is fabless and contracts multiple foundries
# across Taiwan/China/Korea/Singapore for its proprietary BCD-process wafers,
# with a headline long-term 8-inch capacity agreement with Vanguard
# International Semiconductor (VIS). Fabricated wafers are shipped to MPS's
# wholly-owned Chengdu (China) facility for wafer sort and final test;
# packaging/assembly is outsourced to independent OSAT subcontractors in
# China and Malaysia; MPS also opened a Penang (Malaysia) engineering/ops hub.
#
# MODELING NOTES / SIMPLIFICATIONS (documented, not oversights):
#  * The Chengdu wafer-sort/final-test facility and the Penang engineering hub
#    are MPS-OWNED (captive) facilities, not third-party suppliers. The repo's
#    LP schema only has tier1/tier2/tier3 "supplier" slots, so we model these
#    captive back-end nodes as tier-2 nodes — a deliberate schema simplification
#    (see real_world_basis on each). Chengdu is tagged monopoly_bottleneck
#    because it is a single wholly-owned wafer-sort/final-test chokepoint.
#  * The OSAT subcontractors are given GENERIC labels ("Malaysia OSAT Partner",
#    "China OSAT Partner") rather than named real vendors: the source doc does
#    not disclose MPS's specific OSAT partners, so naming one would be an
#    unverifiable real-company claim (inconsistent with this file's disclaimer).
#  * ASML is deliberately EXCLUDED from the tier-3 equipment anchors: MPS's
#    mature-node BCD process does not use EUV lithography, so ASML (EUV-only)
#    does not plausibly feed this supply chain. This exclusion is intentional.
MPS_TIER2_ANCHORS: list[AnchorSpec] = [
    AnchorSpec("Vanguard International Semiconductor", 2, "specialty_wafer_foundry", "Taiwan", "oligopoly", (2, 4), (60, 120), real_world_basis="Long-term 8-inch (200mm) foundry capacity agreement announced 2022 (Singapore/Taiwan fabs)."),
    AnchorSpec("Rest-of-World Foundry Partner", 2, "specialty_wafer_foundry", "South Korea", "oligopoly", (2, 4), (40, 90), real_world_basis="MPS contracts multiple elite foundries across Taiwan/China/Korea/Singapore; generic anchor for the non-VIS foundry base."),
    AnchorSpec("MPS Chengdu Wafer Sort & Final Test", 2, "wafer_sort_final_test", "China", "monopoly_bottleneck", (0, 0), (30, 60), real_world_basis="MPS's wholly-owned 60,000 sq ft Chengdu facility (purchased 2015) for wafer sort and final test; a CAPTIVE facility modeled here as a tier-2 node, not a third-party supplier."),
    AnchorSpec("Malaysia OSAT Partner", 2, "advanced_packaging_osat", "Malaysia", "oligopoly", (2, 4), (40, 80), real_world_basis="Independent OSAT subcontractor(s) in Malaysia perform monolithic packaging/assembly; generic label (specific vendor not publicly disclosed)."),
    AnchorSpec("China OSAT Partner", 2, "advanced_packaging_osat", "China", "diversified_commodity", (3, 5), (40, 80), real_world_basis="Independent OSAT subcontractor(s) in China perform packaging/assembly, serving the 'China for China' domestic base; generic label."),
    AnchorSpec("MPS Penang Engineering Hub", 2, "engineering_ops_support", "Malaysia", "diversified_commodity", (0, 1), (5, 15), real_world_basis="MPS's Penang engineering/operations hub (~60 engineers, advanced labs); a CAPTIVE support facility modeled as a small tier-2 node."),
]
MPS_TIER3_ANCHORS: list[AnchorSpec] = [
    AnchorSpec("Applied Materials", 3, "deposition_etch_equipment", "USA", "oligopoly", (2, 3), (0, 0), feeds=("Vanguard International Semiconductor", "Rest-of-World Foundry Partner")),
    AnchorSpec("Tokyo Electron", 3, "wafer_process_equipment", "Japan", "oligopoly", (2, 3), (0, 0), feeds=("Vanguard International Semiconductor", "Rest-of-World Foundry Partner")),
    AnchorSpec("Shin-Etsu Chemical", 3, "silicon_wafer", "Japan", "oligopoly", (2, 3), (0, 0), feeds=("Vanguard International Semiconductor", "Rest-of-World Foundry Partner")),
    AnchorSpec("Resonac Holdings", 3, "packaging_material", "Japan", "oligopoly", (2, 4), (0, 0), feeds=("Malaysia OSAT Partner", "China OSAT Partner", "MPS Chengdu Wafer Sort & Final Test")),
    AnchorSpec("Amphenol", 3, "connectors_passives", "USA", "diversified_commodity", (4, 7), (0, 0), feeds=("Malaysia OSAT Partner", "China OSAT Partner")),
]

# ---------------------------------------------------------------------------
# Apple-REAL (grounded in Apple's PUBLISHED supplier list, not secondary
# reporting) complex network anchors
# ---------------------------------------------------------------------------
# Unlike every other profile in this file, ``apple_real`` is built from a
# PRIMARY source: Apple's official Supplier List (FY2023), committed to the
# repo as ``Apple-Supplier-List.md``. ``scripts/apple_supplier_list.py`` parses
# that document — every anchor's company_name and region come straight from it.
# A curated lookup in that module classifies the ~40 recognizable names
# (assemblers, OSATs, the TSMC foundry, displays, memory/optics/battery/
# connector/analog vendors) into tier/role/criticality; every other parsed
# supplier becomes a generic tier-3 component anchor tagged with its real
# region. As always, the ATTACHED NUMBERS remain synthetic — only the names
# and regions are real/published.
from scripts.apple_supplier_list import build_apple_real_anchors  # noqa: E402

APPLE_REAL_TIER2_ANCHORS, APPLE_REAL_TIER3_ANCHORS = build_apple_real_anchors()

COMPANY_PROFILES: dict[str, dict[str, list[AnchorSpec]]] = {
    "nvidia": {"tier2": NVIDIA_TIER2_ANCHORS, "tier3": NVIDIA_TIER3_ANCHORS},
    "apple": {"tier2": APPLE_TIER2_ANCHORS, "tier3": APPLE_TIER3_ANCHORS},
    "mps": {"tier2": MPS_TIER2_ANCHORS, "tier3": MPS_TIER3_ANCHORS},
    "apple_real": {"tier2": APPLE_REAL_TIER2_ANCHORS, "tier3": APPLE_REAL_TIER3_ANCHORS},
}

# ---------------------------------------------------------------------------
# Fictional catalogs for the "simple" and "medium" tiers
# ---------------------------------------------------------------------------
MATERIAL_CATALOGS: dict[str, list[str]] = {
    "auto_parts": [
        "steel_stamped_bracket",
        "wiring_harness",
        "rubber_seal",
        "fastener_kit",
        "plastic_trim_molding",
    ],
    "auto_parts_multi_line": [
        "steel_stamped_bracket",
        "wiring_harness",
        "rubber_seal",
        "fastener_kit",
        "plastic_trim_molding",
        "ev_battery_module",
        "power_electronics_pcb",
        "sensor_assembly",
        "aluminum_casting",
        "brake_component",
    ],
}

RAW_MATERIAL_CATALOG: list[str] = [
    "raw_steel_coil",
    "copper_wire",
    "synthetic_rubber_compound",
    "injection_molding_resin",
    "electronic_connector_pins",
    "industrial_adhesive",
    "precision_machined_component",
    "surface_treatment_chemical",
]

FICTIONAL_NAME_PREFIXES: list[str] = [
    "Riverside", "Summit", "Cascade", "Ironclad", "Meridian", "Pinnacle",
    "Crestview", "Union", "Anchor", "Vanguard", "Harborline", "Northgate",
    "Fieldstone", "Redwood", "Silverton", "Bridgeway",
]

FICTIONAL_NAME_SUFFIXES: list[str] = [
    "Machining", "Fabrication", "Components", "Industries", "Manufacturing",
    "Precision", "Systems", "Supply Co.", "Works", "Materials", "Tooling",
    "Molding", "Electronics", "Assembly",
]

PRODUCT_LINE_LABELS: dict[str, list[str]] = {
    "nvidia": [
        "Data Center AI Accelerator",
        "Gaming GPU",
        "Automotive SoC",
        "Networking ASIC",
        "Professional Visualization GPU",
        "Edge AI Module",
    ],
    "apple": [
        "iPhone Line",
        "iPad Line",
        "Mac Line",
        "Apple Watch Line",
        "AirPods Line",
        "Accessories Line",
    ],
    "medium": ["Sedan Assembly Line", "SUV Assembly Line", "EV Assembly Line"],
    "mps": [
        "48V AI Server Power Module",
        "DC-DC Converter IC",
        "Automotive Power IC",
        "Battery Management IC",
        "Motor Driver Module",
        "Intelli-Phase Power Module",
    ],
    "apple_real": [
        "iPhone Line",
        "iPad Line",
        "Mac Line",
        "Apple Watch Line",
        "AirPods Line",
        "Accessories Line",
    ],
}

REGION_WEIGHTS: dict[str, dict[str, float]] = {
    "medium": {
        "China": 0.30,
        "Mexico": 0.15,
        "USA": 0.15,
        "Germany": 0.15,
        "Vietnam": 0.10,
        "Thailand": 0.15,
    },
    "nvidia": {
        "Taiwan": 0.35,
        "South Korea": 0.20,
        "Japan": 0.15,
        "USA": 0.15,
        "Netherlands": 0.05,
        "China": 0.10,
    },
    "apple": {
        "China": 0.45,
        "Taiwan": 0.20,
        "South Korea": 0.15,
        "Japan": 0.10,
        "USA": 0.10,
    },
    # MPS's "China for China" strategy makes China the largest region, with
    # Taiwan (VIS + foundries), Malaysia (OSAT + Penang hub), Korea, and the
    # USA (HQ / equipment vendors) rounding out the network.
    "mps": {
        "China": 0.35,
        "Taiwan": 0.25,
        "Malaysia": 0.20,
        "South Korea": 0.10,
        "USA": 0.10,
    },
    # apple_real: weights approximate the actual manufacturing-region footprint
    # counted from Apple's published FY2023 supplier list (China dominant, then
    # Taiwan/Japan/Vietnam/Korea/US/Thailand). Used for synthetic peer/padding
    # region draws; the named anchors carry their own real region directly.
    "apple_real": {
        "China": 0.45,
        "Taiwan": 0.14,
        "Japan": 0.11,
        "Vietnam": 0.10,
        "South Korea": 0.08,
        "USA": 0.07,
        "Thailand": 0.05,
    },
}

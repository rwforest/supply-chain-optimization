"""Parse Apple's **published** supplier list into complex-tier network anchors.

Unlike the hand-authored anchors in ``scripts/company_profiles.py`` (drawn from
secondary reporting), the ``apple_real`` profile is grounded in a PRIMARY
source: Apple's official Supplier List (FY2023, "represents 98 percent of our
direct spend for materials, manufacturing, and assembly"), committed to this
repo as ``Apple-Supplier-List.md`` (converted from the published PDF). Each
supplier's **name** and **primary manufacturing regions** come straight from
that document.

DISCLAIMER (consistent with ``company_profiles.py``): the supplier names and
their regions are real/published, but every NUMERIC figure attached to the
generated nodes (margin, inventory, demand, capacity, cost, lead time) remains
synthetic and illustrative. The tier/role/criticality assignment is a modeling
choice: a curated lookup classifies the ~40 recognizable names (assemblers,
OSATs, the foundry, displays, memory/optics/battery/connector/analog vendors),
and every other parsed supplier becomes a generic tier-3 component anchor. None
of this represents Apple's actual internal BOM or tier structure.

This module is pure/offline: it reads the committed markdown, imports nothing
heavy, and is fully unit-testable. It is imported by ``company_profiles.py`` to
build ``COMPANY_PROFILES["apple_real"]``.
"""

from __future__ import annotations

import os
import re

# Default path to the committed markdown (repo root, one level above scripts/).
_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Apple-Supplier-List.md",
)

# Region strings as they appear in the "primary locations" column, normalized to
# the region vocabulary used elsewhere in the repo (REGION_WEIGHTS keys).
_COUNTRY_NORM: dict[str, str] = {
    "China mainland": "China",
    "China": "China",
    "Taiwan": "Taiwan",
    "Japan": "Japan",
    "South Korea": "South Korea",
    "United States": "USA",
    "Vietnam": "Vietnam",
    "Thailand": "Thailand",
    "Malaysia": "Malaysia",
    "Singapore": "Singapore",
    "Philippines": "Philippines",
    "India": "India",
    "Germany": "Germany",
    "Belgium": "Belgium",
    "Israel": "Israel",
    "Mexico": "Mexico",
    "Netherlands": "Netherlands",
    "Ireland": "Ireland",
    "United Kingdom": "UK",
    "Czech Republic": "Czech Republic",
    "Austria": "Austria",
    "France": "France",
    "Italy": "Italy",
    "Indonesia": "Indonesia",
    "Cambodia": "Cambodia",
    "Costa Rica": "Costa Rica",
    "Morocco": "Morocco",
    "Malta": "Malta",
    "Portugal": "Portugal",
    "Saudi": "Saudi Arabia",
}
# Match longer strings first so "China mainland" wins over "China".
_COUNTRIES_BY_LEN = sorted(_COUNTRY_NORM, key=len, reverse=True)

# A cell is a company name (not a struck-through clean-energy icon or a
# region-continuation line) if it contains one of these corporate tokens.
_CORP = re.compile(
    r"(Incorporated|Limited|Corporation|Company|Holdings|Co\.,|Co\.|Ltd|GmbH|"
    r"AG\b|N\.V\.|S\.A\.|PLC|Group|Technolog|Precision|Semiconductor|Electronic|"
    r"Industr|Manufactur|Materials|Optical|Solutions|Systems|Chemical|Battery|"
    r"Energy|Glass|Metal|Aluminium|Steel|Titanium|Circuit|Opto|Mining|"
    r"Enterprise|International|Products)"
)


def _clean_name(cell: str) -> str:
    """First ``<br>`` segment of a cell, stripped of strike-through markup."""
    return cell.split("<br>")[0].strip().replace("~~", "").strip().strip("|").strip()


def _looks_like_name(seg: str) -> bool:
    return bool(
        seg
        and 4 <= len(seg) <= 75
        and re.search("[A-Za-z]", seg)
        and _CORP.search(seg)
        and not seg.startswith("**")
        and "CLEAN ENERGY" not in seg
    )


def _regions_in(cell: str) -> list[str]:
    out: list[str] = []
    for c in _COUNTRIES_BY_LEN:
        if c in cell:
            out.append(_COUNTRY_NORM[c])
    # de-dup, preserve order
    return list(dict.fromkeys(out))


def parse_supplier_list(path: str = _DEFAULT_PATH) -> list[dict]:
    """Parse the markdown supplier list into
    ``[{"name": str, "regions": [str, ...]}, ...]``.

    The markdown table's clean-energy-icon column is mangled by the PDF→md
    conversion (struck-through ``~~...~~`` fragments, variable cell counts per
    row), so we don't rely on fixed column positions: the region list is the
    LAST cell (the only one containing country names), and the supplier name is
    the first earlier cell whose cleaned text looks like a company name (skips
    region-continuation cells and icon junk).
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    suppliers: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        regions = _regions_in(cells[-1])
        if not regions:  # last cell isn't a locations cell → header/junk row
            continue
        name = None
        for cell in cells[:-1]:
            cand = _clean_name(cell)
            if cand in _COUNTRY_NORM:  # a region-continuation cell, not a name
                continue
            if _looks_like_name(cand):
                name = cand
                break
        if not name:
            continue
        suppliers.setdefault(name, regions)

    return [{"name": n, "regions": r} for n, r in suppliers.items()]


# ---------------------------------------------------------------------------
# Curated role map: the recognizable names whose supply-chain role is public
# knowledge, mapped to (tier, material_type, criticality). Everything else
# parsed from the list becomes a generic tier-3 component anchor. Matching is
# case-insensitive substring against the parsed supplier name.
# ---------------------------------------------------------------------------
# material_type tags chosen to line up with the derivation role-config
# (scripts/mps_derivation.py APPLE_REAL_CONFIG): "final_assembly" nodes become
# FJSP machines + CVRPTW depots; "advanced_packaging_osat" become test-capable
# FJSP machines; the rest are network structure only.
_CURATED_ROLES: list[tuple[str, int, str, str]] = [
    # (name substring, tier, material_type, criticality)
    # --- Final assemblers / EMS (the CVRPTW dispatch hubs + FJSP machines) ---
    ("Hon Hai", 2, "final_assembly", "oligopoly"),
    ("Pegatron", 2, "final_assembly", "oligopoly"),
    ("Luxshare", 2, "final_assembly", "oligopoly"),
    ("Quanta", 2, "final_assembly", "oligopoly"),
    ("Compal", 2, "final_assembly", "oligopoly"),
    ("Wistron", 2, "final_assembly", "diversified_commodity"),
    ("Jabil", 2, "final_assembly", "diversified_commodity"),
    ("Flex Limited", 2, "final_assembly", "diversified_commodity"),
    ("BYD Company", 2, "final_assembly", "diversified_commodity"),
    ("Cheng Uei", 2, "final_assembly", "diversified_commodity"),
    # --- OSAT (assembly/test) — FJSP test-capable machines ---
    ("Amkor", 2, "advanced_packaging_osat", "oligopoly"),
    ("Advanced Semiconductor Engineering", 2, "advanced_packaging_osat", "oligopoly"),
    ("JCET", 2, "advanced_packaging_osat", "oligopoly"),
    ("United Test and Assembly", 2, "advanced_packaging_osat", "oligopoly"),
    # --- Foundry (sole leading-edge SoC source) ---
    ("Taiwan Semiconductor", 2, "soc_foundry", "monopoly_bottleneck"),
    # --- Displays ---
    ("BOE Technology", 2, "display", "oligopoly"),
    ("LG Display", 2, "display", "oligopoly"),
    ("Japan Display", 2, "display", "oligopoly"),
    # --- Tier-3 components / materials (real regions from the list) ---
    ("SK hynix", 3, "memory", "oligopoly"),
    ("Micron", 3, "memory", "oligopoly"),
    ("Kioxia", 3, "memory", "oligopoly"),
    ("Western Digital", 3, "memory", "oligopoly"),
    ("Samsung Electronics", 3, "memory", "oligopoly"),
    ("Samsung SDI", 3, "battery_cell", "oligopoly"),
    ("LG Energy", 3, "battery_cell", "oligopoly"),
    ("Sunwoda", 3, "battery_cell", "diversified_commodity"),
    ("Desay", 3, "battery_cell", "diversified_commodity"),
    ("Simplo", 3, "battery_cell", "diversified_commodity"),
    ("Largan", 3, "camera_optics", "oligopoly"),
    ("Genius Electronic", 3, "camera_optics", "diversified_commodity"),
    ("Sunny Optical", 3, "camera_optics", "diversified_commodity"),
    ("Cowell", 3, "camera_optics", "diversified_commodity"),
    ("Corning", 3, "cover_glass", "oligopoly"),
    ("Lens Technology", 3, "cover_glass", "diversified_commodity"),
    ("Biel Crystal", 3, "cover_glass", "diversified_commodity"),
    ("Amphenol", 3, "connectors", "diversified_commodity"),
    ("Molex", 3, "connectors", "diversified_commodity"),
    ("Qualcomm", 3, "rf_analog_soc", "oligopoly"),
    ("Broadcom", 3, "rf_analog_soc", "oligopoly"),
    ("Skyworks", 3, "rf_analog_soc", "oligopoly"),
    ("Qorvo", 3, "rf_analog_soc", "oligopoly"),
    ("Texas Instruments", 3, "rf_analog_soc", "oligopoly"),
    ("Analog Devices", 3, "rf_analog_soc", "oligopoly"),
    ("Cirrus Logic", 3, "rf_analog_soc", "oligopoly"),
    ("NXP", 3, "rf_analog_soc", "oligopoly"),
    ("STMicro", 3, "rf_analog_soc", "oligopoly"),
    ("Infineon", 3, "rf_analog_soc", "oligopoly"),
    ("ON Semiconductor", 3, "rf_analog_soc", "diversified_commodity"),
]

# Per-role synthetic fan-out ranges (n_synthetic_peers, n_synthetic_children).
# Big assemblers/foundry fan out hardest (many sub-suppliers); commodity tier-3
# vendors stay small. Chosen so planet-scale scale_factor=25 grows the real
# anchors realistically rather than diluting to generic padding.
_ROLE_FANOUT: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "final_assembly": ((3, 5), (120, 240)),
    "advanced_packaging_osat": ((2, 4), (40, 80)),
    "soc_foundry": ((0, 1), (150, 300)),
    "display": ((2, 4), (50, 100)),
    "memory": ((2, 3), (0, 0)),
    "battery_cell": ((2, 4), (0, 0)),
    "camera_optics": ((2, 3), (0, 0)),
    "cover_glass": ((2, 3), (0, 0)),
    "connectors": ((4, 7), (0, 0)),
    "rf_analog_soc": ((2, 3), (0, 0)),
    "component_generic": ((1, 2), (0, 0)),
}


def _curated_role(name: str) -> tuple[int, str, str] | None:
    low = name.lower()
    for sub, tier, material, crit in _CURATED_ROLES:
        if sub.lower() in low:
            return tier, material, crit
    return None


def build_apple_real_anchors(path: str = _DEFAULT_PATH):
    """Build ``(tier2_anchors, tier3_anchors)`` ``AnchorSpec`` lists from the
    parsed Apple supplier list.

    Curated recognizable names get their public role/tier/criticality; every
    other parsed supplier becomes a generic tier-3 component anchor tagged with
    its real primary region (first region listed). Returned lists are suitable
    for ``COMPANY_PROFILES["apple_real"]``.
    """
    # Imported here (not at module top) to avoid a circular import:
    # company_profiles imports THIS module to call build_apple_real_anchors.
    from scripts.company_profiles import AnchorSpec

    suppliers = parse_supplier_list(path)
    tier2: list[AnchorSpec] = []
    tier3: list[AnchorSpec] = []

    for sup in suppliers:
        name = sup["name"]
        regions = sup["regions"] or ["China"]
        primary_region = regions[0]
        role = _curated_role(name)
        if role is not None:
            tier, material, crit = role
        else:
            tier, material, crit = 3, "component_generic", "diversified_commodity"
        peers, children = _ROLE_FANOUT.get(material, _ROLE_FANOUT["component_generic"])
        spec = AnchorSpec(
            company_name=name,
            tier=tier,
            material_type=material,
            region=primary_region,
            criticality=crit,
            n_synthetic_peers=peers,
            n_synthetic_children=children,
            real_world_basis=(
                f"Apple published Supplier List (FY2023); primary regions: "
                f"{', '.join(regions)}."
            ),
        )
        (tier2 if tier == 2 else tier3).append(spec)

    return tier2, tier3

"""Tests for the Apple published-supplier-list parser and anchor builder
(``scripts/apple_supplier_list.py``) that grounds the ``apple_real`` profile.

Run from the repo root: python -m pytest tests/ -v
"""

from __future__ import annotations

import scripts.apple_supplier_list as asl
import scripts.company_profiles as cp

# The recognizable names the curated role map must classify (substring match).
_EXPECTED_CURATED = [
    "Hon Hai",  # Foxconn
    "Pegatron",
    "Luxshare",
    "Amkor",
    "Advanced Semiconductor Engineering",  # ASE
    "JCET",
    "Taiwan Semiconductor",  # TSMC
    "BOE",
    "LG Display",
    "Corning",
    "Qualcomm",
    "SK hynix",
]


def test_parse_returns_many_clean_suppliers():
    suppliers = asl.parse_supplier_list()
    # The published FY2023 list has ~180-190 rows once continuation lines are
    # folded; assert a healthy count, not an exact one (robust to re-parsing).
    assert len(suppliers) > 120
    for s in suppliers:
        assert s["name"] and len(s["name"]) >= 4
        assert s["regions"], f"{s['name']} has no parsed region"
        # no strike-through junk leaked into a name
        assert "~~" not in s["name"]
        assert "CLEAN ENERGY" not in s["name"]


def test_every_curated_name_is_present_in_the_parse():
    names = [s["name"].lower() for s in asl.parse_supplier_list()]
    for expected in _EXPECTED_CURATED:
        assert any(expected.lower() in n for n in names), f"{expected} not parsed"


def test_regions_are_normalized_to_repo_vocabulary():
    suppliers = asl.parse_supplier_list()
    seen = {r for s in suppliers for r in s["regions"]}
    # normalized names, not raw "China mainland"/"United States"
    assert "China" in seen
    assert "USA" in seen
    assert "China mainland" not in seen
    assert "United States" not in seen


def test_build_anchors_has_assembly_osat_and_foundry():
    tier2, tier3 = asl.build_apple_real_anchors()
    assert tier2 and tier3
    tier2_materials = {a.material_type for a in tier2}
    assert "final_assembly" in tier2_materials
    assert "advanced_packaging_osat" in tier2_materials
    assert "soc_foundry" in tier2_materials
    # TSMC is the sole monopoly bottleneck
    monopoly = [a for a in tier2 if a.criticality == "monopoly_bottleneck"]
    assert len(monopoly) == 1
    assert "Taiwan Semiconductor" in monopoly[0].company_name


def test_build_anchors_all_criticalities_valid():
    tier2, tier3 = asl.build_apple_real_anchors()
    valid = {"monopoly_bottleneck", "oligopoly", "diversified_commodity", "generic"}
    for a in tier2 + tier3:
        assert a.criticality in valid
        assert a.region  # every anchor has a real region
        assert a.tier in (2, 3)


def test_uncurated_suppliers_become_generic_tier3():
    tier2, tier3 = asl.build_apple_real_anchors()
    generic = [a for a in tier3 if a.material_type == "component_generic"]
    # The long tail of unrecognized suppliers should dominate tier-3.
    assert len(generic) > 50


def test_profile_is_registered_and_generates():
    # apple_real must be wired into all the dicts generate_complex_network needs.
    assert "apple_real" in cp.COMPANY_PROFILES
    assert "apple_real" in cp.PRODUCT_LINE_LABELS
    assert "apple_real" in cp.REGION_WEIGHTS
    t2 = cp.COMPANY_PROFILES["apple_real"]["tier2"]
    assert any(a.material_type == "final_assembly" for a in t2)

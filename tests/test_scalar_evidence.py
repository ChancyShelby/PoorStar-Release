"""Tests: test scalar evidence."""

import pytest
from s2clientprotocol import sc2api_pb2 as sc

from poorstar.engine.scalar_evidence import ScalarEvidence


def catalog():
    d = sc.ResponseData()
    d.units.add(unit_id=900, name="Larva")
    d.units.add(unit_id=901, name="Drone")
    return d


def test_presence_zero_and_enemy_selected_do_not_count_as_owned_larvae():
    e = ScalarEvidence(catalog())
    r = sc.ResponseObservation()
    r.observation.player_common.player_id = 1
    r.observation.raw_data.units.add(unit_type=900, alliance=1, display_type=1)
    r.observation.raw_data.units.add(unit_type=900, alliance=4, display_type=2, is_selected=True)
    e.append(r)
    r.observation.player_common.larva_count = 0
    e.append(r)
    d = e.result()
    assert d["scalar_fields"]["larva_count"] == dict(absent=1, present=1, observed_zero=1)
    assert d["larva_comparisons"]["absent_own_visible_larvae_positive"] == 1
    assert d["larva_comparisons"]["present_equals_selected_larvae"] == 1
    assert d["larva_comparisons"]["present_equals_own_larvae"] == 0
    assert all(row["selected_types"] == [] for row in d["selected_type_presence"])


def test_owned_snapshot_is_not_currently_visible_and_selection_is_only_audit():
    e = ScalarEvidence(catalog())
    r = sc.ResponseObservation()
    r.observation.player_common.larva_count = 2
    for display in (1, 2):
        r.observation.raw_data.units.add(
            unit_type=900, alliance=1, display_type=display, is_selected=True
        )
    e.append(r)
    d = e.result()
    assert d["larva_comparisons"]["present_equals_own_larvae"] == 1
    assert d["larva_comparisons"]["present_equals_own_visible_larvae"] == 0
    assert not d["selection_used_as_actor_feature"]


def test_engine_catalog_identity_is_required():
    d = catalog()
    d.units.add(unit_id=902, name="Larva")
    with pytest.raises(ValueError, match="unique engine Larva"):
        ScalarEvidence(d)

from types import SimpleNamespace as S

import pytest

from poorstar.engine.query_evidence import ability_answers


def response(*tags):
    return S(abilities=[S(unit_tag=t, abilities=[]) for t in tags])


def test_present_unit_answer_identity_is_required():
    assert ability_answers(response(12), [12]) == {12: ()}
    with pytest.raises(ValueError, match="identity/order"):
        ability_answers(response(0), [12])


def test_absent_control_zero_identity_is_explicit_only():
    assert ability_answers(response(0), [12], control_tags=[12]) == {12: ()}


def test_count_and_duplicate_request_fail():
    with pytest.raises(ValueError, match="count or request"):
        ability_answers(response(), [12])
    with pytest.raises(ValueError, match="count or request"):
        ability_answers(response(12, 12), [12, 12])


def test_no_order_based_silent_pointer_repair():
    with pytest.raises(ValueError, match="identity/order"):
        ability_answers(response(14, 12), [12, 14])


def test_point_requirements_and_duplicate_abilities():
    r = response(12)
    r.abilities[0].abilities = [S(ability_id=3, requires_point=True)]
    assert ability_answers(r, [12])[12] == ((3, True),)
    r.abilities[0].abilities *= 2
    with pytest.raises(ValueError, match="duplicate ability"):
        ability_answers(r, [12])

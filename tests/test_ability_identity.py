import pytest

from poorstar.actions.abilities import canonical_ability


def test_remap_chain_and_terminal_zero_are_engine_defined():
    assert canonical_ability(12, {12: 14, 14: 19, 19: 0}) == 19
    assert canonical_ability(19, {19: 19}) == 19


def test_missing_and_cycles_are_not_silently_accepted():
    with pytest.raises(ValueError, match="absent"):
        canonical_ability(12, {12: 14})
    with pytest.raises(ValueError, match="cyclic"):
        canonical_ability(12, {12: 14, 14: 12})

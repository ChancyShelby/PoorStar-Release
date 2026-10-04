import json

import pytest

from poorstar.cli.data.build import read_verified, write_report
from poorstar.common.integrity import payload_hash
from poorstar.data.bc.supervision import DISTAR_SKIP, supervision_census
from tests.test_coarse_bc import decision_rows


def test_nested_integer_ability_counts_are_hash_stable_after_json_roundtrip(tmp_path):
    data = {
        "episodes": [
            {
                "requested_ability_counts": {2: 5, 10: 9, 1342: 1},
                "retained_ability_counts": {2: 4, 10: 8, 1342: 1},
            }
        ]
    }
    p = tmp_path / "report.json"
    write_report(p, data)
    value, digest = read_verified(p)
    assert value["episodes"][0]["requested_ability_counts"] == {"2": 5, "10": 9, "1342": 1}
    assert digest == payload_hash(value)
    assert data["episodes"][0]["requested_ability_counts"] == {2: 5, 10: 9, 1342: 1}


def test_json_key_collision_refused_before_publication(tmp_path):
    p = tmp_path / "report.json"
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        write_report(p, {"counts": {2: 5, "2": 9}})
    assert not p.exists() and not (tmp_path / "report.json.partial").exists()


def test_duplicate_json_keys_refused_even_with_digest_of_last_value(tmp_path):
    digest = payload_hash({"counts": {"2": 9}})
    p = tmp_path / "report.json"
    p.write_text('{"counts":{"2":5,"2":9},"content_digest":' + json.dumps(digest) + "}")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        read_verified(p)


def test_census_is_json_native_without_changing_supervision(tmp_path):
    rows, profile, catalog, info = decision_rows()
    census = supervision_census(rows, profile, policy=DISTAR_SKIP)
    assert all(isinstance(k, str) for k in census["requested_ability_counts"])
    assert all(isinstance(k, str) for k in census["retained_ability_counts"])
    assert census["counts"]["rows"] == len(rows) and census["counts"]["requested_commands"] == sum(
        len(r.targets) for r in rows
    )
    assert json.loads(json.dumps(census)) == census
    p = tmp_path / "census.json"
    write_report(p, census)
    assert read_verified(p)[0] == census

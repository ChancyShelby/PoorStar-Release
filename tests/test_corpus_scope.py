import pytest

from poorstar.data.corpus.scope import candidate_scope


def entry(alias, races, *, build=55958):
    return dict(
        alias=alias,
        path=alias,
        metadata=dict(
            file_sha256=alias,
            bytes=10,
            semantic_sha256="match_" + alias,
            base_build=build,
            map_hash="map",
            frames=100,
            players=[
                dict(player_id=i + 1, race=r, account_digest=None) for i, r in enumerate(races)
            ],
        ),
    )


def scope(rows, recovered=()):
    return candidate_scope(
        dict(entries=rows, records=len(rows), content_digest="original"),
        dict(entries=[dict(metadata=dict(file_sha256=x)) for x in recovered], content_digest="old"),
        engine_build=55958,
    )


def test_russian_plural_zvz_counts_two_perspectives_one_match():
    r = scope([entry("z", ["Зерги", "Зерги"])])
    assert r["matchups"] == {"ZvZ": 1}
    assert r["counts"]["compatible_zerg_candidate_matches"] == 1
    assert r["counts"]["compatible_zerg_candidate_perspectives"] == 2
    assert r["entries"][0]["provisional_match_key"] == "match_z"
    assert r["entries"][0]["players"][0]["original_race"] == "Зерги"


def test_unprocessed_records_are_not_declared_fresh():
    r = scope([entry("a", ["Zerg", "Terran"]), entry("b", ["Зерги", "Протоссы"])], ["a"])
    assert r["entries"][0]["history"] == "KNOWN_HISTORICAL_RECOVERY_AT_LEAST"
    assert r["entries"][1]["history"] == "UNKNOWN_NOT_CONFIRMATORY"
    assert not r["training_ready"] and not r["split_frozen"] and not r["rl_ready"]


def test_failure_other_race_and_unsupported_build_all_remain_entries():
    r = scope(
        [
            dict(alias="bad", path="bad", error="MPQ"),
            entry("p", ["Protoss", "Terran"]),
            entry("old", ["Zerg", "Terran"], build=55505),
            entry("unknown", ["unknown", "Zerg"]),
        ]
    )
    assert len(r["entries"]) == 4
    assert not any(e.get("scope_candidate") for e in r["entries"])
    assert r["counts"]["metadata_parse_failure"] == 1
    assert r["counts"]["engine_base_build_unavailable"] == 1
    assert r["counts"]["unknown_race_label"] == 1


def test_inventory_count_mismatch_is_not_silently_accepted():
    with pytest.raises(ValueError, match="count mismatch"):
        candidate_scope(
            dict(entries=[], records=1), dict(entries=[], content_digest="old"), engine_build=55958
        )

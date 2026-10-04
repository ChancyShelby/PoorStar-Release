"""Tests: test corpus partition."""

from copy import deepcopy

import pytest

from poorstar.common.integrity import payload_hash
from poorstar.data.corpus.partition import partition_jobs, validate_partition


def fixture():
    identity = {"plan_digest": "frozen", "split_sha256": "unchanged"}
    jobs = []
    for n in range(30):
        replay = payload_hash(["replay", n])
        group = payload_hash(["native", n])
        for player in (1, 2):
            jobs.append(
                dict(
                    job_id=payload_hash([replay, player]),
                    replay_sha256=replay,
                    player_id=player,
                    native_content_group=group,
                    split=("train", "development", "test")[n % 3],
                )
            )
    return identity, jobs


def test_complete_weighted_partition_repeats_without_mutating_plan_or_split():
    identity, jobs = fixture()
    before = deepcopy(jobs)
    a, r = partition_jobs(identity, jobs, {"cpu": 2, "gpu": 1}, seed=20261004)
    b, s = partition_jobs(identity, jobs, {"gpu": 1, "cpu": 2}, seed=20261004)
    assert a == b and r == s and jobs == before
    assert validate_partition(a, identity, jobs) == 60
    assert r["registered_views"] == 60 and sum(r["lane_view_counts"].values()) == 60
    assert not r["formal_bc_started"] and not r["rl_started"]
    for group in {j["native_content_group"] for j in jobs}:
        pair = {j["job_id"] for j in jobs if j["native_content_group"] == group}
        assert sum(pair <= set(x["job_ids"]) for x in a.values()) == 1


@pytest.mark.parametrize("case", ["overlap", "gap", "split_group", "binding", "duplicate"])
def test_partition_drift_refused(case):
    identity, jobs = fixture()
    a, _ = partition_jobs(identity, jobs, {"cpu": 2, "gpu": 1}, seed=1)
    if case == "overlap":
        a["gpu"]["job_ids"].append(a["cpu"]["job_ids"][0])
    elif case == "gap":
        a["cpu"]["job_ids"].pop()
    elif case == "split_group":
        a["gpu"]["job_ids"].append(a["cpu"]["job_ids"].pop(0))
    elif case == "binding":
        a["cpu"]["data_binding"]["split_sha256"] = "changed"
    elif case == "duplicate":
        a["cpu"]["job_ids"].append(a["cpu"]["job_ids"][0])
    with pytest.raises(ValueError):
        validate_partition(a, identity, jobs)


@pytest.mark.parametrize(
    "weights,seed",
    [
        ({"cpu": 0, "gpu": 1}, 1),
        ({"cpu": 2, "gpu": True}, 1),
        ({"../cpu": 2, "gpu": 1}, 1),
        ({"cpu": 2}, 1),
        ({"cpu": 64, "gpu": 1}, 1),
        ({"cpu": 2, "gpu": 1}, True),
    ],
)
def test_invalid_scheduling_configuration_refused(weights, seed):
    identity, jobs = fixture()
    with pytest.raises(ValueError):
        partition_jobs(identity, jobs, weights, seed=seed)


def test_split_leakage_and_duplicate_jobs_are_refused():
    identity, jobs = fixture()
    jobs[1]["split"] = "test"
    with pytest.raises(ValueError, match="split leakage"):
        partition_jobs(identity, jobs, {"cpu": 2, "gpu": 1}, seed=1)
    identity, jobs = fixture()
    jobs.append(jobs[0])
    with pytest.raises(ValueError, match="duplicate"):
        partition_jobs(identity, jobs, {"cpu": 2, "gpu": 1}, seed=1)

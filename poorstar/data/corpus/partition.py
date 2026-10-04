"""Data: corpus: partition."""

import re
from collections import Counter

from poorstar.common.integrity import payload_hash
from poorstar.data.corpus.assignment import SCHEMA, select_assignment


def partition_jobs(identity, jobs, weights, *, seed):
    """Assign complete native replay groups to weighted, mutually exclusive lanes.

    Weights are fixed before execution. Hash buckets affect wall-clock scheduling,
    not sample selection, extraction semantics, learner order or validation.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer scheduling seed required")
    if not isinstance(weights, dict) or len(weights) < 2:
        raise ValueError("at least two explicitly named lanes required")
    if any(
        type(k) is not str
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", k)
        or type(v) is not int
        or v < 1
        for k, v in weights.items()
    ):
        raise ValueError("safe lane names and positive integer weights required")
    if sum(weights.values()) > 64:
        raise ValueError("scheduling bucket bound exceeded")
    groups = {}
    ids = set()
    for job in jobs:
        if job["job_id"] in ids:
            raise ValueError("duplicate job identity")
        ids.add(job["job_id"])
        native = job.get("native_content_group")
        if type(native) is not str or not re.fullmatch(r"[0-9a-f]{64}", native):
            raise ValueError("native content group identity required")
        group = groups.setdefault(native, [])
        if group and group[0]["split"] != job["split"]:
            raise ValueError("native group split leakage")
        group.append(job)
    if not ids:
        raise ValueError("empty scientific plan")
    buckets = [lane for lane in sorted(weights) for _ in range(weights[lane])]
    lane_for_group = {
        g: buckets[int(payload_hash(["FIXED_GROUP_PARTITION_V1", seed, g]), 16) % len(buckets)]
        for g in groups
    }
    assignments = {}
    for lane in sorted(weights):
        chosen = [j for j in jobs if lane_for_group[j["native_content_group"]] == lane]
        if not chosen:
            raise ValueError("empty lane; preregister another partition before execution")
        manifest = dict(
            schema=SCHEMA,
            data_binding=dict(identity),
            registered_views=len(jobs),
            job_ids=[j["job_id"] for j in chosen],
        )
        assert select_assignment(manifest, identity, jobs) == chosen
        assignments[lane] = manifest
    validate_partition(assignments, identity, jobs)
    report = dict(
        schema="poorstar_c3_fixed_group_partition_v1",
        data_binding=dict(identity),
        registered_views=len(jobs),
        seed=seed,
        weights=dict(weights),
        native_groups=len(groups),
        grouping="NATIVE_CONTENT_GROUP",
        lane_view_counts={k: len(v["job_ids"]) for k, v in assignments.items()},
        lane_split_counts={
            lane: dict(
                Counter(
                    j["split"] for j in jobs if lane_for_group[j["native_content_group"]] == lane
                )
            )
            for lane in assignments
        },
        scientific_scope_unchanged=True,
        split_unchanged=True,
        formal_bc_started=False,
        rl_started=False,
    )
    return assignments, report


def validate_partition(assignments, identity, jobs):
    if len(assignments) < 2:
        raise ValueError("complete multi-lane partition required")
    seen = set()
    group_lane = {}
    for lane, manifest in assignments.items():
        selected = select_assignment(manifest, identity, jobs)
        for job in selected:
            if job["job_id"] in seen:
                raise ValueError("overlapping assignment")
            seen.add(job["job_id"])
            native = job["native_content_group"]
            if group_lane.setdefault(native, lane) != lane:
                raise ValueError("native group split between scheduling lanes")
    if seen != {j["job_id"] for j in jobs}:
        raise ValueError("partition has missing registered jobs")
    return len(seen)

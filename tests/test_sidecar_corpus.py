"""Tests: test sidecar corpus."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest
import torch

from poorstar.data.bc.records import BCShardWriter, freeze_split
from poorstar.data.bc.windows import BCRunPlan, BCWindowDataset
from poorstar.evaluation.sidecar_corpus import (
    COMPONENT_SCHEMA,
    compose_control_sidecar_corpus,
    compose_indexed_control_sidecar_corpus,
)
from poorstar.evaluation.teachers import build_control_sidecar
from poorstar.hierarchy import HierarchyBCRecord, HierarchyBCSequence, HierarchyState
from poorstar.hierarchy.bc import audit_hierarchy_bc_sequence
from poorstar.training import TrainingIdentity
from poorstar.training.checkpoint import capture_rng, file_hash
from poorstar.training.config import ControlSidecar, model_config_sha256
from poorstar.training.contracts import digest
from tests.test_coarse_bc import decision_rows, setup, target
from tests.test_hierarchy_runtime import model_fixture, update_control


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def write_document(path, document):
    document = deepcopy(document)
    document.pop("content_digest", None)
    document["content_digest"] = digest(document)
    path.write_text(json.dumps(document))


def inputs(tmp_path, *, unroll=2, burn=1, arm="persistent_intent", episodes=1):
    rows, profile, catalog, info = decision_rows()
    additional = []
    for loop in (4, 5):
        frame, *_ = setup(loop=loop)
        additional.append(replace(rows[1], scheduler=frame))
    rows = rows + tuple(additional)
    if episodes == 2:
        a, *_ = setup(loop=0, episode="episode2")
        b, *_ = setup(loop=2, episode="episode2")
        extra = [replace(rows[0], scheduler=a, body=b, targets=(target(b),))]
        for loop in (3, 4, 5):
            frame, *_ = setup(loop=loop, episode="episode2")
            extra.append(replace(rows[1], scheduler=frame))
        rows = rows + tuple(extra)
    split = freeze_split([dict(replay_sha256=rows[0].replay_sha256, match_group="match")], seed=1)
    assert split["assignments"][rows[0].replay_sha256] == "train"
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    writer = BCShardWriter(
        tmp_path / "shards" / "train",
        split_manifest=split,
        split="train",
        catalog=catalog,
        game_info=info,
        source_manifest_sha256="a" * 64,
        extractor_sha256="b" * 64,
        maximum_delay=8,
    )
    for row in rows:
        writer.append(row)
    artifact = writer.close()
    build = dict(
        schema="poorstar_c3_sidecar_protocol_fixture_v1",
        split_sha256=split["sha256"],
        shards=[artifact],
    )
    build_path = tmp_path / "build.json"
    write_document(build_path, build)
    dataset = BCWindowDataset(build_path, split_path, unroll_length=unroll, burn_in_length=burn)
    model, _, _ = model_fixture(arm)
    model.eval()
    controls = [
        update_control(model),
        {"manager": None, "tasks": None},
        {
            "manager": None,
            "tasks": None
            if arm == "flat"
            else [{"operation": "keep"}] * model.hierarchy_config.task_slots,
        },
        {"manager": None, "tasks": None},
    ] * episodes
    records = tuple(
        HierarchyBCRecord(
            row,
            scheduler_control=controls[i],
            body_control={"manager": None, "tasks": None} if row.body is not None else None,
            body_execution_task=row.scheduler.key.episode + ":task:1"
            if row.body is not None and arm != "flat"
            else None,
            supervision_id="DECLARED_PROTOCOL_CONTROL:" + str(i),
            supervision_kind="actor_control_trace",
        )
        for i, row in enumerate(rows)
    )
    with torch.no_grad():
        _, audit = audit_hierarchy_bc_sequence(
            model,
            HierarchyBCSequence(records, HierarchyState("episode")),
            profile,
            collect_metrics=False,
        )
    before = {
        (event.episode, event.scheduler_loop): event.prior_state_dict
        for event in audit
        if event.phase == "scheduler"
    }
    plan = BCRunPlan(
        "ENGINEERING", 31, 1, hidden_size=32, unroll_length=unroll, burn_in_length=burn
    )
    plan_doc = dict(plan=asdict(plan), head_weights={}, control_weights={})
    identity = TrainingIdentity(
        "SIDECAR_WINDOW_PROTOCOL_FIXTURE",
        arm,
        31,
        "a" * 64,
        dataset.sha256,
        split["sha256"],
        "c" * 64,
        model_config_sha256(model.config, model.hierarchy_config),
        profile.sha256,
        digest(plan_doc),
    )
    labels = {}
    for record in records:
        label = {
            f.name: deepcopy(getattr(record, f.name))
            for f in fields(record)
            if f.name != "decision"
        }
        labels[ControlSidecar.row_id(record.decision)] = label
    components = []
    for index in range(len(dataset.windows)):
        window = dataset.window(index)
        key = window[0].scheduler.key
        initial = HierarchyState.from_dict(before[(key.episode, key.loop)], model.hierarchy_config)
        annotations = []
        for row in window:
            dependency = lambda frame: [
                dict(
                    loop=frame.key.loop,
                    player_id=frame.key.player_id,
                    observation_sha256=frame.key.observation_sha256,
                    evidence_sha256="d" * 64,
                    availability="ACTOR_VISIBLE_CURRENT_OR_PAST",
                )
            ]
            annotations.append(
                dict(
                    row_id=ControlSidecar.row_id(row),
                    label=labels[ControlSidecar.row_id(row)],
                    scheduler_evidence=dependency(row.scheduler),
                    body_evidence=[] if row.body is None else dependency(row.body),
                )
            )
        evidence = dict(
            schema="poorstar_v3_control_annotation_evidence_v1",
            annotation_policy_sha256="e" * 64,
            human_intent_truth=False,
            future_or_privileged_inputs=False,
            annotation_sha256=digest(annotations),
            initial_control_states_sha256=digest({annotations[0]["row_id"]: initial.to_dict()}),
            row_order=[a["row_id"] for a in annotations],
        )
        evidence_path = tmp_path / ("evidence_" + str(index) + ".json")
        evidence_path.write_text(json.dumps(evidence))
        path = tmp_path / ("component_" + str(index) + ".json")
        result = build_control_sidecar(
            window,
            annotations,
            initial_control_state=initial,
            model=model,
            profile=profile,
            identity=identity,
            evidence_path=evidence_path,
            output_path=path,
            annotation_policy_sha256="e" * 64,
        )
        components.append(dict(window_index=index, path=str(path), sha256=result["sha256"]))
    return dataset, model, identity, components, build_path, split_path, plan_doc


def revise_component(component, change):
    path = Path(component["path"])
    document = json.loads(path.read_text())
    change(document)
    evidence_path = Path(document["provenance_evidence_path"])
    evidence = json.loads(evidence_path.read_text())
    annotations = [
        dict(
            row_id=row_id,
            label=document["records"][row_id],
            scheduler_evidence=document["annotation_dependencies"][row_id]["scheduler"],
            body_evidence=document["annotation_dependencies"][row_id]["body"],
        )
        for row_id in evidence["row_order"]
    ]
    document["annotation_sha256"] = digest(annotations)
    evidence["annotation_sha256"] = digest(annotations)
    evidence["initial_control_states_sha256"] = digest(document["initial_control_states"])
    evidence_path.write_text(json.dumps(evidence))
    document["provenance_evidence_sha256"] = file_hash(evidence_path)
    write_document(path, document)
    component["sha256"] = file_hash(path)


@pytest.mark.parametrize("unroll,burn", [(2, 1), (1, 2)])
def test_full_actual_overlapping_windows_merge_masks_and_prior_body_state_without_adam(
    tmp_path, unroll, burn
):
    dataset, model, identity, components, *_ = inputs(tmp_path, unroll=unroll, burn=burn)
    rng = capture_rng()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    try:
        result = compose_control_sidecar_corpus(
            dataset, components, model=model, identity=identity, output_dir=tmp_path / "corpus"
        )
        reader = ControlSidecar(
            result["control_sidecar_path"],
            replace(identity, control_sidecar_sha256=result["control_sidecar_sha256"]),
            model.hierarchy_config,
        )
        proof = reader.validate_dataset(dataset, model)
        assert proof["windows"] == len(dataset.windows) and proof["original_train_rows"] == 4
        assert (
            proof["optimizer_steps"] == 0
            and not proof["human_intent_truth"]
            and not proof["formal_corpus_qualified"]
        )
        assert proof["window_row_occurrences"] > 4 and proof["denominators"]["act"] == 4
        window = reader.document["corpus_coverage"]["windows"][-1]
        if window["metadata"][0] == 1:
            assert window["initial_state_predecessor"]["phase"] == "body"
            assert window["initial_state_predecessor"]["loop"] == 2
        for i in range(len(dataset.windows)):
            q = reader.sequence(dataset.window(i), window_index=i)
            assert [r.decision.loss for r in q.records] == [r.loss for r in dataset.window(i)]
        assert torch.equal(rng["torch"], capture_rng()["torch"])
        assert all(
            torch.equal(before[n], p) and p.grad is None for n, p in model.named_parameters()
        )
    finally:
        dataset.close()


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "partial",
        "index",
        "duplicate_label",
        "duplicate_dependency",
        "missing_state",
        "future_state",
        "wrong_state",
    ],
)
def test_partial_overlap_or_noncausal_state_never_publishes_complete_artifact(
    tmp_path, failure, indexed
):
    dataset, model, identity, components, *_ = inputs(tmp_path)
    if failure == "partial":
        components.pop()
    elif failure == "index":
        components[1]["window_index"] = 0
    else:

        def change(document):
            first = next(iter(document["records"]))
            state = next(iter(document["initial_control_states"].values()))
            if failure == "duplicate_label":
                document["records"][first]["supervision_id"] = "DIFFERENT_DECLARED_LABEL"
            elif failure == "duplicate_dependency":
                document["annotation_dependencies"][first]["scheduler"][0]["evidence_sha256"] = (
                    "f" * 64
                )
            elif failure == "missing_state":
                document["initial_control_states"] = {}
            elif failure == "future_state":
                state["last_loop"] = 3
                state["intent"]["age"] = 3
                [t.update(age=3) for t in state["tasks"]]
            else:
                state["task_revision"] += 1

        revise_component(components[1], change)
    try:
        with pytest.raises(ValueError):
            compose_control_sidecar_corpus(
                dataset,
                components,
                model=model,
                identity=identity,
                output_dir=tmp_path / "corpus",
                indexed=indexed,
            )
        assert not (tmp_path / "corpus").exists()
        if failure not in ("partial", "index"):
            failures = list(tmp_path.glob(".corpus.attempt-*/failure.json"))
            assert (
                len(failures) == 1
                and not json.loads(failures[0].read_text())["final_artifact_published"]
            )
    finally:
        dataset.close()


def test_reader_checks_only_current_dependencies_and_masks_after_single_global_provenance_pass(
    tmp_path, monkeypatch
):
    dataset, model, identity, components, *_ = inputs(tmp_path)
    try:
        result = compose_control_sidecar_corpus(
            dataset, components, model=model, identity=identity, output_dir=tmp_path / "corpus"
        )
        reader = ControlSidecar(
            result["control_sidecar_path"],
            replace(identity, control_sidecar_sha256=result["control_sidecar_sha256"]),
            model.hierarchy_config,
        )
        import poorstar.evaluation.teachers as teachers

        monkeypatch.setattr(
            teachers,
            "verify_annotation_document",
            lambda *a, **k: pytest.fail("full annotation digest rebuilt during window read"),
        )
        assert len(reader.sequence(dataset.window(1), window_index=1).records) == 3
        wrong = tuple(replace(r, burn_in=False, loss=True) for r in dataset.window(1))
        with pytest.raises(ValueError, match="loss masks"):
            reader.sequence(wrong, window_index=1)
        first = ControlSidecar.row_id(dataset.window(1)[0])
        reader.document["records"][first]["supervision_id"] = "mutated-in-memory"
        with pytest.raises(ValueError, match="changed after initialization"):
            reader.sequence(dataset.window(1), window_index=1)
    finally:
        dataset.close()


def test_component_lineage_and_formal_partial_sidecar_fail_before_optimizer(tmp_path):
    dataset, model, identity, components, *_ = inputs(tmp_path)
    try:
        # Formal annotations must have both typed provenance and full corpus coverage.
        formal = replace(identity, purpose="FORMAL", control_sidecar_sha256=components[0]["sha256"])
        with pytest.raises(ValueError, match="corpus window coverage"):
            ControlSidecar(components[0]["path"], formal, model.hierarchy_config)
        result = compose_control_sidecar_corpus(
            dataset, components, model=model, identity=identity, output_dir=tmp_path / "corpus"
        )
        with pytest.raises(ValueError, match="model samples"):
            ControlSidecar(
                result["control_sidecar_path"],
                replace(
                    identity,
                    purpose="FORMAL",
                    control_sidecar_sha256=result["control_sidecar_sha256"],
                ),
                model.hierarchy_config,
            )
        component = Path(components[1]["path"])
        component.write_text(component.read_text() + " ")
        with pytest.raises(ValueError, match="component bytes changed"):
            ControlSidecar(
                result["control_sidecar_path"],
                replace(identity, control_sidecar_sha256=result["control_sidecar_sha256"]),
                model.hierarchy_config,
            )
    finally:
        dataset.close()


def test_full_teacher_corpus_resets_control_history_at_each_real_episode_boundary(tmp_path):
    dataset, model, identity, components, *_ = inputs(tmp_path, episodes=2)
    try:
        result = compose_control_sidecar_corpus(
            dataset, components, model=model, identity=identity, output_dir=tmp_path / "corpus"
        )
        reader = ControlSidecar(
            result["control_sidecar_path"],
            replace(identity, control_sidecar_sha256=result["control_sidecar_sha256"]),
            model.hierarchy_config,
        )
        proof = reader.validate_dataset(dataset, model)
        assert (
            proof["windows"] == 4
            and proof["original_train_rows"] == 8
            and proof["denominators"]["manager_control"] == 2
        )
        starts = [
            w
            for w in reader.document["corpus_coverage"]["windows"]
            if w["initial_state_predecessor"] is None
        ]
        assert len(starts) == 2
        assert {
            reader.document["initial_control_states"][w["row_order"][0]]["episode"] for w in starts
        } == {"episode", "episode2"}
    finally:
        dataset.close()


def test_teacher_corpus_is_bound_to_exact_window_recipe_and_supervision_policy(tmp_path):
    dataset, model, identity, components, build, split, _ = inputs(tmp_path)
    try:
        result = compose_control_sidecar_corpus(
            dataset, components, model=model, identity=identity, output_dir=tmp_path / "corpus"
        )
        reader = ControlSidecar(
            result["control_sidecar_path"],
            replace(identity, control_sidecar_sha256=result["control_sidecar_sha256"]),
            model.hierarchy_config,
        )
        with pytest.raises(ValueError, match="supervision contract"):
            reader.validate_dataset(dataset, model, supervision_policy="DISTAR_COARSE_SKIP_V1")
        changed = BCWindowDataset(build, split, unroll_length=4, burn_in_length=0)
        try:
            with pytest.raises(ValueError, match="differs from sidecar binding"):
                reader.validate_dataset(changed, model)
        finally:
            changed.close()
    finally:
        dataset.close()


def test_failed_two_file_publication_preserves_staging_without_ready_final(tmp_path, monkeypatch):
    dataset, model, identity, components, *_ = inputs(tmp_path)
    import poorstar.evaluation.sidecar_corpus as module

    monkeypatch.setattr(
        module.os,
        "rename",
        lambda *args: (_ for _ in ()).throw(OSError("declared publication failure")),
    )
    try:
        with pytest.raises(OSError, match="publication failure"):
            compose_control_sidecar_corpus(
                dataset, components, model=model, identity=identity, output_dir=tmp_path / "corpus"
            )
        assert not (tmp_path / "corpus").exists()
        staging = list(tmp_path.glob(".corpus.attempt-*"))
        assert len(staging) == 1
        assert (staging[0] / "provenance.json").exists() and (staging[0] / "failure.json").exists()
    finally:
        dataset.close()


def test_real_composer_cli_and_bc_dry_run_validate_all_windows_before_adam(tmp_path):
    dataset, model, identity, components, build, split, plan = inputs(tmp_path)
    tree = Path(__file__).resolve().parents[1]
    cli = tree / "poorstar/cli/data/annotate.py"
    train = tree / "poorstar/cli/training/train.py"
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            dict(
                files=[
                    dict(path=str(p.relative_to(tree)), sha256=file_hash(p), bytes=p.stat().st_size)
                    for p in (cli, train)
                ],
                scope="ENGINEERING_CLI_FIXTURE_NOT_RELEASE_QUALIFICATION",
            )
        )
    )
    identity = replace(identity, source_sha256=file_hash(source))

    def save(name, value):
        p = tmp_path / name
        p.write_text(json.dumps(value))
        return p

    ident = save("identity.json", asdict(identity))
    cfg = save(
        "model.json", dict(bc=asdict(model.config), hierarchy=asdict(model.hierarchy_config))
    )
    plan_path = save("plan.json", plan)
    mapping = dict(schema=COMPONENT_SCHEMA, dataset_sha256=dataset.sha256, windows=components)
    mapping["content_digest"] = digest(mapping)
    mapping_path = save("components.json", mapping)
    env = os.environ | {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}
    base = [
        "--identity",
        str(ident),
        "--model-config",
        str(cfg),
        "--plan",
        str(plan_path),
        "--source-manifest",
        str(source),
        "--build-report",
        str(build),
        "--split-manifest",
        str(split),
        "--device",
        "cpu",
    ]
    try:
        result = subprocess.run(
            [sys.executable, str(cli)]
            + base
            + ["--components", str(mapping_path), "--output-dir", str(tmp_path / "composed")],
            env=env,
            cwd=tree,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        composed = json.loads(result.stdout.splitlines()[-1])
        assert composed["windows"] == 2
        assert composed["storage"] == "BOUNDED_WINDOW_INDEX_V3"
        ident.write_text(
            json.dumps(
                asdict(replace(identity, control_sidecar_sha256=composed["control_sidecar_sha256"]))
            )
        )
        dry = subprocess.run(
            [sys.executable, str(train), "--stage", "BC"]
            + base
            + [
                "--control-sidecar",
                composed["control_sidecar_path"],
                "--output-dir",
                str(tmp_path / "dry"),
                "--dry-run",
            ],
            env=env,
            cwd=tree,
            capture_output=True,
            text=True,
        )
        assert dry.returncode == 0, dry.stdout + dry.stderr
        proof = json.loads(dry.stdout.splitlines()[-1])
        assert not proof["optimizer_created"]
        assert (
            proof["sample_contract"]["windows"] == 2
            and proof["sample_contract"]["original_train_rows"] == 4
        )
        assert not (tmp_path / "dry").exists()
    finally:
        dataset.close()


def test_indexed_and_merged_windows_have_exact_targets_teacher_loss_and_gradient(tmp_path):
    dataset, model, identity, components, *_ = inputs(tmp_path, unroll=1, burn=2, episodes=2)
    try:
        merged = compose_control_sidecar_corpus(
            dataset, components, model=model, identity=identity, output_dir=tmp_path / "merged"
        )
        indexed = compose_indexed_control_sidecar_corpus(
            dataset, components, model=model, identity=identity, output_dir=tmp_path / "indexed"
        )
        reader = lambda r: ControlSidecar(
            r["control_sidecar_path"],
            replace(identity, control_sidecar_sha256=r["control_sidecar_sha256"]),
            model.hierarchy_config,
        )
        resident = reader(merged)
        bounded = reader(indexed)
        assert not any(
            k in bounded.document
            for k in ("records", "annotation_dependencies", "initial_control_states")
        )
        assert all("row_order" not in w for w in bounded.document["corpus_coverage"]["windows"])
        coverage = bounded.document["corpus_coverage"]
        assert coverage["maximum_retained_overlap_rows"] <= 3
        assert coverage["maximum_retained_control_snapshots"] <= 3
        assert bounded.validate_dataset(dataset, model) == resident.validate_dataset(dataset, model)
        for index in range(len(dataset.windows)):
            rows = dataset.window(index)
            a = resident.sequence(rows, window_index=index)
            b = bounded.sequence(rows, window_index=index)
            assert a.initial_control_state.to_dict() == b.initial_control_state.to_dict()
            window_document = bounded._window_component(index)
            for row in rows:
                key = ControlSidecar.row_id(row)
                assert resident.document["records"][key] == window_document["records"][key]
                assert (
                    resident.document["annotation_dependencies"][key]
                    == window_document["annotation_dependencies"][key]
                )
            assert [
                (
                    r.decision.valid,
                    r.decision.loss,
                    r.decision.burn_in,
                    r.scheduler_control,
                    r.body_control,
                    r.supervision_id,
                )
                for r in a.records
            ] == [
                (
                    r.decision.valid,
                    r.decision.loss,
                    r.decision.burn_in,
                    r.scheduler_control,
                    r.body_control,
                    r.supervision_id,
                )
                for r in b.records
            ]
            model.zero_grad(set_to_none=True)
            first, _ = audit_hierarchy_bc_sequence(model, a, dataset.profile, collect_metrics=False)
            first.loss.backward()
            gradients = {
                n: None if p.grad is None else p.grad.detach().clone()
                for n, p in model.named_parameters()
            }
            model.zero_grad(set_to_none=True)
            second, _ = audit_hierarchy_bc_sequence(
                model, b, dataset.profile, collect_metrics=False
            )
            second.loss.backward()
            assert (
                torch.equal(first.loss, second.loss) and first.denominators == second.denominators
            )
            assert all(
                (gradients[n] is None and p.grad is None)
                or (
                    gradients[n] is not None
                    and p.grad is not None
                    and torch.equal(gradients[n], p.grad)
                )
                for n, p in model.named_parameters()
            )
        assert bounded.maximum_cached_windows == 2 and len(bounded._component_cache) == 2
    finally:
        dataset.close()


def test_indexed_lru_reloads_evicted_component_and_rejects_byte_or_loaded_drift(tmp_path):
    dataset, model, identity, components, *_ = inputs(tmp_path, unroll=1, burn=2)
    try:
        result = compose_indexed_control_sidecar_corpus(
            dataset, components, model=model, identity=identity, output_dir=tmp_path / "indexed"
        )
        reader = ControlSidecar(
            result["control_sidecar_path"],
            replace(identity, control_sidecar_sha256=result["control_sidecar_sha256"]),
            model.hierarchy_config,
        )
        for i in range(3):
            reader.sequence(dataset.window(i), window_index=i)
        assert tuple(reader._component_cache) == (1, 2)
        first = next(iter(reader._component_cache[2]["records"]))
        reader._component_cache[2]["records"][first]["supervision_id"] = "IN_MEMORY_DRIFT"
        with pytest.raises(ValueError, match="changed after initialization"):
            reader.sequence(dataset.window(2), window_index=2)
        path = Path(components[0]["path"])
        path.write_text(path.read_text() + " ")
        with pytest.raises(ValueError, match="component bytes changed"):
            reader.sequence(dataset.window(0), window_index=0)
    finally:
        dataset.close()


def test_indexed_formal_actor_trace_and_changed_components_are_rejected_streaming(tmp_path):
    dataset, model, identity, components, *_ = inputs(tmp_path)
    try:
        result = compose_indexed_control_sidecar_corpus(
            dataset, components, model=model, identity=identity, output_dir=tmp_path / "indexed"
        )
        frozen = replace(identity, control_sidecar_sha256=result["control_sidecar_sha256"])
        with pytest.raises(ValueError, match="model samples"):
            ControlSidecar(
                result["control_sidecar_path"],
                replace(frozen, purpose="FORMAL"),
                model.hierarchy_config,
            )
        path = Path(components[1]["path"])
        path.write_text(path.read_text() + " ")
        with pytest.raises(ValueError, match="component bytes changed"):
            ControlSidecar(result["control_sidecar_path"], frozen, model.hierarchy_config)
    finally:
        dataset.close()

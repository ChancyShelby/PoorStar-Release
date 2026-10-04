"""Evaluation: teachers."""

import json
from dataclasses import asdict, fields
from pathlib import Path

from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import digest, is_sha


def _dependency(items, frame, phase):
    if not items:
        raise ValueError("each annotated control phase requires visible evidence")
    for item in items:
        expected = {"loop", "player_id", "observation_sha256", "evidence_sha256", "availability"}
        if set(item) != expected or item["availability"] != "ACTOR_VISIBLE_CURRENT_OR_PAST":
            raise ValueError(
                "future/privileged/raw-action hindsight is not a causal control dependency"
            )
        if (
            type(item["loop"]) is not int
            or item["loop"] < 0
            or item["loop"] > frame.key.loop
            or item["player_id"] != frame.key.player_id
            or not is_sha(item["observation_sha256"])
            or not is_sha(item["evidence_sha256"])
        ):
            raise ValueError("control evidence time/player/hash mismatch: " + phase)
    if not any(
        i["loop"] == frame.key.loop and i["observation_sha256"] == frame.key.observation_sha256
        for i in items
    ):
        raise ValueError("annotation must bind its exact current phase observation")


def build_control_sidecar(
    rows,
    annotations,
    *,
    initial_control_state,
    model,
    profile,
    identity,
    evidence_path,
    output_path,
    annotation_policy_sha256,
):
    """Validate actual teacher distributions before atomic publication; no Adam.

    Annotation payload: row_id, label, scheduler_evidence, body_evidence.
    Labels name weak_visible_task_labels or declared_external_intervention. An
    actor trace can be an engineering distillation target but never expert truth.
    The writer never overwrites an existing artifact or mutates replay bytes.
    """
    import torch

    from poorstar.hierarchy.bc import HierarchyBCRecord, HierarchyBCSequence, hierarchy_bc_objective
    from poorstar.hierarchy.state import ExecutionFeedback, ExternalOverride
    from poorstar.training.config import ControlSidecar, control_label_from_dict

    rows = tuple(rows)
    if not rows or len(annotations) != len(rows) or not is_sha(annotation_policy_sha256):
        raise ValueError("complete explicit bounded row annotation and frozen policy required")
    if not Path(evidence_path).is_file() or Path(output_path).exists():
        raise ValueError("existing evidence and fresh external sidecar output required")
    evidence = json.loads(Path(evidence_path).read_text())
    if (
        evidence.get("schema") != "poorstar_v3_control_annotation_evidence_v1"
        or evidence.get("annotation_policy_sha256") != annotation_policy_sha256
        or evidence.get("human_intent_truth") is not False
        or evidence.get("future_or_privileged_inputs") is not False
        or evidence.get("annotation_sha256") != digest(annotations)
        or evidence.get("row_order") != [a["row_id"] for a in annotations]
    ):
        raise ValueError("typed annotation provenance does not match actual annotation inputs")
    labels = {}
    records = []
    allowed = {f.name for f in fields(HierarchyBCRecord)} - {"decision"}
    for row, annotation in zip(rows, annotations):
        if set(annotation) != {"row_id", "label", "scheduler_evidence", "body_evidence"}:
            raise ValueError("exact annotation compiler schema required")
        row_id = ControlSidecar.row_id(row)
        if annotation["row_id"] != row_id or row_id in labels:
            raise ValueError("annotation/replay row mismatch or duplicate")
        _dependency(annotation["scheduler_evidence"], row.scheduler, "scheduler")
        if row.body is None:
            if annotation["body_evidence"]:
                raise ValueError("WAIT cannot invent body evidence")
        else:
            _dependency(annotation["body_evidence"], row.body, "body")
        label = dict(annotation["label"])
        if set(label) - allowed or label.get("supervision_kind") not in (
            "weak_visible_task_labels",
            "declared_external_intervention",
            "actor_control_trace",
        ):
            raise ValueError("explicit declared supervision kind required")
        if label["supervision_kind"] == "actor_control_trace" and identity.purpose != "ENGINEERING":
            raise ValueError("sampled actor controls are not formal human expert Intent labels")
        for event in label.get("feedback", []):
            if event["loop"] > row.scheduler.key.loop:
                raise ValueError("future outcome cannot enter BC current feedback")
        labels[row_id] = label
        records.append(HierarchyBCRecord(row, **control_label_from_dict(label)))
    sequence = HierarchyBCSequence(tuple(records), initial_control_state)
    with torch.no_grad():
        objective = hierarchy_bc_objective(model, [sequence], profile, collect_metrics=False)
    states = (
        {}
        if initial_control_state is None
        else {ControlSidecar.row_id(rows[0]): initial_control_state.to_dict()}
    )
    if evidence.get("initial_control_states_sha256") != digest(states):
        raise ValueError("initial control snapshots require exact causal source evidence")
    document = dict(
        schema="poorstar_v3_hierarchy_bc_sidecar_v2",
        binding=dict(
            dataset_sha256=identity.dataset_sha256,
            split_sha256=identity.split_sha256,
            scope_sha256=identity.scope_sha256,
            hierarchy_config_sha256=model.hierarchy_config.sha256,
        ),
        provenance_evidence_path=str(Path(evidence_path).resolve()),
        provenance_evidence_sha256=file_hash(evidence_path),
        records=labels,
        initial_control_states=states,
        annotation_policy_sha256=annotation_policy_sha256,
        annotation_sha256=digest(annotations),
        human_intent_truth=False,
        annotation_dependencies={
            a["row_id"]: dict(scheduler=a["scheduler_evidence"], body=a["body_evidence"])
            for a in annotations
        },
        compiler_validation=dict(
            rows=len(rows),
            denominators=objective.denominators,
            optimizer_steps=0,
            live_frames=0,
            formal_corpus_qualified=False,
        ),
    )
    document["content_digest"] = digest(document)
    verify_annotation_document(document, evidence, rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a racing annotation writer from replacing truth.
    with path.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(
        path=str(path.resolve()),
        sha256=file_hash(path),
        content_digest=document["content_digest"],
        rows=len(rows),
        optimizer_steps=0,
        human_intent_truth=False,
        formal_corpus_qualified=False,
    )


def verify_annotation_document(document, evidence, rows=None):
    """Reader rejects changed provenance and phase dependencies before training."""
    if (
        document.get("human_intent_truth") is not False
        or evidence.get("human_intent_truth") is not False
        or evidence.get("future_or_privileged_inputs") is not False
        or evidence.get("annotation_policy_sha256") != document.get("annotation_policy_sha256")
        or evidence.get("annotation_sha256") != document.get("annotation_sha256")
        or evidence.get("initial_control_states_sha256")
        != digest(document.get("initial_control_states", {}))
    ):
        raise ValueError("sidecar annotation policy/source truth drift")
    if set(document.get("annotation_dependencies", {})) != set(document["records"]):
        raise ValueError("sidecar annotation coverage drift")
    # Preserve original chronological row order in the annotation digest; sorting
    # a mapping must not quietly detach provenance from the actual labels.
    row_order = evidence.get("row_order")
    if (
        not isinstance(row_order, list)
        or len(set(row_order)) != len(row_order)
        or set(row_order) != set(document["records"])
    ):
        raise ValueError("exact original annotation row order required")
    reconstructed = [
        dict(
            row_id=row_id,
            label=document["records"][row_id],
            scheduler_evidence=document["annotation_dependencies"][row_id]["scheduler"],
            body_evidence=document["annotation_dependencies"][row_id]["body"],
        )
        for row_id in row_order
    ]
    if digest(reconstructed) != document["annotation_sha256"]:
        raise ValueError("sidecar records/dependencies no longer match annotation evidence")
    if rows is not None:
        from poorstar.training.config import ControlSidecar

        for row in rows:
            dependencies = document["annotation_dependencies"][ControlSidecar.row_id(row)]
            _dependency(dependencies["scheduler"], row.scheduler, "scheduler")
            if row.body is not None:
                _dependency(dependencies["body"], row.body, "body")
            elif dependencies["body"]:
                raise ValueError("WAIT sidecar contains body provenance")

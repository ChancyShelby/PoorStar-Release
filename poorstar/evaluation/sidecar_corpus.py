"""Evaluation: sidecar corpus."""

import json
import os
import tempfile
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from poorstar.evaluation.teachers import verify_annotation_document
from poorstar.training.checkpoint import capture_rng, file_hash, restore_rng
from poorstar.training.contracts import digest, is_sha

COVERAGE_SCHEMA = "poorstar_v3_control_sidecar_corpus_coverage_v1"
COMPONENT_SCHEMA = "poorstar_v3_control_sidecar_components_v1"
INDEX_SCHEMA = "poorstar_v3_hierarchy_bc_sidecar_index_v3"
INDEX_EVIDENCE_SCHEMA = "poorstar_v3_control_sidecar_index_evidence_v1"


def _read(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None and (
        not is_sha(expected_sha256) or file_hash(path) != expected_sha256
    ):
        raise ValueError("sidecar component bytes changed: " + str(path))
    value = json.loads(path.read_text())
    check = dict(value)
    claimed = check.pop("content_digest", None)
    if digest(check) != claimed:
        raise ValueError("sidecar component content digest changed: " + str(path))
    return value


def row_descriptor(row):
    from poorstar.training.config import ControlSidecar

    return dict(
        row_id=ControlSidecar.row_id(row),
        scheduler=asdict(row.scheduler.key),
        body=None if row.body is None else asdict(row.body.key),
        decision=row.decision,
        delay=row.delay,
        valid=row.valid,
        burn_in=row.burn_in,
        loss=row.loss,
    )


def window_descriptor(dataset, index, rows):
    """Actual original clock/phase identities and the window's loss masks."""
    records = [row_descriptor(row) for row in rows]
    return dict(
        window_index=index,
        metadata=list(dataset.windows[index]),
        rows=len(records),
        row_order=[r["row_id"] for r in records],
        row_digest=digest(records),
        mask_digest=digest([[r["valid"], r["burn_in"], r["loss"]] for r in records]),
    )


def _component(entry, binding):
    if set(entry) != {"window_index", "path", "sha256"} or type(entry["window_index"]) is not int:
        raise ValueError("exact window component path/SHA schema required")
    document = _read(entry["path"], entry["sha256"])
    if (
        document.get("schema") != "poorstar_v3_hierarchy_bc_sidecar_v2"
        or document.get("binding") != binding
    ):
        raise ValueError(
            "every component must be typed v2 with the same frozen data/hierarchy binding"
        )
    if document.get("corpus_coverage") is not None:
        raise ValueError("a window component cannot recursively claim corpus coverage")
    evidence_path = Path(document["provenance_evidence_path"])
    if file_hash(evidence_path) != document["provenance_evidence_sha256"]:
        raise ValueError("window annotation provenance bytes changed")
    evidence = json.loads(evidence_path.read_text())
    verify_annotation_document(document, evidence)
    return (
        document,
        evidence,
        dict(
            window_index=entry["window_index"],
            path=str(Path(entry["path"]).resolve()),
            sha256=entry["sha256"],
            content_digest=document["content_digest"],
            provenance_evidence_path=str(evidence_path.resolve()),
            provenance_evidence_sha256=document["provenance_evidence_sha256"],
        ),
    )


def _initial(document, rows, config):
    from poorstar.hierarchy.state import HierarchyState
    from poorstar.training.config import ControlSidecar

    first = rows[0]
    key = first.scheduler.key
    value = document.get("initial_control_states", {}).get(ControlSidecar.row_id(first))
    if value is None:
        raise ValueError(
            "every corpus window requires an explicit snapshot before its first burn-in row"
        )
    state = HierarchyState.from_dict(value, config)
    if state.episode != key.episode or (
        state.last_loop is not None and state.last_loop >= key.loop
    ):
        raise ValueError(
            "initial snapshot must be same episode and strictly before the first scheduler loop"
        )
    if key.loop == 0 and digest(state.to_dict()) != digest(HierarchyState(key.episode).to_dict()):
        raise ValueError("episode-zero snapshot cannot contain previous control history")
    if key.loop > 0 and state.last_loop is None:
        raise ValueError("midgame snapshot cannot silently reset control history")
    return state


class _OverlapLedger:
    """Exact duplicate annotation checks, retaining only future burn-in rows."""

    def __init__(self):
        self.rows = {}
        self.next_ordinal = 0

    def consume(self, metadata, row_order, document):
        start, active, end = metadata
        if (
            any(type(v) is not int for v in metadata)
            or not 0 <= start <= active < end
            or start > self.next_ordinal
            or end <= self.next_ordinal
            or len(row_order) != end - start
        ):
            raise ValueError("indexed components must extend a continuous original TRAIN prefix")
        for offset, row_id in enumerate(row_order):
            ordinal = start + offset
            fingerprint = digest(
                [row_id, document["records"][row_id], document["annotation_dependencies"][row_id]]
            )
            if ordinal < self.next_ordinal:
                if self.rows.get(ordinal) != fingerprint:
                    raise ValueError("overlapping windows disagree on row labels/dependencies")
            elif ordinal != self.next_ordinal:
                raise ValueError("indexed annotation ordinal gap")
            else:
                self.rows[ordinal] = fingerprint
                self.next_ordinal += 1

    def discard_before(self, ordinal):
        self.rows = {k: v for k, v in self.rows.items() if k >= ordinal}


def verify_index_sidecar_lineage(document, evidence, *, purpose="ENGINEERING"):
    """Stream component identities once; never collect corpus labels or states."""
    coverage = document.get("corpus_coverage")
    check = dict(evidence)
    claimed = check.pop("content_digest", None)
    if (
        document.get("schema") != INDEX_SCHEMA
        or digest(check) != claimed
        or evidence.get("schema") != INDEX_EVIDENCE_SCHEMA
        or not isinstance(coverage, dict)
        or coverage.get("schema") != COVERAGE_SCHEMA
        or coverage.get("status") != "COMPLETE_TYPED_WINDOW_COVERAGE"
        or coverage.get("storage") != "BOUNDED_WINDOW_INDEX_V3"
        or coverage.get("dataset_sha256") != document["binding"]["dataset_sha256"]
        or evidence.get("corpus_coverage_sha256") != digest(coverage)
        or evidence.get("binding_sha256") != digest(document["binding"])
        or evidence.get("annotation_policy_sha256") != document["annotation_policy_sha256"]
        or evidence.get("human_intent_truth") is not False
        or evidence.get("formal_corpus_qualified") is not False
        or coverage.get("formal_corpus_qualified") is not False
        or coverage.get("human_intent_truth") is not False
        or any(
            k in document for k in ("records", "annotation_dependencies", "initial_control_states")
        )
    ):
        raise ValueError("complete bounded index/provenance binding required")
    windows = coverage.get("windows")
    if (
        not isinstance(windows, list)
        or not windows
        or coverage.get("window_count") != len(windows)
        or [w.get("window_index") for w in windows] != list(range(len(windows)))
    ):
        raise ValueError("indexed corpus needs every exact window index")
    overlap = _OverlapLedger()
    occurrences = 0
    previous_start = -1
    for index, window in enumerate(windows):
        component = window["component"]
        original, annotation_evidence, identity = _component(
            {k: component[k] for k in ("window_index", "path", "sha256")}, document["binding"]
        )
        if (
            identity != component
            or component["window_index"] != index
            or original["annotation_policy_sha256"] != document["annotation_policy_sha256"]
            or digest(annotation_evidence["row_order"]) != window["annotation_row_order_sha256"]
            or len(annotation_evidence["row_order"]) != window["rows"]
            or set(original["records"]) != set(annotation_evidence["row_order"])
            or "row_order" in window
            or window["metadata"][0] < previous_start
        ):
            raise ValueError("indexed component row order/policy/identity changed")
        first = annotation_evidence["row_order"][0]
        if (
            set(original["initial_control_states"]) != {first}
            or digest(original["initial_control_states"][first]) != window["initial_state_sha256"]
        ):
            raise ValueError("indexed initial causal snapshot changed")
        if purpose == "FORMAL" and any(
            label.get("supervision_kind") == "actor_control_trace"
            for label in original["records"].values()
        ):
            raise ValueError("model samples cannot become formal human expert Intent labels")
        overlap.consume(window["metadata"], annotation_evidence["row_order"], original)
        floor = (
            windows[index + 1]["metadata"][0] if index + 1 < len(windows) else window["metadata"][2]
        )
        overlap.discard_before(floor)
        previous_start = window["metadata"][0]
        occurrences += window["rows"]
    if (
        overlap.next_ordinal != coverage.get("original_train_rows")
        or coverage.get("unique_rows") != overlap.next_ordinal
        or coverage.get("window_row_occurrences") != occurrences
    ):
        raise ValueError("indexed annotation coverage is partial")


def _publish_index(staging, output_dir, binding, policy, coverage, denominators, windows):
    evidence = dict(
        schema=INDEX_EVIDENCE_SCHEMA,
        binding_sha256=digest(binding),
        annotation_policy_sha256=policy,
        corpus_coverage_sha256=digest(coverage),
        human_intent_truth=False,
        future_or_privileged_inputs=False,
        formal_corpus_qualified=False,
    )
    evidence["content_digest"] = digest(evidence)
    evidence_file = staging / "provenance.json"
    evidence_file.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    document = dict(
        schema=INDEX_SCHEMA,
        binding=binding,
        provenance_evidence_path=str(output_dir / "provenance.json"),
        provenance_evidence_sha256=file_hash(evidence_file),
        annotation_policy_sha256=policy,
        human_intent_truth=False,
        corpus_coverage=coverage,
        compiler_validation=dict(
            windows=len(windows),
            rows=coverage["original_train_rows"],
            denominators=dict(denominators),
            optimizer_steps=0,
            live_frames=0,
            formal_corpus_qualified=False,
        ),
    )
    document["content_digest"] = digest(document)
    verify_index_sidecar_lineage(document, evidence)
    sidecar_file = staging / "control_sidecar.json"
    sidecar_file.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    report = dict(
        schema="poorstar_v3_control_sidecar_corpus_report_v1",
        status="COMPLETE_TYPED_WINDOW_COVERAGE",
        storage="BOUNDED_WINDOW_INDEX_V3",
        dataset_sha256=coverage["dataset_sha256"],
        windows=len(windows),
        unique_rows=coverage["original_train_rows"],
        control_sidecar_path=str(output_dir / "control_sidecar.json"),
        control_sidecar_sha256=file_hash(sidecar_file),
        provenance_evidence_sha256=file_hash(evidence_file),
        optimizer_steps=0,
        live_frames=0,
        human_intent_truth=False,
        formal_corpus_qualified=False,
    )
    report["content_digest"] = digest(report)
    (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for path in staging.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    os.rename(staging, output_dir)
    return report


def compose_indexed_control_sidecar_corpus(*args, **kwargs):
    """Bounded default CLI storage; the legacy resident v2 merge is explicit."""
    if "indexed" in kwargs:
        raise TypeError("indexed storage is fixed by this API")
    return compose_control_sidecar_corpus(*args, **kwargs, indexed=True)


class _Continuity:
    """Retain states only at declared future window starts, not every row."""

    def __init__(self, dataset, config):
        self.starts = {start for start, active, end in dataset.windows}
        self.config = config
        self.next_ordinal = 0
        self.expected = {}
        self.current_state = None
        self.previous_row = None
        self.episode = None
        self.maximum_retained_states = 0

    def before(self, dataset, index, rows, initial):
        from poorstar.hierarchy.state import HierarchyState

        start, active, end = dataset.windows[index]
        if start == self.next_ordinal:
            k = rows[0].scheduler.key
            identity = (rows[0].replay_sha256, k.player_id, k.episode)
            if identity != self.episode:
                if k.loop != 0:
                    raise ValueError("new complete TRAIN episode must start at zero")
                expected = HierarchyState(k.episode).to_dict()
                predecessor = None
            else:
                expected = self.current_state
                predecessor = self.previous_row
            self.expected[start] = (expected, predecessor)
        if start not in self.expected:
            raise ValueError("window starts outside the already validated causal TRAIN prefix")
        expected, predecessor = self.expected[start]
        if digest(initial.to_dict()) != digest(expected):
            raise ValueError("window initial snapshot differs from its actual prior teacher phase")
        return predecessor

    def after(self, dataset, index, rows, audit):
        from poorstar.training.config import ControlSidecar

        start, active, end = dataset.windows[index]
        phases = {}
        for event in audit:
            phases.setdefault(
                (event.replay_sha256, event.episode, event.player_id, event.scheduler_loop), []
            ).append(event)
        for offset, row in enumerate(rows):
            ordinal = start + offset
            k = row.scheduler.key
            events = phases.get((row.replay_sha256, k.episode, k.player_id, k.loop), [])
            expected_phases = ["scheduler", "body"] if row.decision == "ACT" else ["scheduler"]
            if [e.phase for e in events] != expected_phases:
                raise ValueError("complete scheduler/body teacher-state audit required")
            if events[0].phase_loop != k.loop or events[-1].phase_loop != (
                row.body.key.loop if row.body is not None else k.loop
            ):
                raise ValueError("teacher phase clocks differ from original TRAIN frames")
            if ordinal < self.next_ordinal:
                if ordinal in self.expected and digest(events[0].prior_state_dict) != digest(
                    self.expected[ordinal][0]
                ):
                    raise ValueError(
                        "overlapping burn-in controls changed an existing causal snapshot"
                    )
                continue
            if ordinal != self.next_ordinal:
                raise ValueError("window coverage has an original TRAIN ordinal gap")
            identity = (row.replay_sha256, k.player_id, k.episode)
            if identity != self.episode:
                from poorstar.hierarchy.state import HierarchyState

                expected = HierarchyState(k.episode).to_dict()
                predecessor = None
                if k.loop != 0:
                    raise ValueError("new original episode has a nonzero first loop")
            else:
                expected = self.current_state
                predecessor = self.previous_row
            if digest(events[0].prior_state_dict) != digest(expected):
                raise ValueError(
                    "teacher state chain differs from the previous original row ending phase"
                )
            if ordinal in self.starts:
                self.expected[ordinal] = (events[0].prior_state_dict, predecessor)
            self.current_state = events[-1].post_state_dict
            self.previous_row = dict(
                row_id=ControlSidecar.row_id(row),
                replay_sha256=row.replay_sha256,
                episode=k.episode,
                player_id=k.player_id,
                phase=events[-1].phase,
                loop=events[-1].phase_loop,
                observation_sha256=(
                    row.body if row.body is not None else row.scheduler
                ).key.observation_sha256,
                post_state_sha256=digest(self.current_state),
            )
            self.episode = identity
            self.next_ordinal += 1
        if self.next_ordinal != end:
            raise ValueError("window does not extend the complete original TRAIN prefix")
        # Starts are nondecreasing; expired teacher snapshots cannot be needed
        # by a later burn-in window. Retain only the bounded future overlap.
        floor = dataset.windows[index + 1][0] if index + 1 < len(dataset.windows) else end
        self.expected = {k: v for k, v in self.expected.items() if k >= floor}
        self.maximum_retained_states = max(self.maximum_retained_states, len(self.expected))


def verify_corpus_sidecar_lineage(document, evidence):
    """One initialization pass over frozen component identities and annotations."""
    coverage = document.get("corpus_coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("schema") != COVERAGE_SCHEMA
        or coverage.get("status") != "COMPLETE_TYPED_WINDOW_COVERAGE"
        or evidence.get("corpus_coverage_sha256") != digest(coverage)
        or coverage.get("dataset_sha256") != document["binding"]["dataset_sha256"]
        or coverage.get("formal_corpus_qualified") is not False
        or coverage.get("human_intent_truth") is not False
    ):
        raise ValueError("complete typed corpus coverage/provenance binding required")
    windows = coverage.get("windows")
    components = coverage.get("components")
    if (
        not isinstance(windows, list)
        or not windows
        or not isinstance(components, list)
        or len(windows) != len(components)
        or coverage.get("window_count") != len(windows)
        or [w.get("window_index") for w in windows] != list(range(len(windows)))
        or [c.get("window_index") for c in components] != list(range(len(windows)))
    ):
        raise ValueError("corpus coverage requires every exact window index")
    represented = set()
    state_keys = set()
    for component, window in zip(components, windows):
        entry = {key: component[key] for key in ("window_index", "path", "sha256")}
        original, original_evidence, identity = _component(entry, document["binding"])
        if (
            identity != component
            or original["annotation_policy_sha256"] != document["annotation_policy_sha256"]
        ):
            raise ValueError("window component identity/policy lineage changed")
        if original_evidence["row_order"] != window["row_order"] or set(original["records"]) != set(
            window["row_order"]
        ):
            raise ValueError("window row order detached from its original annotation component")
        first = window["row_order"][0]
        state_keys.add(first)
        if (
            set(original["initial_control_states"]) != {first}
            or digest(original["initial_control_states"][first]) != window["initial_state_sha256"]
            or digest(document["initial_control_states"].get(first))
            != window["initial_state_sha256"]
        ):
            raise ValueError("window initial state detached from component/merged state lineage")
        for row_id in window["row_order"]:
            if original["records"][row_id] != document["records"].get(row_id) or original[
                "annotation_dependencies"
            ][row_id] != document["annotation_dependencies"].get(row_id):
                raise ValueError(
                    "duplicate or merged row labels/dependencies disagree with component provenance"
                )
            represented.add(row_id)
    if (
        represented != set(document["records"])
        or state_keys != set(document["initial_control_states"])
        or coverage.get("unique_rows") != len(represented)
        or coverage.get("original_train_rows") != len(represented)
    ):
        raise ValueError("corpus rows/snapshots are partial or contain unregistered labels")


def validate_dataset_sidecar(
    reader,
    dataset,
    model,
    *,
    head_weights=None,
    control_weights=None,
    supervision_policy="STRICT_WHOLE_BODY_V1",
):
    """Stream ALL actual windows before Adam, including masks and causal states."""
    import torch

    from poorstar.hierarchy.bc import audit_hierarchy_bc_sequence

    if (
        dataset.allowed_split != "train"
        or dataset.sha256 != reader.document["binding"]["dataset_sha256"]
        or dataset.split["sha256"] != reader.document["binding"]["split_sha256"]
        or model.hierarchy_config.sha256 != reader.config.sha256
    ):
        raise ValueError("actual TRAIN dataset/model differs from sidecar binding")
    coverage = reader.document.get("corpus_coverage")
    if coverage is not None and (
        coverage["window_count"] != len(dataset.windows)
        or coverage["original_train_rows"] != dataset.rows
        or coverage["window_metadata_sha256"] != digest(dataset.windows)
    ):
        raise ValueError("sidecar corpus was composed for another exact window recipe")
    if coverage is not None:
        from poorstar.training.config import model_config_sha256

        if (
            coverage["validation_model_config_sha256"]
            != model_config_sha256(model.config, model.hierarchy_config)
            or coverage["validation_supervision_policy"] != supervision_policy
        ):
            raise ValueError("teacher validation architecture/supervision contract changed")
    continuity = _Continuity(dataset, reader.config)
    denominators = Counter()
    occurrences = 0
    rng = capture_rng()
    training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for index in range(len(dataset.windows)):
                rows = dataset.window(index)
                sequence = reader.sequence(rows, window_index=index)
                initial = sequence.initial_control_state
                if initial is None:
                    raise ValueError("complete window needs its explicit causal snapshot")
                predecessor = continuity.before(dataset, index, rows, initial)
                result, audit = audit_hierarchy_bc_sequence(
                    model,
                    sequence,
                    dataset.profile,
                    head_weights=head_weights,
                    control_weights=control_weights,
                    collect_metrics=False,
                    supervision_policy=supervision_policy,
                )
                continuity.after(dataset, index, rows, audit)
                if (
                    coverage is not None
                    and coverage["windows"][index]["initial_state_predecessor"] != predecessor
                ):
                    raise ValueError(
                        "snapshot predecessor differs from actual original teacher phase"
                    )
                denominators.update(result.denominators)
                occurrences += len(rows)
    finally:
        model.train(training)
        restore_rng(rng)
    if continuity.next_ordinal != dataset.rows:
        raise ValueError("not every original TRAIN row was causally validated")
    return dict(
        status="PASS",
        scope="ALL_ACTUAL_TRAIN_WINDOWS_TEACHER_CONTRACT",
        windows=len(dataset.windows),
        original_train_rows=dataset.rows,
        window_row_occurrences=occurrences,
        denominators=dict(denominators),
        corpus_coverage_bound=coverage is not None,
        optimizer_steps=0,
        human_intent_truth=False,
        formal_corpus_qualified=False,
    )


def compose_control_sidecar_corpus(
    dataset,
    components,
    *,
    model,
    identity,
    output_dir,
    head_weights=None,
    control_weights=None,
    supervision_policy="STRICT_WHOLE_BODY_V1",
    indexed=False,
):
    """Publish evidence+sidecar as one fresh directory; failed staging is kept."""
    import torch

    from poorstar.hierarchy.bc import (
        HierarchyBCRecord,
        HierarchyBCSequence,
        audit_hierarchy_bc_sequence,
    )
    from poorstar.training.config import (
        ControlSidecar,
        control_label_from_dict,
        model_config_sha256,
    )

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if (
        dataset.allowed_split != "train"
        or dataset.sha256 != identity.dataset_sha256
        or dataset.split["sha256"] != identity.split_sha256
        or dataset.profile.sha256 != identity.profile_sha256
        or model_config_sha256(model.config, model.hierarchy_config) != identity.model_config_sha256
        or dataset.maximum_delay != model.config.maximum_delay
    ):
        raise ValueError("complete frozen TRAIN data/profile/model/window identity required")
    components = tuple(components)
    if [c.get("window_index") for c in components] != list(range(len(dataset.windows))):
        raise ValueError("one frozen component for every exact window index 0..N-1 required")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="." + output_dir.name + ".attempt-", dir=output_dir.parent)
    )
    rng = capture_rng()
    training = model.training
    binding = dict(
        dataset_sha256=identity.dataset_sha256,
        split_sha256=identity.split_sha256,
        scope_sha256=identity.scope_sha256,
        hierarchy_config_sha256=model.hierarchy_config.sha256,
    )
    labels = {}
    dependencies = {}
    states = {}
    windows = []
    lineage = []
    policy = None
    continuity = _Continuity(dataset, model.hierarchy_config)
    denominators = Counter()
    occurrences = 0
    overlap = _OverlapLedger()
    maximum_retained_rows = 0
    try:
        model.eval()
        with torch.no_grad():
            for index, component in enumerate(components):
                document, evidence, source = _component(component, binding)
                rows = dataset.window(index)
                description = window_descriptor(dataset, index, rows)
                if evidence["row_order"] != description["row_order"] or set(
                    document["records"]
                ) != set(description["row_order"]):
                    raise ValueError("component does not label exactly its actual TRAIN window")
                if policy is None:
                    policy = document["annotation_policy_sha256"]
                elif policy != document["annotation_policy_sha256"]:
                    raise ValueError("annotation policy differs between corpus windows")
                verify_annotation_document(document, evidence, rows)
                first = description["row_order"][0]
                if set(document.get("initial_control_states", {})) != {first}:
                    raise ValueError(
                        "each component must declare only its actual first burn-in row snapshot"
                    )
                initial = _initial(document, rows, model.hierarchy_config)
                predecessor = continuity.before(dataset, index, rows, initial)
                for row_id in description["row_order"]:
                    label = document["records"][row_id]
                    dependency = document["annotation_dependencies"][row_id]
                    if (
                        identity.purpose == "FORMAL"
                        and label.get("supervision_kind") == "actor_control_trace"
                    ):
                        raise ValueError(
                            "sampled actor controls are engineering targets, not formal expert truth"
                        )
                    if (
                        not indexed
                        and row_id in labels
                        and (labels[row_id] != label or dependencies[row_id] != dependency)
                    ):
                        raise ValueError("overlapping windows disagree on row labels/dependencies")
                    if not indexed:
                        labels[row_id] = label
                        dependencies[row_id] = dependency
                overlap.consume(dataset.windows[index], evidence["row_order"], document)
                maximum_retained_rows = max(maximum_retained_rows, len(overlap.rows))
                sequence = HierarchyBCSequence(
                    tuple(
                        HierarchyBCRecord(
                            row,
                            **control_label_from_dict(
                                document["records"][ControlSidecar.row_id(row)]
                            ),
                        )
                        for row in rows
                    ),
                    initial,
                )
                result, audit = audit_hierarchy_bc_sequence(
                    model,
                    sequence,
                    dataset.profile,
                    head_weights=head_weights,
                    control_weights=control_weights,
                    collect_metrics=False,
                    supervision_policy=supervision_policy,
                )
                continuity.after(dataset, index, rows, audit)
                if first in states and digest(states[first]) != digest(initial.to_dict()):
                    raise ValueError("duplicate window starts disagree on their control snapshot")
                if not indexed:
                    states[first] = initial.to_dict()
                    lineage.append(source)
                description.update(
                    initial_state_sha256=digest(initial.to_dict()),
                    initial_state_predecessor=predecessor,
                    teacher_denominators=result.denominators,
                )
                if indexed:
                    description.pop("row_order")
                    description.update(
                        component=source, annotation_row_order_sha256=digest(evidence["row_order"])
                    )
                windows.append(description)
                denominators.update(result.denominators)
                occurrences += len(rows)
                floor = (
                    dataset.windows[index + 1][0]
                    if index + 1 < len(dataset.windows)
                    else dataset.windows[index][2]
                )
                overlap.discard_before(floor)
        if continuity.next_ordinal != dataset.rows or (not indexed and len(labels) != dataset.rows):
            raise ValueError("partial TRAIN prefix cannot become a complete corpus artifact")
        coverage = dict(
            schema=COVERAGE_SCHEMA,
            status="COMPLETE_TYPED_WINDOW_COVERAGE",
            dataset_sha256=dataset.sha256,
            window_metadata_sha256=digest(dataset.windows),
            window_count=len(windows),
            original_train_rows=dataset.rows,
            unique_rows=dataset.rows,
            window_row_occurrences=occurrences,
            windows=windows,
            storage="BOUNDED_WINDOW_INDEX_V3" if indexed else "RESIDENT_MERGED_V2",
            maximum_retained_overlap_rows=maximum_retained_rows,
            maximum_retained_control_snapshots=continuity.maximum_retained_states,
            validation_model_config_sha256=identity.model_config_sha256,
            validation_supervision_policy=supervision_policy,
            optimizer_steps=0,
            human_intent_truth=False,
            formal_corpus_qualified=False,
        )
        if indexed:
            return _publish_index(
                staging, output_dir, binding, policy, coverage, denominators, windows
            )
        coverage["components"] = lineage
        annotations = [
            dict(
                row_id=row_id,
                label=label,
                scheduler_evidence=dependencies[row_id]["scheduler"],
                body_evidence=dependencies[row_id]["body"],
            )
            for row_id, label in labels.items()
        ]
        evidence = dict(
            schema="poorstar_v3_control_annotation_evidence_v1",
            annotation_policy_sha256=policy,
            human_intent_truth=False,
            future_or_privileged_inputs=False,
            annotation_sha256=digest(annotations),
            initial_control_states_sha256=digest(states),
            row_order=list(labels),
            corpus_coverage_sha256=digest(coverage),
        )
        evidence_file = staging / "provenance.json"
        evidence_file.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
        document = dict(
            schema="poorstar_v3_hierarchy_bc_sidecar_v2",
            binding=binding,
            provenance_evidence_path=str(output_dir / "provenance.json"),
            provenance_evidence_sha256=file_hash(evidence_file),
            records=labels,
            initial_control_states=states,
            annotation_policy_sha256=policy,
            annotation_sha256=digest(annotations),
            human_intent_truth=False,
            annotation_dependencies=dependencies,
            corpus_coverage=coverage,
            compiler_validation=dict(
                windows=len(windows),
                rows=len(labels),
                window_row_occurrences=occurrences,
                denominators=dict(denominators),
                optimizer_steps=0,
                live_frames=0,
                formal_corpus_qualified=False,
            ),
        )
        document["content_digest"] = digest(document)
        verify_annotation_document(document, evidence)
        verify_corpus_sidecar_lineage(document, evidence)
        sidecar_file = staging / "control_sidecar.json"
        sidecar_file.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        report = dict(
            schema="poorstar_v3_control_sidecar_corpus_report_v1",
            status="COMPLETE_TYPED_WINDOW_COVERAGE",
            dataset_sha256=dataset.sha256,
            control_sidecar_path=str(output_dir / "control_sidecar.json"),
            control_sidecar_sha256=file_hash(sidecar_file),
            provenance_evidence_sha256=file_hash(evidence_file),
            windows=len(windows),
            unique_rows=len(labels),
            optimizer_steps=0,
            live_frames=0,
            human_intent_truth=False,
            formal_corpus_qualified=False,
        )
        report["content_digest"] = digest(report)
        (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.rename(staging, output_dir)
        return report
    except BaseException as error:
        failure = dict(
            schema="poorstar_v3_control_sidecar_corpus_failure_v1",
            status="FAIL",
            final_artifact_published=False,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            validated_windows=len(windows),
            optimizer_steps=0,
            human_intent_truth=False,
            formal_corpus_qualified=False,
        )
        failure["content_digest"] = digest(failure)
        (staging / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        model.train(training)
        restore_rng(rng)

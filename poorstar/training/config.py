"""Training: config."""

import json
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

from poorstar.training.contracts import digest, is_sha


def read_model_config(path):
    from poorstar.hierarchy.state import HierarchyConfig
    from poorstar.models.policy import BCModelConfig

    document = json.loads(Path(path).read_text())
    if set(document) != {"bc", "hierarchy"}:
        raise ValueError("explicit lower/hierarchy model configuration required")
    low = dict(document["bc"])
    high = dict(document["hierarchy"])
    for name in ("abilities", "unit_types", "upgrades", "position_grid"):
        if name in low:
            low[name] = tuple(low[name])
    for name in (
        "preference_names",
        "preference_levels",
        "commitment_horizons",
        "task_horizons",
        "task_types",
        "target_regions",
        "goal_kinds",
        "goal_quantities",
    ):
        if name in high:
            high[name] = tuple(high[name])
    return BCModelConfig(**low), HierarchyConfig(**high)


def model_config_sha256(low, high):
    return digest([low.sha256, high.sha256])


def specialist_override(config, intent_profile, intervention_id):
    """Supported fixed profile; unsupported renewal policies fail pre-engine."""
    from poorstar.hierarchy.state import ExternalOverride

    if (
        config.arm != "persistent_intent"
        or config.manager_mode != "external"
        or not config.persistence
        or config.commitment_mode != "minimum_hold"
        or set(intent_profile) != set(config.preference_names)
    ):
        raise ValueError(
            "specialist requires persistent external minimum-hold Manager and complete named profile; renewal policy unavailable"
        )
    result = ExternalOverride(
        tuple(intent_profile[n] for n in config.preference_names),
        config.commitment_horizons[-1],
        0.5,
        intervention_id,
    )
    result.validate(config)
    return result


def specialist_program(config, intent_profile, episode):
    """Frozen specialist profile, with the arm's declared renewal semantics."""
    from poorstar.hierarchy.interventions import ExternalPreferenceProgram, PreferenceIntervention

    if set(intent_profile) != set(config.preference_names):
        raise ValueError("complete named specialist Preference profile required")
    horizon = (
        config.fixed_commitment_horizon
        if config.commitment_mode == "fixed_dwell"
        else config.commitment_horizons[-1]
    )
    return ExternalPreferenceProgram(
        config,
        episode=episode,
        interventions=(
            PreferenceIntervention(
                0, tuple(intent_profile[n] for n in config.preference_names), horizon
            ),
        ),
    )


def control_label_from_dict(label):
    """One typed JSON boundary for compiler and learner; no label inference."""
    from poorstar.evaluation.confidence import ConfidenceTarget
    from poorstar.hierarchy.state import DispatchObservation, ExecutionFeedback, ExternalOverride

    label = dict(label)
    for name in ("feedback", "body_feedback"):
        label[name] = tuple(ExecutionFeedback(**event) for event in label.get(name, []))
    for name in ("scheduler_dispatch_observation", "body_dispatch_observation"):
        if label.get(name) is not None:
            label[name] = DispatchObservation.from_dict(label[name])
    if label.get("confidence_target") is not None:
        target = dict(label["confidence_target"])
        target["evidence_sha256"] = tuple(target["evidence_sha256"])
        label["confidence_target"] = ConfidenceTarget(**target)
    if label.get("external_override") is not None:
        override = dict(label["external_override"])
        override["preference"] = tuple(override["preference"])
        label["external_override"] = ExternalOverride(**override)
    return label


class ControlSidecar:
    """Separate declared high-level targets, never injected into actor observation."""

    def __init__(self, path, identity, hierarchy_config):
        from poorstar.training.checkpoint import file_hash

        if file_hash(path) != identity.control_sidecar_sha256:
            raise ValueError("high-level label artifact is not frozen in training identity")
        document = json.loads(Path(path).read_text())
        check = dict(document)
        content = check.pop("content_digest", None)
        if digest(check) != content or document.get("schema") not in (
            "poorstar_v3_hierarchy_bc_sidecar_v1",
            "poorstar_v3_hierarchy_bc_sidecar_v2",
            "poorstar_v3_hierarchy_bc_sidecar_index_v3",
        ):
            raise ValueError("control sidecar content/schema drift")
        binding = dict(
            dataset_sha256=identity.dataset_sha256,
            split_sha256=identity.split_sha256,
            scope_sha256=identity.scope_sha256,
            hierarchy_config_sha256=hierarchy_config.sha256,
        )
        if document.get("binding") != binding or not is_sha(
            document.get("provenance_evidence_sha256")
        ):
            raise ValueError("high-level label data/split/scope/provenance binding required")
        evidence_path = Path(document.get("provenance_evidence_path", ""))
        if (
            not evidence_path.is_file()
            or file_hash(evidence_path) != document["provenance_evidence_sha256"]
        ):
            raise ValueError("high-level label provenance evidence is missing or changed")
        self.document = document
        self.config = hierarchy_config
        self._indexed = document["schema"] == "poorstar_v3_hierarchy_bc_sidecar_index_v3"
        if self._indexed:
            from poorstar.evaluation.sidecar_corpus import verify_index_sidecar_lineage

            # One bounded streaming pass over original component files. No
            # corpus-wide records, dependencies or state snapshots are loaded.
            verify_index_sidecar_lineage(
                document, json.loads(evidence_path.read_text()), purpose=identity.purpose
            )
            self._coverage = document["corpus_coverage"]
            self._window_digests = tuple(digest(w) for w in self._coverage["windows"])
            self._component_cache = OrderedDict()
            self.maximum_cached_windows = 0
            return
        if document["schema"].endswith("_v2"):
            from poorstar.evaluation.teachers import verify_annotation_document

            evidence = json.loads(evidence_path.read_text())
            # One complete annotation/provenance digest pass at initialization.
            # Window reads never rebuild the full annotation list or reread files.
            verify_annotation_document(document, evidence)
            if document.get("corpus_coverage") is not None:
                from poorstar.evaluation.sidecar_corpus import verify_corpus_sidecar_lineage

                verify_corpus_sidecar_lineage(document, evidence)
        elif identity.purpose == "FORMAL":
            raise ValueError(
                "formal high-level BC requires typed causal annotation provenance, not opaque legacy evidence"
            )
        if (
            identity.purpose == "FORMAL"
            and hierarchy_config.arm != "flat"
            and document.get("corpus_coverage") is None
        ):
            raise ValueError("formal high-level BC requires complete typed corpus window coverage")
        if identity.purpose == "FORMAL" and any(
            r.get("supervision_kind") == "actor_control_trace" for r in document["records"].values()
        ):
            raise ValueError("model samples cannot become formal human expert Intent labels")
        self._record_digests = {
            key: digest([label, document.get("annotation_dependencies", {}).get(key)])
            for key, label in document["records"].items()
        }
        self._state_digests = {
            key: digest(value) for key, value in document.get("initial_control_states", {}).items()
        }
        self._coverage = document.get("corpus_coverage")
        self._window_digests = (
            None if self._coverage is None else tuple(digest(w) for w in self._coverage["windows"])
        )

    @staticmethod
    def row_id(row):
        k = row.scheduler.key
        return digest([row.replay_sha256, k.episode, k.player_id, k.loop, k.observation_sha256])

    def _window_component(self, index):
        from poorstar.evaluation.sidecar_corpus import _component

        if index not in self._component_cache:
            # Evict before loading so even a transient new JSON document does
            # not add a third cached window to the resident annotation set.
            if len(self._component_cache) >= 2:
                self._component_cache.popitem(last=False)
            component = self._coverage["windows"][index]["component"]
            document, _, identity = _component(
                {k: component[k] for k in ("window_index", "path", "sha256")},
                self.document["binding"],
            )
            if identity != component:
                raise ValueError("indexed component identity changed")
            # LRU capacity is a fixed two windows, independent of corpus size.
            self._component_cache[index] = document
            self.maximum_cached_windows = max(
                self.maximum_cached_windows, len(self._component_cache)
            )
        self._component_cache.move_to_end(index)
        document = self._component_cache[index]
        check = dict(document)
        claimed = check.pop("content_digest", None)
        if digest(check) != claimed:
            raise ValueError("loaded indexed window labels changed after initialization")
        return document

    def sequence(self, rows, *, window_index=None):
        from poorstar.hierarchy.bc import HierarchyBCRecord, HierarchyBCSequence
        from poorstar.hierarchy.state import ExecutionFeedback, ExternalOverride, HierarchyState

        rows = tuple(rows)
        if not rows:
            raise ValueError("sidecar sequence cannot be empty")
        if self._coverage is not None:
            from poorstar.evaluation.sidecar_corpus import row_descriptor

            if type(window_index) is not int or not 0 <= window_index < len(
                self._coverage["windows"]
            ):
                raise ValueError(
                    "corpus sidecar sequence requires its exact registered window index"
                )
            expected = self._coverage["windows"][window_index]
            if digest(expected) != self._window_digests[window_index]:
                raise ValueError("loaded window coverage changed after initialization")
            description = [row_descriptor(row) for row in rows]
            if (
                digest(description) != expected["row_digest"]
                or digest([[row.valid, row.burn_in, row.loss] for row in rows])
                != expected["mask_digest"]
            ):
                raise ValueError(
                    "current window rows/phase clocks/loss masks differ from corpus coverage"
                )
        document = self._window_component(window_index) if self._indexed else self.document
        records = []
        for row in rows:
            row_id = self.row_id(row)
            label = document["records"].get(row_id)
            if label is None:
                raise ValueError("high-level supervision unavailable for this row")
            dependencies = document.get("annotation_dependencies", {}).get(row_id)
            if not self._indexed and digest([label, dependencies]) != self._record_digests.get(
                row_id
            ):
                raise ValueError(
                    "loaded annotation labels/dependencies changed after initialization"
                )
            if document["schema"].endswith("_v2"):
                from poorstar.evaluation.teachers import _dependency

                _dependency(dependencies["scheduler"], row.scheduler, "scheduler")
                if row.body is not None:
                    _dependency(dependencies["body"], row.body, "body")
                elif dependencies["body"]:
                    raise ValueError("WAIT sidecar contains body provenance")
            records.append(HierarchyBCRecord(row, **control_label_from_dict(label)))
        first = self.row_id(rows[0])
        state = document.get("initial_control_states", {}).get(first)
        if not self._indexed and (
            None if state is None else digest(state)
        ) != self._state_digests.get(first):
            raise ValueError("loaded causal snapshot changed after initialization")
        initial = None if state is None else HierarchyState.from_dict(state, self.config)
        key = rows[0].scheduler.key
        if initial is not None and (
            initial.episode != key.episode
            or initial.last_loop is not None
            and initial.last_loop >= key.loop
        ):
            raise ValueError(
                "window snapshot must precede its first scheduler loop in the same episode"
            )
        return HierarchyBCSequence(tuple(records), initial)

    def validate_dataset(
        self,
        dataset,
        model,
        *,
        head_weights=None,
        control_weights=None,
        supervision_policy="STRICT_WHOLE_BODY_V1",
    ):
        from poorstar.evaluation.sidecar_corpus import validate_dataset_sidecar

        return validate_dataset_sidecar(
            self,
            dataset,
            model,
            head_weights=head_weights,
            control_weights=control_weights,
            supervision_policy=supervision_policy,
        )

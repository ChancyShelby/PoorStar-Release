"""Evaluation: measurement bridge."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from poorstar.common.integrity import source_integrity
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import TrainingIdentity, digest, is_sha


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str

    def read(self):
        if not isinstance(self.path, str) or not self.path or not is_sha(self.sha256):
            raise ValueError("typed artifact path and file SHA256 required")
        if file_hash(self.path) != self.sha256:
            raise ValueError("actual evaluator artifact file drift: " + self.path)
        return json.loads(Path(self.path).read_text())


def _ref(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("exact artifact reference required")
    return ArtifactRef(**value)


def _document(value, schema):
    body = dict(value)
    expected = body.pop("content_digest", None)
    if value.get("schema") != schema or digest(body) != expected:
        raise ValueError("frozen document schema/content drift: " + schema)


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("measured finite scalar unavailable")
    return value


def _positive(value):
    if type(value) is not int or value < 1:
        raise ValueError("frozen positive integer budget required")
    return value


def _validate_launch_proof(report, identity, execution_sha256):
    """Re-read the real phase authorization, including every bound gate file."""
    from poorstar.training.gates import validate_launch

    proof = report.get("launch_preflight")
    if not isinstance(proof, dict) or set(proof) != {"path", "sha256", "content_digest", "stage"}:
        raise ValueError("actual formal evaluator phase authorization is missing")
    _ref(dict(path=proof["path"], sha256=proof["sha256"])).read()
    authorized = validate_launch(
        proof["path"], identity, report["stage"], execution_config_sha256=execution_sha256
    )
    if authorized["content_digest"] != proof["content_digest"] or proof["stage"] != report["stage"]:
        raise ValueError("actual formal evaluator phase authorization drift")


def _load_run(binding, plan, source_check, source_manifest_sha256):
    """Validate the producer's actual stage report, exports, and metric ledger."""
    from poorstar.training.export import load_actor

    required = {"stage_report", "stage_plan", "identity", "stage", "binding"}
    if set(binding) != required:
        raise ValueError("complete frozen evaluator run binding required")
    report = _ref(binding["stage_report"]).read()
    _document(report, "poorstar_v3_stage_actor_collection_report_v1")
    stage_plan = _ref(binding["stage_plan"]).read()
    _document(stage_plan, "poorstar_v3_stage_actor_plan_v1")
    from poorstar.training.stage_actor import StageActorPlan

    numeric = dict(stage_plan)
    numeric.pop("content_digest")
    frozen = StageActorPlan(**numeric)
    if (
        report["plan_sha256"] != frozen.sha256
        or report["execution_config_sha256"] != report["plan_sha256"]
    ):
        raise ValueError("actual stage collection execution configuration drift")
    identity = TrainingIdentity(**binding["identity"])
    if (
        report["identity"] != asdict(identity)
        or report["stage"] != binding["stage"]
        or report["binding"] != binding["binding"]
        or report["purpose"] != plan["purpose"]
        or report["provenance"] != "evaluator"
        or identity.purpose != plan["purpose"]
    ):
        raise ValueError("actual evaluator identity/stage/cell binding drift")
    actual_binding = dict(
        map_name=frozen.map_name,
        seed=frozen.seed,
        collection_id=frozen.collection_id,
        maximum_loops=frozen.maximum_loops,
        maximum_compute_millis=frozen.maximum_compute_millis,
        parameter_budget=frozen.parameter_budget,
        execution=frozen.execution,
    )
    if (
        frozen.identity != asdict(identity)
        or frozen.phase != report["stage"]
        or frozen.provenance != "evaluator"
        or actual_binding != report["binding"]
    ):
        raise ValueError("producer collection plan differs from actual evaluator binding")
    if (
        identity.source_sha256 != source_manifest_sha256
        or identity.scope_sha256 != plan["scope_sha256"]
        or identity.split_sha256 != plan["split_sha256"]
    ):
        raise ValueError("actual evaluator source/scope/split differs from frozen study")
    if (
        report["source_preflight"] != source_check
        or report["source_postflight"] != source_check
        or source_check["problems"]
    ):
        raise ValueError("actual evaluator source pre/postflight differs")
    if report["all_owned_sessions_closed"] is not True or any(
        p.get("exit_code") is None for p in report["owned_process_results"]
    ):
        raise ValueError("actual evaluator owned processes not closed")
    if plan["purpose"] == "FORMAL" and (
        report.get("fixture_scope") is not None
        or type(report.get("actual_engine_frames")) is not int
        or report["actual_engine_frames"] < 1
    ):
        raise ValueError("protocol fixture cannot become formal evaluator evidence")
    if plan["purpose"] == "FORMAL":
        _validate_launch_proof(report, identity, frozen.sha256)
    actual_actors = {}
    for name in ("learner", "opponent"):
        actor = report["actors"][name]
        model, actor_identity, export = load_actor(
            actor["weights_path"], sha256=actor["weights_sha256"]
        )
        from poorstar.training.job_worker import actor_parameter_counts
        from poorstar.training.stages import actor_counters

        actor_counters(export)
        if actor["parameter_counts"] != actor_parameter_counts(model, export["training_phase"]):
            raise ValueError("actual actor parameter allocation drift")
        for key in ("policy_version", "parameter_sha256", "training_phase", "cumulative_counters"):
            if actor[key] != export[key]:
                raise ValueError("actual immutable actor export binding drift: " + key)
        if actor["identity"] != asdict(actor_identity) or not is_sha(actor["snapshot_id"]):
            raise ValueError("actual immutable actor identity/snapshot drift")
        if actor["snapshot_id"] != getattr(frozen, name + "_snapshot_id"):
            raise ValueError("actual immutable snapshot differs from frozen opponent/learner")
        if name == "learner" and (
            actor_identity != identity or export["training_phase"] != report["stage"]
        ):
            raise ValueError("learner export differs from evaluator stage identity")
        if any(
            getattr(actor_identity, n) != getattr(identity, n)
            for n in ("scope_sha256", "profile_sha256", "matchup")
        ):
            raise ValueError("actual opponent actor information/action domain differs")
        actual_actors[name] = actor
        if name == "learner":
            unit_types = model.config.unit_types
    b = report["binding"]
    execution = b["execution"]
    if (
        type(b["seed"]) is not int
        or b["seed"] < 0
        or not b["map_name"]
        or not b["collection_id"]
        or report["environment_loops"] is None
        or report["environment_loops"] > _positive(b["maximum_loops"])
        or report["compute_millis"] > _positive(b["maximum_compute_millis"])
    ):
        raise ValueError("actual evaluator seed/map/frame/compute budget unresolved")
    _positive(execution["measurement_period"])
    if execution.get("metric_program_sha256") != plan["metric_program_sha256"]:
        raise ValueError(
            "producer actual metric program differs from frozen semantic metric program"
        )
    budget = b["parameter_budget"]
    for name, actor in actual_actors.items():
        counts = actor["parameter_counts"]
        if counts["parameter_count"] > budget["maximum_parameters"] or (
            name == "learner"
            and counts["trainable_parameter_count"] > budget["maximum_trainable_parameters"]
        ):
            raise ValueError("actual evaluator exceeds frozen parameter budget")
    if report["status"] != "COMPLETE":
        raise ValueError("actual registered evaluator run is " + report["status"])
    artifacts = report["artifacts"]
    collector = _ref(artifacts["collector_report"]).read()
    _document(collector, "poorstar_v3_two_player_actor_report_v1")
    metrics = _ref(artifacts["evaluator_metrics"]).read()
    _document(metrics, "poorstar_v3_visible_evaluator_metrics_v1")
    if (
        collector["provenance"] != metrics["provenance"]
        or metrics["provenance"] != "evaluator"
        or collector["checkpoint_sha256"] != actual_actors["learner"]["parameter_sha256"]
        or collector["environment_loops"] != report["environment_loops"]
        or metrics["physical_environment_loops"] != report["environment_loops"]
        or not collector["terminated"]
        or collector["truncated"]
        or collector["boundary_reason"] != "engine_terminal"
        or collector["reward"] not in (-1.0, 0.0, 1.0)
        or metrics["terminal_outcome"] != collector["reward"]
    ):
        raise ValueError("actual complete natural-terminal evaluator artifact required")
    observations = metrics["observations"]
    if not observations:
        raise ValueError("actual visible evaluator sample ledger is missing")
    player = observations[0]["player_id"]
    loops = []
    for row in observations:
        value = dict(row)
        sha = value.pop("content_digest", None)
        if (
            digest(value) != sha
            or row["player_id"] != player
            or not is_sha(row["observation_sha256"])
        ):
            raise ValueError("actual visible sample content/player drift")
        if type(row["loop"]) is not int or row["loop"] < 0 or loops and row["loop"] <= loops[-1]:
            raise ValueError("actual visible sample clocks are not chronological")
        loops.append(row["loop"])
    period = execution["measurement_period"]
    end = collector["environment_loops"]
    expected_loops = list(range(0, end, period)) + [end]
    if loops != expected_loops:
        raise ValueError("incomplete registered fixed-clock visible sample ledger")
    controls = metrics["controls"]
    for row in controls:
        value = dict(row)
        sha = value.pop("evidence_sha256", None)
        if (
            digest(value) != sha
            or not 0 <= row["loop"] <= end
            or not is_sha(row["observation_sha256"])
        ):
            raise ValueError("actual control ledger content/clock drift")
    receipts = [v for v in collector["receipts"] if v["player_id"] == player]
    actual_counts = dict(
        commands=sum(v["commands"] for v in receipts),
        accepted=sum(v["accepted"] for v in receipts),
        rejected=sum(v["commands"] - v["accepted"] for v in receipts),
    )
    if metrics["dispatch_counts"] != actual_counts or metrics["goal_observations"] != [
        v for v in collector["goal_observations"] if v["player_id"] == player
    ]:
        raise ValueError("metric ledger differs from actual dispatch/task receipts")
    from poorstar.evaluation.confidence import ConfidenceWindow, build_confidence_target

    confirmations = [
        dict(
            episode=collector["episode_id"],
            parent_revision=g["parent_revision"],
            loop=g["loop"],
            task_id=g["task_id"],
            status="confirmed",
            evidence_sha256=g["evidence_id"],
        )
        for g in metrics["goal_observations"]
        if g["outcome"] == "confirmed"
    ]
    seen_windows = set()
    for row in metrics.get("confidence_predictions", []):
        body = dict(row)
        body.pop("target")
        expected = body.pop("evidence_sha256")
        window = ConfidenceWindow(**row["window"])
        window.validate()
        matching = [
            c
            for c in controls
            if c["phase"] == "scheduler"
            and c["operation"] == "update"
            and c["loop"] == window.start_loop
            and c["observation_sha256"] == window.prediction_observation_sha256
        ]
        if (
            digest(body) != expected
            or row["window_sha256"] != window.sha256
            or window.sha256 in seen_windows
            or row["player_id"] != player
            or window.episode != collector["episode_id"]
            or len(matching) != 1
            or row["prediction_kind"] != "PURE_PRECHOICE_MANAGER_BETA_MEAN"
            or not 0 < _number(row["prediction"]) < 1
        ):
            raise ValueError("actual pure-prechoice confidence prediction/window ledger drift")
        seen_windows.add(window.sha256)
        target = build_confidence_target(
            window,
            confirmations=confirmations,
            cutoff_loop=end,
            cutoff_evidence_sha256=collector["content_digest"],
        )
        if digest(asdict(target)) != digest(row["target"]):
            raise ValueError(
                "actual confidence target differs from visible confirmation/censor evidence"
            )
    return dict(
        report=report,
        collector=collector,
        metrics=metrics,
        actors=actual_actors,
        player_id=player,
        unit_types=unit_types,
    )


def scenario_sha256(run):
    """Actual fixed-opponent/map/seed/runtime scenario, without a causal claim."""
    b = run["report"]["binding"]
    return digest(
        dict(
            seed=b["seed"],
            map_name=b["map_name"],
            version=b["execution"]["version"],
            opponent_sha256=run["actors"]["opponent"]["weights_sha256"],
        )
    )


def _metric(run, selector):
    kind = selector.get("kind")
    metrics = run["metrics"]
    if kind == "terminal_outcome" and set(selector) == {"kind"}:
        return _number(metrics["terminal_outcome"])
    if kind == "dispatch_rate" and set(selector) == {"kind", "numerator"}:
        if (
            selector["numerator"] not in ("accepted", "rejected")
            or metrics["dispatch_counts"]["commands"] == 0
        ):
            raise ValueError("actual nonempty dispatch denominator required")
        return _number(
            metrics["dispatch_counts"][selector["numerator"]]
            / metrics["dispatch_counts"]["commands"]
        )
    if kind in ("visible_scalar", "completed_own_unit_types"):
        required = (
            {"kind", "loop", "name"} if kind == "visible_scalar" else {"kind", "loop", "unit_types"}
        )
        if set(selector) != required or type(selector["loop"]) is not int:
            raise ValueError("exact frozen sample selector required")
        rows = [r for r in metrics["observations"] if r["loop"] == selector["loop"]]
        if len(rows) != 1:
            raise ValueError("exact measured loop absent; interpolation is forbidden")
        if kind == "visible_scalar":
            if not isinstance(selector["name"], str) or selector["name"] not in rows[0]["metrics"]:
                raise ValueError("registered visible scalar unavailable")
            return _number(rows[0]["metrics"][selector["name"]])
        types = selector["unit_types"]
        counts = rows[0]["completed_own_unit_types"]
        if (
            not rows[0]["complete_entity_coverage"]
            or counts is None
            or not types
            or len(set(types)) != len(types)
            or any(type(v) is not int or v not in run["unit_types"] for v in types)
        ):
            raise ValueError("complete measured own-unit census and catalog IDs required")
        # Zero is an actual completed, complete-coverage census of this type.
        return sum(counts.get(str(v), 0) for v in types)
    raise ValueError("unsupported/unregistered metric selector")


def _anchor(run, selector):
    if (
        set(selector) != {"kind", "index"}
        or type(selector["index"]) is not int
        or selector["index"] < 0
    ):
        raise ValueError("actual collected anchor selector required")
    key = {"intervention": "registered_interventions", "initial": "initial_inputs"}.get(
        selector["kind"]
    )
    rows = (
        []
        if key is None
        else [r for r in run["collector"].get(key, []) if r["player_id"] == run["player_id"]]
    )
    if selector["index"] >= len(rows):
        raise ValueError("actual causal/initial anchor unavailable; restarted seed is insufficient")
    row = rows[selector["index"]]
    if not all(
        is_sha(row.get(n))
        for n in ("visible_input_sha256", "pre_intervention_action_prefix_sha256")
    ):
        raise ValueError("actual collected neural-input/action-prefix anchor unavailable")
    return row


def _terminal_panel(ids, runs):
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("complete unique registered natural-terminal panel required")
    panel = [runs[n] for n in ids]
    return panel, sum(_metric(v, {"kind": "terminal_outcome"}) for v in panel) / len(panel)


def _cell(analysis, selector, runs):
    kind = analysis["kind"]
    selected = selector.get("kind")
    if kind in ("paired", "dose") and selected == "paired":
        required = {
            "kind",
            "run_id",
            "comparison",
            "treatment",
            "scenario_sha256",
            "anchor",
            "metrics",
            "foundation_sha256",
        }
        if set(selector) != required:
            raise ValueError("exact paired cell selector required")
        run = runs[selector["run_id"]]
        anchor = _anchor(run, selector["anchor"])
        causal = (
            kind == "dose"
            or analysis["parameters"].get("pairing_mode", "SAME_CHECKPOINT_CAUSAL")
            == "SAME_CHECKPOINT_CAUSAL"
        )
        if causal and selector["anchor"]["kind"] != "intervention":
            raise ValueError(
                "same-checkpoint causal treatment requires an actual registered intervention"
            )
        if causal and not is_sha(anchor.get("policy_context_sha256")):
            raise ValueError(
                "actual complete pre-intervention neural/control context anchor unavailable"
            )
        if selector["scenario_sha256"] != scenario_sha256(run):
            raise ValueError("actual paired fixed scenario differs from registration")
        foundation = selector["foundation_sha256"]
        if foundation is not None and (
            foundation == "0" * 64 or foundation != run["report"]["identity"]["initializer_sha256"]
        ):
            raise ValueError("actual exported common initializer/foundation unavailable")
        return dict(
            comparison=selector["comparison"],
            treatment=selector["treatment"],
            seed=run["report"]["binding"]["seed"],
            opponent_sha256=run["actors"]["opponent"]["weights_sha256"],
            map_name=run["report"]["binding"]["map_name"],
            scenario_sha256=selector["scenario_sha256"],
            checkpoint_sha256=run["actors"]["learner"]["parameter_sha256"],
            intervention_loop=anchor.get("actual_scheduler_loop", anchor.get("loop")),
            shared_prefix_sha256=anchor["pre_intervention_action_prefix_sha256"],
            intervention_observation_sha256=anchor["visible_input_sha256"],
            metrics={n: _metric(run, s) for n, s in selector["metrics"].items()},
            evidence_sha256=digest(
                dict(stage_report=run["report"]["content_digest"], selector=selector)
            ),
            provenance="evaluator",
            status="COMPLETE",
            foundation_sha256=foundation,
        )
    if kind in ("intent_dynamics", "task_goals") and selected in ("control", "goal"):
        if (
            set(selector) != {"kind", "run_id", "index"}
            or type(selector["index"]) is not int
            or selector["index"] < 0
        ):
            raise ValueError("exact actual ledger event index required")
        run = runs[selector["run_id"]]
        ledger = (
            [
                r
                for r in run["metrics"]["controls"]
                if r["phase"] == "scheduler" and r["operation"] is not None
            ]
            if selected == "control"
            else run["metrics"]["goal_observations"]
        )
        if selector["index"] >= len(ledger):
            raise ValueError("registered actual ledger event absent")
        row = ledger[selector["index"]]
        if selected == "control":
            if row["preference"] is None:
                raise ValueError("actual terminated Intent has no Preference; do not invent one")
            return {
                n: row[n]
                for n in (
                    "episode",
                    "loop",
                    "preference",
                    "operation",
                    "manager_decision",
                    "evidence_sha256",
                )
            }
        return dict(row)
    if (
        kind == "learning_curve"
        and selected == "outcome_panel"
        and set(selector) == {"kind", "run_ids"}
    ):
        panel, score = _terminal_panel(selector["run_ids"], runs)
        frames = {r["actors"]["learner"]["cumulative_counters"]["environment_loops"] for r in panel}
        checkpoints = {r["actors"]["learner"]["parameter_sha256"] for r in panel}
        if len(frames) != 1 or len(checkpoints) != 1:
            raise ValueError(
                "one actual checkpoint/frame counter per complete curve panel required"
            )
        return [frames.pop(), score]
    if kind in ("population_coverage", "restricted_response_gain") and selected == "payoff_row":
        if set(selector) != {"kind", "policy_snapshot_id", "opponents"} or not is_sha(
            selector["policy_snapshot_id"]
        ):
            raise ValueError("typed actual population snapshot/opponent matrix required")
        if set(selector["opponents"]) != set(analysis["parameters"]["opponent_ids"]):
            raise ValueError("complete frozen fixed-opponent payoff matrix required")
        result = {}
        for opponent, ids in selector["opponents"].items():
            panel, score = _terminal_panel(ids, runs)
            if any(
                r["actors"]["learner"]["snapshot_id"] != selector["policy_snapshot_id"]
                or r["actors"]["opponent"]["snapshot_id"] != opponent
                for r in panel
            ):
                raise ValueError("actual immutable payoff population/opponent differs")
            result[opponent] = score
        return result
    if kind == "confidence" and selected == "confidence":
        if (
            set(selector) != {"kind", "run_id", "index", "window_sha256"}
            or type(selector["index"]) is not int
            or selector["index"] < 0
        ):
            raise ValueError("typed confidence prediction index/window identity required")
        ledger = runs[selector["run_id"]]["metrics"].get("confidence_predictions", [])
        if selector["index"] >= len(ledger):
            raise ValueError("actual confidence prediction/outcome ledger is missing")
        row = ledger[selector["index"]]
        if selector["window_sha256"] != row["window_sha256"]:
            raise ValueError("confidence selector differs from actual commitment window")
        return dict(prediction=row["prediction"], target=row["target"])
    raise ValueError("unsupported typed analysis cell selector")


def _used_runs(selector):
    if "run_id" in selector:
        return [selector["run_id"]]
    if "run_ids" in selector:
        return selector["run_ids"]
    if "opponents" in selector:
        return [n for ids in selector["opponents"].values() for n in ids]
    return []


def assemble_measurements(plan_path, mapping_path):
    """Return successful measurements or a content-bound blocked-panel report."""
    plan = json.loads(Path(plan_path).read_text())
    mapping = json.loads(Path(mapping_path).read_text())
    _document(plan, "poorstar_v3_research_evaluation_plan_v1")
    _document(mapping, "poorstar_v3_measurement_bridge_mapping_v1")
    if (
        mapping["evaluation_plan_sha256"] != plan["content_digest"]
        or mapping["evaluation_plan_file_sha256"] != file_hash(plan_path)
        or mapping["purpose"] != plan["purpose"]
    ):
        raise ValueError("frozen bridge mapping differs from registered evaluator plan")
    source = mapping["source"]
    manifest = _ref(source["manifest"])
    manifest.read()
    source_check = source_integrity(Path(source["code_root"]), Path(manifest.path))
    if source_check["problems"] or manifest.sha256 != plan["source_sha256"]:
        raise ValueError("actual frozen source manifest/code drift")
    metric = _ref(mapping["metric_program"])
    metric_path = Path(metric.path).resolve()
    root = Path(source["code_root"]).resolve()
    if (
        metric.sha256 != plan["metric_program_sha256"]
        or file_hash(metric_path) != metric.sha256
        or not metric_path.is_relative_to(root)
    ):
        raise ValueError("actual metric program source hash/tree differs from frozen plan")
    entries = json.loads(Path(manifest.path).read_text())["files"]
    if not any(
        (root / e["path"]).resolve() == metric_path and e["sha256"] == metric.sha256
        for e in entries
    ):
        raise ValueError("metric program is not covered by actual frozen source manifest")
    if mapping["matched_budget_sha256"] != plan["matched_budget_sha256"]:
        raise ValueError("frozen matched experimental budget mapping drift")
    analyses = {a["analysis_id"]: a for a in plan["analyses"]}
    if len(analyses) != len(plan["analyses"]) or set(mapping["cells"]) != set(analyses):
        raise ValueError("all registered analyses must be mapped exactly once")
    refs = []
    for a, cells in mapping["cells"].items():
        if set(cells) != set(analyses[a]["expected_cell_ids"]) or len(cells) != len(
            analyses[a]["expected_cell_ids"]
        ):
            raise ValueError("complete frozen analysis seed/opponent/map cell mapping required")
        refs.extend(n for s in cells.values() for n in _used_runs(s))
    if set(refs) != set(mapping["runs"]) or not mapping["runs"]:
        raise ValueError(
            "complete registered run matrix required; unknown/unused runs cannot disappear"
        )
    reports = [v["stage_report"]["sha256"] for v in mapping["runs"].values()]
    if len(set(reports)) != len(reports):
        raise ValueError("duplicate evaluator run artifact cannot increase sample size")
    runs = {}
    failures = {}
    blocked = {}
    values = {}
    causal_context_evidence = {}
    for name, binding in mapping["runs"].items():
        try:
            runs[name] = _load_run(binding, plan, source_check, manifest.sha256)
        except (ValueError, KeyError, TypeError, OSError) as error:
            failures[name] = dict(
                error_type=type(error).__name__, reason=str(error), binding=binding
            )
    actual_games = [r["report"]["artifacts"]["collector_report"]["sha256"] for r in runs.values()]
    if len(set(actual_games)) != len(actual_games):
        raise ValueError("duplicate actual collector artifact cannot increase sample size")
    for name, analysis in analyses.items():
        values[name] = {}
        ledger_indices = {}
        paired_contexts = {}
        for cell, selector in mapping["cells"][name].items():
            try:
                failed = [n for n in _used_runs(selector) if n in failures]
                if failed:
                    raise ValueError(
                        "actual registered evaluator runs blocked: " + ",".join(failed)
                    )
                values[name][cell] = _cell(analysis, selector, runs)
                if analysis["kind"] in ("paired", "dose") and (
                    analysis["kind"] == "dose"
                    or analysis["parameters"].get("pairing_mode", "SAME_CHECKPOINT_CAUSAL")
                    == "SAME_CHECKPOINT_CAUSAL"
                ):
                    from poorstar.evaluation.evaluation import PairedResult

                    row = PairedResult(**values[name][cell])
                    anchor = _anchor(runs[selector["run_id"]], selector["anchor"])
                    context = anchor["policy_context_sha256"]
                    if row.pair_key in paired_contexts and paired_contexts[row.pair_key] != context:
                        raise ValueError(
                            "actual recurrent/control/feedback/receipt/allocation context differs across causal pair"
                        )
                    paired_contexts[row.pair_key] = context
                    causal_context_evidence[name + ":" + cell] = context
                if selector.get("kind") in ("control", "goal", "confidence"):
                    ledger_indices.setdefault(selector["run_id"], []).append(selector["index"])
            except (ValueError, KeyError, TypeError, OSError) as error:
                blocked[name + ":" + cell] = dict(
                    error_type=type(error).__name__, reason=str(error), selector=selector
                )
        for run_id, indices in ledger_indices.items():
            run = runs[run_id]
            selector_kind = {
                "intent_dynamics": "control",
                "task_goals": "goal",
                "confidence": "confidence",
            }[analysis["kind"]]
            ledger = (
                [
                    r
                    for r in run["metrics"]["controls"]
                    if r["phase"] == "scheduler" and r["operation"] is not None
                ]
                if selector_kind == "control"
                else run["metrics"].get("confidence_predictions", [])
                if selector_kind == "confidence"
                else run["metrics"]["goal_observations"]
            )
            if sorted(indices) != list(range(len(ledger))):
                blocked[name + ":ledger:" + run_id] = dict(
                    reason="complete actual ledger must be accounted exactly once",
                    actual_events=len(ledger),
                    mapped_indices=indices,
                )
    result = dict(
        schema="poorstar_v3_measurement_bridge_blocked_v1"
        if blocked or failures
        else "poorstar_v3_research_measurements_v1",
        purpose=plan["purpose"],
        evaluation_plan_sha256=plan["content_digest"],
        provenance="evaluator",
        analyses=values,
        assembly_status="BLOCKED" if blocked or failures else "ASSEMBLED",
        blocked_cells=blocked,
        failed_runs=failures,
        mapping_file_sha256=file_hash(mapping_path),
        mapping_content_digest=mapping["content_digest"],
        source_integrity=source_check,
        actual_metric_program_sha256=metric.sha256,
        all_registered_run_ids=sorted(mapping["runs"]),
        fixed_clock_samples_are_continuous_truth=False,
        actual_causal_policy_context_sha256=causal_context_evidence,
        causal_pairing_inferred_from_seed=False,
        formal_training_started=False,
    )
    result["content_digest"] = digest(result)
    if not blocked and not failures:
        from poorstar.evaluation.runner import evaluate_registered

        try:
            # Preserve the existing analyzer's exact pairing, ledger, curve,
            # and full-matrix semantics before publishing consumable input.
            evaluate_registered(plan, result)
        except (ValueError, KeyError, TypeError) as error:
            result.pop("content_digest")
            result["schema"] = "poorstar_v3_measurement_bridge_blocked_v1"
            result["assembly_status"] = "BLOCKED"
            result["blocked_cells"]["registered_analysis"] = dict(
                error_type=type(error).__name__, reason=str(error)
            )
            result["content_digest"] = digest(result)
    return result

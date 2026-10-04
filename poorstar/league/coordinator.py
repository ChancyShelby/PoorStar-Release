"""League: coordinator."""

import json
import math
import socket
import time
from dataclasses import asdict
from pathlib import Path

from poorstar.league.contracts import WorkerRequest, WorkerResult
from poorstar.league.ownership import (
    CoordinatorOwner,
    authorize_worker,
    read_document,
    verify_stage_gates,
    verify_worker_lease,
    write_document,
)
from poorstar.league.promotion import evaluate_promotion
from poorstar.league.scheduler import LeagueConfig, LeagueScheduler
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import TrainingIdentity, digest


class LeagueCoordinator:
    def __init__(self, root, registry, identity, plan, *, fixture=False):
        self.root, self.registry, self.identity, self.plan = Path(root), registry, identity, plan
        self.fixture = fixture
        self.plan_sha256 = plan.sha256
        self.owner = CoordinatorOwner(self.root, self.plan_sha256)
        self.scheduler = None
        try:
            config = LeagueConfig(
                plan.seed,
                plan.budget.training_game_loops,
                plan.budget.evaluation_game_loops,
                plan.budget.maximum_job_loops,
                plan.budget.population_limit,
                kind=plan.kind,
            )
            self.scheduler = LeagueScheduler(
                self.root / "league.sqlite", registry, identity, config
            )
            self.db = self.scheduler.db
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS league_state (key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS operations (id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,"
                "spec TEXT NOT NULL,status TEXT NOT NULL,result TEXT)"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY,operation_id TEXT NOT NULL,"
                "ordinal INTEGER NOT NULL,request TEXT NOT NULL,status TEXT NOT NULL,result TEXT)"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS events (ordinal INTEGER PRIMARY KEY,event TEXT NOT NULL)"
            )
            plan_value = json.loads(json.dumps(asdict(plan)))
            binding = dict(
                identity_sha256=identity.sha256,
                plan=plan_value,
                fixture=fixture,
                host=socket.gethostname(),
            )
            old = self._state("binding")
            if old is not None and old != binding:
                raise ValueError("league run/host/fixture/plan identity drift")
            self._validate_population()
            if old is None:
                self.db.execute("BEGIN IMMEDIATE")
                try:
                    self._set_state("binding", binding)
                    self._set_state("current", {m.name: m.snapshot_id for m in plan.members})
                    self._set_state("promoted", {m.name: m.snapshot_id for m in plan.members})
                    self._set_state("training_history", list(plan.historical))
                    for spec in self._operation_specs():
                        self.db.execute(
                            "INSERT INTO operations VALUES (?,?,?,'READY',NULL)",
                            (digest(spec), spec["ordinal"], json.dumps(spec, sort_keys=True)),
                        )
                    self._event(
                        "INITIALIZED",
                        dict(
                            plan_sha256=self.plan_sha256,
                            matched_budget_sha256=plan.matched_budget_sha256,
                        ),
                    )
                    self.db.execute("COMMIT")
                except BaseException:
                    self.db.execute("ROLLBACK")
                    raise
            if not (self.root / "run_plan.json").exists():
                write_document(self.root / "run_plan.json", plan_value, immutable=True)
            if read_document(self.root / "run_plan.json") != plan_value:
                raise ValueError("frozen plan file drift")
        except BaseException:
            if self.scheduler is not None:
                self.scheduler.close()
            self.owner.close()
            raise

    def close(self):
        self.scheduler.close()
        self.owner.close()

    def _state(self, name):
        row = self.db.execute("SELECT value FROM league_state WHERE key=?", (name,)).fetchone()
        return None if row is None else json.loads(row[0])

    def _set_state(self, name, value):
        self.db.execute(
            "INSERT INTO league_state VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (name, json.dumps(value, sort_keys=True)),
        )

    def _event(self, kind, value):
        ordinal = self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        self.db.execute(
            "INSERT INTO events VALUES (?,?)",
            (ordinal, json.dumps(dict(kind=kind, value=value), sort_keys=True)),
        )

    def _validate_population(self):
        for m in self.plan.members:
            snapshot = self.registry.read(m.snapshot_id)
            if snapshot.get("training_identity") != m.identity:
                raise ValueError("population exact exported identity required")
            role = snapshot["role"]
            if role not in ("main", "main_exploiter", "league_exploiter", "intent_specialist"):
                raise ValueError("active population role required")
            if self.plan.kind == "ordinary" and role == "intent_specialist":
                raise ValueError("ordinary league cannot include Intent specialists")
            if (
                snapshot["scope_sha256"] != self.identity.scope_sha256
                or snapshot["profile_sha256"] != self.identity.profile_sha256
            ):
                raise ValueError("active population scope/action profile mismatch")
        if self.plan.kind == "intent_specialized" and not any(
            self.registry.read(m.snapshot_id)["role"] == "intent_specialist"
            for m in self.plan.members
        ):
            raise ValueError("specialized league requires an explicitly declared specialist")
        for group, role in ((self.plan.anchors, "anchor"), (self.plan.historical, "historical")):
            for identifier in group:
                record = self.registry.read(identifier)
                if (
                    record["role"] != role
                    or record["scope_sha256"] != self.identity.scope_sha256
                    or record["profile_sha256"] != self.identity.profile_sha256
                ):
                    raise ValueError("fixed anchor/history role or scope drift")

    def _operation_specs(self):
        ordinal = 0
        for round_id in range(self.plan.rounds):
            for member in self.plan.members:
                for game in range(self.plan.training_games_per_member):
                    yield dict(
                        ordinal=ordinal,
                        kind="actor",
                        phase="train",
                        round=round_id,
                        member=member.name,
                        game=game,
                    )
                    ordinal += 1
            for member in self.plan.members:
                yield dict(
                    ordinal=ordinal,
                    kind="learner",
                    phase="update",
                    round=round_id,
                    member=member.name,
                )
                ordinal += 1
            for member in self.plan.members:
                for target in ("incumbent", "candidate"):
                    for opponent in self.plan.anchors + self.plan.historical:
                        for map_name in self.plan.maps:
                            for seed in self.plan.evaluation_seeds:
                                yield dict(
                                    ordinal=ordinal,
                                    kind="actor",
                                    phase="eval",
                                    round=round_id,
                                    member=member.name,
                                    target=target,
                                    opponent=opponent,
                                    map_name=map_name,
                                    seed=seed,
                                )
                                ordinal += 1
                yield dict(
                    ordinal=ordinal,
                    kind="promotion",
                    phase="promotion",
                    round=round_id,
                    member=member.name,
                )
                ordinal += 1

    def _usage(self):
        usage = dict(
            actor_compute_millis=0, evaluation_compute_millis=0, learner_compute_millis=0, updates=0
        )
        for request, status, result in self.db.execute(
            "SELECT request,status,result FROM attempts"
        ):
            request = WorkerRequest(**json.loads(request))
            value = None if result is None else json.loads(result)
            category = (
                "learner_compute_millis"
                if request.kind == "learner"
                else "evaluation_compute_millis"
                if request.payload["job"]["provenance"] == "evaluator"
                else "actor_compute_millis"
            )
            usage[category] += (
                request.maximum_compute_millis
                if status == "RUNNING"
                else value["charged_compute_millis"]
            )
            if request.kind == "learner":
                usage["updates"] += (
                    request.payload["requested_updates"]
                    if status == "RUNNING"
                    else value["charged_updates"]
                )
        return usage

    def _member(self, name):
        return next(m for m in self.plan.members if m.name == name)

    def _snapshot_payload(self, identifier):
        record = self.registry.read(identifier)
        return dict(
            snapshot_id=identifier,
            weights_path=str(self.registry.weights_path(identifier).resolve()),
            weights_sha256=record["weights_sha256"],
            identity=record["training_identity"],
            policy_version=record["policy_version"],
            intent_profile=record["intent_profile"],
        )

    def _request(self, operation_id, spec):
        count = self.db.execute(
            "SELECT COUNT(*) FROM attempts WHERE operation_id=?", (operation_id,)
        ).fetchone()[0]
        if count >= self.plan.maximum_infrastructure_attempts:
            raise ValueError("frozen infrastructure retry count exhausted")
        usage = self._usage()
        category = (
            "learner_compute_millis"
            if spec["kind"] == "learner"
            else "evaluation_compute_millis"
            if spec["phase"] == "eval"
            else "actor_compute_millis"
        )
        maximum = (
            self.plan.budget.maximum_learner_compute_millis
            if spec["kind"] == "learner"
            else self.plan.budget.maximum_actor_compute_millis
        )
        if usage[category] + maximum > getattr(self.plan.budget, category):
            raise ValueError("frozen league compute budget fully reserved/exhausted")
        if (
            spec["kind"] == "learner"
            and usage["updates"] + self.plan.updates_per_member > self.plan.budget.maximum_updates
        ):
            raise ValueError("frozen league update budget fully reserved/exhausted")
        member = self._member(spec["member"])
        current = self._state("current")
        promoted = self._state("promoted")
        round_key = "round:" + str(spec["round"]) + ":" + member.name
        state = self._state(round_key)
        if state is None:
            state = dict(incumbent=promoted[member.name])
            self._set_state(round_key, state)
        if spec["kind"] == "actor":
            provenance = "learner" if spec["phase"] == "train" else "evaluator"
            if provenance == "learner":
                learner = current[member.name]
                # Roles define opponent pools, never privileged observation or rewards.
                role = self.registry.read(learner)["role"]
                if role == "main_exploiter":
                    opponents = [
                        identifier
                        for name, identifier in promoted.items()
                        if self.registry.read(identifier)["role"] == "main" and name != member.name
                    ]
                else:
                    opponents = list(
                        dict.fromkeys(
                            [v for name, v in promoted.items() if name != member.name]
                            + [v for v in self._state("training_history") if v != learner]
                        )
                    )
                if not opponents:
                    raise ValueError("declared training role has no eligible opponent")
                map_name = self.plan.maps[spec["game"] % len(self.plan.maps)]
                seed = int(digest([self.plan.seed, spec])[:8], 16)
                fixed = None
            else:
                learner = state[spec["target"]]
                opponents = [spec["opponent"]]
                map_name, seed, fixed = spec["map_name"], spec["seed"], spec["opponent"]
            job = self.scheduler.schedule(
                learner,
                opponents,
                provenance=provenance,
                map_name=map_name,
                seed=seed,
                operation_key=operation_id + ":" + str(count),
                fixed_opponent=fixed,
            )
            token = digest([self.owner.owner, operation_id, count])
            record = self.scheduler.job(job["job_id"])
            if record["status"] != "QUEUED":
                raise ValueError("operation job is already owned or committed")
            payload = dict(
                job=job,
                provenance=provenance,
                snapshots={
                    name: self._snapshot_payload(identifier)
                    for name, identifier in (("learner", learner), ("opponent", job["opponent"]))
                },
            )
            maximum = self.plan.budget.maximum_actor_compute_millis
            execution = self.plan.actor_execution
        else:
            token = digest([self.owner.owner, operation_id, count])
            execution = self.plan.learner_execution
            inputs = []
            for raw_spec, raw_result in self.db.execute(
                "SELECT spec,result FROM operations WHERE status='COMPLETE'"
            ):
                previous = json.loads(raw_spec)
                if (
                    previous["phase"] == "train"
                    and previous["round"] == spec["round"]
                    and previous["member"] == member.name
                ):
                    result = json.loads(raw_result)
                    if (
                        result["worker_status"] not in ("COMPLETE", "TRUNCATED")
                        or "rollout_index" not in result["artifacts"]
                    ):
                        raise ValueError(
                            "learner cannot use unresolved/truncated/evaluator trajectory"
                        )
                    inputs.append(result["artifacts"]["rollout_index"])
            if len(inputs) != self.plan.training_games_per_member:
                raise ValueError("complete training-only rollout inputs required before update")
            previous = self._snapshot_payload(current[member.name])
            payload = dict(
                provenance="learner",
                previous_snapshot=previous,
                rollout_indices=inputs,
                requested_updates=self.plan.updates_per_member,
                expected_next_policy_version=previous["policy_version"]
                + self.plan.updates_per_member,
            )
            previous_training = self._state("training:" + member.name)
            if previous_training is not None:
                payload["previous_training"] = previous_training
            maximum = self.plan.budget.maximum_learner_compute_millis
        payload["member_name"] = member.name
        payload["parameter_budget"] = dict(
            parameter_count=member.parameter_count,
            trainable_parameter_count=member.trainable_parameter_count,
            maximum_parameters=self.plan.budget.maximum_parameters,
            maximum_trainable_parameters=self.plan.budget.maximum_trainable_parameters,
        )
        request = WorkerRequest(
            spec["kind"],
            self.plan_sha256,
            member.identity,
            operation_id,
            digest([operation_id, count]),
            token,
            self.owner.owner,
            str((self.root / "attempts" / operation_id / str(count)).resolve()),
            maximum,
            payload,
            execution,
        )
        return request, count

    def _check_artifacts(self, request, result):
        for artifact in result.artifacts.values():
            if file_hash(artifact["path"]) != artifact["sha256"]:
                raise ValueError("worker artifact content drift")
        if result.status == "ERROR":
            return
        name = "report" if request.kind == "actor" else "training_report"
        if name not in result.artifacts:
            raise ValueError("complete worker requires report evidence")
        report = read_document(result.artifacts[name]["path"])
        if request.kind == "actor":
            job = request.payload["job"]
            if (
                report.get("provenance") != job["provenance"]
                or report.get("environment_loops") != result.environment_loops
            ):
                raise ValueError("actor result measured frames/provenance mismatch")
            if result.status == "COMPLETE" and (
                not report.get("terminated") or report.get("reward") not in (-1.0, 0.0, 1.0)
            ):
                raise ValueError("natural engine terminal required for completed payoff")
            if result.status == "TRUNCATED" and (
                report.get("terminated") or not report.get("truncated") or report.get("reward") != 0
            ):
                raise ValueError(
                    "truncation needs zero outcome reward and declared bootstrap boundary"
                )
            if job["provenance"] == "learner" and "rollout_index" not in result.artifacts:
                raise ValueError("training actor requires immutable rollout index")
            if "rollout_index" in result.artifacts:
                index = read_document(result.artifacts["rollout_index"]["path"])
                if (
                    index.get("job_id") != job["job_id"]
                    or index.get("identity_sha256") != TrainingIdentity(**request.identity).sha256
                ):
                    raise ValueError("rollout index actor identity/job drift")
        else:
            if "actor_export" not in result.artifacts:
                raise ValueError("learner update requires immutable actor export")
            if (
                report.get("identity_sha256") != TrainingIdentity(**request.identity).sha256
                or report.get("updates") != request.payload["requested_updates"]
                or report.get("policy_version") != request.payload["expected_next_policy_version"]
            ):
                raise ValueError("learner report update/identity/version mismatch")

    def _commit(self, request, result, elapsed_millis):
        result.validate(request)
        self._check_artifacts(request, result)
        charged = max(result.compute_millis, elapsed_millis)
        status = result.status
        if charged > request.maximum_compute_millis:
            status = "ERROR"  # Actual overspend stays visible; never clip accounting.
        if request.kind == "actor" and (
            result.environment_loops is not None
            and result.environment_loops > request.payload["job"]["maximum_loops"]
        ):
            raise ValueError("actor exceeded reserved frame cap")
        value = dict(
            worker_status=status,
            artifacts=result.artifacts,
            charged_compute_millis=charged,
            charged_updates=request.payload.get("requested_updates", 0),
            environment_loops=result.environment_loops,
            request_sha256=request.sha256,
        )
        if request.kind == "actor":
            report = (
                read_document(result.artifacts["report"]["path"])
                if "report" in result.artifacts
                else None
            )
            verdict = (
                {-1.0: "loss", 0.0: "tie", 1.0: "win"}[report["reward"]]
                if status == "COMPLETE"
                else "truncated"
                if status == "TRUNCATED"
                else "error"
            )
            evidence = file_hash(Path(request.output_dir) / "result.json")
            state = self.scheduler.job(request.payload["job"]["job_id"])
            if state["status"] == "QUEUED":
                self.scheduler.claim(request.claim_token, job_id=request.payload["job"]["job_id"])
                state = self.scheduler.job(request.payload["job"]["job_id"])
            if state["status"] == "RUNNING":
                self.scheduler.finish(
                    request.payload["job"]["job_id"],
                    request.claim_token,
                    verdict=verdict,
                    environment_loops=result.environment_loops,
                    evidence_sha256=evidence,
                    error="" if status != "ERROR" else "worker error/compute cap exceeded",
                )
            elif state["result"]["evidence_sha256"] != evidence:
                raise ValueError("crash recovery committed job evidence mismatch")
            value["verdict"] = verdict
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if request.kind == "learner" and status == "COMPLETE":
                old = self.registry.read(request.payload["previous_snapshot"]["snapshot_id"])
                snapshot = self.registry.register(
                    result.artifacts["actor_export"]["path"],
                    identity=TrainingIdentity(**request.identity),
                    policy_version=request.payload["expected_next_policy_version"],
                    role=old["role"],
                    intent_profile=old["intent_profile"],
                )
                spec = json.loads(
                    self.db.execute(
                        "SELECT spec FROM operations WHERE id=?", (request.operation_id,)
                    ).fetchone()[0]
                )
                current = self._state("current")
                current[spec["member"]] = snapshot["snapshot_id"]
                self._set_state("current", current)
                history = self._state("training_history")
                if request.payload["previous_snapshot"]["snapshot_id"] not in history:
                    history.append(request.payload["previous_snapshot"]["snapshot_id"])
                    self._set_state("training_history", history)
                key = "round:" + str(spec["round"]) + ":" + spec["member"]
                state = self._state(key)
                state["candidate"] = snapshot["snapshot_id"]
                self._set_state(key, state)
                value["snapshot_id"] = snapshot["snapshot_id"]
                if (
                    "checkpoint_manifest" in result.artifacts
                    and "accumulated_rollout_index" in result.artifacts
                ):
                    self._set_state(
                        "training:" + spec["member"],
                        {
                            name: result.artifacts[name]
                            for name in ("checkpoint_manifest", "accumulated_rollout_index")
                        },
                    )
            self.db.execute(
                "UPDATE attempts SET status='COMMITTED',result=? WHERE id=?",
                (json.dumps(value, sort_keys=True), request.attempt_id),
            )
            next_status = "READY" if status == "ERROR" else "COMPLETE"
            self.db.execute(
                "UPDATE operations SET status=?,result=? WHERE id=?",
                (next_status, json.dumps(value, sort_keys=True), request.operation_id),
            )
            self._event("WORKER_COMMITTED", dict(attempt_id=request.attempt_id, value=value))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return value

    def recover(self):
        """Never reclaim a live worker, never return its reserved frames for free."""
        recovered = 0
        for identifier, raw in self.db.execute(
            "SELECT id,request FROM attempts WHERE status='RUNNING'"
        ).fetchall():
            request = WorkerRequest(**json.loads(raw))
            directory = Path(request.output_dir)
            elapsed = 0
            if not self.fixture:
                from poorstar.league.dispatch import verify_closed_attempt

                elapsed = verify_closed_attempt(request)
            if (directory / "result.json").exists():
                result = WorkerResult(**read_document(directory / "result.json"))
                self._commit(request, result, elapsed)
                recovered += 1
                continue
            result = WorkerResult(
                request.sha256,
                request.operation_id,
                request.attempt_id,
                request.claim_token,
                request.kind,
                "ERROR",
                request.maximum_compute_millis,
                None if request.kind == "actor" else 0,
                True,
                {},
            )
            write_document(directory / "result.json", asdict(result), immutable=True)
            self._commit(request, result, elapsed)
            recovered += 1
        return recovered

    def run_next(self, dispatcher):
        if self.plan.sha256 != self.plan_sha256:
            raise ValueError("mutable run plan drift")
        if self.db.execute("SELECT COUNT(*) FROM attempts WHERE status='RUNNING'").fetchone()[0]:
            raise ValueError("recover existing owned attempt before advancing loop")
        row = self.db.execute(
            "SELECT id,spec FROM operations WHERE status!='COMPLETE' ORDER BY ordinal LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        operation_id, spec = row[0], json.loads(row[1])
        if spec["kind"] == "promotion":
            panels = {target: [] for target in ("incumbent", "candidate")}
            for raw, raw_result in self.db.execute(
                "SELECT spec,result FROM operations WHERE status='COMPLETE'"
            ):
                prior = json.loads(raw)
                if (
                    prior["phase"] == "eval"
                    and prior["member"] == spec["member"]
                    and prior["round"] == spec["round"]
                ):
                    value = json.loads(raw_result)
                    panels[prior["target"]].append(
                        dict(
                            opponent=prior["opponent"],
                            map_name=prior["map_name"],
                            seed=prior["seed"],
                            provenance="evaluator",
                            verdict=value["verdict"],
                        )
                    )
            result = evaluate_promotion(self.plan, panels["incumbent"], panels["candidate"])
            self.db.execute("BEGIN IMMEDIATE")
            try:
                if result["promoted"]:
                    state = self._state("round:" + str(spec["round"]) + ":" + spec["member"])
                    promoted = self._state("promoted")
                    promoted[spec["member"]] = state["candidate"]
                    self._set_state("promoted", promoted)
                self.db.execute(
                    "UPDATE operations SET status='COMPLETE',result=? WHERE id=?",
                    (json.dumps(result, sort_keys=True), operation_id),
                )
                self._event("PROMOTION_EVALUATED", dict(operation_id=operation_id, result=result))
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            return result
        if not self.fixture:
            execution = (
                self.plan.actor_execution
                if spec["kind"] == "actor"
                else self.plan.learner_execution
            )
            verify_stage_gates(
                TrainingIdentity(**self._member(spec["member"]).identity),
                self.plan_sha256,
                spec["kind"],
                execution,
            )
        request, count = self._request(operation_id, spec)
        directory = Path(request.output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        write_document(directory / "request.json", asdict(request), immutable=True)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,'RUNNING',NULL)",
                (
                    request.attempt_id,
                    operation_id,
                    count,
                    json.dumps(asdict(request), sort_keys=True),
                ),
            )
            self.db.execute("UPDATE operations SET status='RUNNING' WHERE id=?", (operation_id,))
            self._event(
                "WORKER_RESERVED",
                dict(attempt_id=request.attempt_id, request_sha256=request.sha256),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        write_document(
            self.root / "dispatch_lease.json",
            dict(
                request_sha256=request.sha256,
                operation_id=operation_id,
                attempt_id=request.attempt_id,
                claim_token=request.claim_token,
                owner=request.owner,
                request_path=str(directory / "request.json"),
            ),
        )
        if (
            request.kind == "actor"
            and self.scheduler.claim(request.claim_token, job_id=request.payload["job"]["job_id"])
            != request.payload["job"]
        ):
            raise ValueError("exact operation actor job claim failed")
        verify_worker_lease(request)
        if not self.fixture:
            authorize_worker(request)
        started = time.monotonic()
        result = dispatcher(request)
        verify_worker_lease(request)
        elapsed = math.ceil((time.monotonic() - started) * 1000)
        result.validate(request)
        if not (directory / "result.json").exists():
            write_document(directory / "result.json", asdict(result), immutable=True)
        elif read_document(directory / "result.json") != asdict(result):
            raise ValueError("dispatcher immutable result drift")
        return self._commit(request, result, elapsed)

    def report(self):
        statuses = dict(self.db.execute("SELECT status,COUNT(*) FROM operations GROUP BY status"))
        scheduler = self.scheduler.report()
        usage = self._usage()
        caps = {
            name: getattr(self.plan.budget, name)
            for name in (
                "actor_compute_millis",
                "evaluation_compute_millis",
                "learner_compute_millis",
            )
        }
        exceeded = {name: usage[name] > cap for name, cap in caps.items()}
        return dict(
            schema="poorstar_v3_league_loop_report_v1",
            identity_sha256=self.identity.sha256,
            plan_sha256=self.plan_sha256,
            matched_budget_sha256=self.plan.matched_budget_sha256,
            operations=statuses,
            scheduler=scheduler,
            usage=usage,
            compute_caps=caps,
            compute_exceeded=exceeded,
            current=self._state("current"),
            promoted=self._state("promoted"),
            training_history=self._state("training_history"),
            fixture=self.fixture,
            engine_started=not self.fixture
            and self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] > 0,
            run_complete=statuses.get("COMPLETE", 0)
            == self.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0],
            capability_evaluated=False,
            science_h4_e6_confirmed=False,
            pipeline_stage_unlocked=False,
        )

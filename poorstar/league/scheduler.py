"""League: scheduler."""

import json
import math
import random
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from poorstar.training.contracts import digest


@dataclass(frozen=True)
class LeagueConfig:
    seed: int
    training_game_loops: int
    evaluation_game_loops: int
    maximum_job_loops: int
    population_limit: int
    pfsp_power: float = 2.0
    probability_floor: float = 0.01
    kind: str = "ordinary"

    def __post_init__(self):
        for n in (
            "seed",
            "training_game_loops",
            "evaluation_game_loops",
            "maximum_job_loops",
            "population_limit",
        ):
            if type(getattr(self, n)) is not int or getattr(self, n) < (0 if n == "seed" else 1):
                raise ValueError("explicit population/frame budgets required")
        if self.kind not in ("ordinary", "intent_specialized"):
            raise ValueError("declared matched league arm required")
        if (
            not math.isfinite(self.pfsp_power)
            or self.pfsp_power <= 0
            or not 0 < self.probability_floor <= 1
        ):
            raise ValueError("finite declared PFSP coefficients required")

    @property
    def sha256(self):
        return digest(asdict(self))


class LeagueScheduler:
    def __init__(self, path, registry, identity, config):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.registry, self.identity, self.config = registry, identity, config
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL, "
            "spec TEXT NOT NULL, status TEXT NOT NULL, owner TEXT, started REAL, result TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS payoff (provenance TEXT, learner TEXT, opponent TEXT, "
            "games INTEGER, score REAL, PRIMARY KEY(provenance,learner,opponent))"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS population (snapshot_id TEXT PRIMARY KEY)")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            binding = json.dumps(
                dict(identity_sha256=identity.sha256, config=asdict(config)), sort_keys=True
            )
            old = self.db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
            if old and old[0] != binding:
                raise ValueError("league identity/budget drift")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('binding',?)", (binding,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def close(self):
        self.db.close()

    def payoff(self, learner, opponent, *, provenance="learner"):
        if provenance not in ("learner", "evaluator"):
            raise ValueError("explicit payoff provenance required")
        row = self.db.execute(
            "SELECT games,score FROM payoff WHERE provenance=? AND learner=? AND opponent=?",
            (provenance, learner, opponent),
        ).fetchone()
        return dict(games=row[0], win_rate=row[1] / row[0]) if row else dict(games=0, win_rate=0.5)

    def _spent_and_reserved(self, provenance):
        spent = reserved = 0
        for spec, status, result in self.db.execute("SELECT spec,status,result FROM jobs"):
            spec = json.loads(spec)
            if spec["provenance"] != provenance:
                continue
            if status in ("QUEUED", "RUNNING"):
                reserved += spec["maximum_loops"]
            elif result is not None:
                spent += json.loads(result)["charged_loops"]
        return spent, reserved

    def schedule(
        self,
        learner_id,
        opponent_ids,
        *,
        provenance="learner",
        map_name,
        seed=None,
        operation_key=None,
        fixed_opponent=None,
    ):
        if provenance not in ("learner", "evaluator") or not map_name or not opponent_ids:
            raise ValueError("explicit job provenance, map and opponent pool required")
        learner = self.registry.read(learner_id)
        opponents = tuple(sorted(set(opponent_ids)))
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("nonnegative actual engine/sampling seed required")
        if fixed_opponent is not None and (
            provenance != "evaluator" or fixed_opponent not in opponents
        ):
            raise ValueError("fixed panel opponent requires evaluator provenance")
        if learner_id in opponents:
            raise ValueError("self-match requires separately registered player snapshot roles")
        for candidate in (learner, *(self.registry.read(i) for i in opponents)):
            if (
                candidate["scope_sha256"] != self.identity.scope_sha256
                or candidate["matchup"] != "ZvZ"
                or candidate["profile_sha256"] != self.identity.profile_sha256
            ):
                raise ValueError("snapshot scope/action-domain mismatch")
            if self.config.kind == "ordinary" and candidate["role"] == "intent_specialist":
                raise ValueError("specialist may not enter ordinary league arm")
        if len(set(opponents) | {learner_id}) > self.config.population_limit:
            raise ValueError("matched population budget exceeded")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if operation_key is not None:
                for (raw,) in self.db.execute("SELECT spec FROM jobs"):
                    existing = json.loads(raw)
                    if existing.get("operation_key") == operation_key:
                        if (
                            existing["learner"] != learner_id
                            or existing["opponent"] not in opponents
                            or existing["provenance"] != provenance
                            or existing["map_name"] != map_name
                            or (seed is not None and existing["seed"] != seed)
                            or (
                                fixed_opponent is not None
                                and existing["opponent"] != fixed_opponent
                            )
                        ):
                            raise ValueError("idempotent operation scheduling drift")
                        self.db.execute("COMMIT")
                        return existing
            spent, reserved = self._spent_and_reserved(provenance)
            enrolled = {r[0] for r in self.db.execute("SELECT snapshot_id FROM population")}
            population = enrolled | set(opponents) | {learner_id}
            if len(population) > self.config.population_limit:
                raise ValueError("global registered snapshot population budget exceeded")
            self.db.executemany(
                "INSERT OR IGNORE INTO population VALUES (?)", [(s,) for s in population]
            )
            budget = (
                self.config.training_game_loops
                if provenance == "learner"
                else self.config.evaluation_game_loops
            )
            if spent + reserved + self.config.maximum_job_loops > budget:
                raise ValueError("league frame budget fully reserved/exhausted")
            ordinal = self.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            rng = random.Random(f"{self.config.seed}:{ordinal}")
            # Evaluation pool is fixed by the caller and uniform. Its results
            # never affect training PFSP weights.
            weights = [
                1.0
                if provenance == "evaluator"
                else max(
                    self.config.probability_floor,
                    (1 - self.payoff(learner_id, o)["win_rate"]) ** self.config.pfsp_power,
                )
                for o in opponents
            ]
            opponent_id = fixed_opponent or rng.choices(opponents, weights=weights, k=1)[0]
            spec = dict(
                schema="poorstar_v3_league_job_v1",
                identity_sha256=self.identity.sha256,
                league_config_sha256=self.config.sha256,
                learner=learner_id,
                opponent=opponent_id,
                provenance=provenance,
                map_name=map_name,
                seed=rng.randrange(2**31) if seed is None else seed,
                maximum_loops=self.config.maximum_job_loops,
                ordinal=ordinal,
            )
            if operation_key is not None:
                spec["operation_key"] = operation_key
            job_id = digest(spec)
            spec["job_id"] = job_id
            self.db.execute(
                "INSERT INTO jobs VALUES (?,?,?,'QUEUED',NULL,NULL,NULL)",
                (job_id, ordinal, json.dumps(spec, sort_keys=True)),
            )
            self.db.execute("COMMIT")
            return spec
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def claim(self, owner, *, job_id=None):
        if not isinstance(owner, str) or not owner:
            raise ValueError("explicit actor ownership required")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = (
                self.db.execute(
                    "SELECT id,spec FROM jobs WHERE status='QUEUED' ORDER BY ordinal LIMIT 1"
                ).fetchone()
                if job_id is None
                else self.db.execute(
                    "SELECT id,spec FROM jobs WHERE status='QUEUED' AND id=?", (job_id,)
                ).fetchone()
            )
            if row:
                self.db.execute(
                    "UPDATE jobs SET status='RUNNING',owner=?,started=? WHERE id=?",
                    (owner, time.time(), row[0]),
                )
            self.db.execute("COMMIT")
            return None if row is None else json.loads(row[1])
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def job(self, job_id):
        row = self.db.execute(
            "SELECT spec,status,owner,result FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown league job")
        return dict(
            spec=json.loads(row[0]),
            status=row[1],
            owner=row[2],
            result=None if row[3] is None else json.loads(row[3]),
        )

    def finish(self, job_id, owner, *, verdict, environment_loops=None, evidence_sha256, error=""):
        from poorstar.training.contracts import is_sha

        if verdict not in ("win", "loss", "tie", "truncated", "error") or not is_sha(
            evidence_sha256
        ):
            raise ValueError("resolved outcome/infrastructure result and evidence required")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT spec,status,owner FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row or row[1:] != ("RUNNING", owner):
                raise ValueError("job not running under this actor owner")
            spec = json.loads(row[0])
            if environment_loops is None and verdict not in ("truncated", "error"):
                raise ValueError("natural result requires actual measured frame count")
            if environment_loops is not None and (
                type(environment_loops) is not int
                or not 0 <= environment_loops <= spec["maximum_loops"]
            ):
                raise ValueError("actual frames outside reserved job budget")
            charged = spec["maximum_loops"] if environment_loops is None else environment_loops
            result = dict(
                verdict=verdict,
                environment_loops=environment_loops,
                charged_loops=charged,
                evidence_sha256=evidence_sha256,
                error=error,
            )
            status = "COMPLETE" if verdict in ("win", "loss", "tie") else "INCOMPLETE"
            self.db.execute(
                "UPDATE jobs SET status=?,result=? WHERE id=?",
                (status, json.dumps(result, sort_keys=True), job_id),
            )
            if status == "COMPLETE":
                score = {"win": 1.0, "tie": 0.5, "loss": 0.0}[verdict]
                for a, b, value in (
                    (spec["learner"], spec["opponent"], score),
                    (spec["opponent"], spec["learner"], 1 - score),
                ):
                    self.db.execute(
                        "INSERT INTO payoff VALUES (?,?,?,1,?) ON CONFLICT(provenance,learner,opponent) "
                        "DO UPDATE SET games=games+1,score=score+excluded.score",
                        (spec["provenance"], a, b, value),
                    )
            self.db.execute("COMMIT")
            return result
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def report(self):
        statuses = dict(self.db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
        return dict(
            schema="poorstar_v3_league_scheduler_report_v1",
            identity_sha256=self.identity.sha256,
            config_sha256=self.config.sha256,
            statuses=statuses,
            registered_snapshot_population=self.db.execute(
                "SELECT COUNT(*) FROM population"
            ).fetchone()[0],
            budgets={
                p: dict(zip(("spent", "reserved"), self._spent_and_reserved(p)))
                for p in ("learner", "evaluator")
            },
            capability_evaluated=False,
            promoted=False,
        )

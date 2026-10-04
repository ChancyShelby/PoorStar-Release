"""Evaluation: evaluation."""

import math
import random
from dataclasses import asdict, dataclass

from poorstar.training.contracts import digest, is_sha


def _finite(values):
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("finite measured scalars required")


@dataclass(frozen=True)
class PairedResult:
    comparison: str
    treatment: str
    seed: int
    opponent_sha256: str
    map_name: str
    scenario_sha256: str
    checkpoint_sha256: str
    intervention_loop: int
    shared_prefix_sha256: str
    intervention_observation_sha256: str
    metrics: dict
    evidence_sha256: str
    provenance: str = "evaluator"
    status: str = "COMPLETE"
    foundation_sha256: str | None = None

    def validate(self):
        if (
            not self.comparison
            or not self.treatment
            or not self.map_name
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.intervention_loop) is not int
            or self.intervention_loop < 0
            or self.provenance != "evaluator"
            or self.status not in ("COMPLETE", "ERROR", "TRUNCATED")
        ):
            raise ValueError("independent exact paired evaluator identity required")
        for name in (
            "opponent_sha256",
            "scenario_sha256",
            "checkpoint_sha256",
            "shared_prefix_sha256",
            "intervention_observation_sha256",
            "evidence_sha256",
        ):
            if not is_sha(getattr(self, name)):
                raise ValueError("content-bound paired evidence required: " + name)
        if not isinstance(self.metrics, dict) or any(
            not isinstance(k, str) or not k for k in self.metrics
        ):
            raise ValueError("named measured metrics required")
        _finite(self.metrics.values())
        if self.foundation_sha256 is not None and not is_sha(self.foundation_sha256):
            raise ValueError("content-bound shared foundation required")

    @property
    def pair_key(self):
        return (
            self.comparison,
            self.seed,
            self.opponent_sha256,
            self.map_name,
            self.scenario_sha256,
            self.checkpoint_sha256,
            self.intervention_loop,
            self.shared_prefix_sha256,
            self.intervention_observation_sha256,
        )


def paired_effects(
    results,
    *,
    baseline,
    treatment,
    metric,
    direction,
    noninferiority_margin=None,
    bootstrap_seed=0,
    bootstrap_repeats=1000,
    pairing_mode="SAME_CHECKPOINT_CAUSAL",
):
    """Cluster by training/evaluation seed, not by correlated games of one seed."""
    if direction not in (-1, 1) or type(bootstrap_repeats) is not int or bootstrap_repeats < 1:
        raise ValueError("frozen effect direction and bootstrap rule required")
    results = tuple(results)
    if baseline == treatment or len({r.comparison for r in results}) != 1:
        raise ValueError("one explicit comparison and two distinct treatments required")
    if noninferiority_margin is not None:
        _finite([noninferiority_margin])
        if noninferiority_margin < 0:
            raise ValueError("nonnegative preregistered noninferiority margin required")
    if pairing_mode not in ("SAME_CHECKPOINT_CAUSAL", "TRAINED_ARM_MATCHED_SCENARIO"):
        raise ValueError("explicit frozen causal or trained-arm comparison required")
    groups = {}
    all_evidence = []
    for result in results:
        result.validate()
        if result.treatment not in (baseline, treatment):
            raise ValueError("unexpected treatment in frozen comparison")
        if pairing_mode == "TRAINED_ARM_MATCHED_SCENARIO":
            if result.foundation_sha256 is None:
                raise ValueError("trained-arm comparison requires exact common foundation")
            key = (
                result.comparison,
                result.seed,
                result.opponent_sha256,
                result.map_name,
                result.scenario_sha256,
                result.foundation_sha256,
                result.intervention_loop,
            )
        else:
            key = result.pair_key
        group = groups.setdefault(key, {})
        if result.treatment in group:
            raise ValueError("duplicate paired result cannot increase sample size")
        group[result.treatment] = result
        all_evidence.append(result.evidence_sha256)
    if not groups or any(set(g) != {baseline, treatment} for g in groups.values()):
        raise ValueError(
            "complete baseline/treatment paired matrix required; missing games cannot be silently dropped"
        )
    pairs = []
    failures = []
    for key, group in sorted(groups.items()):
        a, b = group[baseline], group[treatment]
        if a.status != "COMPLETE" or b.status != "COMPLETE":
            failures.append(
                dict(key=list(key), baseline_status=a.status, treatment_status=b.status)
            )
            continue
        if metric not in a.metrics or metric not in b.metrics:
            raise ValueError("required paired metric unavailable")
        pairs.append((a.seed, direction * (b.metrics[metric] - a.metrics[metric])))
    seeds = sorted({seed for seed, _ in pairs})
    per_seed = {
        seed: sum(x for s, x in pairs if s == seed) / sum(s == seed for s, _ in pairs)
        for seed in seeds
    }
    mean = None if not seeds else sum(per_seed.values()) / len(seeds)
    interval = None
    if len(seeds) > 1 and not failures:
        rng = random.Random(bootstrap_seed)
        samples = sorted(
            sum(per_seed[rng.choice(seeds)] for _ in seeds) / len(seeds)
            for _ in range(bootstrap_repeats)
        )
        interval = [
            samples[int(0.025 * (len(samples) - 1))],
            samples[int(0.975 * (len(samples) - 1))],
        ]
    return dict(
        status="INCOMPLETE_OR_FAILED_MATRIX" if failures else "MEASURED",
        metric=metric,
        pairing_mode=pairing_mode,
        direction=direction,
        pairs=len(pairs),
        seed_clusters=len(seeds),
        seed_means=per_seed,
        mean_directed_effect=mean,
        cluster_bootstrap_95_interval=interval,
        noninferiority_margin=noninferiority_margin,
        noninferiority_pass=None
        if interval is None or noninferiority_margin is None
        else interval[0] >= -noninferiority_margin,
        failed_pairs=failures,
        all_evidence_sha256=sorted(all_evidence),
        hypothesis_proven=False,
    )


def dose_and_cross_effects(results, *, baseline, treatments, metric_directions):
    """treatments maps treatment IDs to declared (dimension, dose); no relabeling."""
    if not treatments or not metric_directions:
        raise ValueError("frozen preference doses and semantic metrics required")
    for dimension, dose in treatments.values():
        if not dimension or not math.isfinite(dose) or not 0 <= dose <= 1:
            raise ValueError("named bounded Preference dose required")
    allowed = {baseline, *treatments}
    if any(r.treatment not in allowed for r in results):
        raise ValueError("undeclared intervention cannot enter cross-effect matrix")
    cells = {}
    by_dimension = {}
    for treatment, (dimension, dose) in sorted(treatments.items()):
        if (dimension, dose) in by_dimension:
            raise ValueError("duplicate dimension/dose registration")
        by_dimension[(dimension, dose)] = treatment
        for metric, direction in metric_directions.items():
            cell = paired_effects(
                [r for r in results if r.treatment in (baseline, treatment)],
                baseline=baseline,
                treatment=treatment,
                metric=metric,
                direction=direction,
            )
            cells[treatment + ":" + metric] = cell
    monotonicity = {}
    for dimension in sorted({x[0] for x in treatments.values()}):
        doses = sorted(
            (dose, treatment) for treatment, (d, dose) in treatments.items() if d == dimension
        )
        for metric in metric_directions:
            values = [cells[t + ":" + metric]["mean_directed_effect"] for _, t in doses]
            monotonicity[dimension + ":" + metric] = (
                None
                if any(x is None for x in values)
                or any(cells[t + ":" + metric]["status"] != "MEASURED" for _, t in doses)
                else all(b >= a for a, b in zip(values, values[1:]))
            )
    return dict(cells=cells, directed_dose_monotonicity=monotonicity, hypothesis_proven=False)


def intent_dynamics(records, *, preference_names, probability_bin_edges, oscillation_distance):
    """Actual trainable choices only; forced keeps do not inflate frequency."""
    names = tuple(preference_names)
    edges = tuple(probability_bin_edges)
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or tuple(sorted(set(edges))) != edges:
        raise ValueError("frozen complete preference bins required")
    _finite([oscillation_distance])
    if oscillation_distance < 0:
        raise ValueError("nonnegative frozen oscillation threshold required")
    episodes = {}
    evidence = []
    expected = {"episode", "loop", "preference", "operation", "manager_decision", "evidence_sha256"}
    for record in records:
        if (
            set(record) != expected
            or not record["episode"]
            or not is_sha(record["evidence_sha256"])
        ):
            raise ValueError("explicit causal manager trace required")
        if (
            type(record["loop"]) is not int
            or record["loop"] < 0
            or type(record["manager_decision"]) is not bool
        ):
            raise ValueError("manager clocks and actual eligible decision flag required")
        if record["operation"] not in ("keep", "update", "terminate", "external"):
            raise ValueError("declared manager operation required")
        if len(record["preference"]) != len(names):
            raise ValueError("named Preference dimension mismatch")
        _finite(record["preference"])
        if any(not 0 <= x <= 1 for x in record["preference"]):
            raise ValueError("Preference outside declared bounds")
        episode = episodes.setdefault(record["episode"], [])
        if episode and record["loop"] <= episode[-1]["loop"]:
            raise ValueError("duplicate/nonchronological manager trace")
        episode.append(record)
        evidence.append(record["evidence_sha256"])
    updates = []
    eligible = 0
    forced = 0
    duration = 0
    oscillations = 0
    candidates = 0
    distance = lambda a, b: sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    for episode in episodes.values():
        duration += episode[-1]["loop"] - episode[0]["loop"]
        eligible += sum(r["manager_decision"] for r in episode)
        forced += sum(r["operation"] == "keep" and not r["manager_decision"] for r in episode)
        values = [
            r["preference"] for r in episode if r["manager_decision"] and r["operation"] == "update"
        ]
        updates += values
        for a, b, c in zip(values, values[1:], values[2:]):
            candidates += 1
            oscillations += distance(a, c) <= oscillation_distance < distance(a, b)
    dimensions = {}
    for i, name in enumerate(names):
        values = [row[i] for row in updates]
        bins = [0] * (len(edges) - 1)
        for value in values:
            index = next(
                j
                for j, (lo, hi) in enumerate(zip(edges, edges[1:]))
                if lo <= value < hi or j == len(bins) - 1 and value == 1
            )
            bins[index] += 1
        dimensions[name] = dict(
            update_count=len(values),
            distinct_values=len(set(values)),
            bins=bins,
            entropy=None
            if not values
            else -sum(n / len(values) * math.log(n / len(values)) for n in bins if n),
            span=None if not values else max(values) - min(values),
        )
    return dict(
        eligible_manager_decisions=eligible,
        forced_keep_receipts=forced,
        learned_updates=len(updates),
        measured_span_loops=duration,
        updates_per_loop=None if duration == 0 else len(updates) / duration,
        aba_oscillations=oscillations,
        aba_candidates=candidates,
        dimensions=dimensions,
        evidence_sha256=evidence,
        collapse_thresholds_applied=False,
    )


def learning_curve(points, *, frame_budget, attainment_threshold):
    if type(frame_budget) is not int or frame_budget < 1:
        raise ValueError("matched positive physical frame budget required")
    _finite([attainment_threshold])
    if not points or points[0][0] != 0 or points[-1][0] != frame_budget:
        raise ValueError("complete fixed-budget curve including frame zero required")
    previous = -1
    for frames, score in points:
        if type(frames) is not int or not previous < frames <= frame_budget:
            raise ValueError("strictly chronological physical environment frame counts required")
        _finite([score])
        previous = frames
    area = sum((b - a) * (x + y) / 2 for (a, x), (b, y) in zip(points, points[1:]))
    reached = next((frames for frames, score in points if score >= attainment_threshold), None)
    return dict(
        frame_budget=frame_budget,
        normalized_auc=area / frame_budget,
        first_measured_threshold_frames=reached,
        threshold=attainment_threshold,
        threshold_censored=reached is None,
        final_score=points[-1][1],
        interpolated_attainment=False,
    )


def task_goal_metrics(events, *, declared_task_ids):
    """Observed confirmations and censoring; absence is not an observed failure."""
    task_ids = tuple(declared_task_ids)
    if not task_ids or len(set(task_ids)) != len(task_ids) or any(not t for t in task_ids):
        raise ValueError("complete predeclared task ledger required")
    by_task = {t: [] for t in task_ids}
    seen = set()
    for event in events:
        if (
            event["task_id"] not in by_task
            or event["outcome"] not in ("progress", "confirmed", "censored")
            or type(event["loop"]) is not int
            or event["loop"] < 0
            or not is_sha(event["evidence_id"])
        ):
            raise ValueError("typed player-visible task evidence required")
        key = (event["task_id"], event["loop"], event["outcome"], event["evidence_id"])
        if key in seen:
            raise ValueError("duplicate task evidence cannot increase confidence")
        seen.add(key)
        by_task[event["task_id"]].append(event)
    confirmed = {
        t: min(e["loop"] for e in rows if e["outcome"] == "confirmed")
        for t, rows in by_task.items()
        if any(e["outcome"] == "confirmed" for e in rows)
    }
    unresolved = sorted(set(task_ids) - set(confirmed))
    return dict(
        declared_tasks=len(task_ids),
        observed_confirmed_tasks=len(confirmed),
        earliest_confirmation_loops=confirmed,
        unresolved_tasks=unresolved,
        explicitly_censored_tasks=sorted(
            t for t in unresolved if any(e["outcome"] == "censored" for e in by_task[t])
        ),
        observed_confirmation_lower_bound=len(confirmed) / len(task_ids),
        complete_window_success_rate=None,
        absence_is_failure=False,
        command_causation_proven=False,
    )


def population_coverage(payoff, *, opponent_ids, cluster_count, iterations=100):
    """Frozen payoff-profile clustering; not full-game exploitability."""
    policies = sorted(payoff)
    opponents = tuple(opponent_ids)
    if (
        not policies
        or not opponents
        or len(set(opponents)) != len(opponents)
        or type(cluster_count) is not int
        or not 1 <= cluster_count <= len(policies)
        or type(iterations) is not int
        or iterations < 1
    ):
        raise ValueError("frozen finite population/opponent/clustering budget required")
    matrix = []
    for policy in policies:
        if set(payoff[policy]) != set(opponents):
            raise ValueError("complete fixed evaluator payoff matrix required")
        row = [payoff[policy][opponent] for opponent in opponents]
        _finite(row)
        if any(not -1 <= x <= 1 for x in row):
            raise ValueError("outcome payoff outside [-1,1]")
        matrix.append(row)
    distance = lambda a, b: sum((x - y) ** 2 for x, y in zip(a, b))
    centers = [matrix[0]]
    for _ in range(1, cluster_count):
        centers.append(max(matrix, key=lambda row: min(distance(row, c) for c in centers)))
    assignments = []
    for _ in range(iterations):
        assignments = [
            min(range(cluster_count), key=lambda j: distance(row, centers[j])) for row in matrix
        ]
        centers = [
            [
                sum(matrix[i][j] for i in range(len(matrix)) if assignments[i] == k)
                / assignments.count(k)
                for j in range(len(opponents))
            ]
            if k in assignments
            else centers[k]
            for k in range(cluster_count)
        ]
    return dict(
        cluster_assignments=dict(zip(policies, assignments)),
        occupied_clusters=len(set(assignments)),
        registered_cluster_count=cluster_count,
        worst_fixed_opponent_payoff={p: min(row) for p, row in zip(policies, matrix)},
        pairwise_squared_profile_distance={
            a + ":" + b: distance(matrix[i], matrix[j])
            for i, a in enumerate(policies)
            for j, b in enumerate(policies)
            if i < j
        },
        full_game_exploitability_measured=False,
    )


def restricted_response_gain(payoff, mixture, *, opponent_ids):
    if set(payoff) != set(mixture) or not mixture:
        raise ValueError("content-bound population mixture required")
    _finite(mixture.values())
    if any(x < 0 for x in mixture.values()) or not math.isclose(
        sum(mixture.values()), 1.0, abs_tol=1e-8
    ):
        raise ValueError("declared normalized policy mixture required")
    opponents = tuple(opponent_ids)
    if not opponents or any(set(row) != set(opponents) for row in payoff.values()):
        raise ValueError("complete registered response-set payoff matrix required")
    for row in payoff.values():
        _finite(row.values())
        if any(not -1 <= x <= 1 for x in row.values()):
            raise ValueError("natural outcome payoff required")
    scores = {o: sum(mixture[p] * payoff[p][o] for p in mixture) for o in opponents}
    return dict(
        worst_registered_response=min(scores, key=scores.get),
        worst_response_payoff=min(scores.values()),
        response_scores=scores,
        uniform_opponent_payoff=sum(scores.values()) / len(scores),
        registered_response_gain=sum(scores.values()) / len(scores) - min(scores.values()),
        interpretation="EMPIRICAL_REGISTERED_RESPONSE_SET_ONLY_NOT_SC2_EXPLOITABILITY",
    )

"""League: promotion."""

import math


def wilson(score, games, z):
    if games <= 0:
        raise ValueError("nonempty independent evaluator cell required")
    p = score / games
    denominator = 1 + z * z / games
    middle = (p + z * z / (2 * games)) / denominator
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return max(0.0, middle - half), min(1.0, middle + half)


def evaluate_promotion(plan, incumbent_records, candidate_records):
    expected = {
        (opponent, map_name, seed)
        for opponent in plan.anchors + plan.historical
        for map_name in plan.maps
        for seed in plan.evaluation_seeds
    }

    def panel(records):
        values = {}
        for record in records:
            key = (record["opponent"], record["map_name"], record["seed"])
            if key in values or key not in expected or record.get("provenance") != "evaluator":
                raise ValueError("duplicate/foreign/training result in frozen evaluation panel")
            if record.get("verdict") not in ("win", "loss", "tie"):
                return None
            values[key] = {"win": 1.0, "tie": 0.5, "loss": 0.0}[record["verdict"]]
        return values if values.keys() == expected else None

    old, new = panel(incumbent_records), panel(candidate_records)
    if old is None or new is None:
        return dict(
            status="NOT_EVALUATED",
            promoted=False,
            reason="incomplete natural-terminal evaluator panel",
        )
    cells = []
    passed = True
    for opponent in plan.anchors + plan.historical:
        for map_name in plan.maps:
            keys = [(opponent, map_name, seed) for seed in plan.evaluation_seeds]
            count = len(keys)
            old_score = sum(old[k] for k in keys)
            new_score = sum(new[k] for k in keys)
            old_interval = wilson(old_score, count, plan.promotion.wilson_z)
            new_interval = wilson(new_score, count, plan.promotion.wilson_z)
            anchor = opponent in plan.anchors
            threshold = (
                plan.promotion.minimum_anchor_win_rate
                if anchor
                else plan.promotion.minimum_history_win_rate
            )
            # Conservative independent intervals; shared seed panel is retained
            # for downstream paired scientific statistics, not invented here.
            gate = count >= plan.promotion.minimum_games_per_cell and new_interval[0] >= threshold
            if anchor:
                gate = (
                    gate
                    and new_interval[0] - old_interval[1]
                    >= -plan.promotion.anchor_noninferiority_margin
                )
            passed &= gate
            cells.append(
                dict(
                    opponent=opponent,
                    map_name=map_name,
                    games=count,
                    anchor=anchor,
                    incumbent_score=old_score,
                    candidate_score=new_score,
                    incumbent_interval=old_interval,
                    candidate_interval=new_interval,
                    gate="PASS" if gate else "FAIL",
                )
            )
    return dict(
        status="PASS" if passed else "FAIL",
        promoted=passed,
        cells=cells,
        science_h4_e6_confirmed=False,
        pipeline_stage_unlocked=False,
    )

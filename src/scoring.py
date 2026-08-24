from typing import Mapping


DEFAULT_SCORING = {
    "strikeouts": 3.0,
    "outs": 1.0,
    "hits": -1.3,
    "earned_runs": -3.0,
    "walks": -1.3,
    "hit_batters": -1.3,
}


def calculate_points(predictions, scoring=None):
    """Calculate fantasy points from component predictions."""

    scoring_values = DEFAULT_SCORING.copy()

    if scoring is not None:
        scoring_values.update(scoring)

    required = ["strikeouts", "outs", "hits", "earned_runs", "walks", "hit_batters"]

    missing = [key for key in required if key not in predictions]

    if missing:
        raise ValueError(f"Missing prediction components required for scoring: {missing}")

    points = 0.0

    for component in required:
        value = predictions[component]
        points += float(value) * float(scoring_values[component])

    return points
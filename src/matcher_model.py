"""The trained matcher as plain coefficients: encode a pair's features and apply a logistic function.

Training uses scikit-learn (train_matcher.py). Serving needs nothing but this file and the coefficients
JSON, the same split nomenklatura's RegressionV1 makes.

Inputs, twelve per pair:
  five name scores in [0, 1]: name_score, skeleton_score, weighted_score, phonetic_score, arabic_score
  seven indicators for the attribute states: birth_agree, birth_close, birth_conflict, country_agree,
  country_conflict, identifier_agree, identifier_conflict. "missing" is the reference level, so a
  pair with nothing to compare scores on the name scores alone.

Standard library only.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "pairs_lr_v1.json"

NAME_SCORES = ("name_score", "skeleton_score", "weighted_score", "phonetic_score", "arabic_score")
STATE_LEVELS = {
    "birth_state": ("agree", "close", "conflict"),
    "country_state": ("agree", "conflict"),
    "identifier_state": ("agree", "conflict"),
}
STATE_PREFIX = {"birth_state": "birth", "country_state": "country", "identifier_state": "identifier"}
INPUTS = NAME_SCORES + tuple(
    f"{STATE_PREFIX[state]}_{level}" for state, levels in STATE_LEVELS.items() for level in levels
)
# The eight features of the feature table, and the inputs each one supplies. Dropping a feature drops these.
FEATURE_INPUTS = {name: (name,) for name in NAME_SCORES} | {
    "birth_state": ("birth_agree", "birth_close", "birth_conflict"),
    "country_state": ("country_agree", "country_conflict"),
    "identifier_state": ("identifier_agree", "identifier_conflict"),
}


def encode(features: Mapping[str, float | str]) -> dict[str, float]:
    """The twelve model inputs from a pair's eight features."""
    row = {name: float(features[name]) for name in NAME_SCORES}
    for state, levels in STATE_LEVELS.items():
        for level in levels:
            row[f"{STATE_PREFIX[state]}_{level}"] = 1.0 if features[state] == level else 0.0
    return row


class LogisticMatcher:
    def __init__(self, coefficients: Mapping[str, float], intercept: float) -> None:
        self.coefficients = dict(coefficients)
        self.intercept = intercept

    def probability(self, features: Mapping[str, float | str]) -> float:
        """Probability that the two entities are the same."""
        row = encode(features)
        z = self.intercept + sum(weight * row[name] for name, weight in self.coefficients.items())
        return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "LogisticMatcher":
        data = json.loads(path.read_text())
        return cls(data["coefficients"], data["intercept"])

"""Tests for the threshold-tuning helpers in src/evaluate_screening.py, on hand-built rows.

Run:
    pytest tests/test_evaluate_screening.py
"""

import pytest

import evaluate_screening as es


def row(script, best, group=None):
    """A negative row as final_scores returns it: (kind, group, true score, best score, top-1, script)."""
    return ("synthetic", group or script, None, best, False, script)


def test_rate():
    rows = [row("Latin", s) for s in (0.9, 0.8, 0.7, 0.6)]
    assert es.rate(rows, 0.7) == 0.75  # 0.9, 0.8 and 0.7 reach it
    assert es.rate(rows, 1.0) == 0.0
    assert es.rate([], 0.5) == 0.0


def test_choose_thresholds_takes_the_lowest_threshold_at_or_under_the_target():
    rows = [row("Latin", s) for s in (0.9, 0.8, 0.7, 0.6)]  # 1 of 4 reaches 0.81 to 0.90
    rows += [row("Arabic", 1.0)] * 4  # perfect scores can never be alerted away
    rows += [row("other", 0.3)]  # never reaches the grid
    chosen = es.choose_thresholds(rows, target=0.25)
    assert chosen["Latin"] == 0.81
    assert chosen["Arabic"] == 1.0  # even a threshold of 1.0 keeps all four: the end of the grid
    assert chosen["other"] == 0.5  # already at or under the target at the lowest threshold
    assert chosen["Cyrillic"] == 0.5  # no negatives at all


def test_spread_ignores_missing_scripts():
    fpr = {"Latin": 0.1, "Cyrillic": 0.3, "Arabic": 0.2, "other": None}
    assert es.spread(fpr, ("Latin", "Cyrillic", "Arabic", "other")) == pytest.approx(0.2)
    assert es.spread({"Latin": None}, ("Latin",)) == 0.0

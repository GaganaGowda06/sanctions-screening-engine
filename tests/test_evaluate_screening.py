"""Tests for the threshold-tuning helpers in src/evaluate_screening.py, on hand-built rows.

Run:
    pytest tests/test_evaluate_screening.py
"""

import pytest

import evaluate_screening as es


def pos(script, score, group="variant"):
    """A positive row as final_scores returns it: (group, group, true score, best other, top-1, script)."""
    return (group, group, score, 0.0, True, script)


def neg(script, best):
    """A negative row: (kind, group, true score, best score, top-1, script)."""
    return ("synthetic", script, None, best, False, script)


def test_recall_at_counts_headline_positives_of_one_script():
    rows = [pos("Latin", 0.95), pos("Latin", 0.9), pos("Latin", 0.85), pos("Cyrillic", 0.7)]
    rows += [pos("Latin", 0.99, group="other_to_latin")]  # not a headline group
    assert es.recall_at(rows, "Latin", 0.9) == pytest.approx(2 / 3)
    assert es.recall_at(rows, "Cyrillic", 0.9) == 0.0
    assert es.recall_at(rows, "Arabic", 0.5) == 0.0  # no positives


def test_fpr_at():
    rows = [neg("Latin", s) for s in (0.92, 0.8, 0.7, 0.6)] + [neg("Arabic", 1.0)]
    assert es.fpr_at(rows, "Latin", 0.8) == 0.5  # 0.92 and 0.8 reach it
    assert es.fpr_at(rows, "Latin", 0.99) == 0.0
    assert es.fpr_at(rows, "Cyrillic", 0.5) == 0.0  # no negatives


def test_script_threshold_has_the_lowest_fpr_that_keeps_recall():
    pos_rows = [pos("Latin", s) for s in (0.95, 0.9, 0.85)]  # at 0.9: 2 of 3 found
    neg_rows = [neg("Latin", s) for s in (0.92, 0.8, 0.7, 0.6)]
    chosen = es.choose_script_thresholds(pos_rows, neg_rows, global_threshold=0.9)
    # Recall stays at 2/3 or better up to 0.90. The false positive rate is lowest (1 of 4) from 0.81 to 0.90,
    # and the lower end of that stretch is taken: it finds all 3 positives at the same false positive rate.
    assert chosen["Latin"] == 0.81


def test_script_threshold_never_reaches_a_perfect_only_score():
    pos_rows = [pos("Arabic", 1.0), pos("Arabic", 1.0)]
    neg_rows = [neg("Arabic", 1.0), neg("Arabic", 0.6), neg("Arabic", 0.5)]
    chosen = es.choose_script_thresholds(pos_rows, neg_rows, global_threshold=0.87)
    # Every positive is a perfect score, so any threshold up to 0.99 keeps recall. The perfect-score
    # false positive cannot be removed, and the rest are gone from 0.61.
    assert chosen["Arabic"] == 0.61
    assert max(es.ev.THRESHOLDS) == 0.99 and chosen["Arabic"] < 1.0


def test_other_scripts_and_scripts_without_positives_use_the_global_threshold():
    chosen = es.choose_script_thresholds([pos("Latin", 0.9)], [neg("Latin", 0.8), neg("other", 0.9)], 0.87)
    assert chosen["other"] == 0.87 and chosen["Cyrillic"] == 0.87 and chosen["Arabic"] == 0.87


def test_a_script_keeps_the_global_threshold_when_nothing_better_keeps_recall():
    # Recall drops as soon as the threshold rises, so 0.87 is the highest feasible; below it more false positives.
    pos_rows = [pos("Cyrillic", 0.87), pos("Cyrillic", 0.99)]
    neg_rows = [neg("Cyrillic", 0.86), neg("Cyrillic", 0.5)]
    assert es.choose_script_thresholds(pos_rows, neg_rows, 0.87)["Cyrillic"] == 0.87


def test_spread_ignores_missing_scripts():
    fpr = {"Latin": 0.1, "Cyrillic": 0.3, "Arabic": 0.2, "other": None}
    assert es.spread(fpr, ("Latin", "Cyrillic", "Arabic", "other")) == pytest.approx(0.2)
    assert es.spread({"Latin": None}, ("Latin",)) == 0.0


def test_view_swaps_in_the_components_of_one_blend():
    q = {"name": "x", "comps_0.5": ["half"], "comps_1.0": ["whole"]}
    assert es.view([q], 0.5)[0]["comps"] == ["half"] and es.view([q], 1.0)[0]["comps"] == ["whole"]
    assert "comps" not in q  # the original is untouched

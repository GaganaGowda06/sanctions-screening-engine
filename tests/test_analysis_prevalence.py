"""Tests for the prevalence arithmetic (analysis/prevalence.py), worked by hand.

Run:
    pytest tests/test_analysis_prevalence.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import prevalence  # noqa: E402


def test_precision_at_half_prevalence_is_the_headline_precision():
    # 2,000 listed, 1,474 hit, 132 wrong-entity; 2,000 clean, 187 false alerts: the measured 1474 / 1793.
    r, w, f = 1474 / 2000, 132 / 2000, 187 / 2000
    row = prevalence.rates_at(r, w, f, 0.5)
    assert row["precision"] == pytest.approx(1474 / 1793, abs=1e-12)
    assert row["alert_rate"] == pytest.approx((1474 + 132 + 187) / 4000, abs=1e-12)


def test_a_hand_worked_example():
    # r = 0.8, w = 0.1, f = 0.05, p = 0.01: true 0.008, wrong 0.001, false 0.0495 -> precision 0.008 / 0.0585
    row = prevalence.rates_at(0.8, 0.1, 0.05, 0.01)
    assert row["precision"] == pytest.approx(0.008 / 0.0585, abs=1e-12)
    assert row["alert_rate"] == pytest.approx(0.01 * 0.9 + 0.99 * 0.05, abs=1e-12)
    per = row["per_100000_customers"]
    assert per["listed_parties"] == pytest.approx(1000) and per["true_alerts"] == pytest.approx(800)
    assert per["wrong_entity_alerts"] == pytest.approx(100) and per["false_alerts_on_clean_customers"] == pytest.approx(
        4950
    )
    assert per["alerts_to_review"] == pytest.approx(5850)
    assert row["alerts_reviewed_per_true_hit"] == pytest.approx(5850 / 800)


def test_precision_falls_as_prevalence_falls_with_the_rates_fixed():
    values = [prevalence.rates_at(0.74, 0.07, 0.09, p)["precision"] for p in (0.5, 1e-2, 1e-3, 1e-4, 1e-5)]
    assert values == sorted(values, reverse=True)
    assert values[-1] < 1e-3  # about one alert in ten thousand is real at one listed customer in 100,000


def test_the_alert_rate_tends_to_the_false_alert_rate_when_nobody_is_listed():
    assert prevalence.rates_at(0.74, 0.07, 0.09, 1e-9)["alert_rate"] == pytest.approx(0.09, abs=1e-8)


def test_the_false_alert_rate_needed_for_a_target_precision_inverts_the_precision_formula():
    for p in (1e-2, 1e-3, 1e-4):
        needed = prevalence.fpr_needed_for_precision(0.74, 0.07, p, 0.5)
        assert prevalence.rates_at(0.74, 0.07, needed, p)["precision"] == pytest.approx(0.5, abs=1e-9)
    assert prevalence.fpr_needed_for_precision(0.74, 0.07, 1e-4, 0.5) < 1e-4  # far under one clean customer in 10,000

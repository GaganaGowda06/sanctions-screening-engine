"""Tests for drift against a baseline (src/audit_drift.py) and the drift CLI command, on hand-built reports.

Run:
    pytest tests/test_audit_drift.py
"""

import json
import math

import numpy as np
import pytest
from test_replay import tree

import audit_drift
import screen_cli
from audit import AuditLog
from audit_drift import (
    BaselineMismatch,
    DriftConfig,
    compare_to_baseline,
    load_baseline,
    psi,
    psi_status,
    save_baseline,
    status_for,
)

CTX = "a" * 64  # run context hashes (the exact commit...)
OTHER = "b" * 64
DEC = "d" * 64  # ...and decision context hashes (what decides decisions)
OTHER_DEC = "e" * 64
EPS = 0.0001
CONFIG = DriftConfig()

# Three fixed score shapes (20 bins, 100 decisions each) so a script's distribution is the same in any batch.
SHAPES = {
    "Latin": [0, 0, 20, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 5],
    "Arabic": [0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 5, 15, 10, 10],
    "Cyrillic": [10, 20, 20, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 5],
}
RATES = {"Latin": 0.25, "Arabic": 0.40, "Cyrillic": 0.20}


def scaled(shape: list[int], n: int) -> list[int]:
    return [c * n // 100 for c in shape]


def group(counts: dict[str, int | str], context: str = CTX, near_rate: float = 0.10, rates: dict | None = None,
          rate_scale: float = 1.0, decision: str = DEC) -> dict:  # fmt: skip
    """A group as compute_metrics writes it: visible cells carry a histogram, hidden cells only a string count."""
    rates = rates or RATES
    per_script, total, bins = {}, 0, [0] * 20
    for script, n in counts.items():
        if isinstance(n, str):
            per_script[script] = {"count": n}
            continue
        counts_here = scaled(SHAPES[script], n)
        per_script[script] = {
            "count": n,
            "alert_rate": rates[script] * rate_scale,
            "score_histogram": {"counts": counts_here, "no_candidates": 0},
        }
        total += n
        bins = [a + b for a, b in zip(bins, counts_here, strict=True)]
    return {
        "run_context_hash": context,
        "decision_context_hash": decision,
        "count": total,
        "alerts": 0,
        "alert_rate": 0.0,
        "score_histogram": {"counts": bins, "no_candidates": 0},
        "near_threshold": {"count": 0, "rate": near_rate},
        "per_script": per_script,
        "duration_ms": {"p50": 1, "p95": 2, "p99": 3},
    }


def report_of(*groups: dict) -> dict:
    return {"report_schema_version": 2, "groups": list(groups)}


def baseline_of(g: dict, context: dict | None = None) -> dict:
    return {
        "baseline_schema_version": 1,
        "decision_context_hash": g["decision_context_hash"],
        "run_context_hash": g["run_context_hash"],
        "context": context,
        "sample": {},
        "report": report_of(g),
    }


BASE_COUNTS = {"Latin": 1200, "Arabic": 400, "Cyrillic": 600}  # every cell well over min_psi_count
SHIFT_COUNTS = {"Latin": 600, "Arabic": 1200, "Cyrillic": 400}  # the same kinds of query, a different mix
BASE = group(BASE_COUNTS)


# ---------- PSI ----------


def test_psi_of_a_hand_worked_example():
    # shares .5 .5 against .25 .75: (.25-.5)ln(.5) + (.75-.5)ln(1.5)
    expected = (0.25 - 0.5) * math.log(0.5) + (0.75 - 0.5) * math.log(1.5)
    assert psi([100, 100, 0], [50, 150, 0], EPS) == pytest.approx(expected, abs=1e-12)
    assert expected == pytest.approx(0.2746530721670274)


def test_an_empty_bin_uses_epsilon_on_both_sides():
    base, live = [50, 50, 0], [50, 0, 50]
    expected = 2 * (0.5 - EPS) * math.log(0.5 / EPS)  # the two bins that disagree contribute the same amount
    assert psi(base, live, EPS) == pytest.approx(expected, abs=1e-12)
    assert psi(base, live, EPS) == pytest.approx(8.5154, abs=1e-3)


def test_psi_is_zero_for_identical_shares_symmetric_and_scale_free():
    assert psi([3, 5, 2], [3, 5, 2], EPS) == 0.0
    assert psi([30, 50, 20], [3, 5, 2], EPS) == pytest.approx(0.0, abs=1e-15)  # only the shares matter
    a, b = [10, 30, 60], [40, 40, 20]
    assert psi(a, b, EPS) == pytest.approx(psi(b, a, EPS), abs=1e-15)
    assert psi([], [1], EPS) is None or psi([0, 0], [1, 1], EPS) is None
    assert psi([0, 0], [1, 1], EPS) is None


def test_status_thresholds_are_applied_at_the_boundaries():
    assert [status_for(v, 0.1, 0.25) for v in (0.0, 0.0999999, 0.1, 0.2499999, 0.25, 3.0)] == [
        "ok", "ok", "warn", "warn", "alert", "alert",
    ]  # fmt: skip


def test_psi_status_is_above_the_limit_not_at_it():
    assert [psi_status(v, 0.1, 0.25) for v in (0.0, 0.0999999, 0.1, 0.1000001, 0.25, 0.2500001)] == [
        "ok", "ok", "ok", "warn", "warn", "alert",
    ]  # fmt: skip


def test_a_computed_psi_exactly_at_the_limits_is_not_past_them(monkeypatch):
    monkeypatch.setattr(audit_drift, "null_distribution", lambda *a, **k: np.zeros(1000))  # no null: floors decide
    live = group(SHIFT_COUNTS)
    value = compare_to_baseline(report_of(live), baseline_of(BASE), CONFIG)["score_psi"]["value"]
    assert value > 0
    for warn, alert, expected in (
        (value, value * 2, "ok"),  # exactly at the warn limit: not above it
        (value * 0.999999, value * 2, "warn"),
        (value / 2, value, "warn"),  # exactly at the alert limit: warn, not alert
        (value / 3, value * 0.999999, "alert"),
    ):
        config = DriftConfig(psi_warn=warn, psi_alert=alert)
        assert compare_to_baseline(report_of(live), baseline_of(BASE), config)["score_psi"]["status"] == expected


def test_alert_rate_status_needs_both_a_ratio_and_an_absolute_difference():
    config = DriftConfig()

    def status(base_rate, live_rate):
        from audit_drift import alert_rate_result

        return alert_rate_result(base_rate, live_rate, config, thin=False)["status"]

    assert status(0.25, 0.5) == "alert"  # ratio exactly 2.0, difference 0.25
    assert status(0.25, 0.4375) == "warn"  # ratio 1.75
    assert status(0.25, 0.3125) == "ok"  # ratio 1.25, below the warn ratio
    assert status(0.25, 0.125) == "alert"  # a fall counts: the ratio taken as larger of r and 1/r
    assert status(0.01, 0.03) == "ok"  # ratio 3, but the difference 0.02 is under the noise floor
    assert status(0.0, 0.2) == "alert" and status(0.2, 0.0) == "alert"
    assert status(0.0, 0.0) == "ok"
    assert status(0.25, 0.25) == "ok"


def test_near_threshold_status_at_the_boundary():
    live = group(BASE_COUNTS, near_rate=0.28125)
    base = group(BASE_COUNTS, near_rate=0.25)
    for warn, alert, expected in ((0.03125, 0.0625, "warn"), (0.0, 0.03125, "alert"), (0.0313, 0.0626, "ok")):
        if warn == 0.0:
            warn = 0.01
        result = compare_to_baseline(report_of(live), baseline_of(base), DriftConfig(
            near_threshold_diff_warn=warn, near_threshold_diff_alert=alert))  # fmt: skip
        assert result["near_threshold"]["abs_diff"] == 0.03125
        assert result["near_threshold"]["status"] == expected


# ---------- the comparison ----------


def test_an_identical_group_is_ok_everywhere():
    result = compare_to_baseline(report_of(BASE), baseline_of(BASE), CONFIG)
    assert result["overall"] == "ok"
    assert result["score_psi"]["value"] == 0.0 and result["input_mix_psi"]["value"] == 0.0
    for script in ("Latin", "Arabic", "Cyrillic"):
        entry = result["per_script"][script]
        assert entry["score_psi"]["value"] == 0.0 and entry["alert_rate"]["abs_diff"] == 0.0
        assert entry["alert_rate"]["ratio"] == 1.0


def test_per_script_alert_rates_are_reported_as_a_difference_and_a_ratio():
    live = group(BASE_COUNTS, rates={"Latin": 0.25, "Arabic": 0.60, "Cyrillic": 0.20})
    entry = compare_to_baseline(report_of(live), baseline_of(BASE), CONFIG)["per_script"]["Arabic"]["alert_rate"]
    assert (entry["baseline"], entry["live"]) == (0.40, 0.60)
    assert entry["abs_diff"] == pytest.approx(0.20) and entry["ratio"] == pytest.approx(1.5)
    assert entry["status"] == "warn"  # ratio 1.5 and difference 0.2: warn ratio, below the 2.0 alert ratio


def test_a_mix_only_shift_moves_input_mix_psi_but_not_within_script_psi():
    live = group(SHIFT_COUNTS)  # same shapes and rates per script, new proportions
    result = compare_to_baseline(report_of(live), baseline_of(BASE), CONFIG)
    assert result["input_mix_psi"]["value"] > CONFIG.psi_alert and result["input_mix_psi"]["status"] == "alert"
    for script in ("Latin", "Arabic", "Cyrillic"):
        assert result["per_script"][script]["score_psi"]["value"] == pytest.approx(0.0, abs=1e-12)
        assert result["per_script"][script]["score_psi"]["status"] == "ok"
        assert result["per_script"][script]["alert_rate"]["status"] == "ok"
    assert result["score_psi"]["value"] > 0  # the overall score distribution moves because the mix did
    assert result["overall"] == "alert"


def test_a_within_script_shift_moves_that_scripts_psi_and_not_the_mix():
    live = group(BASE_COUNTS)
    live["per_script"]["Arabic"]["score_histogram"]["counts"] = scaled(
        SHAPES["Latin"], 100
    )  # Arabic now scores like Latin
    result = compare_to_baseline(report_of(live), baseline_of(BASE), CONFIG)
    assert result["input_mix_psi"]["value"] == 0.0
    assert (
        result["per_script"]["Arabic"]["score_psi"]["status"] == "alert"
        or result["per_script"]["Arabic"]["score_psi"]["value"] > 0.1
    )
    assert result["per_script"]["Cyrillic"]["score_psi"]["value"] == 0.0


# ---------- refusals and thin data ----------


def test_a_report_from_another_decision_context_is_refused_with_a_clear_message():
    other = group(BASE_COUNTS, context=OTHER, decision=OTHER_DEC)
    with pytest.raises(BaselineMismatch) as stop:
        compare_to_baseline(report_of(other), baseline_of(BASE), CONFIG)
    text = str(stop.value)
    assert "only applies to the decision context it was measured under" in text
    assert DEC[:12] in text and OTHER_DEC[:12] in text and "refusing to compare" in text


def test_a_report_from_the_same_decisions_at_another_commit_is_compared_not_refused():
    # same decision context, a different run context (another git commit): the baseline still applies
    later_commit = group(BASE_COUNTS, context=OTHER, decision=DEC)
    result = compare_to_baseline(report_of(later_commit), baseline_of(BASE), CONFIG)
    assert result["overall"] == "ok" and result["run_context_hash"] == OTHER
    assert result["baseline_run_context_hash"] == CTX and result["decision_context_hash"] == DEC


def test_when_two_commits_made_the_same_decisions_the_larger_group_is_compared_and_the_other_named():
    small = group({"Latin": 600, "Arabic": 200, "Cyrillic": 300}, context=OTHER)
    big = group(BASE_COUNTS, context="c" * 64)
    result = compare_to_baseline(report_of(small, big), baseline_of(BASE), CONFIG)
    assert result["run_context_hash"] == "c" * 64 and result["other_matching_contexts"] == [OTHER]


def test_the_refusal_names_the_fields_that_differ_when_both_contexts_are_known():
    other = group({"Latin": 600}, context=OTHER, decision=OTHER_DEC)
    stored = {"git_sha": "1" * 40, "config": {"global_threshold": 0.86}, "snapshot_id": "s1"}
    live_ctx = {"git_sha": "2" * 40, "config": {"global_threshold": 0.85}, "snapshot_id": "s1"}
    with pytest.raises(BaselineMismatch) as stop:
        compare_to_baseline(report_of(other), baseline_of(BASE, stored), CONFIG, live_contexts={OTHER: live_ctx})
    assert "config.global_threshold: stored 0.86, live 0.85" in str(stop.value)
    assert "git_sha" not in str(stop.value)  # the commit is not a decision field


def test_a_report_with_no_decisions_is_refused():
    with pytest.raises(BaselineMismatch, match="no decisions"):
        compare_to_baseline(report_of(), baseline_of(BASE), CONFIG)


def test_a_group_below_min_psi_count_reports_insufficient_data_not_a_psi():
    live = group({"Latin": 90, "Arabic": 30, "Cyrillic": 30})  # 150 decisions
    result = compare_to_baseline(report_of(live), baseline_of(BASE), CONFIG)
    assert result["count_live"] == 150
    for key in ("score_psi", "input_mix_psi"):
        assert result[key]["value"] is None and result[key]["status"] == "insufficient_data"
        assert "min_psi_count 200" in result[key]["reason"]
    assert result["near_threshold"]["status"] == "insufficient_data"
    assert result["overall"] == "insufficient_data"
    assert result["note_thin_data"] is not None


def test_the_min_psi_count_boundary_is_inclusive():
    just_enough = group({"Latin": 100, "Arabic": 50, "Cyrillic": 50})  # exactly 200
    result = compare_to_baseline(report_of(just_enough), baseline_of(BASE), CONFIG)
    assert result["score_psi"]["value"] is not None
    thin = group({"Latin": 100, "Arabic": 50, "Cyrillic": 49})
    assert compare_to_baseline(report_of(thin), baseline_of(BASE), CONFIG)["score_psi"]["value"] is None


def test_a_thin_script_cell_gets_no_psi_while_the_group_does():
    live = group({"Latin": 1200, "Arabic": 100, "Cyrillic": 600})  # Arabic: 100 live decisions, under 200
    result = compare_to_baseline(report_of(live), baseline_of(BASE), CONFIG)
    arabic = result["per_script"]["Arabic"]
    assert (
        arabic["score_psi"]["status"] == "insufficient_data" and arabic["alert_rate"]["status"] == "insufficient_data"
    )
    assert arabic["score_psi"]["value"] is None
    assert result["per_script"]["Latin"]["score_psi"]["value"] is not None
    assert result["score_psi"]["value"] is not None and result["overall"] in ("ok", "warn", "alert")


def test_a_suppressed_group_is_insufficient_data():
    hidden = {"run_context_hash": CTX, "decision_context_hash": DEC, "count": "<10"}
    result = compare_to_baseline(report_of(hidden), baseline_of(BASE), CONFIG)
    assert result["overall"] == "insufficient_data" and "suppressed" in result["reason"]
    result = compare_to_baseline(report_of(BASE), baseline_of(hidden), CONFIG)
    assert result["overall"] == "insufficient_data"


def test_cells_that_are_suppressed_in_either_report_are_excluded_from_the_per_script_comparison():
    live = group({"Latin": 1200, "Arabic": "<10", "Cyrillic": 600})
    live["count"] = 2200  # the hidden Arabic cell holds the other decisions
    result = compare_to_baseline(report_of(live), baseline_of(BASE), CONFIG)
    assert set(result["per_script"]) == {"Latin", "Cyrillic"}
    assert "Arabic" in result["skipped_scripts"] and "not visible" in result["skipped_scripts"]["Arabic"]
    base = group({"Latin": 1200, "Arabic": "withheld", "Cyrillic": 600})
    base["count"] = 2200
    result = compare_to_baseline(report_of(BASE), baseline_of(base), CONFIG)
    assert "Arabic" not in result["per_script"] and "Arabic" in result["skipped_scripts"]


def test_the_input_mix_puts_hidden_decisions_in_one_bucket():
    live = group({"Latin": 1200, "Arabic": "<10", "Cyrillic": 600})
    live["count"] = 2200
    result = compare_to_baseline(report_of(live), baseline_of(BASE), CONFIG)
    assert result["input_mix_psi"]["categories"] == ["Arabic", "Cyrillic", "Latin", "suppressed"]
    assert result["input_mix_psi"]["value"] is not None


# ---------- config ----------


def test_the_shipped_config_has_the_conventional_psi_limits_and_a_200_minimum():
    config = DriftConfig.load()
    assert (config.psi_warn, config.psi_alert, config.min_psi_count) == (0.1, 0.25, 200)
    assert config.alert_rate_ratio_warn < config.alert_rate_ratio_alert
    assert DriftConfig.load(config_path_missing()) == DriftConfig()


def config_path_missing():
    from pathlib import Path

    return Path("/nonexistent/monitoring.json")


@pytest.mark.parametrize(
    "kwargs",
    [{"psi_warn": 0.3, "psi_alert": 0.2}, {"psi_warn": 0}, {"min_psi_count": 0}, {"psi_epsilon": 0},
     {"alert_rate_ratio_warn": 3.0}, {"near_threshold_diff_warn": 0.5}],
)  # fmt: skip
def test_bad_drift_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        DriftConfig(**kwargs)


# ---------- baseline files ----------


def test_a_baseline_is_saved_once_and_loaded_back(tmp_path):
    report = report_of(BASE)
    path = save_baseline(report, {"git_sha": "x"}, {"seed": 11}, tmp_path / "baseline")
    assert path.name == f"{DEC}.json"  # named by the decision context hash
    loaded = load_baseline(path)
    assert loaded["decision_context_hash"] == DEC and loaded["run_context_hash"] == CTX
    assert loaded["report"] == report and loaded["sample"] == {"seed": 11}
    with pytest.raises(FileExistsError):  # a frozen baseline is never silently replaced
        save_baseline(report, {}, {}, tmp_path / "baseline")
    path.write_text(json.dumps({"baseline_schema_version": 99}))
    with pytest.raises(ValueError, match="not a baseline file"):
        load_baseline(path)


# ---------- the CLI ----------


def setup_cli(tmp_path, live: dict, baseline_group: dict = BASE):
    audit = AuditLog.create_new(tmp_path / "audit", fsync=False)
    baseline_dir = tmp_path / "baseline"
    save_baseline(report_of(baseline_group), {"git_sha": "x"}, {}, baseline_dir)
    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps(report_of(live)))
    return audit, baseline_dir, report_file


def test_cli_drift_exit_codes_follow_the_status(tmp_path, capsys):
    audit, baseline_dir, report_file = setup_cli(tmp_path, BASE)
    args = ["drift", "--report", str(report_file), "--baseline-dir", str(baseline_dir)]
    assert screen_cli.main(args) == 0
    assert "overall ok" in capsys.readouterr().out
    report_file.write_text(json.dumps(report_of(group(SHIFT_COUNTS))))
    assert screen_cli.main(args) == 20
    out = capsys.readouterr().out
    assert "overall alert" in out and "input-mix PSI" in out and "[alert]" in out
    report_file.write_text(json.dumps(report_of(group({"Latin": 90, "Arabic": 30, "Cyrillic": 30}))))
    assert screen_cli.main(args) == 3 and "insufficient_data" in capsys.readouterr().out
    warn = group(BASE_COUNTS, rates={"Latin": 0.25, "Arabic": 0.60, "Cyrillic": 0.20})
    report_file.write_text(json.dumps(report_of(warn)))
    assert screen_cli.main(args) == 10


def test_cli_drift_json_output(tmp_path, capsys):
    _, baseline_dir, report_file = setup_cli(tmp_path, BASE)
    assert screen_cli.main(["drift", "--report", str(report_file), "--baseline-dir", str(baseline_dir), "--json"]) == 0
    (result,) = json.loads(capsys.readouterr().out)
    assert result["decision_context_hash"] == DEC and result["score_psi"]["value"] == 0.0


def test_cli_drift_refuses_a_context_with_no_baseline_and_a_bad_report(tmp_path, capsys):
    _, baseline_dir, report_file = setup_cli(tmp_path, group({"Latin": 600}, context=OTHER, decision=OTHER_DEC))
    code = screen_cli.main(["drift", "--report", str(report_file), "--baseline-dir", str(baseline_dir)])
    err = capsys.readouterr().err
    assert code == screen_cli.EXIT_USAGE and "no baseline" in err and "only applies to the decision context" in err
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert (
        screen_cli.main(["drift", "--report", str(bad), "--baseline-dir", str(baseline_dir)]) == screen_cli.EXIT_USAGE
    )
    explicit = baseline_dir / f"{DEC}.json"
    assert (
        screen_cli.main(["drift", "--report", str(report_file), "--baseline", str(explicit)]) == screen_cli.EXIT_USAGE
    )
    assert "refusing to compare" in capsys.readouterr().err


def test_cli_drift_leaves_the_audit_log_the_baseline_and_the_report_byte_identical(tmp_path):
    audit, baseline_dir, report_file = setup_cli(tmp_path, group(SHIFT_COUNTS))
    before = (tree(tmp_path / "audit"), tree(baseline_dir), report_file.read_bytes())
    screen_cli.main(["drift", "--report", str(report_file), "--baseline-dir", str(baseline_dir),
                     "--audit-dir", str(tmp_path / "audit")])  # fmt: skip
    assert (tree(tmp_path / "audit"), tree(baseline_dir), report_file.read_bytes()) == before
    assert audit.headers.read_all() == []


def test_cli_drift_prints_no_names_only_numbers_and_statuses(tmp_path, capsys):
    _, baseline_dir, report_file = setup_cli(tmp_path, group(SHIFT_COUNTS))
    screen_cli.main(["drift", "--report", str(report_file), "--baseline-dir", str(baseline_dir)])
    out = capsys.readouterr().out
    assert out.count("\n") < 20 and "request" not in out.lower()


# ---------- the sampling-noise floor ----------


def test_split_half_psi_measures_what_no_change_looks_like():
    import run_drift_batches

    uniform = [200] * 20  # 4000 decisions spread evenly over the 20 bins
    floor = run_drift_batches.split_half_psi(uniform, seed=3, reps=200)
    assert floor["decisions_per_half"] == 2000
    assert floor["median"] <= floor["p90"] <= floor["p95"]
    assert 0.01 < floor["median"] < 0.03  # about (bins - 1) * (1/2000 + 1/2000) = 0.019
    assert run_drift_batches.split_half_psi(uniform, seed=3, reps=200) == floor  # seeded
    small = run_drift_batches.split_half_psi([10] * 20, seed=3, reps=200)  # 200 decisions: far noisier
    assert small["median"] > 5 * floor["median"] and small["median"] > 0.1


# ---------- limits from the pooled two-sample null ----------


def tilted(amplitude: float, n: int) -> list[int]:
    """n decisions over 20 bins, alternately above and below an even spread by `amplitude`."""
    return [round(n / 20 * (1 + amplitude * (-1) ** i)) for i in range(20)]


FLAT = [100] * 20  # a baseline of 2000 decisions spread evenly


def flat(n: int) -> list[int]:
    return [n // 20] * 20


def test_the_pooled_null_is_seeded_ordered_and_shrinks_with_the_sample_size():
    from audit_drift import null_psi

    p95, p99 = null_psi(FLAT, flat(250), CONFIG, "score")
    assert p95 < p99 and null_psi(FLAT, flat(250), CONFIG, "score") == (p95, p99)  # fixed seed
    assert null_psi(FLAT, flat(250), CONFIG, "input_mix") != (p95, p99)  # each comparison has its own fixed stream
    big95, big99 = null_psi(FLAT, flat(10_000), CONFIG, "score")
    assert big95 < p95 / 2 and big99 < p99 / 2
    # (bins - 1) * (1/n_live + 1/n_base) is the expected PSI of two samples from one distribution: about 0.086 here
    assert 0.05 < p95 < 0.30


def test_the_null_includes_the_noise_of_the_baseline_as_well_as_the_live_sample():
    from audit_drift import null_psi

    live = flat(500)
    small_base = null_psi([5] * 20, live, CONFIG, "score")  # a baseline of 100 decisions
    large_base = null_psi([250] * 20, live, CONFIG, "score")  # a baseline of 5000
    assert small_base[0] > 2 * large_base[0] and small_base[1] > 2 * large_base[1]
    # expected pure-noise PSI is (19) * (1/100 + 1/500) = 0.23 against 19 * (1/5000 + 1/500) = 0.042
    assert 0.15 < small_base[0] and large_base[0] < 0.08


def test_the_null_is_drawn_from_the_pooled_proportions_so_a_bin_empty_in_both_is_never_drawn():
    from audit_drift import null_distribution

    base = [100, 100] + [0] * 18
    values = null_distribution(base, [90, 110] + [0] * 18, CONFIG, "score")
    assert len(values) == 1000 and list(values) == sorted(values)
    assert values[949] < 0.1  # two bins only: little noise


def test_the_pooled_null_moves_with_the_live_data_it_is_pooled_from():
    from audit_drift import null_psi

    same = null_psi(FLAT, flat(500), CONFIG, "score")
    spiky = null_psi(FLAT, tilted(0.9, 500), CONFIG, "score")  # pooled shares are lumpier, so draws are noisier
    assert spiky != same


def test_every_psi_comparison_reports_its_n_its_null_its_limits_and_the_corrected_ones():
    result = compare_to_baseline(report_of(group(SHIFT_COUNTS)), baseline_of(BASE), CONFIG)
    entries = [result["score_psi"], result["input_mix_psi"]] + [e["score_psi"] for e in result["per_script"].values()]
    assert len(entries) == 5
    for item in entries:
        needed = {"value", "status", "n", "n_baseline", "null_p95", "null_p99", "warn_limit", "alert_limit"}
        assert needed | {"corrected_warn_limit", "corrected_alert_limit", "corrected_status"} <= set(item)
        assert item["warn_limit"] == max(CONFIG.psi_warn, item["null_p95"])
        assert item["alert_limit"] == max(CONFIG.psi_alert, item["null_p99"])
        assert item["status"] == psi_status(item["value"], item["warn_limit"], item["alert_limit"])
        assert (
            item["corrected_warn_limit"] >= item["warn_limit"] and item["corrected_alert_limit"] >= item["alert_limit"]
        )
        assert item["corrected_status"] == psi_status(
            item["value"], item["corrected_warn_limit"], item["corrected_alert_limit"]
        )
    assert result["score_psi"]["n"] == 2200 and result["input_mix_psi"]["n"] == 2200
    assert result["per_script"]["Arabic"]["score_psi"]["n"] == 1200 and result["score_psi"]["n_baseline"] == 2200


def test_k_and_the_corrected_percentiles_are_reported():
    result = compare_to_baseline(report_of(group(SHIFT_COUNTS)), baseline_of(BASE), CONFIG)
    m = result["multiplicity"]
    assert m["k"] == 5  # the score, the script mix, and three scripts with enough decisions
    assert (m["alpha_warn"], m["alpha_alert"]) == (0.05, 0.01)
    assert m["warn_percentile"] == pytest.approx(99.0) and m["alert_percentile"] == pytest.approx(99.8)
    assert "worst of k" in m["why"]
    fewer = group({"Latin": 1200, "Arabic": 100, "Cyrillic": 600})  # Arabic is thin: k drops to 4
    assert compare_to_baseline(report_of(fewer), baseline_of(BASE), CONFIG)["multiplicity"]["k"] == 4
    assert compare_to_baseline(report_of(group({"Latin": 90})), baseline_of(BASE), CONFIG)["multiplicity"] is None


def test_the_corrected_fractions_are_bonferroni():
    from audit_drift import corrected_fractions

    assert corrected_fractions(1, CONFIG) == (0.95, 0.99)
    assert corrected_fractions(5, CONFIG) == pytest.approx((0.99, 0.998))
    assert corrected_fractions(20, CONFIG) == pytest.approx((0.9975, 0.9995))


def only_latin(shape: list[int], n: int = 2000) -> dict:
    g = group({"Latin": n})
    g["score_histogram"]["counts"] = list(shape)
    g["per_script"]["Latin"]["score_histogram"]["counts"] = list(shape)
    return g


def test_the_overall_status_uses_the_corrected_percentiles_while_each_comparison_keeps_its_own(monkeypatch):
    config = DriftConfig(psi_warn=0.000001, psi_alert=0.000002, min_psi_count=50)
    base = only_latin(flat(2000))
    live = only_latin(tilted(0.4, 2000))
    monkeypatch.setattr(audit_drift, "null_distribution", lambda *a, **k: np.zeros(1000))
    v = compare_to_baseline(report_of(live), baseline_of(base), config)["score_psi"]["value"]
    assert v > 0.05
    # a made-up null whose uncorrected 95th percentile is below v and whose Bonferroni (k = 3) one is above it
    ramp = np.linspace(0, v / 0.97, 1000)
    monkeypatch.setattr(audit_drift, "null_distribution", lambda *a, **k: ramp)
    result = compare_to_baseline(report_of(live), baseline_of(base), config)
    assert result["multiplicity"]["k"] == 3  # score, script mix, Latin
    assert result["score_psi"]["status"] == "warn" and result["score_psi"]["corrected_status"] == "ok"
    assert result["per_script"]["Latin"]["score_psi"]["status"] == "warn"  # per-comparison: uncorrected
    assert result["overall_uncorrected"] == "warn" and result["overall"] == "ok"  # overall: corrected
    # a null whose every limit is below v: even the corrected percentiles are crossed
    low = np.linspace(0, v / 1.2, 1000)
    monkeypatch.setattr(audit_drift, "null_distribution", lambda *a, **k: low)
    result = compare_to_baseline(report_of(live), baseline_of(base), config)
    assert result["score_psi"]["corrected_status"] == "alert" and result["overall"] == "alert"


def test_the_floors_still_apply_to_the_corrected_limits(monkeypatch):
    monkeypatch.setattr(audit_drift, "null_distribution", lambda *a, **k: np.zeros(1000))
    result = compare_to_baseline(report_of(group(SHIFT_COUNTS)), baseline_of(BASE), CONFIG)
    item = result["score_psi"]
    assert item["corrected_warn_limit"] == 0.1 and item["corrected_alert_limit"] == 0.25


def test_the_same_psi_is_ok_on_a_small_sample_and_a_warn_on_a_large_one():
    from audit_drift import psi_result

    config = DriftConfig(min_psi_count=50)
    small = psi_result(FLAT, tilted(0.4, 100), config, 2000, 100, "score")
    large = psi_result(FLAT, tilted(0.4, 10_000), config, 2000, 10_000, "score")
    assert small["value"] == pytest.approx(large["value"], rel=1e-9)  # the same shares, so the same PSI
    assert 0.1 < small["value"] < 0.25
    assert small["warn_limit"] > small["value"] and small["status"] == "ok"  # noise alone gets this far at n = 100
    assert large["warn_limit"] == 0.1 and large["status"] == "warn"  # at n = 10,000 it cannot: a real shift


def test_the_alert_limit_is_the_larger_of_the_convention_and_the_null_p99():
    from audit_drift import psi_result

    config = DriftConfig(min_psi_count=50)
    result = psi_result(FLAT, tilted(0.7, 100), config, 2000, 100, "score")  # PSI about 0.6
    assert result["alert_limit"] == max(0.25, result["null_p99"])
    assert result["status"] == psi_status(result["value"], result["warn_limit"], result["alert_limit"])
    big = psi_result(FLAT, tilted(0.7, 10_000), config, 2000, 10_000, "score")
    assert big["status"] == "alert" and big["alert_limit"] == 0.25


def test_a_mix_only_shift_is_still_an_alert_at_these_sizes_even_after_the_correction():
    result = compare_to_baseline(report_of(group(SHIFT_COUNTS)), baseline_of(BASE), CONFIG)
    mix = result["input_mix_psi"]
    assert mix["status"] == "alert" and mix["corrected_status"] == "alert" and mix["value"] > 10 * mix["null_p99"]
    assert result["overall"] == "alert"


def test_the_null_settings_come_from_the_config():
    config = DriftConfig.load()
    assert (config.psi_null_draws, config.psi_null_seed) == (1000, 11)
    assert (config.psi_warn_percentile, config.psi_alert_percentile) == (95, 99)
    assert (config.psi_alpha_warn, config.psi_alpha_alert) == (0.05, 0.01)
    bad = (
        {"psi_null_draws": 10},
        {"psi_warn_percentile": 99, "psi_alert_percentile": 95},
        {"psi_alert_percentile": 100},
        {"psi_alpha_warn": 0.01, "psi_alpha_alert": 0.05},
        {"psi_alpha_alert": 0},
    )
    for kwargs in bad:
        with pytest.raises(ValueError):
            DriftConfig(**kwargs)


def test_a_different_seed_gives_a_different_but_similar_null():
    from audit_drift import null_psi

    a = null_psi(FLAT, flat(300), DriftConfig(psi_null_seed=1), "score")
    b = null_psi(FLAT, flat(300), DriftConfig(psi_null_seed=2), "score")
    assert a != b and abs(a[0] - b[0]) < 0.04


# ---------- durations are information only ----------


def test_durations_are_shown_beside_the_baseline_and_never_compared():
    live = group(BASE_COUNTS)
    live["duration_ms"] = {"p50": 5000, "p95": 9000, "p99": 20000}  # a hundred times slower
    result = compare_to_baseline(report_of(live), baseline_of(BASE), CONFIG)
    duration = result["duration_ms"]
    assert duration["informational"] is True and "status" not in duration and "machine-dependent" in duration["note"]
    assert duration["baseline"] == {"p50": 1, "p95": 2, "p99": 3} and duration["live"]["p99"] == 20000
    assert result["overall"] == "ok"  # no effect on any status


def test_cli_drift_labels_durations_informational(tmp_path, capsys):
    live = group(BASE_COUNTS)
    _, baseline_dir, report_file = setup_cli(tmp_path, live)
    screen_cli.main(["drift", "--report", str(report_file), "--baseline-dir", str(baseline_dir)])
    out = capsys.readouterr().out
    assert "duration_ms" in out and "informational" in out and "not compared" in out
    assert "sampling null p95" in out and "limits warn" in out

"""Tests for src/evaluate_final.py. They run on a tiny fake hold-out, never the real one.

Run:
    pytest tests/test_evaluate_final.py
"""

import hashlib
import json

import pytest

import evaluate_final as final
import evaluate_scoring as ev


def test_wilson_matches_known_values():
    low, high = final.wilson(0, 10)
    assert (low, high) == (0.0, pytest.approx(0.2775, abs=1e-3))
    low, high = final.wilson(10, 10)
    assert (low, high) == (pytest.approx(0.7225, abs=1e-3), 1.0)
    low, high = final.wilson(50, 100)
    assert (low, high) == (pytest.approx(0.4038, abs=1e-3), pytest.approx(0.5962, abs=1e-3))
    assert final.wilson(0, 0) == (0.0, 0.0)


def test_estimate():
    assert final.estimate(5, 10) == {"value": 0.5, "ci95": [0.2366, 0.7634], "k": 5, "n": 10}


def test_two_proportion_p():
    assert final.two_proportion_p(30, 100, 30, 100) == pytest.approx(1.0)
    # z = (0.3 - 0.5) / sqrt(0.4 * 0.6 * (1/100 + 1/100)) = -2.887, p = 0.0039
    assert final.two_proportion_p(30, 100, 50, 100) == pytest.approx(0.0039, abs=5e-4)
    assert final.two_proportion_p(0, 0, 5, 10) == 1.0


def test_bootstrap_f1_is_deterministic_and_brackets_the_point_estimate():
    codes = [1] * 50 + [0] * 50  # recall 0.5
    no_false_alarms = [False] * 100  # precision 1.0, so F1 = 2 * 0.5 / 1.5 = 0.6667
    low, high = final.bootstrap_f1(codes, no_false_alarms)
    assert low < 2 / 3 < high
    assert (low, high) == final.bootstrap_f1(codes, no_false_alarms)
    assert final.bootstrap_f1([1] * 20, [False] * 20) == (1.0, 1.0)


def test_compare_flags_a_test_value_outside_the_interval():
    est = {"value": 0.60, "ci95": [0.55, 0.65], "k": 60, "n": 100}
    inside = final.compare("x", est, 0.62, 100)
    outside = final.compare("x", est, 0.80, 100)
    assert inside["test_inside_holdout_ci"] is True and inside["difference"] == pytest.approx(-0.02)
    assert outside["test_inside_holdout_ci"] is False and outside["p_value"] < 0.01
    assert final.compare("f1", {"value": 0.7, "ci95": [0.6, 0.8]}, 0.7, None)["p_value"] is None


def person(i, names, dob="1980", country="ru"):
    return {"id": f"e{i}", "schema": "Person", "target": True,
            "properties": {"name": names, "birthDate": [dob], "nationality": [country]}}  # fmt: skip


def fake_inputs(tmp_path):
    people = [
        person(0, ["Ivan Petrov", "Иван Петров"]),
        person(1, ["Anna Sidorova", "Анна Сидорова"]),
        person(2, ["Boris Kuznetsov"]),
        person(3, ["Mohammed Ali", "Muhammad Ali"], "1970", "sa"),
        person(4, ["Maria Garcia"], "1990", "es"),
        person(5, ["Karim Nasser"], "1960", "eg"),
    ]
    entities = tmp_path / "entities.ftm.json"
    entities.write_text("\n".join(json.dumps(p) for p in people) + "\n")

    def q(kind, group, name, true_id=None, dob=None, country=None, **extra):
        return {"split": "final", "kind": kind, "group": group, "true_id": true_id, "name": name,
                "dob": dob, "country": country, **extra}  # fmt: skip

    queries = [
        q("positive", "variant", "Mohammed Ali", "e3", "1970", "sa"),
        q("positive", "cyr_to_latin", "Иван Петров", "e0", "1980", None),
        q("positive", "other_to_latin", "张伟", "e1"),
        q("synthetic", "Latin", "Zorblax Qwerty", dob="1975", country="us"),
        q("near_miss", "Latin", "Maria Garcia", dob="1950", country="us", entity="e4"),
    ]
    kept = {"e3": ["Muhammad Ali"], "e0": ["Ivan Petrov"], "e1": ["Anna Sidorova"]}
    holdout = tmp_path / "final_holdout.json"
    holdout.write_text(ev.holdout_text(queries, kept))
    checksum = tmp_path / "final_holdout.sha256"
    checksum.write_text(hashlib.sha256(holdout.read_bytes()).hexdigest() + "\n")
    return entities, holdout, checksum


def run(tmp_path, entities, holdout, checksum, *extra):
    out = tmp_path / "final_evaluation.json"
    final.main(["--input", str(entities), "--holdout", str(holdout), "--checksum", str(checksum),
                "--output", str(out), *extra])  # fmt: skip
    return out


def test_end_to_end_on_a_fake_holdout(tmp_path):
    entities, holdout, checksum = fake_inputs(tmp_path)
    report = json.loads(run(tmp_path, entities, holdout, checksum).read_text())
    assert report["checksum"]["ok"] is True
    assert report["engine_check"] == {"queries": 5, "mismatches": 0}
    assert report["queries"] == {"positive_headline": 2, "negative": 2, "positive_other_to_latin": 1}
    assert set(report["overall"]) == {"recall", "precision", "f1", "top1_accuracy", "false_positive_rate"}
    for name in ("recall", "false_positive_rate"):
        low, high = report["overall"][name]["ci95"]
        assert low <= report["overall"][name]["value"] <= high
    assert set(report["by_script"]) == {"Latin", "Cyrillic", "Arabic"}
    assert "variant" in report["by_direction"] and "cyr_to_latin" in report["by_direction"]
    assert report["false_positives_per_1000"]["all"]["names"] == 2
    figures = {row["figure"] for row in report["test_vs_holdout"]}
    assert {"recall", "precision", "f1", "top-1 accuracy", "false positive rate", "FPR, Latin queries"} <= figures
    assert report["frozen_config"]["global_threshold"] == 0.86
    assert report["frozen_config"]["per_script_thresholds"] is False
    # No Cyrillic or Arabic negatives in the fake set: reported as empty, not as a crash.
    assert report["false_positives_per_1000"]["Cyrillic queries"]["names"] == 0
    assert report["by_script"]["Arabic"]["false_positive_rate"]["value"] is None


def test_a_later_holdout_can_be_compared_with_an_earlier_one(tmp_path):
    entities, holdout, checksum = fake_inputs(tmp_path)
    first = run(tmp_path, entities, holdout, checksum)
    second = tmp_path / "second.json"
    final.main(["--input", str(entities), "--holdout", str(holdout), "--checksum", str(checksum),
                "--output", str(second), "--compare-with", str(first)])  # fmt: skip
    versus = json.loads(second.read_text())["versus_earlier_holdout"]
    assert set(versus) == {"recall", "precision", "f1", "top1_accuracy", "false_positive_rate"}
    assert all(v["difference"] == 0.0 and v["intervals_overlap"] for v in versus.values())  # the same queries twice


def test_refuses_to_run_twice(tmp_path):
    entities, holdout, checksum = fake_inputs(tmp_path)
    out = run(tmp_path, entities, holdout, checksum)
    before = out.read_text()
    with pytest.raises(SystemExit, match="not run twice"):
        run(tmp_path, entities, holdout, checksum)
    assert out.read_text() == before


def test_refuses_to_score_when_the_checksum_does_not_match(tmp_path):
    entities, holdout, checksum = fake_inputs(tmp_path)
    checksum.write_text("0" * 64 + "\n")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        run(tmp_path, entities, holdout, checksum)
    assert not (tmp_path / "final_evaluation.json").exists()


def test_refuses_to_run_with_a_config_that_is_not_the_frozen_one(tmp_path):
    entities, holdout, checksum = fake_inputs(tmp_path)
    config = json.loads(final.CONFIG_PATH.read_text())
    config["global_threshold"] = 0.7
    other = tmp_path / "config.json"
    other.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="not the frozen configuration"):
        run(tmp_path, entities, holdout, checksum, "--config", str(other))


def test_the_shipped_config_is_the_frozen_one():
    from screening import ScreeningConfig

    assert ScreeningConfig.load() == final.FROZEN

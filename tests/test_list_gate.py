"""Tests for the list-update regression gate (src/list_gate.py) on small hand-built snapshots.

Run:
    pytest tests/test_list_gate.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import list_gate as lg
import screen_cli
from list_gate import GateConfig, SnapshotRefused

CONS, VOWELS = "bcdfghjklmnprstvz", "aeiou"


def word(i: int, length: int = 6) -> str:
    """A pronounceable, distinct made-up word for each i."""
    out = []
    for k in range(length):
        pool = CONS if k % 2 == 0 else VOWELS
        out.append(pool[i % len(pool)])
        i //= len(pool)
    return "".join(out).capitalize()


def person(entity_id: str, names: list[str], target: bool = True, referents: list[str] | None = None, **extra) -> dict:
    return {"id": entity_id, "names": names, "target": target, "referents": referents or [], **extra}


def write_snapshot(directory: Path, entities: list[dict], version: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for e in entities:
        props = {"name": e["names"]}
        if e.get("birth_dates"):
            props["birthDate"] = e["birth_dates"]
        if e.get("countries"):
            props["country"] = e["countries"]
        lines.append(
            json.dumps(
                {
                    "id": e["id"],
                    "schema": "Person",
                    "target": e["target"],
                    "referents": e["referents"],
                    "properties": props,
                },
                ensure_ascii=False,
            )
        )
    body = ("\n".join(lines) + "\n").encode("utf-8")
    path = directory / "entities.ftm.json"
    path.write_bytes(body)
    (directory / "VERSION").write_text(f"version={version}\nsha1={hashlib.sha1(body).hexdigest()}\n")  # noqa: S324
    return path


def write_holdout(path: Path, queries: list[dict], kept: dict[str, list[str]]) -> Path:
    path.write_text(json.dumps({"queries": queries, "kept_names": kept}, ensure_ascii=False))
    return path


def positive(true_id: str, name: str, group: str = "variant") -> dict:
    return {"kind": "positive", "group": group, "true_id": true_id, "name": name, "dob": None, "country": None}


def negative(name: str, script: str = "Latin", kind: str = "synthetic", imitates: str | None = None) -> dict:
    q = {"kind": kind, "group": script, "true_id": None, "name": name, "dob": None, "country": None}
    if imitates:
        q["entity"] = imitates
    return q


OLD_V, NEW_V = "20261006000000-aaa", "20261008000000-bbb"


def variants(i: int) -> tuple[str, str, str]:
    """(query alias, kept alias, base) for entity i: two close spellings of one made-up name."""
    base = f"{word(i, 6)} {word(i + 500, 7)}"
    return base + "n", base, base


# ---------- hand-built snapshots covering every resolution case ----------


@pytest.fixture
def world(tmp_path):
    names = {k: variants(i) for i, k in enumerate(["k", "r1", "r2", "amb", "d1", "d2"], start=1)}
    old_entities = [
        person("old-k", [names["k"][0], names["k"][1]]),
        person("old-r1", [names["r1"][0], names["r1"][1]]),
        person("old-r2", [names["r2"][0], names["r2"][1]], referents=["new-r2"]),  # the old entity names its successor
        person("old-amb", [names["amb"][0], names["amb"][1]]),
        person("old-d1", [names["d1"][0], names["d1"][1]]),
        person("old-d2", [names["d2"][0], names["d2"][1]]),
        person("old-gone", ["Quorlex Fantabrim"]),
        person("old-changed", ["Mivrol Tazbek"]),
    ]
    new_entities = [
        person("old-k", [names["k"][0], names["k"][1]]),
        person("new-r1", [names["r1"][0], names["r1"][1]], referents=["old-r1"]),  # the new entity names the old ID
        person("new-r2", [names["r2"][0], names["r2"][1]]),
        person("new-amb-b", [names["amb"][0], names["amb"][1]], referents=["old-amb"]),
        person("new-amb-a", [names["amb"][0], names["amb"][1]], referents=["old-amb"]),
        person("old-d2", [names["d2"][0], names["d2"][1]], target=False),  # still in the file, no longer a target
        person("old-changed", ["Mivrol Tazbek", "Mivrol Tazbeck"]),
        person("brand-new", ["Hovnat Elpris"]),
    ]
    old = write_snapshot(tmp_path / "old", old_entities, OLD_V)
    new = write_snapshot(tmp_path / "new", new_entities, NEW_V)
    queries = [
        positive("old-k", names["k"][0]),
        positive("old-r1", names["r1"][0]),
        positive("old-r2", names["r2"][0]),
        positive("old-amb", names["amb"][0]),
        positive("old-d1", names["d1"][0]),  # delisted: absent from the new file
        positive("old-d2", names["d2"][0]),  # delisted: present but not a target
        negative("Zebulon Quixotic"),
        negative(
            word(77, 8) + " " + word(88, 7), "Latin", "near_miss", imitates="old-d1"
        ),  # imitates a delisted entity
        negative("محمد زيد", "Arabic"),
        negative("Иван Зайцев", "Cyrillic"),
    ]
    kept = {
        "old-k": [names["k"][1]], "old-r1": [names["r1"][1]], "old-r2": [names["r2"][1]],
        "old-amb": [names["amb"][1]], "old-d1": [names["d1"][1]], "old-d2": [names["d2"][1]],
    }  # fmt: skip
    holdout = write_holdout(tmp_path / "holdout.json", queries, kept)
    return {"old": old, "new": new, "holdout": holdout, "tmp": tmp_path, "names": names}


def resolved(world):
    old, new = lg.read_snapshot(world["old"]), lg.read_snapshot(world["new"])
    queries, _ = lg.load_holdout(world["holdout"])
    return old, new, *lg.resolve_holdout(queries, old, new)


def test_each_case_is_kept_remapped_either_way_or_dropped(world):
    _, _, rows, counts = resolved(world)
    mapped = {q["true_id"]: q["new_id"] for q in rows if q["kind"] == "positive"}
    assert mapped == {
        "old-k": "old-k",  # present in the new snapshot: kept
        "old-r1": "new-r1",  # a new entity's referents contain the old ID
        "old-r2": "new-r2",  # the old entity's referents name a new ID
        "old-amb": "new-amb-a",  # two candidates: the smallest ID
    }
    assert counts["positives"] == 6 and counts["kept"] == 1 and counts["remapped"] == 3
    assert counts["remapped_ambiguous"] == 1 and counts["dropped"] == 2
    assert counts["dropped_by_reason"] == {"dropped_absent": 1, "dropped_not_a_target": 1}
    assert counts["dropped_by_script"] == {"Latin": 2}
    by_script = counts["by_script"]
    assert sum(c["kept"] for c in by_script.values()) == 1 and sum(c["remapped"] for c in by_script.values()) == 3
    assert by_script["Latin"]["dropped"] == 2  # kept, remapped and dropped add up to the positives, per script
    assert sum(sum(c.values()) for c in by_script.values()) == counts["positives"]


def test_negatives_are_never_dropped_even_when_the_entity_they_imitate_is_delisted(world):
    _, _, rows, counts = resolved(world)
    negatives = [q for q in rows if q["kind"] != "positive"]
    assert len(negatives) == counts["negatives"] == 4 and counts["negatives_dropped"] == 0
    assert any(q.get("entity") == "old-d1" for q in negatives)  # the near-miss for a delisted entity is still there
    assert all(q["new_id"] is None for q in negatives)


def test_negatives_survive_even_when_every_positive_is_dropped(tmp_path):
    old = write_snapshot(tmp_path / "o", [person("a", ["Aaa Bbb"]), person("b", ["Ccc Ddd"])], OLD_V)
    new = write_snapshot(tmp_path / "n", [person("zzz", ["Zzz Yyy"])], NEW_V)
    queries = [positive("a", "Aaa Bbb"), positive("b", "Ccc Ddd"), negative("Eee Fff"), negative("Ggg Hhh")]
    rows, counts = lg.resolve_holdout(
        [{**q, "script": "Latin"} for q in queries], lg.read_snapshot(old), lg.read_snapshot(new)
    )
    assert counts["dropped"] == 2 and counts["kept"] == counts["remapped"] == 0
    assert [q["kind"] for q in rows] == ["synthetic", "synthetic"] and counts["negatives_dropped"] == 0


def test_the_list_diff_counts_added_removed_and_changed(world):
    old, new = lg.read_snapshot(world["old"]), lg.read_snapshot(world["new"])
    diff = lg.diff_lists(old, new)
    # old list: k r1 r2 amb d1 d2 gone changed (8).
    # new list: k new-r1 new-r2 amb-b amb-a changed brand-new (7); d2 is still in the file but is not a target
    assert (diff["screening_list_old"], diff["screening_list_new"]) == (8, 7)
    assert diff["removed"] == 6 and diff["added"] == 5  # old-r1 r2 amb d1 d2 gone left; new-r1 r2 amb-a amb-b brand-new
    assert diff["changed"] == 1  # old-changed gained an alias
    assert diff["unchanged"] == 1  # old-k
    assert diff["entity_lines_old"] == 8 and diff["entity_lines_new"] == 8


def test_the_hold_out_rule_is_applied_to_the_new_snapshots_names():
    names = ["Anna Petrova", "Anna Petrovna", "Анна Петрова", "محمد علي"]
    # the Cyrillic spelling transliterates to the tested alias, so the variant rule holds it out as well
    assert lg.kept_names_for("variant", "Anna Petrova", names) == ["Anna Petrovna", "محمد علي"]
    assert lg.kept_names_for("cyr_to_latin", "Анна Петрова", names) == ["Anna Petrova", "Anna Petrovna"]
    assert lg.kept_names_for("latin_to_ar", "Anna Petrova", names) == ["محمد علي"]
    assert lg.kept_names_for("ar_to_latin", "محمد علي", names) == ["Anna Petrova", "Anna Petrovna"]


def test_genuinely_new_entities_are_not_successors_of_old_ones(world):
    old, new = lg.read_snapshot(world["old"]), lg.read_snapshot(world["new"])
    # new-r1, new-amb-a/b name an old ID, new-r2 is named by an old entity: successors. brand-new is genuinely new.
    assert lg.new_only_ids(old, new) == {"brand-new"}


# ---------- verification and labels ----------


def test_a_snapshot_without_a_version_file_or_with_a_wrong_checksum_is_refused(tmp_path):
    path = write_snapshot(tmp_path / "s", [person("a", ["Aaa Bbb"])], OLD_V)
    (path.with_name("VERSION")).unlink()
    with pytest.raises(SnapshotRefused, match="cannot be verified"):
        lg.verify_snapshot(path)
    path = write_snapshot(tmp_path / "s2", [person("a", ["Aaa Bbb"])], OLD_V)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(SnapshotRefused, match="does not match the SHA-1"):
        lg.verify_snapshot(path)
    with pytest.raises(SnapshotRefused, match="not a file"):
        lg.verify_snapshot(tmp_path / "nowhere.json")


def test_the_same_snapshot_twice_is_refused(world):
    with pytest.raises(SnapshotRefused, match="nothing to compare"):
        lg.run_gate(world["old"], world["old"], world["holdout"])


def test_an_older_new_snapshot_is_labelled_a_reverse_time_churn_test_everywhere(world):
    forward = lg.run_gate(world["old"], world["new"], world["holdout"])
    reverse = lg.run_gate(world["new"], world["old"], world["holdout"])
    assert forward["label"] == "forward list update"
    assert reverse["label"] == "REVERSE-TIME CHURN TEST" and "OLDER" in reverse["direction"]
    assert all(line.startswith("[REVERSE-TIME CHURN TEST] ") for line in lg.summary_lines(reverse))
    assert not any("REVERSE" in line for line in lg.summary_lines(forward))
    for section in ("list_diff", "resolution", "queries_in_resolved_set", "metrics_on_resolved_set", "gate"):
        assert reverse[section]["label"] == "REVERSE-TIME CHURN TEST"
    assert (
        "REVERSE_TIME" in lg.default_report_path(reverse).name and "REVERSE" not in lg.default_report_path(forward).name
    )


def test_every_report_names_the_holdout_and_engine_behind_its_headline_numbers(world):
    report = lg.run_gate(world["old"], world["new"], world["holdout"])
    p = report["provenance"]
    assert p["holdout"] == world["holdout"].name and len(p["holdout_sha256"]) == 64
    assert "third" in p["holdout_is"] and "rarity-weighted" in p["engine"]
    assert set(p["not_the_same_as"]) == {"F1 0.777", "F1 0.776", "Arabic burden 22.6%"}


# ---------- the paired bootstrap, by hand ----------


def arrays(**kw) -> dict:
    base = {
        k: np.array([], dtype=float)
        for k in ("pos_tp_old", "pos_tp_new", "pos_wrong_old", "pos_wrong_new", "neg_old", "neg_new")
    }
    base["arabic"] = np.array([], dtype=bool)
    return {**base, **{k: np.array(v, dtype=float if k != "arabic" else bool) for k, v in kw.items()}}


def test_the_paired_bootstrap_matches_a_hand_calculation_on_two_positives():
    # Two positives. Old finds the first only, new finds both; no negatives, no wrong alerts.
    # F1 old = 2*1*0.5/1.5 = 2/3, F1 new = 1, so the point delta is 1/3.
    # A resample of 2 is (A,A) (A,B) (B,A) (B,B), each 1/4. Old F1 is 1, 2/3, 2/3, 0 (no true alert: precision 0).
    # New F1 is 1 in all four. Deltas: 0, 1/3, 1/3, 1. So 25% of draws are 0 and 25% are 1:
    # the 2.5th percentile is 0 and the 97.5th is 1, whatever the seed.
    a = arrays(pos_tp_old=[1, 0], pos_tp_new=[1, 1], pos_wrong_old=[0, 0], pos_wrong_new=[0, 0])
    for seed in (1, 2, 3):
        boot = lg.paired_bootstrap(a, draws=1000, seed=seed)
        assert boot["f1"]["old"] == pytest.approx(2 / 3) and boot["f1"]["new"] == 1.0
        assert boot["f1"]["delta"] == pytest.approx(1 / 3)
        assert boot["f1"]["delta_ci95"] == [0.0, 1.0]
        assert boot["arabic_burden"] is None


def test_the_bootstrap_distribution_has_the_hand_calculated_mean():
    a = arrays(pos_tp_old=[1, 0], pos_tp_new=[1, 1], pos_wrong_old=[0, 0], pos_wrong_new=[0, 0])
    rng_draws = lg.paired_bootstrap(a, draws=20000, seed=5)
    # the draws are 0, 1/3, 1/3 or 1 with probabilities 1/4, 1/4, 1/4, 1/4; the interval edges are 0 and 1 for any
    # number of draws, because a quarter of the mass sits at each end
    assert rng_draws["f1"]["delta_ci95"] == [0.0, 1.0]


def test_identical_runs_give_a_zero_delta_with_a_zero_width_interval():
    a = arrays(
        pos_tp_old=[1, 1, 0], pos_tp_new=[1, 1, 0], pos_wrong_old=[0, 0, 1], pos_wrong_new=[0, 0, 1],
        neg_old=[0, 1, 0, 0], neg_new=[0, 1, 0, 0], arabic=[True, True, False, False],
    )  # fmt: skip
    boot = lg.paired_bootstrap(a, draws=1000, seed=9)
    assert boot["f1"]["delta"] == 0.0 and boot["f1"]["delta_ci95"] == [0.0, 0.0]
    assert boot["arabic_burden"]["delta"] == 0.0 and boot["arabic_burden"]["delta_ci95"] == [0.0, 0.0]


def test_the_arabic_burden_delta_by_hand():
    # Three Arabic negatives, none alerting before and all alerting now: every resample has burden 0 -> 1, delta 1.
    a = arrays(
        neg_old=[0, 0, 0, 0], neg_new=[1, 1, 1, 0], arabic=[True, True, True, False],
        pos_tp_old=[1], pos_tp_new=[1], pos_wrong_old=[0], pos_wrong_new=[0],
    )  # fmt: skip
    boot = lg.paired_bootstrap(a, draws=1000, seed=4)
    burden = boot["arabic_burden"]
    assert (burden["n"], burden["old"], burden["new"], burden["delta"]) == (3, 0.0, 1.0, 1.0)
    assert burden["delta_ci95"] == [1.0, 1.0]


def test_the_same_seed_gives_the_same_interval_and_another_seed_a_similar_one():
    rng = np.random.default_rng(0)
    old = (rng.random(300) < 0.7).astype(float)
    new = np.where(rng.random(300) < 0.1, 1 - old, old)
    a = arrays(pos_tp_old=old, pos_tp_new=new, pos_wrong_old=np.zeros(300), pos_wrong_new=np.zeros(300),
               neg_old=np.zeros(50), neg_new=np.zeros(50), arabic=np.zeros(50, dtype=bool))  # fmt: skip
    one, again, other = (lg.paired_bootstrap(a, 1000, s) for s in (1, 1, 2))
    assert one == again
    assert abs(one["f1"]["delta_ci95"][0] - other["f1"]["delta_ci95"][0]) < 0.02


# ---------- the gate rule at the tolerance boundaries ----------

TOL = GateConfig(f1_tolerance=0.01, arabic_burden_tolerance=0.02)


def boot_with(f1_ci, burden_ci=None) -> dict:
    return {
        "f1": {"delta_ci95": list(f1_ci)},
        "arabic_burden": None if burden_ci is None else {"delta_ci95": list(burden_ci)},
    }


@pytest.mark.parametrize(
    ("f1_ci", "burden_ci", "passed"),
    [
        ((-0.05, -0.0100001), None, False),  # entirely below -tolerance
        ((-0.05, -0.01), None, True),  # the upper end touches -tolerance: passes
        ((-0.05, -0.0099), None, True),
        ((-0.05, 0.02), None, True),  # an interval that includes zero passes
        ((0.0, 0.0), (0.0200001, 0.09), False),  # entirely above +tolerance
        ((0.0, 0.0), (0.02, 0.09), True),  # the lower end touches +tolerance: passes
        ((0.0, 0.0), (-0.01, 0.5), True),
        ((-0.2, -0.11), (0.03, 0.04), False),  # both
        ((0.01, 0.2), (-0.2, -0.1), True),  # improvements never fail
    ],
)
def test_the_gate_passes_and_fails_exactly_at_the_tolerances(f1_ci, burden_ci, passed):
    verdict = lg.decide(boot_with(f1_ci, burden_ci), TOL)
    assert verdict["passed"] is passed
    assert bool(verdict["reasons"]) is (not passed)
    assert "judgement" in verdict["tolerances_are"]


def test_both_reasons_are_named_when_both_tolerances_are_broken():
    verdict = lg.decide(boot_with((-0.3, -0.2), (0.1, 0.2)), TOL)
    assert len(verdict["reasons"]) == 2
    assert "F1 interval" in verdict["reasons"][0] and "Arabic burden interval" in verdict["reasons"][1]


def test_gate_settings_come_from_the_config_file_and_bad_values_are_refused():
    config = GateConfig.load()
    assert (config.f1_tolerance, config.arabic_burden_tolerance) == (0.01, 0.02)
    assert (config.bootstrap_draws, config.seed) == (1000, 11)
    assert GateConfig.load(Path("/nonexistent.json")) == GateConfig()
    for kwargs in ({"f1_tolerance": -0.1}, {"arabic_burden_tolerance": 1.0}, {"bootstrap_draws": 10}):
        with pytest.raises(ValueError):
            GateConfig(**kwargs)


# ---------- end to end on small snapshots ----------


def numbered_snapshot(directory: Path, version: str, n: int, shift: str = "", extra: list[dict] | None = None) -> Path:
    entities = [
        person(f"e{i}", [f"{word(i, 6)} {word(i + 700, 6)}{shift}", f"{word(i, 6)} {word(i + 700, 6)}x{shift}"])
        for i in range(n)
    ]
    return write_snapshot(directory, entities + (extra or []), version)


def numbered_holdout(path: Path, n: int, arabic: list[str] = ()) -> Path:
    queries = [positive(f"e{i}", f"{word(i, 6)} {word(i + 700, 6)}") for i in range(n)]
    queries += [negative(f"{word(i + 3000, 7)} {word(i + 4000, 6)}") for i in range(n // 2)]
    queries += [negative(a, "Arabic") for a in arabic]
    kept = {f"e{i}": [f"{word(i, 6)} {word(i + 700, 6)}x"] for i in range(n)}
    return write_holdout(path, queries, kept)


def test_an_unchanged_list_passes_with_a_zero_delta_and_a_report_without_names(tmp_path):
    old = numbered_snapshot(tmp_path / "o", OLD_V, 40)
    new = numbered_snapshot(tmp_path / "n", NEW_V, 40)
    holdout = numbered_holdout(tmp_path / "h.json", 40)
    report = lg.run_gate(old, new, holdout)
    assert report["gate"]["passed"] and report["metrics_on_resolved_set"]["f1"]["delta"] == 0.0
    assert report["resolution"]["kept"] == 40 and report["resolution"]["dropped"] == 0
    assert report["list_diff"]["added"] == report["list_diff"]["removed"] == report["list_diff"]["changed"] == 0
    text = json.dumps(report)
    queries, _ = lg.load_holdout(holdout)
    assert not any(q["name"] in text for q in queries)  # no query text
    assert "e0" not in report and '"e1"' not in text  # no entity IDs
    assert report["list_growth_alerts_on_negatives"]["new_only_entities_on_the_list"] == 0


def test_losing_the_aliases_that_find_the_positives_fails_the_gate(tmp_path):
    n = 40
    old = numbered_snapshot(tmp_path / "o", OLD_V, n)
    # same entities, but every alias is now an unrelated name: none of the positives is found any more
    changed = [person(f"e{i}", [f"{word(i + 9000, 7)} {word(i + 9500, 6)}"]) for i in range(n)]
    new = write_snapshot(tmp_path / "n", changed, NEW_V)
    report = lg.run_gate(old, new, numbered_holdout(tmp_path / "h.json", n))
    f1 = report["metrics_on_resolved_set"]["f1"]
    assert f1["old"] > 0.9 and f1["new"] == 0.0
    assert f1["delta_ci95"][1] < -GateConfig().f1_tolerance
    assert not report["gate"]["passed"] and "F1 interval" in report["gate"]["reasons"][0]
    assert report["resolution"]["dropped"] == 0  # they are kept (same IDs), they are just no longer found


def test_new_entities_that_make_arabic_names_alert_fail_the_burden_check_and_are_counted(tmp_path):
    n = 40
    arabic = [f"محمد {chr(0x0627 + i)}{chr(0x0628 + i)}{chr(0x062C + i)}" for i in range(30)]
    old = numbered_snapshot(tmp_path / "o", OLD_V, n)
    brand_new = [person(f"new-{i}", [name]) for i, name in enumerate(arabic)]  # exist only in the new snapshot
    new = numbered_snapshot(tmp_path / "n", NEW_V, n, extra=brand_new)
    report = lg.run_gate(old, new, numbered_holdout(tmp_path / "h.json", n, arabic))
    burden = report["metrics_on_resolved_set"]["arabic_burden"]
    assert burden["n"] == 30 and burden["old"] == 0.0 and burden["new"] == 1.0
    assert not report["gate"]["passed"] and any("Arabic burden" in r for r in report["gate"]["reasons"])
    growth = report["list_growth_alerts_on_negatives"]
    assert growth["new_only_entities_on_the_list"] == 30
    assert growth["arabic_script_synthetic_negatives"]["new_alerts_from_new_only_entities"] == 30
    assert growth["all_negatives"]["new_alerts"] == 30 and growth["all_negatives"]["alerts_that_went_away"] == 0


def test_alerts_that_come_from_successor_entities_are_not_counted_as_new_entities(tmp_path):
    n = 40
    old = numbered_snapshot(tmp_path / "o", OLD_V, n)
    # the Arabic names alert in the new list through entities that merely replace old ones (referents name old IDs)
    arabic = [f"محمد {chr(0x0627 + i)}{chr(0x0628 + i)}{chr(0x062C + i)}" for i in range(10)]
    successors = [person(f"succ-{i}", [name], referents=[f"e{i}"]) for i, name in enumerate(arabic)]
    new = numbered_snapshot(tmp_path / "n", NEW_V, n, extra=successors)
    report = lg.run_gate(old, new, numbered_holdout(tmp_path / "h.json", n, arabic))
    growth = report["list_growth_alerts_on_negatives"]
    assert (
        growth["all_negatives"]["new_alerts"] == 10
        and growth["all_negatives"]["new_alerts_from_new_only_entities"] == 0
    )


def test_a_delisting_is_dropped_and_counted_by_script_not_failed_on(tmp_path):
    n = 40
    old = numbered_snapshot(tmp_path / "o", OLD_V, n)
    entities = [
        person(f"e{i}", [f"{word(i, 6)} {word(i + 700, 6)}", f"{word(i, 6)} {word(i + 700, 6)}x"]) for i in range(30)
    ]
    new = write_snapshot(tmp_path / "n", entities, NEW_V)  # e30..e39 are delisted
    report = lg.run_gate(old, new, numbered_holdout(tmp_path / "h.json", n))
    assert report["resolution"]["dropped"] == 10 and report["resolution"]["dropped_by_script"] == {"Latin": 10}
    assert report["queries_in_resolved_set"]["positives"] == 30 and report["gate"]["passed"]
    assert report["list_diff"]["removed"] == 10


# ---------- nothing is written to an audit directory; the CLI ----------


def test_the_gate_writes_nothing_to_an_audit_directory(tmp_path, monkeypatch):
    from test_replay import tree

    from audit import AuditLog

    audit_dir = tmp_path / "audit"
    AuditLog.create_new(audit_dir, fsync=False)
    monkeypatch.setenv("SCREENING_AUDIT_DIR", str(audit_dir))
    before = tree(audit_dir)
    old = numbered_snapshot(tmp_path / "o", OLD_V, 30)
    new = numbered_snapshot(tmp_path / "n", NEW_V, 30)
    lg.run_gate(old, new, numbered_holdout(tmp_path / "h.json", 30))
    holdout2 = str(numbered_holdout(tmp_path / "h2.json", 30))
    code = screen_cli.main(
        ["gate", "--new-snapshot", str(new), "--old-snapshot", str(old), "--holdout", holdout2,
         "--out", str(tmp_path / "report.json")]
    )  # fmt: skip
    assert code == 0 and tree(audit_dir) == before


def test_the_gate_module_does_not_import_the_audit_modules():
    import ast

    tree_ = ast.parse(Path(lg.__file__).read_text())
    imported = {n.module.split(".")[0] for n in ast.walk(tree_) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name.split(".")[0] for n in ast.walk(tree_) if isinstance(n, ast.Import) for a in n.names}
    assert not imported & {"audit", "audit_verify", "audited_screening", "audit_recover", "audit_purge", "replay"}


def test_cli_gate_exit_codes_pass_fail_and_refused(tmp_path, capsys):
    n = 40
    old = numbered_snapshot(tmp_path / "o", OLD_V, n)
    good = numbered_snapshot(tmp_path / "g", NEW_V, n)
    bad = write_snapshot(
        tmp_path / "b",
        [person(f"e{i}", [f"{word(i + 9000, 7)} {word(i + 9500, 6)}"]) for i in range(n)],
        "20261009000000-ccc",
    )
    holdout = str(numbered_holdout(tmp_path / "h.json", n))
    base = ["gate", "--old-snapshot", str(old), "--holdout", holdout]
    assert screen_cli.main([*base, "--new-snapshot", str(good), "--out", str(tmp_path / "r1.json")]) == 0
    assert "gate: PASS" in capsys.readouterr().out
    assert screen_cli.main([*base, "--new-snapshot", str(bad), "--out", str(tmp_path / "r2.json")]) == 1
    assert "gate: FAIL" in capsys.readouterr().out
    assert (
        screen_cli.main([*base, "--new-snapshot", str(old), "--out", str(tmp_path / "r3.json")]) == 2
    )  # same snapshot
    assert "nothing to compare" in capsys.readouterr().err
    assert screen_cli.main([*base, "--new-snapshot", str(tmp_path / "missing.json")]) == 2
    (good.with_name("VERSION")).unlink()
    assert screen_cli.main([*base, "--new-snapshot", str(good), "--out", str(tmp_path / "r4.json")]) == 2
    assert "cannot be verified" in capsys.readouterr().err
    assert not (tmp_path / "r3.json").exists() and not (tmp_path / "r4.json").exists()


def test_a_report_is_never_overwritten(tmp_path, capsys):
    old = numbered_snapshot(tmp_path / "o", OLD_V, 30)
    new = numbered_snapshot(tmp_path / "n", NEW_V, 30)
    holdout = str(numbered_holdout(tmp_path / "h.json", 30))
    args = ["gate", "--old-snapshot", str(old), "--new-snapshot", str(new), "--holdout", holdout,
            "--out", str(tmp_path / "report.json")]  # fmt: skip
    assert screen_cli.main(args) == 0
    first = (tmp_path / "report.json").read_text()
    assert screen_cli.main(args) == 2 and (tmp_path / "report.json").read_text() == first
    assert "refusing to run the gate" in capsys.readouterr().err

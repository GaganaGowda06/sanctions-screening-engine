"""Constructed drift batches on the real hold-out, and the no-name check on committed baselines.

These need the downloaded list and the hold-out, so they are skipped where data/ is absent (CI). The batch test takes
several minutes and is marked slow: run it with `pytest -m slow`. It is not production traffic: the batches are
seeded samples of the spent hold-out screened through the audited path (see src/run_drift_batches.py).

Run:
    pytest tests/test_drift_batches.py            # the baseline name check
    pytest -m slow tests/test_drift_batches.py    # the constructed batches
"""

import json
from pathlib import Path

import pytest

import run_drift_batches
from audit_drift import BASELINE_DIR, load_baseline
from make_baseline import HOLDOUT, load_queries, sample_indices
from normalise import normalise
from profile_data import DEFAULT_INPUT
from scoring import scoring_form

REAL_LIST = Path(DEFAULT_INPUT)
needs_holdout = pytest.mark.skipif(not HOLDOUT.exists(), reason="the hold-out is not on disk")
needs_data = pytest.mark.skipif(
    not REAL_LIST.exists() or not HOLDOUT.exists(), reason="the list or the hold-out is not on disk"
)
HASH_FIELDS = {"run_context_hash", "file_sha256", "holdout_sha256", "git_sha", "snapshot_sha256"}
TIME_FIELDS = {"generated_at_utc", "first_timestamp_utc", "last_timestamp_utc"}  # digits by chance, e.g. "1998"


def strings_of(value: object, skip_keys: frozenset[str] = frozenset()):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            if k not in skip_keys:
                yield from strings_of(v, skip_keys)
    elif isinstance(value, list):
        for v in value:
            yield from strings_of(v, skip_keys)


def name_derived(queries: list[dict], indices: list[int]) -> set[str]:
    out = set()
    for i in indices:
        q = queries[i]
        out |= {q["name"], normalise(q["name"]), scoring_form(q["name"]), *scoring_form(q["name"]).split()}
        out |= {s for s in (q["dob"], q["country"]) if s}
    return {s for s in out if s}


def check_no_name_derived(baseline: dict, forbidden: set[str]) -> None:
    """No string value in the baseline is a name-derived value, and none of the sample's name-derived strings occurs
    inside the report or the sample description. The `context` section (git SHA, library versions, settings) is the
    engine's own and holds no customer data; it is excluded from the substring scan because a name token can
    coincide with a word in it (a listed name containing "python" against "cpython"), but its values still must not
    equal a name-derived string."""
    for leaf in strings_of(baseline, frozenset(HASH_FIELDS)):
        assert leaf not in forbidden, leaf
    scanned = {k: v for k, v in baseline.items() if k not in ("context", *HASH_FIELDS)}
    # string VALUES only (a name token can coincide with a JSON key: "rabi" inside "arabic"), leaving out hashes and
    # timestamps, whose digits can contain a four-digit birth year by chance. They are still compared leaf by leaf.
    scrubbed = " ".join(strings_of(scanned, frozenset(HASH_FIELDS | TIME_FIELDS))).lower()
    for item in forbidden:
        if len(item) >= 4:  # shorter strings could sit inside a hex hash by chance
            assert item.lower() not in scrubbed, item
    assert "request_id" not in json.dumps(baseline)
    for leaf in strings_of(baseline):
        assert not (len(leaf) == 32 and set(leaf) <= set("0123456789abcdef")), "a request ID is in the baseline"


@needs_holdout
@pytest.mark.parametrize(
    "path", sorted(BASELINE_DIR.glob("*.json")) or [None], ids=lambda p: p.name[:12] if p else "none"
)
def test_a_committed_baseline_holds_nothing_name_derived(path):
    if path is None:
        pytest.skip("no baseline file in baseline/")
    baseline = load_baseline(path)
    sample = baseline["sample"]
    queries = load_queries(HOLDOUT)
    indices = sample_indices(queries, sample["size"], sample["seed"])
    assert sample["holdout"] == HOLDOUT.name and path.stem == baseline["decision_context_hash"]
    check_no_name_derived(baseline, name_derived(queries, indices))


@pytest.fixture(scope="module")
def batches(tmp_path_factory):
    """One run of the three batches (several minutes), shared by the tests below."""
    work = tmp_path_factory.mktemp("drift_batches")
    return run_drift_batches.run_batches(REAL_LIST, work), work


@pytest.mark.slow
@needs_data
def test_the_split_half_noise_floor_is_measured_and_reported(batches):
    out, _ = batches
    floor = out["noise_floor_of_the_baseline_sample"]
    assert floor["all"]["decisions_per_half"] > 0 and floor["all"]["median"] >= 0
    assert all(v["p95"] >= v["median"] >= 0 for k, v in floor.items() if k != "what_this_is")


@pytest.mark.slow
@needs_data
def test_fifty_stable_batches_rarely_raise_an_overall_warning(batches):
    """Replaces the earlier single-stable-batch test, which asserted that one fresh sample of 800 decisions has overall
    status ok. A single draw cannot be asserted against a probabilistic limit: even with correct limits a clean batch
    sometimes warns, so one batch passing or failing says little. The expectation is now about the rate over 50 seeded
    stable batches: the overall status is warn or worse in at most 10% of them and alert in at most 2%. This is a
    change of expectation, made after the single batch came out warn, and it is recorded here and in the project notes.
    The original single stable batch is still run, and its result is in the report (see the next test)."""
    out, _ = batches
    repeated = out["repeated_stable"]
    assert repeated["batches"] == 50
    assert repeated["overall_warn_or_worse_rate"] <= 0.10, repeated
    assert repeated["overall_alert_rate"] <= 0.02, repeated


@pytest.mark.slow
@needs_data
def test_the_original_single_stable_batch_is_still_run_and_its_result_reported(batches):
    out, _ = batches
    single = out["scenarios"]["stable"]
    assert single["size"] == 800 and "overall" in single["observed"]  # recorded whatever the status was
    assert single["result"]["multiplicity"] is not None and single["result"]["overall_uncorrected"] is not None


@pytest.mark.slow
@needs_data
def test_a_script_mix_shift_moves_the_mix_psi_and_leaves_within_script_psi_ok(batches):
    out, _ = batches
    shift = out["scenarios"]["mix_shift"]["observed"]
    assert shift["input_mix_psi"]["status"] == "alert", shift
    assert shift["within_script_score_psi"], "at least one script should have enough decisions for a PSI"
    assert all(v["status"] == "ok" for v in shift["within_script_score_psi"].values()), shift


@pytest.mark.slow
@needs_data
def test_the_mix_shift_does_move_the_input_mix_psi_far_past_the_alert_limit(batches):
    """The part of the mix-shift expectation that held: the mix moved, and the drift check saw it."""
    out, _ = batches
    mix = out["scenarios"]["mix_shift"]["result"]["input_mix_psi"]
    assert mix["status"] == "alert" and mix["value"] > 2 * 0.25


@pytest.mark.slow
@needs_data
def test_degraded_inputs_move_the_score_distribution_and_the_near_threshold_rate(batches):
    out, _ = batches
    degraded = out["scenarios"]["degraded"]["observed"]
    assert degraded["score_psi"]["status"] in ("warn", "alert"), degraded
    assert degraded["near_threshold"]["status"] in ("warn", "alert"), degraded
    assert out["scenarios"]["degraded"]["expected_pattern_seen"]


@pytest.mark.slow
@needs_data
def test_the_batch_baseline_and_report_hold_nothing_name_derived(batches):
    out, work = batches
    queries = load_queries(HOLDOUT)
    indices = sample_indices(queries, out["baseline"]["size"], out["seed"])
    forbidden = name_derived(queries, indices)
    baseline = load_baseline(next((work / "baseline").glob("*.json")))
    check_no_name_derived(baseline, forbidden)
    # string values only, leaving out hashes, timestamps and the report's own fixed prose ("not production
    # traffic" would otherwise collide with a listed name that contains "production")
    prose = frozenset(HASH_FIELDS | TIME_FIELDS | {"what_this_is", "expected"})
    text = " ".join(strings_of(out, prose)).lower()
    assert not any(f.lower() in text for f in forbidden if len(f) >= 6)


# ---------- the check's exclusions are an explicit list, and it catches what it should ----------


def tiny_baseline() -> dict:
    return {
        "run_context_hash": "a" * 64,
        "decision_context_hash": "b" * 64,
        "context": {"git_sha": "c" * 40, "python_implementation": "cpython"},
        "sample": {"holdout": "h.json", "holdout_sha256": "d" * 64, "seed": 1, "size": 10, "script_mix": {"Latin": 10}},
        "report": {
            "generated_at_utc": "2026-10-08T10:00:00+00:00",
            "config": {"file_sha256": "e" * 64},
            "groups": [{"run_context_hash": "a" * 64, "per_script": {"Latin": {"count": 10}}}],
        },
    }


def test_the_exclusions_are_exact_field_names_not_patterns():
    assert HASH_FIELDS == {"run_context_hash", "file_sha256", "holdout_sha256", "git_sha", "snapshot_sha256"}
    assert TIME_FIELDS == {"generated_at_utc", "first_timestamp_utc", "last_timestamp_utc"}
    assert not any(any(ch in name for ch in "*?[]") for name in HASH_FIELDS | TIME_FIELDS)  # no wildcards
    check_no_name_derived(tiny_baseline(), {"Mohammed Ali", "mohammed", "1970-01-01", "iq"})  # a clean one passes


@pytest.mark.parametrize(
    "path",
    [
        ("sample", "note"),  # a field nobody excluded
        ("report", "extra"),
        ("report", "groups", 0, "per_script", "Latin", "label"),
        ("sample", "holdout_sha256x"),  # looks like an excluded name but is not one
        ("report", "config", "sha256"),  # nor is this
    ],
)
def test_a_name_derived_value_in_a_field_that_is_not_excluded_is_caught(path):
    baseline = tiny_baseline()
    node = baseline
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = "Visiting Mohammed Ali Hassan"
    with pytest.raises(AssertionError):
        check_no_name_derived(baseline, {"Mohammed Ali Hassan", "mohammed", "1970"})


def test_a_name_that_is_exactly_a_value_is_caught_even_in_the_context_section():
    baseline = tiny_baseline()
    baseline["context"]["python_implementation"] = (
        "mohammed"  # the context section is exempt from the substring scan only
    )
    with pytest.raises(AssertionError):
        check_no_name_derived(baseline, {"mohammed"})


def test_a_request_id_is_caught_wherever_it_is():
    baseline = tiny_baseline()
    baseline["report"]["groups"][0]["per_script"]["Latin"]["id"] = "f" * 32
    with pytest.raises(AssertionError, match="request ID"):
        check_no_name_derived(baseline, set())


def test_the_excluded_fields_are_still_compared_leaf_by_leaf():
    baseline = tiny_baseline()
    baseline["report"]["generated_at_utc"] = "mohammed"  # excluded from the substring scan, not from equality
    with pytest.raises(AssertionError):
        check_no_name_derived(baseline, {"mohammed"})

"""Tests for the decision context hash: what decides a decision, as opposed to the exact commit and record format.

Baselines match on decision_context_hash; replay still needs the full context hash.

Run:
    pytest tests/test_decision_context.py
"""

import dataclasses
import json
import shutil

import pytest
from test_replay import CLOCK, make_engine, tree

import run_context as rc
from audit_drift import BaselineMismatch, compare_to_baseline, save_baseline
from audit_metrics import MetricsConfig, compute_metrics
from replay import IDENTICAL, REFUSED_CONTEXT, replay
from screening import ScreeningConfig

NO_SHRINK = MetricsConfig(0.05, 1)


@pytest.fixture
def ctx(tmp_path):
    return make_engine(tmp_path).run_context


def changed(context: rc.RunContext, **fields) -> rc.RunContext:
    return dataclasses.replace(context, **fields)


# ---------- what the hash covers and what it leaves out ----------


@pytest.mark.parametrize(
    "fields",
    [
        {"git_sha": "b" * 40},
        {"git_dirty": True},
        {"audit_schema_version": 99},
        {"top_candidates": 3},
        {"pins_file_sha256": "f" * 64},
        {"lockfile_sha256": "f" * 64},
        {"config_file_sha256": "f" * 64},
        {"python_implementation": "pypy"},
        {"schema_version": 99},
    ],
)
def test_the_decision_hash_ignores_commit_dirty_flag_audit_schema_and_the_like(ctx, fields):
    other = changed(ctx, **fields)
    assert other.decision_context_hash == ctx.decision_context_hash
    assert other.context_hash != ctx.context_hash  # replay's full hash still sees it


def artefact_edit(ctx: rc.RunContext, name: str) -> dict:
    return {"artefacts": tuple((k, v + "x" if k == name else v) for k, v in ctx.artefacts)}


@pytest.mark.parametrize("source", sorted(rc.SCORE_PATH_SOURCES))
def test_each_scoring_and_normalisation_source_hash_is_covered(ctx, source):
    assert changed(ctx, **artefact_edit(ctx, source)).decision_context_hash != ctx.decision_context_hash


def test_the_audit_schema_source_is_not_a_decision_input_but_is_in_the_full_hash(ctx):
    other = changed(ctx, **artefact_edit(ctx, "audit_schema.py"))
    assert other.decision_context_hash == ctx.decision_context_hash and other.context_hash != ctx.context_hash


@pytest.mark.parametrize(
    "fields",
    [
        {"snapshot_id": "20990101000000-new"},
        {"snapshot_sha256": "f" * 64},
        {"rarity_table_sha256": "f" * 64},
        {"python_version": "3.99.0"},
        {"unidata_version": "99.0.0"},
        {"libraries": (("rapidfuzz", "9.9.9"),)},
    ],
)
def test_each_decision_input_changes_the_decision_hash_and_the_full_hash(ctx, fields):
    other = changed(ctx, **fields)
    assert other.decision_context_hash != ctx.decision_context_hash and other.context_hash != ctx.context_hash


def test_config_and_threshold_and_engine_settings_including_the_candidate_cap_are_covered(ctx):
    config = json.loads(ctx.config)
    config["global_threshold"] = 0.85
    settings = json.loads(ctx.engine_settings)
    cap = dict(settings, candidate_cap=settings["candidate_cap"] + 1)
    for fields in ({"config": rc.canonical_json(config)}, {"engine_settings": rc.canonical_json(cap)}):
        assert changed(ctx, **fields).decision_context_hash != ctx.decision_context_hash
    assert "candidate_cap" in ctx.decision_context()["engine_settings"]


def test_the_decision_context_lists_exactly_what_decides_decisions(ctx):
    assert set(ctx.decision_context()) == {
        "config", "engine_settings", "snapshot_id", "snapshot_sha256", "rarity_table_sha256", "scoring_sources",
        "libraries", "python_version", "unidata_version",
    }  # fmt: skip
    assert set(ctx.decision_context()["scoring_sources"]) == set(rc.SCORE_PATH_SOURCES)
    assert "git_sha" not in json.dumps(ctx.decision_context())


# ---------- the stored context file ----------


def test_the_stored_context_file_holds_the_decision_hash_and_checks_it(tmp_path):
    engine = make_engine(tmp_path)
    path = engine.contexts_dir / f"{engine.context_hash}.json"
    stored = json.loads(path.read_text())
    assert stored["decision_context_hash"] == engine.run_context.decision_context_hash
    assert rc.load_context(engine.contexts_dir, engine.context_hash) == engine.run_context
    stored["decision_context_hash"] = "0" * 64
    path.write_text(json.dumps(stored))
    with pytest.raises(rc.AuditError, match="does not match the context it sits in"):
        rc.load_context(engine.contexts_dir, engine.context_hash)


def test_a_context_dict_without_the_decision_hash_still_loads():
    # contexts written before the field existed: the hash is derived, so nothing is lost
    context = rc.RunContext.from_dict({k: v for k, v in make_context_dict().items() if k != "decision_context_hash"})
    assert context.decision_context_hash


def make_context_dict() -> dict:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        return make_engine(Path(d)).run_context.as_dict()


# ---------- baseline matching against replay ----------


def screened_log(tmp_path, *engines_git):
    """One log, an engine per entry (a git function, or a ScreeningConfig), each screening a few names."""
    first = make_engine(tmp_path, git=engines_git[0] if callable(engines_git[0]) else None)
    engines = [first]
    for spec in engines_git[1:]:
        kwargs = {"git": spec} if callable(spec) else {"config": spec}
        engines.append(make_engine(tmp_path, log=first.log, **kwargs))
    ids = []
    for engine in engines:
        for name in ("Ivan Petrov", "Mohammed Ali Hassan", "Zebulon Quixotic"):
            ids.append((engine, engine.audited_screen(name).request_id))
    return first.log, engines, ids


def test_metrics_groups_record_their_decision_context_hash(tmp_path):
    log, (engine,), _ = screened_log(tmp_path, lambda: ("a" * 40, False))
    report, _ = compute_metrics(log.headers.path.parent, 0, len(log.headers.read_all()) - 1, NO_SHRINK, CLOCK)
    (group,) = report["groups"]
    assert group["run_context_hash"] == engine.context_hash
    assert group["decision_context_hash"] == engine.run_context.decision_context_hash


def baseline_from(report: dict, engine, tmp_path) -> dict:
    """A baseline built from the first group of a report, as make_baseline would."""
    first = {**report, "groups": [report["groups"][0]]}
    path = save_baseline(first, engine.run_context.as_dict(), {"seed": 0}, tmp_path / "baseline")
    return json.loads(path.read_text())


def test_a_new_commit_keeps_baseline_matching_but_fails_replay(tmp_path):
    log, (old, new), ids = screened_log(
        tmp_path, lambda: ("a" * 40, False), lambda: ("b" * 40, False)
    )  # same code and list; only the git commit differs
    assert old.run_context.decision_context_hash == new.run_context.decision_context_hash
    assert old.context_hash != new.context_hash

    report, _ = compute_metrics(log.headers.path.parent, 0, len(log.headers.read_all()) - 1, NO_SHRINK, CLOCK)
    assert len(report["groups"]) == 2  # two run contexts, never mixed...
    assert (
        report["groups"][0]["decision_context_hash"] == report["groups"][1]["decision_context_hash"]
    )  # ...one decision context
    baseline = baseline_from(report, old, tmp_path)
    assert baseline["decision_context_hash"] == old.run_context.decision_context_hash
    result = compare_to_baseline(report, baseline)  # matches: the baseline applies to the new commit's traffic too
    assert result["decision_context_hash"] == baseline["decision_context_hash"]
    assert result["other_matching_contexts"]  # the second commit's group was matched too

    # but replay of a record made at the old commit, by an engine at the new commit, is refused
    old_record = next(rid for engine, rid in ids if engine is old)
    refused = replay(new, old_record)
    assert refused.status == REFUSED_CONTEXT and any(d.startswith("git_sha:") for d in refused.differences)
    assert replay(old, old_record).status == IDENTICAL


def test_a_dirty_flag_alone_also_keeps_baseline_matching_and_fails_replay(tmp_path):
    log, (clean, dirty), ids = screened_log(tmp_path, lambda: ("a" * 40, False), lambda: ("a" * 40, True))
    assert clean.run_context.decision_context_hash == dirty.run_context.decision_context_hash
    report, _ = compute_metrics(log.headers.path.parent, 0, len(log.headers.read_all()) - 1, NO_SHRINK, CLOCK)
    compare_to_baseline(report, baseline_from(report, clean, tmp_path))
    assert replay(dirty, ids[0][1]).status == REFUSED_CONTEXT


def test_a_threshold_change_breaks_both_baseline_matching_and_replay(tmp_path):
    log, (old, new), ids = screened_log(tmp_path, lambda: ("a" * 40, False), ScreeningConfig(global_threshold=0.85))
    assert old.run_context.decision_context_hash != new.run_context.decision_context_hash
    report, _ = compute_metrics(log.headers.path.parent, 0, len(log.headers.read_all()) - 1, NO_SHRINK, CLOCK)
    old_group = {**report, "groups": [g for g in report["groups"] if g["run_context_hash"] == old.context_hash]}
    new_group = {**report, "groups": [g for g in report["groups"] if g["run_context_hash"] == new.context_hash]}
    baseline = baseline_from(old_group, old, tmp_path)
    with pytest.raises(BaselineMismatch, match="decision context"):
        compare_to_baseline(new_group, baseline)
    assert replay(new, ids[0][1]).status == REFUSED_CONTEXT


def test_the_refusal_names_the_threshold_but_not_the_commit(tmp_path):
    log, (old, new), _ = screened_log(tmp_path, lambda: ("a" * 40, False), ScreeningConfig(global_threshold=0.85))
    report, _ = compute_metrics(log.headers.path.parent, 0, len(log.headers.read_all()) - 1, NO_SHRINK, CLOCK)
    old_report = {**report, "groups": [g for g in report["groups"] if g["run_context_hash"] == old.context_hash]}
    new_report = {**report, "groups": [g for g in report["groups"] if g["run_context_hash"] == new.context_hash]}
    contexts = {new.context_hash: new.run_context.as_dict()}
    with pytest.raises(BaselineMismatch) as stop:
        compare_to_baseline(new_report, baseline_from(old_report, old, tmp_path), live_contexts=contexts)
    assert "config.global_threshold: stored 0.86, live 0.85" in str(stop.value)


def test_drift_matching_leaves_the_audit_store_untouched(tmp_path):
    log, (old, new), _ = screened_log(tmp_path, lambda: ("a" * 40, False), lambda: ("b" * 40, False))
    report, _ = compute_metrics(log.headers.path.parent, 0, len(log.headers.read_all()) - 1, NO_SHRINK, CLOCK)
    before = tree(log.headers.path.parent)
    compare_to_baseline(report, baseline_from(report, old, tmp_path))
    assert tree(log.headers.path.parent) == before
    shutil.rmtree(tmp_path / "baseline")

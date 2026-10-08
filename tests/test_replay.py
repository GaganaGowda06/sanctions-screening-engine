"""Tests for src/replay.py, the context files, and the CLI replay command.

Small engines are built in memory over a temporary audit store. Changing the threshold, or a source file, gives a
second engine with another run context over the same store.

Run:
    pytest tests/test_replay.py
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

import audit
import run_context as rc
import screen_cli
from audit import AuditError, AuditLog, HeaderLog
from audited_screening import AuditedEngine
from replay import DIFFERENT, IDENTICAL, NOT_REPLAYABLE, REFUSED_CONTEXT, REFUSED_TAMPERED, UNKNOWN_REQUEST, replay
from screening import ScreeningConfig, ScreeningEngine

ENTITIES = [
    {"id": "e1", "names": ["Mohammed Ali Hassan"], "birth_dates": ["1970-01-01"], "countries": ["iq"]},
    {"id": "e2", "names": ["Ivan Petrov"], "birth_dates": ["1980-05-05"], "countries": ["ru"]},
    {"id": "e3", "names": ["محمد علي حسن"], "birth_dates": [], "countries": []},
    {"id": "e4", "names": ["Acme Trading Limited"], "birth_dates": [], "countries": []},
]
CLOCK = lambda: datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)  # noqa: E731
QUERIES = [
    ("Mohammed Ali Hassan", "1970-01-01", "iq"),
    ("Mohammed Ali Hassan", None, None),
    ("Muhammad Ali Hasan", None, "iq"),
    ("محمد علي حسن", None, None),
    ("Ivan Petrov", "1980", "ru"),
    ("Ivan Petrov", "1950-02-02", "us"),
    ("Zebulon Quixotic", None, None),
    ("Acme Trading", None, None),
    ("Ivan Hassan", None, None),
    ("Petrov Mohammed", None, None),
]


def write_list(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    body = b'{"id":"e1"}\n'
    path = directory / "entities.ftm.json"
    path.write_bytes(body)
    (directory / "VERSION").write_text(f"version=20260101000000-abc\nsha1={hashlib.sha1(body).hexdigest()}\n")  # noqa: S324
    return path


def make_engine(
    tmp_path, *, log=None, config=None, src_dir=None, for_screening=True, git=None, versions=None
) -> AuditedEngine:
    engine = ScreeningEngine.from_entities(ENTITIES, config or ScreeningConfig())
    kwargs = {"src_dir": src_dir} if src_dir else {}
    context = rc.build_run_context(
        engine,
        write_list(tmp_path / "list"),
        git=git or (lambda: ("a" * 40, False)),
        versions=versions or (lambda: {"rapidfuzz": "3.14.6"}),
        **kwargs,
    )
    log = log or AuditLog.create_new(tmp_path / "audit", fsync=False)
    return AuditedEngine(engine, context, log, clock=CLOCK, for_screening=for_screening)


@pytest.fixture
def recorded(tmp_path):
    engine = make_engine(tmp_path)
    ids = [engine.audited_screen(n, b, c, customer_ref=f"c-{i}").request_id for i, (n, b, c) in enumerate(QUERIES)]
    return engine, ids


def tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# ---------- identical replay ----------


def test_replaying_every_record_of_a_batch_log_is_identical(recorded):
    engine, ids = recorded
    results = [replay(engine, rid) for rid in ids]
    assert [r.status for r in results] == [IDENTICAL] * len(ids), [r for r in results if not r.identical][:2]
    assert all(r.differences == () and r.identical for r in results)
    assert {"alert", "clear"} <= {h["decision"] for h in engine.log.headers.read_all() if h["kind"] == "decision"}


def test_a_fresh_engine_built_the_same_way_replays_the_records_too(recorded, tmp_path):
    engine, ids = recorded
    later = make_engine(tmp_path, log=engine.log, for_screening=False)
    assert later.context_hash == engine.context_hash
    assert all(replay(later, rid).identical for rid in ids)


def test_replay_leaves_the_whole_audit_store_byte_identical_and_never_calls_the_logging_path(recorded, monkeypatch):
    engine, ids = recorded

    def boom(*a, **k):
        raise AssertionError("replay must not write to the log")

    monkeypatch.setattr(AuditLog, "write", boom)
    monkeypatch.setattr(HeaderLog, "append", boom)
    monkeypatch.setattr(HeaderLog, "append_locked", boom)
    before = tree(engine.log.headers.path.parent)
    for rid in ids:
        assert replay(engine, rid).identical
    assert replay(engine, "f" * 32).status == UNKNOWN_REQUEST
    assert tree(engine.log.headers.path.parent) == before


def test_components_is_called_once_per_replay(recorded):
    engine, ids = recorded
    calls = []
    original = engine.engine.screener.components
    engine.engine.screener.components = lambda *a, **k: calls.append(1) or original(*a, **k)
    replay(engine, ids[0])
    assert len(calls) == 1


# ---------- a different context is refused ----------


def test_a_changed_threshold_makes_replay_refuse_with_a_diff_naming_the_threshold(recorded, tmp_path):
    engine, ids = recorded
    other = make_engine(tmp_path, log=engine.log, config=ScreeningConfig(global_threshold=0.85), for_screening=False)
    result = replay(other, ids[0])
    assert result.status == REFUSED_CONTEXT and not result.identical
    assert any(d.startswith("config.global_threshold: stored 0.86, live 0.85") for d in result.differences)
    assert result.reproduce["git_sha"] == "a" * 40
    assert result.reproduce["snapshot_id"] == "20260101000000-abc"
    assert "different context" in result.reason


def test_the_refused_replay_gives_no_answer_for_any_record(recorded, tmp_path):
    engine, ids = recorded
    other = make_engine(tmp_path, log=engine.log, config=ScreeningConfig(global_threshold=0.5), for_screening=False)
    assert {replay(other, rid).status for rid in ids} == {REFUSED_CONTEXT}


def test_a_changed_artefact_is_named_in_the_diff(recorded, tmp_path):
    engine, ids = recorded
    edited = tmp_path / "edited_src"
    edited.mkdir()
    for name in rc.HASHED_SOURCES:
        shutil.copy(rc.SRC_DIR / name, edited / name)
    (edited / "normalise.py").write_text((edited / "normalise.py").read_text() + "\n# edited\n")
    other = make_engine(tmp_path, log=engine.log, src_dir=edited, for_screening=False)
    result = replay(other, ids[0])
    assert result.status == REFUSED_CONTEXT
    assert [d for d in result.differences if d.startswith("artefacts.")] == [
        next(d for d in result.differences if d.startswith("artefacts.normalise.py:"))
    ]
    assert not any(d.startswith("artefacts.scoring.py") for d in result.differences)


def test_a_changed_engine_setting_and_a_changed_snapshot_are_named(recorded, tmp_path):
    engine, ids = recorded
    other_engine = ScreeningEngine.from_entities(ENTITIES[:3], ScreeningConfig())
    other_context = rc.build_run_context(
        other_engine,
        write_list(tmp_path / "list2"),
        git=lambda: ("b" * 40, True),
        versions=lambda: {"rapidfuzz": "3.15.0"},
    )
    other = AuditedEngine(other_engine, other_context, engine.log, for_screening=False)
    result = replay(other, ids[0])
    text = "\n".join(result.differences)
    assert "git_sha: stored" in text and "git_dirty: stored False, live True" in text
    assert "libraries.rapidfuzz: stored '3.14.6', live '3.15.0'" in text
    assert "rarity_table_sha256" in text


def test_changing_top_candidates_is_refused_with_a_diff_naming_it(recorded, tmp_path, monkeypatch):
    engine, ids = recorded
    monkeypatch.setattr(rc, "TOP_CANDIDATES", 5)  # the constant in audit_schema, as run_context reads it
    other = make_engine(tmp_path, log=engine.log, for_screening=False)
    assert other.top_candidates == 5
    result = replay(other, ids[0])
    assert result.status == REFUSED_CONTEXT
    assert "top_candidates: stored 10, live 5" in result.differences


def test_a_changed_schema_file_is_named_and_storage_code_is_not_hashed(recorded, tmp_path):
    engine, ids = recorded
    edited = tmp_path / "edited_src"
    edited.mkdir()
    for name in rc.HASHED_SOURCES:
        shutil.copy(rc.SRC_DIR / name, edited / name)
    (edited / "audit_schema.py").write_text((edited / "audit_schema.py").read_text() + "\n# edited\n")
    other = make_engine(tmp_path, log=engine.log, src_dir=edited, for_screening=False)
    result = replay(other, ids[0])
    assert result.status == REFUSED_CONTEXT
    assert any(d.startswith("artefacts.audit_schema.py:") for d in result.differences)
    hashed = {name for name, _ in engine.run_context.artefacts}
    assert "audit_schema.py" in hashed
    assert not hashed & {"audit.py", "audit_verify.py", "replay.py", "audited_screening.py"}  # how it is stored


def test_a_missing_context_file_is_refused_with_the_reason(recorded, tmp_path):
    engine, ids = recorded
    (engine.contexts_dir / f"{engine.context_hash}.json").unlink()
    other = make_engine(tmp_path, log=engine.log, config=ScreeningConfig(global_threshold=0.85), for_screening=False)
    result = replay(other, ids[0])
    assert result.status == REFUSED_CONTEXT and "no context file" in result.reason and result.differences == ()


def test_a_changed_context_file_is_refused(recorded, tmp_path):
    engine, ids = recorded
    path = engine.contexts_dir / f"{engine.context_hash}.json"
    stored = json.loads(path.read_text())
    stored["config"]["global_threshold"] = 0.1
    path.write_text(json.dumps(stored))
    other = make_engine(tmp_path, log=engine.log, config=ScreeningConfig(global_threshold=0.85), for_screening=False)
    result = replay(other, ids[0])
    assert result.status == REFUSED_CONTEXT
    # an edit to a decision field is caught by the decision hash stored beside it, any other by the context hash
    assert "does not match the context it sits in" in result.reason or "does not hash to its name" in result.reason


# ---------- the record itself ----------


def test_a_tampered_payload_is_refused_on_its_digest(recorded):
    engine, ids = recorded
    path = engine.log.payloads.path_for(ids[0])
    stored = json.loads(path.read_text())
    stored["data"]["raw_input"]["name"] = "Someone Else"
    path.write_text(json.dumps(stored))
    result = replay(engine, ids[0])
    assert result.status == REFUSED_TAMPERED and "digest" in result.reason
    assert replay(engine, ids[1]).identical  # the others are unaffected


def test_a_replaced_salt_is_refused(recorded):
    engine, ids = recorded
    path = engine.log.payloads.path_for(ids[0])
    stored = json.loads(path.read_text())
    stored["salt_hex"] = "00" * 32
    path.write_text(json.dumps(stored))
    assert replay(engine, ids[0]).status == REFUSED_TAMPERED


def test_an_unreadable_payload_is_refused(recorded):
    engine, ids = recorded
    engine.log.payloads.path_for(ids[0]).write_text("{broken")
    assert replay(engine, ids[0]).status == REFUSED_TAMPERED


def test_a_missing_payload_is_reported_as_not_replayable(recorded):
    engine, ids = recorded
    engine.log.payloads.path_for(ids[2]).unlink()
    result = replay(engine, ids[2])
    assert result.status == NOT_REPLAYABLE and not result.identical
    assert "payload_missing" in result.reason and "cannot be replayed" in result.reason


def test_an_edited_header_is_refused(recorded):
    engine, ids = recorded
    path = engine.log.headers.path
    lines = path.read_bytes().split(b"\n")[:-1]
    record = json.loads(lines[3])
    record["decision"] = "alert" if record["decision"] == "clear" else "clear"
    lines[3] = audit.canonical_json(record).encode()
    path.write_bytes(b"".join(line + b"\n" for line in lines))
    assert replay(engine, record["request_id"]).status == REFUSED_TAMPERED


def test_an_unknown_request_id(recorded):
    engine, _ = recorded
    assert replay(engine, "0" * 32).status == UNKNOWN_REQUEST


def test_a_recorded_decision_that_does_not_match_the_replay_is_reported_field_by_field(recorded):
    """Same context, but the header was rewritten and rehashed consistently, as a forger who knew the format would."""
    engine, ids = recorded
    path = engine.log.headers.path
    lines = path.read_bytes().split(b"\n")[:-1]
    for i, line in enumerate(lines):
        record = json.loads(line)
        if record["request_id"] == ids[0]:
            record["final_score"] = 0.123
            record["candidates"][0]["name_score"] = 0.5
            record["record_hash"] = audit.compute_record_hash(record)
            lines[i] = audit.canonical_json(record).encode()
    path.write_bytes(b"".join(line + b"\n" for line in lines))
    result = replay(engine, ids[0])
    assert result.status == DIFFERENT
    assert any(d.startswith("final_score: recorded 0.123") for d in result.differences)
    assert any("name_score" in d for d in result.differences)


# ---------- context files ----------


def test_the_context_is_written_once_under_its_hash_and_holds_no_personal_data(recorded):
    engine, ids = recorded
    files = list(engine.contexts_dir.glob("*.json"))
    assert [f.stem for f in files] == [engine.context_hash]
    stored = rc.load_context(engine.contexts_dir, engine.context_hash)
    assert stored == engine.run_context
    text = files[0].read_text()
    assert "Hassan" not in text and "Petrov" not in text and all(r not in text for r in ids)


def test_building_a_second_engine_with_the_same_context_rewrites_nothing(recorded, tmp_path):
    engine, _ = recorded
    path = engine.contexts_dir / f"{engine.context_hash}.json"
    before = path.stat().st_mtime_ns
    make_engine(tmp_path, log=engine.log)
    assert path.stat().st_mtime_ns == before and len(list(engine.contexts_dir.glob("*.json"))) == 1


def test_a_context_file_with_different_content_stops_the_engine_from_starting(recorded, tmp_path):
    engine, _ = recorded
    (engine.contexts_dir / f"{engine.context_hash}.json").write_text("{}")
    with pytest.raises(AuditError, match="different content"):
        make_engine(tmp_path, log=engine.log)


def test_engines_storing_the_same_new_context_at_once_never_see_a_half_written_file(recorded, tmp_path):
    import threading

    engine, _ = recorded
    other = make_engine(tmp_path, log=engine.log, config=ScreeningConfig(global_threshold=0.7), for_screening=False)
    errors = []

    def store() -> None:
        try:
            rc.store_context(engine.contexts_dir, other.run_context)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    for _ in range(20):
        for stale in engine.contexts_dir.glob(f"{other.context_hash}.json"):
            stale.unlink()
        threads = [threading.Thread(target=store) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert errors == []
    assert rc.load_context(engine.contexts_dir, other.context_hash) == other.run_context
    assert not list(engine.contexts_dir.glob("*.tmp"))  # no temporary files left behind


def test_two_contexts_give_two_files_and_verify_chain_is_unaffected(recorded, tmp_path):
    from audit_verify import verify_chain

    engine, _ = recorded
    make_engine(tmp_path, log=engine.log, config=ScreeningConfig(global_threshold=0.85))
    assert len(list(engine.contexts_dir.glob("*.json"))) == 2
    assert verify_chain(engine.log).ok


def test_diff_contexts_walks_nested_fields():
    a = {"x": 1, "n": {"k": "a" * 64, "same": 1}, "gone": 1}
    b = {"x": 2, "n": {"k": "b" * 64, "same": 1}, "new": 1}
    assert rc.diff_contexts(a, b) == [
        "gone: stored 1, live <absent>",
        "n.k: stored 'aaaaaaaaaaaa...', live 'bbbbbbbbbbbb...'",
        "new: stored <absent>, live 1",
        "x: stored 1, live 2",
    ]


# ---------- the CLI ----------


def cli_world(tmp_path, monkeypatch):
    monkeypatch.delenv("SCREENING_AUDIT_DIR", raising=False)
    entities = write_list(tmp_path / "list")
    rows = [
        {"id": "e1", "schema": "Person", "target": True, "properties": {"name": ["Mohammed Ali Hassan"]}},
        {"id": "e2", "schema": "Person", "target": True, "properties": {"name": ["Ivan Petrov"]}},
    ]
    body = "".join(json.dumps(r) + "\n" for r in rows).encode()
    entities.write_bytes(body)
    (entities.parent / "VERSION").write_text(f"version=cli\nsha1={hashlib.sha1(body).hexdigest()}\n")  # noqa: S324
    return entities, tmp_path / "audit"


def test_cli_replay_exits_0_on_an_identical_result_and_prints_no_names(tmp_path, monkeypatch, capsys):
    entities, audit_dir = cli_world(tmp_path, monkeypatch)
    base = ["--input", str(entities), "--audit-dir", str(audit_dir)]
    assert screen_cli.main(["init", "--audit-dir", str(audit_dir)]) == 0
    capsys.readouterr()
    assert screen_cli.main(["screen", "Mohammed Ali Hassan", "--json", *base]) == 0
    request_id = json.loads(capsys.readouterr().out)["request_id"]
    before = tree(audit_dir)
    assert screen_cli.main(["replay", request_id, *base]) == 0
    out = capsys.readouterr().out
    assert out.startswith("identical:") and "Hassan" not in out
    assert tree(audit_dir) == before  # replay changed nothing, not even a context file


def test_cli_replay_exits_nonzero_for_a_different_context_unknown_request_and_missing_payload(
    tmp_path, monkeypatch, capsys
):
    entities, audit_dir = cli_world(tmp_path, monkeypatch)
    base = ["--input", str(entities), "--audit-dir", str(audit_dir)]
    screen_cli.main(["init", "--audit-dir", str(audit_dir)])
    capsys.readouterr()
    screen_cli.main(["screen", "Mohammed Ali Hassan", "--json", *base])
    request_id = json.loads(capsys.readouterr().out)["request_id"]

    assert screen_cli.main(["replay", "0" * 32, *base]) == 1
    assert capsys.readouterr().out.startswith("unknown_request")

    real_load = ScreeningConfig.load
    monkeypatch.setattr(ScreeningConfig, "load", classmethod(lambda cls, *a: ScreeningConfig(global_threshold=0.85)))
    assert screen_cli.main(["replay", request_id, *base]) == 1
    out = capsys.readouterr().out
    assert out.startswith("refused_context") and "global_threshold" in out and "to reproduce: check out git" in out
    monkeypatch.setattr(ScreeningConfig, "load", real_load)

    (audit_dir / "payloads" / f"{request_id}.json").unlink()
    assert screen_cli.main(["replay", request_id, *base]) == 1
    assert capsys.readouterr().out.startswith("not_replayable")


def test_cli_replay_without_an_audit_directory_or_a_store_refuses(tmp_path, monkeypatch, capsys):
    entities, audit_dir = cli_world(tmp_path, monkeypatch)
    assert screen_cli.main(["replay", "0" * 32, "--input", str(entities)]) == 2
    assert screen_cli.main(["replay", "0" * 32, "--input", str(entities), "--audit-dir", str(audit_dir)]) == 1
    assert "log_missing" in capsys.readouterr().err
    assert not audit_dir.exists()

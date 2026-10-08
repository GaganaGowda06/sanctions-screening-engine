"""Tests for src/screen_cli.py: the production CLI writes an audit record before it prints a decision.

The list is a few entities in a temporary file (with its VERSION file), so the whole path runs for real: engine
build, run context, screening, audit header and payload, chain.

Run:
    pytest tests/test_screen_cli.py
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import audit
import audited_screening
import screen_cli
from audit import AuditError, AuditLog
from audit_verify import verify_chain

SRC = Path(screen_cli.__file__).resolve().parent
FTM = [
    {"id": "e1", "schema": "Person", "target": True,
     "properties": {"name": ["Mohammed Ali Hassan"], "birthDate": ["1970-01-01"], "country": ["iq"]}},
    {"id": "e2", "schema": "Person", "target": True,
     "properties": {"name": ["Ivan Petrov"], "birthDate": ["1980-05-05"], "country": ["ru"]}},
    {"id": "e3", "schema": "Person", "target": True, "properties": {"name": ["محمد علي حسن"]}},
    {"id": "e4", "schema": "Company", "target": True, "properties": {"name": ["Acme Trading Limited"]}},
]  # fmt: skip


def decisions(log):
    """The decision records of a log, leaving out engine_start and recovery events."""
    return [h for h in log.headers.read_all() if h["kind"] == "decision"]


@pytest.fixture
def bare(tmp_path, monkeypatch):
    monkeypatch.delenv(audited_screening.AUDIT_DIR_ENV, raising=False)
    directory = tmp_path / "list"
    directory.mkdir()
    body = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in FTM).encode("utf-8")
    entities = directory / "entities.ftm.json"
    entities.write_bytes(body)
    (directory / "VERSION").write_text(f"version=20260101000000-test\nsha1={hashlib.sha1(body).hexdigest()}\n")  # noqa: S324
    audit_dir = tmp_path / "audit"
    return {"entities": entities, "audit_dir": audit_dir, "tmp": tmp_path}


@pytest.fixture
def world(bare):
    """The same, with the audit store created the way an operator would: by the init command."""
    assert screen_cli.main(["init", "--audit-dir", str(bare["audit_dir"])]) == 0
    return bare


def run(world, *args: str, audit_dir: bool = True) -> list[str]:
    argv = ["screen", *args, "--input", str(world["entities"])]
    if audit_dir:
        argv += ["--audit-dir", str(world["audit_dir"])]
    return argv


def log_of(world) -> AuditLog:
    return AuditLog.in_directory(world["audit_dir"], fsync=False)


def snapshot(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


# ---------- one name, one record ----------


def test_one_name_writes_exactly_one_verifiable_record(world, capsys):
    code = screen_cli.main(run(world, "Mohammed Ali Hassan", "--birth-date", "1970-01-01", "--country", "iq", "--json"))
    out = capsys.readouterr().out
    assert code == 0
    printed = json.loads(out)
    log = log_of(world)
    (header,) = decisions(log)
    assert header["request_id"] == printed["request_id"]
    assert header["decision"] == "alert" and printed["matches"][0]["entity_id"] == "e1"
    result = verify_chain(log)
    assert result.ok and result.records == 2 and result.payloads_checked == 1  # engine_start + the decision
    payload = log.payloads.read(header["request_id"]).data
    assert payload["raw_input"]["name"] == "Mohammed Ali Hassan" and payload["customer_ref"] is None


def test_the_text_output_names_the_record_and_a_clear_name_is_recorded_too(world, capsys):
    assert screen_cli.main(run(world, "Zebulon Quixotic")) == 0
    out = capsys.readouterr().out
    log = log_of(world)
    (header,) = decisions(log)
    assert "No alert" in out and header["request_id"] in out and header["decision"] == "clear"
    assert verify_chain(log).ok


def test_customer_ref_goes_to_the_payload_only(world, capsys):
    assert screen_cli.main(run(world, "Ivan Petrov", "--customer-ref", "cust-123", "--json")) == 0
    printed = json.loads(capsys.readouterr().out)
    log = log_of(world)
    (header,) = decisions(log)
    assert log.payloads.read(printed["request_id"]).data["customer_ref"] == "cust-123"
    assert "cust-123" not in json.dumps(header)


def test_limit_only_trims_the_display_not_the_record(world, capsys):
    screen_cli.main(run(world, "Mohammed Ali Hassan", "--limit", "0", "--json"))
    printed = json.loads(capsys.readouterr().out)
    (header,) = decisions(log_of(world))
    assert printed["matches"] == [] and header["decision"] == "alert"


# ---------- an audit failure gives no decision ----------


def test_an_audit_failure_gives_a_nonzero_exit_and_no_decision_output(world, capsys, monkeypatch):
    def fail(self, header, payload):
        raise AuditError("disk full")

    monkeypatch.setattr(AuditLog, "write", fail)
    code = screen_cli.main(run(world, "Mohammed Ali Hassan", "--json"))
    captured = capsys.readouterr()
    assert code == screen_cli.EXIT_AUDIT_FAILURE != 0
    assert captured.out == ""
    assert "no decision produced" in captured.err and "disk full" in captured.err
    assert decisions(log_of(world)) == []


def test_an_unusable_audit_store_refuses_to_start_with_no_output(world, capsys):
    (world["audit_dir"] / "headers.jsonl").unlink()
    (world["audit_dir"] / "headers.jsonl").mkdir()  # a directory where the log file should be
    (world["audit_dir"] / "headers.jsonl.head").unlink()
    code = screen_cli.main(run(world, "Mohammed Ali Hassan"))
    captured = capsys.readouterr()
    assert code == screen_cli.EXIT_AUDIT_FAILURE and captured.out == ""
    assert "refusing to start" in captured.err


def test_a_tampered_log_stops_screening_with_no_decision(world, capsys):
    assert screen_cli.main(run(world, "Ivan Petrov")) == 0
    capsys.readouterr()
    log = log_of(world)
    lines = log.headers.path.read_bytes().split(b"\n")[:-1]
    record = json.loads(lines[-1])
    assert record["final_score"] != 0.0
    record["final_score"] = 0.0
    log.headers.path.write_bytes(audit.canonical_json(record).encode() + b"\n")
    code = screen_cli.main(run(world, "Ivan Petrov"))
    captured = capsys.readouterr()
    # the engine cannot even start: its engine_start record would be chained after an edited one
    assert code != 0 and captured.out == "" and "does not match its own hash" in captured.err


# ---------- no audit directory, no start ----------


def test_an_unset_audit_directory_refuses_to_start_before_building_the_list(world, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the list must not be built")

    monkeypatch.setattr(audited_screening.AuditedEngine, "from_file", classmethod(boom))
    before = snapshot(world["tmp"])
    code = screen_cli.main(run(world, "Mohammed Ali Hassan", audit_dir=False))
    captured = capsys.readouterr()
    assert code == screen_cli.EXIT_USAGE == 2
    assert captured.out == "" and "refusing to start" in captured.err
    assert snapshot(world["tmp"]) == before


def test_an_empty_audit_directory_setting_is_also_refused(world, capsys, monkeypatch):
    monkeypatch.setenv(audited_screening.AUDIT_DIR_ENV, "")
    assert screen_cli.main(run(world, "Ivan Petrov", audit_dir=False)) == 2
    assert capsys.readouterr().out == ""


def test_the_environment_variable_works_and_the_flag_wins(world, capsys, monkeypatch):
    other = world["tmp"] / "from-env"
    AuditLog.create_new(other, fsync=False)
    monkeypatch.setenv(audited_screening.AUDIT_DIR_ENV, str(other))
    assert screen_cli.main(run(world, "Ivan Petrov", audit_dir=False)) == 0
    assert len(decisions(AuditLog.in_directory(other))) == 1
    assert screen_cli.main(run(world, "Ivan Petrov")) == 0  # --audit-dir given too
    assert len(decisions(log_of(world))) == 1
    assert len(decisions(AuditLog.in_directory(other))) == 1


def test_there_is_no_default_audit_directory(world):
    with pytest.raises(audit.AuditConfigError):
        audited_screening.resolve_audit_dir(None)
    with pytest.raises(audit.AuditConfigError):
        audited_screening.AuditedEngine.from_file(world["entities"])


def test_a_separate_process_without_the_setting_exits_2_and_prints_nothing(world):
    done = subprocess.run(
        [sys.executable, str(SRC / "screen_cli.py"), "screen", "Ivan Petrov", "--input", str(world["entities"])],
        capture_output=True, text=True, cwd=SRC, env={"PYTHONPATH": str(SRC), "PATH": ""},
    )  # fmt: skip
    assert done.returncode == 2 and done.stdout == "" and "refusing to start" in done.stderr


# ---------- batch ----------


def write_batch(world, items) -> Path:
    path = world["tmp"] / "batch.jsonl"
    path.write_text("".join((i if isinstance(i, str) else json.dumps(i, ensure_ascii=False)) + "\n" for i in items))
    return path


def test_a_batch_of_n_names_gives_n_records_and_a_clean_chain(world, capsys):
    items = [
        {"name": "Mohammed Ali Hassan", "birth_date": "1970-01-01", "country": "iq", "customer_ref": "c-1"},
        {"name": "Ivan Petrov", "customer_ref": "c-2"},
        {"name": "محمد علي حسن"},
        {"name": "Zebulon Quixotic"},
        {"name": "Acme Trading"},
        {"name": "Muhammad Ali Hasan", "country": "iq", "customer_ref": "c-6"},
        {"name": "Nobody Atall"},
    ]
    code = screen_cli.main(run(world, "--batch", str(write_batch(world, items)), "--json"))
    lines = capsys.readouterr().out.splitlines()
    assert code == 0 and len(lines) == len(items)
    printed = [json.loads(line) for line in lines]
    log = log_of(world)
    headers = decisions(log)
    assert [h["request_id"] for h in headers] == [p["request_id"] for p in printed]
    result = verify_chain(log)
    assert result.ok and result.records == len(items) + 1 and result.payloads_checked == len(items)  # + engine_start
    refs = [log.payloads.read(h["request_id"]).data["customer_ref"] for h in headers]
    assert refs == ["c-1", "c-2", None, None, None, "c-6", None]
    assert [log.payloads.read(h["request_id"]).data["raw_input"]["name"] for h in headers] == [i["name"] for i in items]


def test_a_batch_can_come_from_standard_input(world, capsys, monkeypatch):
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO('{"name": "Ivan Petrov"}\n\n{"name": "Zebulon Quixotic"}\n'))
    assert screen_cli.main(run(world, "--batch", "-")) == 0
    assert len(decisions(log_of(world))) == 2 and capsys.readouterr().out.count("record ") == 2


@pytest.mark.parametrize(
    "bad",
    ['{"name": "Ivan Petrov"', '{"nom": "x"}', '["Ivan Petrov"]', '{"name": ""}', '{"name": "x", "country": 5}',
     '{"name": "x", "extra": 1}'],
)  # fmt: skip
def test_a_malformed_batch_line_stops_the_run_before_anything_is_screened(world, capsys, bad):
    path = write_batch(world, [{"name": "Ivan Petrov"}, bad])
    code = screen_cli.main(run(world, "--batch", str(path)))
    captured = capsys.readouterr()
    assert code == 2 and captured.out == "" and "line 2" in captured.err
    assert decisions(log_of(world)) == []  # nothing was screened or recorded


def test_an_empty_batch_is_refused(world, capsys):
    path = world["tmp"] / "empty.jsonl"
    path.write_text("\n\n")
    assert screen_cli.main(run(world, "--batch", str(path))) == 2
    assert capsys.readouterr().out == ""


def test_a_batch_stops_at_the_first_audit_failure_and_prints_no_decision_after_it(world, capsys, monkeypatch):
    real, calls = AuditLog.write, []

    def flaky(self, header, payload):
        calls.append(1)
        if len(calls) == 3:
            raise AuditError("disk full")
        return real(self, header, payload)

    monkeypatch.setattr(AuditLog, "write", flaky)
    items = [{"name": n} for n in ("Ivan Petrov", "Mohammed Ali Hassan", "Acme Trading", "Zebulon Quixotic", "X Y")]
    code = screen_cli.main(run(world, "--batch", str(write_batch(world, items)), "--json"))
    captured = capsys.readouterr()
    assert code == screen_cli.EXIT_AUDIT_FAILURE
    assert len(captured.out.splitlines()) == 2 and "no decision produced" in captured.err
    assert "2 of 5 names were screened and recorded before the failure" in captured.err
    log = log_of(world)
    assert len(decisions(log)) == 2 and verify_chain(log).ok
    assert len(calls) == 3  # the fourth and fifth names were never screened


def test_a_name_and_a_batch_together_are_refused(world, capsys):
    with pytest.raises(SystemExit) as stop:
        screen_cli.main(run(world, "Ivan Petrov", "--batch", "x.jsonl"))
    assert stop.value.code == 2
    with pytest.raises(SystemExit):
        screen_cli.main(run(world, "--batch", "x.jsonl", "--country", "iq"))
    with pytest.raises(SystemExit):
        screen_cli.main(run(world))


# ---------- init, and refusing a store that is not there ----------


def test_init_creates_a_store_that_verifies_clean_and_then_screening_gives_a_clean_chain(bare, capsys):
    assert screen_cli.main(["init", "--audit-dir", str(bare["audit_dir"])]) == 0
    assert "initialised" in capsys.readouterr().out
    log = log_of(bare)
    assert verify_chain(log).ok and verify_chain(log).records == 0
    assert log.headers.read_checkpoint() == {"count": 0, "record_hash": audit.GENESIS_PREV_HASH}
    for name in ("Ivan Petrov", "Mohammed Ali Hassan", "Zebulon Quixotic"):
        assert screen_cli.main(run(bare, name)) == 0
    result = verify_chain(log)
    assert result.ok and result.records == 4 and result.payloads_checked == 3  # 3 decisions + engine_start


def test_init_refuses_a_directory_that_exists_and_is_not_empty(world, capsys):
    assert screen_cli.main(run(world, "Ivan Petrov")) == 0
    capsys.readouterr()
    before = {p: p.read_bytes() for p in world["audit_dir"].rglob("*") if p.is_file()}
    code = screen_cli.main(["init", "--audit-dir", str(world["audit_dir"])])
    captured = capsys.readouterr()
    assert code == screen_cli.EXIT_AUDIT_FAILURE and "not empty" in captured.err and captured.out == ""
    assert {p: p.read_bytes() for p in world["audit_dir"].rglob("*") if p.is_file()} == before
    other = world["tmp"] / "unrelated"
    other.mkdir()
    (other / "file.txt").write_text("not an audit store")
    assert screen_cli.main(["init", "--audit-dir", str(other)]) == screen_cli.EXIT_AUDIT_FAILURE


def test_init_accepts_an_existing_empty_directory_and_needs_a_directory_setting(bare, capsys):
    bare["audit_dir"].mkdir()
    assert screen_cli.main(["init", "--audit-dir", str(bare["audit_dir"])]) == 0
    assert screen_cli.main(["init"]) == screen_cli.EXIT_USAGE
    assert "refusing to start" in capsys.readouterr().err


def test_screening_with_no_store_refuses_names_log_missing_and_init_and_creates_nothing(bare, capsys):
    code = screen_cli.main(run(bare, "Mohammed Ali Hassan", "--json"))
    captured = capsys.readouterr()
    assert code == screen_cli.EXIT_AUDIT_FAILURE == 1
    assert captured.out == ""
    assert "log_missing" in captured.err and "screen_cli.py init" in captured.err
    assert not bare["audit_dir"].exists()


def test_wiping_the_audit_directory_makes_screening_refuse_not_start_fresh(world, capsys):
    assert screen_cli.main(run(world, "Ivan Petrov")) == 0
    capsys.readouterr()
    shutil.rmtree(world["audit_dir"])
    code = screen_cli.main(run(world, "Mohammed Ali Hassan"))
    captured = capsys.readouterr()
    assert code == 1 and captured.out == ""
    assert "log_missing" in captured.err and "screen_cli.py init" in captured.err
    assert not world["audit_dir"].exists()  # it was not quietly recreated


def test_emptying_the_audit_directory_makes_screening_refuse_too(world, capsys):
    assert screen_cli.main(run(world, "Ivan Petrov")) == 0
    capsys.readouterr()
    shutil.rmtree(world["audit_dir"])
    world["audit_dir"].mkdir()
    code = screen_cli.main(run(world, "Mohammed Ali Hassan"))
    captured = capsys.readouterr()
    assert code == 1 and captured.out == "" and "log_missing" in captured.err
    assert list(world["audit_dir"].iterdir()) == []


def test_a_wipe_in_the_middle_of_a_batch_stops_it_with_the_count_of_names_recorded(world, capsys, monkeypatch):
    real, calls = AuditLog.write, []

    def wipe_then_write(self, header, payload):
        calls.append(1)
        if len(calls) == 3:
            shutil.rmtree(world["audit_dir"])
        return real(self, header, payload)

    monkeypatch.setattr(AuditLog, "write", wipe_then_write)
    items = [{"name": n} for n in ("Ivan Petrov", "Acme Trading", "Mohammed Ali Hassan", "Zebulon Quixotic")]
    code = screen_cli.main(run(world, "--batch", str(write_batch(world, items)), "--json"))
    captured = capsys.readouterr()
    assert code == 1 and len(captured.out.splitlines()) == 2
    assert "log_missing" in captured.err and "2 of 4 names were screened and recorded" in captured.err


def test_a_partial_batch_exits_nonzero_and_reports_how_many_were_recorded(world, capsys, monkeypatch):
    real, calls = AuditLog.write, []

    def fail_on_fourth(self, header, payload):
        calls.append(1)
        if len(calls) == 4:
            raise AuditError("disk full")
        return real(self, header, payload)

    monkeypatch.setattr(AuditLog, "write", fail_on_fourth)
    items = [{"name": f"Zebulon Quixotic {i}"} for i in range(6)]
    code = screen_cli.main(run(world, "--batch", str(write_batch(world, items))))
    captured = capsys.readouterr()
    assert code != 0
    assert "3 of 6 names were screened and recorded before the failure; 3 were not screened" in captured.err
    assert captured.out.count("record ") == 3
    log = log_of(world)
    assert len(decisions(log)) == 3 and verify_chain(log).ok


# ---------- explicit subcommands ----------


@pytest.mark.parametrize("name", ["init", "replay", "screen"])
def test_a_customer_called_like_a_subcommand_is_screened_and_recorded(world, capsys, name):
    assert screen_cli.main(run(world, name, "--json")) == 0
    printed = json.loads(capsys.readouterr().out)
    log = log_of(world)
    (header,) = [h for h in decisions(log) if h["request_id"] == printed["request_id"]]
    assert log.payloads.read(header["request_id"]).data["raw_input"]["name"] == name
    assert verify_chain(log).ok


def test_a_bare_name_with_no_subcommand_exits_2_with_a_usage_message(world, capsys):
    with pytest.raises(SystemExit) as stop:
        screen_cli.main(["Ivan Petrov", "--input", str(world["entities"]), "--audit-dir", str(world["audit_dir"])])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert "usage:" in err and "invalid choice" in err
    assert decisions(log_of(world)) == []


def test_no_arguments_at_all_exits_2(capsys):
    with pytest.raises(SystemExit) as stop:
        screen_cli.main([])
    assert stop.value.code == 2 and "usage:" in capsys.readouterr().err


def test_a_separate_process_with_a_bare_name_exits_2_and_prints_nothing_on_stdout(world):
    done = subprocess.run(
        [sys.executable, str(SRC / "screen_cli.py"), "Ivan Petrov", "--audit-dir", str(world["audit_dir"])],
        capture_output=True, text=True, cwd=SRC, env={"PYTHONPATH": str(SRC), "PATH": ""},
    )  # fmt: skip
    assert done.returncode == 2 and done.stdout == "" and "usage:" in done.stderr

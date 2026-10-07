"""Tests for src/audited_screening.py.

Small engines are built in memory, so the header, payload and equivalence checks run anywhere. The equivalence
test over the three hold-outs needs the downloaded list and is skipped without it, like the real-list context test.

Run:
    pytest tests/test_audited_screening.py
"""

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

import audit
import audited_screening as asc
import run_context as rc
from audit import AuditError, AuditLog
from audited_screening import AuditedEngine
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
    ("", None, None),
]


def write_list(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    body = b'{"id":"e1"}\n'
    path = directory / "entities.ftm.json"
    path.write_bytes(body)
    (directory / "VERSION").write_text(f"version=20260101000000-abc\nsha1={hashlib.sha1(body).hexdigest()}\n")  # noqa: S324
    return path


def make_engine(tmp_path, entities=ENTITIES, config=None, **kwargs) -> AuditedEngine:
    engine = ScreeningEngine.from_entities(entities, config or ScreeningConfig())
    context = rc.build_run_context(
        engine, write_list(tmp_path / "list"), git=lambda: ("a" * 40, False), versions=lambda: {"rapidfuzz": "3.14.6"}
    )
    kwargs.setdefault("clock", CLOCK)
    return AuditedEngine(engine, context, AuditLog.in_directory(tmp_path / "audit", fsync=False), **kwargs)


def string_leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from string_leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from string_leaves(v)


# ---------- the same decision as screen() ----------


@pytest.mark.parametrize(("name", "birth_date", "country"), QUERIES)
def test_audited_screen_gives_the_same_matches_as_screen(tmp_path, name, birth_date, country):
    audited = make_engine(tmp_path)
    plain = audited.engine.screen(name, birth_date, country)
    result = audited.screen(name, birth_date, country)
    assert result.matches == plain
    assert [m.score for m in result.matches] == [m.score for m in plain]


def test_decisions_agree_and_some_queries_alert_and_some_clear(tmp_path):
    audited = make_engine(tmp_path)
    decisions = []
    for name, birth_date, country in QUERIES:
        result = audited.screen(name, birth_date, country)
        header = audited.log.headers.read_all()[-1]
        assert header["request_id"] == result.request_id
        assert header["decision"] == ("alert" if result.matches else "clear")
        assert header["matched_entity_ids"] == [m.entity_id for m in result.matches]
        decisions.append(header["decision"])
    assert {"alert", "clear"} <= set(decisions)


def test_components_is_called_exactly_once_per_screen(tmp_path):
    audited = make_engine(tmp_path)
    screener, calls = audited.engine.screener, []
    original = screener.components
    screener.components = lambda *a, **k: calls.append(1) or original(*a, **k)
    audited.screen("Mohammed Ali Hassan", "1970-01-01", "iq")
    assert len(calls) == 1


def test_the_threshold_is_the_one_for_the_query_script(tmp_path):
    config = ScreeningConfig(
        thresholds={"Latin": 0.80, "Cyrillic": 0.88, "Arabic": 0.91, "other": 0.86}, per_script=True
    )
    audited = make_engine(tmp_path, config=config)
    audited.screen("Mohammed Ali Hassan")
    audited.screen("محمد علي حسن")
    latin, arabic = audited.log.headers.read_all()
    assert latin["threshold"] == 0.80 and arabic["threshold"] == 0.91


# ---------- the header ----------


def forbidden_strings(name, birth_date, country, normalised, payload):
    out = {name, normalised["normalise"], normalised["scoring_form"], normalised["skeleton"], *payload["name_tokens"]}
    out |= set(payload["retrieval_keys"])
    out |= {s for s in (birth_date, country) if s}
    return {s for s in out if s}


@pytest.mark.parametrize(("name", "birth_date", "country"), [q for q in QUERIES if q[0]])
def test_the_header_contains_nothing_derived_from_the_name_birth_date_or_country(tmp_path, name, birth_date, country):
    audited = make_engine(tmp_path)
    result = audited.screen(name, birth_date, country)
    header = audited.log.headers.read_all()[-1]
    payload = audited.log.payloads.read(result.request_id).data
    forbidden = forbidden_strings(name, birth_date, country, payload["normalised"], payload)
    assert forbidden, "the payload should hold the name-derived strings"
    assert set(header) == set(audit.HEADER_FIELDS)  # the log writes keys sorted
    text = json.dumps(header, ensure_ascii=False).lower()
    for leaf in string_leaves(header):
        assert leaf not in forbidden, leaf
        assert leaf.lower() not in {f.lower() for f in forbidden}, leaf
    for item in forbidden:
        if len(item) >= 4:  # shorter strings could sit inside a hex hash by chance
            assert item.lower() not in text, item
    assert "query_script" not in header and "script" not in " ".join(header)


def test_the_payload_holds_the_name_derived_fields(tmp_path):
    audited = make_engine(tmp_path)
    result = audited.screen("Muhammad Ali Hasan", "1970-01-01", "iq")
    data = audited.log.payloads.read(result.request_id).data
    assert data["raw_input"] == {"name": "Muhammad Ali Hasan", "birth_date": "1970-01-01", "country": "iq"}
    assert data["query_script"] == "Latin"
    assert data["normalised"]["scoring_form"] == "muhammad ali hasan"
    assert data["name_tokens"] == ["muhammad", "ali", "hasan"]
    assert data["retrieval_keys"] and all(k[:2] in ("t:", "s:") for k in data["retrieval_keys"])
    assert [m["entity_id"] for m in data["matches"]] == [m.entity_id for m in result.matches]


def test_header_digest_matches_the_stored_payload(tmp_path):
    audited = make_engine(tmp_path)
    result = audited.screen("Mohammed Ali Hassan", "1970-01-01", "iq")
    header = audited.log.headers.read_all()[-1]
    record = audited.log.payloads.read(result.request_id)
    assert record.digest() == header["payload_digest"]
    assert len(record.salt) >= 16


def test_the_same_query_twice_gives_different_salts_and_digests_but_the_same_scores(tmp_path):
    audited = make_engine(tmp_path)
    a = audited.screen("Mohammed Ali Hassan")
    b = audited.screen("Mohammed Ali Hassan")
    ha, hb = audited.log.headers.read_all()
    assert a.request_id != b.request_id
    assert audited.log.payloads.read(a.request_id).salt != audited.log.payloads.read(b.request_id).salt
    assert ha["payload_digest"] != hb["payload_digest"]
    assert ha["candidates"] == hb["candidates"] and ha["final_score"] == hb["final_score"]


def test_the_clear_header_still_logs_candidates(tmp_path):
    audited = make_engine(tmp_path)
    audited.screen("Ivan Hassan")
    header = audited.log.headers.read_all()[-1]
    assert header["candidates_scored"] >= 1 and len(header["candidates"]) >= 1
    assert all(c["at_or_above_threshold"] is False for c in header["candidates"]) or header["decision"] == "alert"


def test_the_header_candidates_carry_numeric_scores_and_the_matched_entity(tmp_path):
    audited = make_engine(tmp_path)
    audited.screen("Mohammed Ali Hassan", "1970-01-01", "iq")
    header = audited.log.headers.read_all()[-1]
    top = header["candidates"][0]
    assert top["entity_id"] == "e1" == header["matched_entity_id"]
    for field in ("score", "name_score", "skeleton_score", "birth_date_adjustment", "country_adjustment"):
        assert isinstance(top[field], float)
    assert top["score"] == header["final_score"] >= header["threshold"]


def test_all_hits_are_logged_even_past_the_top_ten(tmp_path):
    many = [{"id": f"h{i:02d}", "names": ["Mohammed Ali Hassan"]} for i in range(15)]
    many += [{"id": f"l{i:02d}", "names": [f"Mohammed Ali Zzz{i}qq"]} for i in range(25)]
    audited = make_engine(tmp_path, entities=many)
    result = audited.screen("Mohammed Ali Hassan")
    header = audited.log.headers.read_all()[-1]
    assert len(result.matches) == 15
    assert header["matched_entity_ids"] == sorted(header["matched_entity_ids"])
    assert sum(c["at_or_above_threshold"] for c in header["candidates"]) == 15
    assert len(header["candidates"]) >= 15
    assert header["candidates_scored"] > len(header["candidates"]) or header["candidates_scored"] == 15


def test_only_ten_are_logged_when_nothing_reaches_the_threshold(tmp_path):
    many = [{"id": f"l{i:02d}", "names": [f"Mohammed Ali Zzz{i}qq"]} for i in range(30)]
    audited = make_engine(tmp_path, entities=many)
    result = audited.screen("Mohammed Ali Hassan")
    header = audited.log.headers.read_all()[-1]
    assert result.matches == [] and header["decision"] == "clear"
    assert header["candidates_scored"] >= 11
    assert len(header["candidates"]) == 10


def test_the_timestamp_uses_the_injected_clock_and_there_is_no_retention_date(tmp_path):
    audited = make_engine(tmp_path)
    audited.screen("Ivan Petrov")
    header = audited.log.headers.read_all()[-1]
    assert header["timestamp_utc"] == "2026-10-07T12:00:00.000000+00:00"
    assert "retention_until" not in header


def test_customer_ref_is_in_the_payload_only(tmp_path):
    audited = make_engine(tmp_path)
    result = audited.screen("Mohammed Ali Hassan", "1970-01-01", "iq", customer_ref="cust-7f3a9")
    header = audited.log.headers.read_all()[-1]
    assert audited.log.payloads.read(result.request_id).data["customer_ref"] == "cust-7f3a9"
    assert "cust-7f3a9" not in json.dumps(header) and "customer_ref" not in header


def test_customer_ref_is_optional_and_must_be_a_string(tmp_path):
    audited = make_engine(tmp_path)
    result = audited.screen("Ivan Petrov")
    assert audited.log.payloads.read(result.request_id).data["customer_ref"] is None
    with pytest.raises(ValueError):
        audited.screen("Ivan Petrov", customer_ref=12345)  # type: ignore[arg-type]
    assert len(audited.log.headers.read_all()) == 1  # the refused call wrote nothing


# ---------- the run context is held, not rebuilt ----------


def test_the_header_carries_the_hash_of_the_context_the_engine_holds(tmp_path, monkeypatch):
    audited = make_engine(tmp_path)

    def boom(*a, **k):
        raise AssertionError("the run context must not be rebuilt at log time")

    monkeypatch.setattr(asc, "build_run_context", boom)
    monkeypatch.setattr(rc, "build_run_context", boom)
    audited.screen("Ivan Petrov")
    assert (
        audited.log.headers.read_all()[-1]["run_context_hash"]
        == audited.run_context.context_hash
        == audited.context_hash
    )
    assert re.fullmatch(r"[0-9a-f]{64}", audited.context_hash)


def test_the_held_hash_does_not_change_when_a_source_file_would(tmp_path):
    audited = make_engine(tmp_path)
    first = audited.context_hash
    audited.screen("Ivan Petrov")
    assert audited.log.headers.read_all()[-1]["run_context_hash"] == first


# ---------- a logging failure fails the call ----------


def test_a_header_write_failure_fails_the_call_and_leaves_no_payload(tmp_path):
    audited = make_engine(tmp_path)
    (tmp_path / "audit").mkdir(exist_ok=True)
    (tmp_path / "audit" / "headers.jsonl").mkdir()  # a directory where the log file should be
    with pytest.raises(AuditError):
        audited.screen("Mohammed Ali Hassan")
    payloads = tmp_path / "audit" / "payloads"
    assert not payloads.exists() or not list(payloads.iterdir())


def test_a_payload_write_failure_fails_the_call_and_writes_no_header(tmp_path):
    audited = make_engine(tmp_path)
    (tmp_path / "audit").mkdir(exist_ok=True)
    (tmp_path / "audit" / "payloads").write_text("a file where the store directory should be")
    with pytest.raises(AuditError):
        audited.screen("Mohammed Ali Hassan")
    assert audited.log.headers.read_all() == []


def test_an_unencodable_name_fails_the_call(tmp_path):
    audited = make_engine(tmp_path)
    with pytest.raises(AuditError):
        audited.screen("bad \ud800 name")
    assert audited.log.headers.read_all() == []


# ---------- all three hold-outs, on the real list ----------

ROOT = rc.ROOT
HOLDOUTS = [
    ROOT / "holdout" / n for n in ("final_holdout.json", "screening_holdout_v2.json", "screening_holdout_v3.json")
]
REAL_LIST = ROOT / "data" / "sanctions" / "entities.ftm.json"


needs_data = pytest.mark.skipif(
    not REAL_LIST.exists() or not all(p.exists() for p in HOLDOUTS), reason="the list or a hold-out is not on disk"
)
SAMPLE_PER_HOLDOUT = 200
SAMPLE_SEED = 20261007


def holdout_queries(sample: int | None) -> list[tuple[str, dict]]:
    """Every query of the three hold-outs, or a fixed seeded sample of `sample` per hold-out."""
    out = []
    for path in HOLDOUTS:
        queries = json.loads(path.read_text())["queries"]
        if sample is not None:
            queries = random.Random(f"{SAMPLE_SEED}-{path.name}").sample(queries, sample)
        out += [(path.name, q) for q in queries]
    return out


def check_equivalence(tmp_path, queries: list[tuple[str, dict]]) -> tuple[int, int]:
    audited = AuditedEngine.from_file(REAL_LIST, tmp_path / "audit", clock=CLOCK)
    audited.log.headers.fsync = audited.log.payloads.fsync = False  # speed only
    alerts = 0
    for holdout, q in queries:
        plain = audited.engine.screen(q["name"], q["dob"], q["country"])
        result = audited.screen(q["name"], q["dob"], q["country"])
        assert result.matches == plain, (holdout, q["name"])
        assert [(m.entity_id, m.score) for m in result.matches] == [(m.entity_id, m.score) for m in plain]
        with open(audited.log.headers.path, "rb") as fh:
            header = json.loads(fh.readlines()[-1])
        assert header["request_id"] == result.request_id
        assert header["decision"] == ("alert" if plain else "clear")
        assert header["matched_entity_ids"] == [m.entity_id for m in plain]
        alerts += bool(plain)
    return len(queries), alerts


@needs_data
def test_audited_screen_equals_screen_on_a_seeded_sample_of_each_holdout(tmp_path):
    checked, alerts = check_equivalence(tmp_path, holdout_queries(SAMPLE_PER_HOLDOUT))
    assert checked == 3 * SAMPLE_PER_HOLDOUT and 0 < alerts < checked


@pytest.mark.slow
@needs_data
def test_audited_screen_equals_screen_on_every_query_of_all_three_holdouts(tmp_path):
    checked, alerts = check_equivalence(tmp_path, holdout_queries(None))
    assert checked == 13_800 and 0 < alerts < checked

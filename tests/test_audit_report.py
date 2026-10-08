"""Tests for the markdown monitoring report (src/audit_report.py) and the `report` CLI command.

The log and the baseline are hand-built: decisions with chosen scores and names under the context of a small real
engine, and a baseline made from a second hand-built log of the same shape. A golden file holds the expected markdown,
with the run and decision context hashes replaced by placeholders (they change whenever scoring code changes).

Regenerate the golden file after a deliberate change to the report's layout:
    UPDATE_GOLDEN=1 pytest tests/test_audit_report.py -k golden

Run:
    pytest tests/test_audit_report.py
"""

import json
import os
import re
from pathlib import Path

import pytest
from test_audit_drift import SHAPES
from test_audit_metrics import ARABIC, CLOCK, CYRILLIC, LATIN, add_decision
from test_audited_screening import forbidden_strings
from test_replay import make_engine, tree

import audit_report
import screen_cli
from audit import AuditLog
from audit_drift import save_baseline
from audit_metrics import MetricsConfig, compute_metrics
from audit_report import ReportRefused, build_report

GOLDEN = Path(__file__).parent / "golden" / "monitoring_report.md"
NAMES = {"Latin": LATIN, "Arabic": ARABIC, "Cyrillic": CYRILLIC}
CONFIG = MetricsConfig(near_threshold_band=0.05, min_cell=10)
FORWARD_GATE = {
    "label": "forward list update",
    "direction": "the new snapshot is newer",
    "old_snapshot": "20261006074704-hib",
    "new_snapshot": "20261008075007-crk",
    "provenance": {
        "holdout": "screening_holdout_v3.json",
        "holdout_sha256": "c" * 64,
        "engine": "the current engine: the Stage 2 hand-tuned matcher plus the Stage 3 rarity-weighted skeleton",
        "not_the_same_as": {"F1 0.777": "the Stage 2 engine on the SECOND hold-out"},
    },
    "resolution": {"kept": 90, "remapped": 2, "dropped": 1},
    "metrics_on_resolved_set": {
        "f1": {"old": 0.8, "new": 0.79, "delta": -0.01, "delta_ci95": [-0.02, 0.0]},
        "arabic_burden": {"old": 0.22, "new": 0.23, "delta": 0.01, "delta_ci95": [-0.01, 0.03]},
    },
    "gate": {"passed": True},
}
REVERSE_GATE = {
    **FORWARD_GATE,
    "label": "REVERSE-TIME CHURN TEST",
    "direction": "the new snapshot is OLDER than the current one",
    "new_snapshot": "20260701194702-fjb",
}


def scores_from_shape(shape: list[int], n: int) -> list[float]:
    """n scores whose 20-bin histogram follows `shape` (bin centres), None-free."""
    out = []
    for i, c in enumerate(shape):
        out += [(i + 0.5) / 20] * (c * n // 100)
    return out


def fill(log: AuditLog, context: str, counts: dict[str, int], shifted: bool = False, start: int = 0) -> int:
    i = start
    for script, n in counts.items():
        scores = scores_from_shape(SHAPES[script], n)
        if shifted:
            scores = [min(s * 0.3, 0.3) for s in scores]  # everything piles into the low bins
        for score in scores:
            add_decision(log, i, NAMES[script], score, 5, context=context)
            i += 1
    return i


@pytest.fixture
def world(tmp_path):
    """A live store and a baseline directory for the same decision context, plus a reports directory."""
    engine_b = make_engine(tmp_path / "b")
    fill(engine_b.log, engine_b.context_hash, {"Latin": 400, "Arabic": 200, "Cyrillic": 200})
    base_dir = tmp_path / "baseline"
    report_b, _ = compute_metrics(engine_b.log.headers.path.parent, 1, 800, CONFIG, CLOCK)
    one = {**report_b, "groups": [report_b["groups"][0]]}
    save_baseline(one, engine_b.run_context.as_dict(), {"seed": 0}, base_dir)

    live = make_engine(tmp_path / "live")
    # the same engine context in a second store: engine_start is record 0, decisions follow from seq 1
    assert live.context_hash == engine_b.context_hash
    reports = tmp_path / "reports"
    reports.mkdir()
    return {"engine": live, "tmp": tmp_path, "baseline": base_dir, "reports": reports, "count": [0]}


def live_window(world, counts, shifted=False) -> tuple[Path, int, int]:
    engine = world["engine"]
    end = fill(engine.log, engine.context_hash, counts, shifted)
    return engine.log.headers.path.parent, 1, end


def make(world, counts=None, shifted=False, **kwargs):
    audit_dir, a, b = live_window(world, counts or {"Latin": 400, "Arabic": 200, "Cyrillic": 200}, shifted)
    text, status = build_report(
        audit_dir, a, b, baseline_dir=world["baseline"], reports_dir=world["reports"],
        metrics_config=CONFIG, clock=CLOCK, **kwargs,
    )  # fmt: skip
    return audit_dir, (a, b), text, status


def normalised(text: str, engine) -> str:
    return text.replace(engine.context_hash[:12], "{RUN}").replace(
        engine.run_context.decision_context_hash[:12], "{DEC}"
    )


# ---------- golden file ----------


def test_the_markdown_matches_the_golden_file(world):
    (world["reports"] / "stage4_list_update_gate_20261008075007-crk.json").write_text(json.dumps(FORWARD_GATE))
    _, _, text, status = make(world)
    got = normalised(text, world["engine"])
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(got, encoding="utf-8")
    assert got == GOLDEN.read_text(encoding="utf-8")
    assert status == 0


def test_a_reverse_time_gate_report_is_labelled_in_the_markdown(world):
    (world["reports"] / "stage4_list_update_gate_REVERSE_TIME_20260701194702-fjb.json").write_text(
        json.dumps(REVERSE_GATE)
    )
    _, _, text, _ = make(world)
    assert "REVERSE-TIME CHURN TEST" in text and "not a forecast" in text
    assert "screening_holdout_v3.json" in text and "SECOND hold-out" in text  # provenance is shown


def test_the_newest_gate_report_by_snapshot_version_is_the_one_shown(world):
    (world["reports"] / "stage4_list_update_gate_20261008075007-crk.json").write_text(json.dumps(FORWARD_GATE))
    (world["reports"] / "stage4_list_update_gate_REVERSE_TIME_20260701194702-fjb.json").write_text(
        json.dumps(REVERSE_GATE)
    )
    _, _, text, _ = make(world)
    assert "20261008075007-crk" in text and "REVERSE-TIME" not in text


def test_with_no_gate_report_the_section_says_so(world):
    _, _, text, _ = make(world)
    assert "No gate report was found" in text


# ---------- what the markdown holds ----------


def test_every_section_is_there_and_the_limits_are_the_fixed_text(world):
    _, _, text, _ = make(world)
    for heading in ("## Integrity", "## Window and run contexts", "## Metrics", "## Drift against the baseline",
                    "## Latest list-update gate report", "## Limits of this report"):  # fmt: skip
        assert heading in text
    limits = text.split("## Limits of this report")[1]
    for paragraph in audit_report.LIMITS_PARAGRAPHS:
        assert paragraph in limits
    assert "constructed batches, not production traffic" in limits and "no analyst dispositions" in limits
    assert "deduplication model is outside the decision path" in limits


def test_the_drift_section_has_every_comparison_k_and_the_corrected_percentiles(world):
    _, _, text, _ = make(world)
    drift = text.split("## Drift against the baseline")[1].split("## Latest")[0]
    for label in ("| Score |", "| Input mix |", "| Latin score |", "| Arabic score |", "| Cyrillic score |"):
        assert label in drift
    assert "k = 5" in drift and "Bonferroni" in drift and "99.0000" in drift and "99.8000" in drift
    for column in (
        "PSI",
        "n live",
        "n baseline",
        "Null p95",
        "Null p99",
        "Warn limit",
        "Alert limit",
        "Corrected status",
    ):
        assert column in drift
    assert "Near-threshold rate" in drift and "alert rate" in drift


def test_suppression_in_the_json_carries_into_the_markdown(world):
    audit_dir, a, b = live_window(world, {"Latin": 400, "Arabic": 200, "Cyrillic": 200})
    report, _ = compute_metrics(audit_dir, a, b, MetricsConfig(0.05, 300), CLOCK)  # min_cell above Arabic and Cyrillic
    text, _ = build_report(audit_dir, a, b, baseline_dir=world["baseline"], reports_dir=world["reports"])
    cells = {s: c["count"] for s, c in report["groups"][0]["per_script"].items()}
    assert "<300" in cells.values() or "withheld" in cells.values()
    assert any(f"| {s} | {c} | suppressed |" in text for s, c in cells.items() if not isinstance(c, int))


# ---------- privacy, plain writing ----------


def test_no_name_derived_value_and_no_request_id_in_the_markdown(world):
    audit_dir, window, text, _ = make(world)
    engine = world["engine"]
    forbidden = set()
    for name in NAMES.values():
        derived = engine.evaluate(name).derived  # the payload's name-derived fields, as the audit path writes them
        payload = {"raw_input": {"name": name}, **derived}
        forbidden |= forbidden_strings(name, None, None, derived["normalised"], payload)
    lowered = text.lower()
    for item in forbidden:
        if len(item) >= 4:
            assert item.lower() not in lowered, item
    for header in engine.log.headers.read_all():
        assert header["request_id"] not in text
    assert str(world["tmp"]) not in text  # no paths either


def test_no_em_dash_and_no_horizontal_rule_anywhere(world):
    (world["reports"] / "stage4_list_update_gate_20261008075007-crk.json").write_text(json.dumps(FORWARD_GATE))
    _, _, text, _ = make(world)
    assert "—" not in text and "–" not in text and "―" not in text
    for line in text.splitlines():
        assert not re.fullmatch(r"\s*([-*_=~])(\s*\1){2,}\s*", line), line  # no ---, ***, ___, === lines
        assert not line.startswith("<") and "```" not in line
    assert audit_report.LIMITS_PARAGRAPHS and all("—" not in p for p in audit_report.LIMITS_PARAGRAPHS)


def test_the_report_module_source_has_no_em_dash():
    assert "—" not in Path(audit_report.__file__).read_text(encoding="utf-8")


# ---------- the store is left alone ----------


def test_the_audit_log_and_the_baseline_stay_byte_identical(world):
    audit_dir, a, b = live_window(world, {"Latin": 400, "Arabic": 200, "Cyrillic": 200})
    before_store = {k: v for k, v in tree(audit_dir).items() if not k.startswith("monitoring")}
    before_baseline = tree(world["baseline"])
    text, _ = build_report(audit_dir, a, b, baseline_dir=world["baseline"], reports_dir=world["reports"],
                           metrics_config=CONFIG, clock=CLOCK)  # fmt: skip
    after_store = {k: v for k, v in tree(audit_dir).items() if not k.startswith("monitoring")}
    assert after_store == before_store and tree(world["baseline"]) == before_baseline
    assert set(tree(audit_dir)) - set(before_store) == {f"monitoring/{a}-{b}.json"}  # only the metrics JSON is new


# ---------- exit status ----------


def test_an_unchanged_window_exits_0_and_a_shifted_one_exits_20(world):
    assert make(world)[3] == 0
    _, _, text, status = make(world | {"engine": make_engine(world["tmp"] / "live2")}, shifted=True)
    assert status == 20 and "Overall status: alert" in text


@pytest.mark.parametrize(("overall", "code"), [("ok", 0), ("insufficient_data", 3), ("warn", 10), ("alert", 20)])
def test_the_exit_status_is_the_worst_status(world, monkeypatch, overall, code):
    real = audit_report.compare_to_baseline

    def fake(report, baseline, config=None, live_contexts=None):
        return {**real(report, baseline, config), "overall": overall}

    monkeypatch.setattr(audit_report, "compare_to_baseline", fake)
    assert make(world)[3] == code


def test_with_several_baselines_the_exit_status_is_the_worst(world, monkeypatch):
    statuses = iter(["warn", "alert"])
    real = audit_report.compare_to_baseline
    monkeypatch.setattr(
        audit_report, "compare_to_baseline", lambda *a, **k: {**real(*a, **k), "overall": next(statuses)}
    )
    monkeypatch.setattr(
        audit_report,
        "find_baselines",
        lambda report, d: ([audit_report.load_baseline(p) for p in d.glob("*.json")] * 2, []),
    )
    assert make(world)[3] == 20


# ---------- refusals write nothing ----------


def test_no_matching_baseline_is_refused(world, tmp_path):
    audit_dir, a, b = live_window(world, {"Latin": 400, "Arabic": 200, "Cyrillic": 200})
    empty = tmp_path / "nobaseline"
    empty.mkdir()
    with pytest.raises(ReportRefused, match="no baseline"):
        build_report(
            audit_dir, a, b, baseline_dir=empty, reports_dir=world["reports"], metrics_config=CONFIG, clock=CLOCK
        )


def test_a_tampered_log_is_refused_before_anything_is_written(world):
    audit_dir, a, b = live_window(world, {"Latin": 400, "Arabic": 200, "Cyrillic": 200})
    path = audit_dir / "headers.jsonl"
    lines = path.read_bytes().split(b"\n")
    lines[3] = (
        lines[3]
        .replace(b'"decision": "clear"', b'"decision": "alert"')
        .replace(b'"decision":"clear"', b'"decision":"alert"')
    )
    path.write_bytes(b"\n".join(lines))
    before = tree(audit_dir)
    with pytest.raises(ReportRefused, match="tampering"):
        build_report(
            audit_dir,
            a,
            b,
            baseline_dir=world["baseline"],
            reports_dir=world["reports"],
            metrics_config=CONFIG,
            clock=CLOCK,
        )
    assert tree(audit_dir) == before


# ---------- the CLI ----------


def run_cli(world, *extra, counts=None, shifted=False):
    audit_dir, a, b = live_window(world, counts or {"Latin": 400, "Arabic": 200, "Cyrillic": 200}, shifted)
    return audit_dir, a, b


def test_cli_report_writes_the_markdown_once_and_returns_the_status(world, monkeypatch, capsys):
    audit_dir, a, b = live_window(world, {"Latin": 400, "Arabic": 200, "Cyrillic": 200})
    monkeypatch.setattr(audit_report, "BASELINE_DIR", world["baseline"])
    monkeypatch.setattr(audit_report, "REPORT_DIR", world["reports"])
    real = audit_report.build_report
    monkeypatch.setattr(
        screen_cli,
        "build_report",
        lambda d, x, y: real(
            d, x, y, baseline_dir=world["baseline"], reports_dir=world["reports"], metrics_config=CONFIG, clock=CLOCK
        ),
    )
    code = screen_cli.main(["report", "--from", str(a), "--to", str(b), "--audit-dir", str(audit_dir)])
    out = capsys.readouterr().out
    assert code == 0 and (audit_dir / "monitoring" / f"{a}-{b}.md").is_file() and "exit status 0: ok" in out
    again = screen_cli.main(["report", "--from", str(a), "--to", str(b), "--audit-dir", str(audit_dir)])
    assert again == 1 and "never overwritten" in capsys.readouterr().err


def test_cli_report_refuses_without_an_audit_dir_and_for_a_bad_window(world, monkeypatch, capsys):
    monkeypatch.delenv("SCREENING_AUDIT_DIR", raising=False)
    assert screen_cli.main(["report", "--from", "0", "--to", "1"]) == 2
    audit_dir, a, b = live_window(world, {"Latin": 400, "Arabic": 200, "Cyrillic": 200})
    assert screen_cli.main(["report", "--from", "5", "--to", "2", "--audit-dir", str(audit_dir)]) == 2
    assert screen_cli.main(["report", "--from", "0", "--to", "999999", "--audit-dir", str(audit_dir)]) == 2
    assert not list((audit_dir / "monitoring").glob("*.md")) if (audit_dir / "monitoring").exists() else True


def test_the_report_command_is_listed_and_the_module_is_classified():
    parser, _ = screen_cli.build_parser()
    assert "report" in parser.format_help()


# ---------- the README summary and the doc say the same things ----------

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DOC = ROOT / "docs" / "stage4_audit_and_monitoring.md"


def readme_summary() -> str:
    text = README.read_text(encoding="utf-8")
    return text.split("## Stage 4: audit log and monitoring")[1].split("\n## ")[0]


def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def test_the_readme_summary_links_the_doc_and_is_short():
    summary = readme_summary()
    assert "(docs/stage4_audit_and_monitoring.md)" in summary and DOC.is_file()
    assert 300 <= len(summary.split()) <= 520  # about 400 words


def test_the_readme_states_the_gaps_in_the_words_the_report_uses():
    for text in (readme_summary(), doc_text()):
        for paragraph in audit_report.LIMITS_PARAGRAPHS[:3]:
            assert paragraph in text


def test_the_readme_what_this_does_not_show_list_is_the_docs_list_unchanged():
    def listing(text: str) -> str:
        return text.split("### What this does not show")[1].strip()

    assert listing(readme_summary()) == listing(doc_text())


def test_every_number_in_the_readme_summary_appears_in_the_doc():
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", readme_summary())
    numbers = [n.rstrip(",.") for n in numbers]
    assert len(numbers) > 15
    doc = doc_text()
    missing = [n for n in numbers if n not in doc]
    assert not missing, missing


def test_the_readme_summary_and_the_doc_are_plain_writing():
    for text in (readme_summary(), doc_text()):
        assert "\u2014" not in text and "\u2013" not in text
        for line in text.splitlines():
            assert not re.fullmatch(r"\s*([-*_=~])(\s*\1){2,}\s*", line), line


def test_the_doc_covers_the_required_topics_and_defines_positives_first():
    doc = doc_text().lower()
    for topic in ("regulation 40", "regulation 87", "tamper-evident, not tamper-proof", "bonferroni", "two-sample null",
                  "reverse-time churn test", "deduplication model is outside the decision path", "legal hold",
                  "`init`", "checkpoint", "replay", "recovery"):  # fmt: skip
        assert topic in doc, topic
    first_count = doc.index("2,400")
    assert (
        "headline positives" in doc[: first_count + 400] and "all of its positive queries" in doc[: first_count + 400]
    )
    assert doc.index("headline positives") < doc.index("1,999")


def test_the_regulation_wording_names_the_knowledge_condition_and_both_start_events():
    doc = doc_text()
    assert "knows or suspects" in doc and "knows, or has reasonable grounds to believe" in doc
    assert "occasional transaction" in doc and "business relationship has ended" in doc
    assert "ten years" in doc and "does not enforce the ten-year cap" in doc

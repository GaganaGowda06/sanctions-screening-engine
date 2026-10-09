"""The short README at the repository root: every figure in it ties to a report or a doc.

Reads the committed reports and docs only (no data/ needed).

Run:
    pytest tests/test_readme.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
DOCS = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "docs").glob("*.md")}
LIMITS = DOCS["limitations_evidence.md"]


def report(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def section(title: str) -> str:
    return README.split(f"## {title}")[1].split("\n## ")[0]


def test_the_readme_is_short_and_has_its_sections_in_order():
    assert 600 <= len(README.split()) <= 1100
    titles = re.findall(r"^## (.+)$", README, flags=re.M)
    assert titles == ["Headline numbers", "The stages", "Limits", "How to run", "Licence and attribution"]
    opening = README.split("\n## ")[0].split("\n", 1)[1].strip()
    assert len(opening.split()) < 100
    for stage in ("Data.", "Matcher.", "Arabic fairness.", "Audit and monitoring.", "Analysis and validation."):
        assert f"**{stage}**" in section("The stages")


def test_the_headline_f1_and_arabic_burden_match_the_provenance_report():
    third = report("stage4_f1_provenance.json")["third_holdout"]
    current, stage2 = third["current_engine"], third["stage2_engine"]
    assert f"0.80 ({current['f1']:.4f})" in README and f"{stage2['f1']:.4f}" in README
    assert round(current["f1"], 2) == 0.80
    assert f"{stage2['arabic_burden'] * 100:.1f}% to {current['arabic_burden'] * 100:.1f}%" in README
    assert current["headline_positives"] == current["headline_negatives"] == 2000 and "2,000 constructed" in README


def test_the_pairs_figures_match_the_scores_and_the_sweep_reports():
    scores = report("pairs_external_scores.json")
    sweep = report("pairs_external_threshold_sweep.json")
    assert f"{scores['pairs_scored']:,}" == "439,813" and "439,813" in README
    hard = sweep["no_shared_name"]
    assert f"{hard['pairs']:,}" in README
    assert f"{hard['at_086']['f1']:.3f} ({hard['at_086']['f1']:.4f})" in README
    allp = sweep["all_pairs"]
    for f1 in (allp["at_086"]["f1"], allp["best"]["f1"], hard["best"]["f1"]):
        assert f"{f1:.4f}" in README
    assert allp["best"]["threshold"] == hard["best"]["threshold"] == 0.85 and "peaks at 0.85" in README
    assert abs(allp["best"]["threshold"] - 0.86) < 0.0100001  # within one step of 0.86


def test_the_other_figures_in_the_limits_are_in_the_evidence_docs():
    for fragment in ("0.7953", "1 alert in 100", "1 in 1,000", "7% to 9%", "0.7775", "99.75%", "0.0881"):
        assert fragment in LIMITS, fragment
    for fragment in ("0.7953", "1 alert in 100", "1 in 1,000", "7% to 9%", "0.7775", "99.75%"):
        assert fragment in README, fragment
    assert "the full name scores 1.0" in README


def test_every_number_in_the_readme_appears_in_a_doc_or_is_a_listed_non_claim():
    non_claims = {"3.14", "350", "0.50", "0.99", "2", "1"}
    numbers = {n.rstrip(",.") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", README)}
    text = "\n".join(DOCS.values())
    missing = sorted(n for n in numbers - non_claims if n not in text)
    assert not missing, missing


def test_the_what_this_does_not_show_list_is_the_stage_4_docs_list_unchanged():
    doc = DOCS["stage4_audit_and_monitoring.md"].split("### What this does not show")[1].strip().split("\n## ")[0]
    assert doc.strip() in README


def test_links_and_commands_point_at_files_that_exist():
    for target in re.findall(r"\]\(([^)#]+)\)", README):
        if not target.startswith("http"):
            assert (ROOT / target).is_file(), target
    for path in ("scripts/demo.py", "app/streamlit_app.py", "requirements.txt", "src/download_data.py"):
        assert path in README and (ROOT / path).is_file(), path


def test_the_licence_and_attribution_are_stated():
    text = section("Licence and attribution")
    assert "CC BY-NC 4.0" in text and "non-commercial" in text and "OpenSanctions" in text
    assert "arXiv:2603.11051" in text and "derived names, birth dates and IDs" in text
    assert "code is MIT licensed" in text and "covers the code only, not the data" in text
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")


def test_the_readme_is_plain_writing():
    assert "—" not in README and "–" not in README
    for line in README.splitlines():
        assert not re.fullmatch(r"\s*([-*_=~])(\s*\1){2,}\s*", line), line

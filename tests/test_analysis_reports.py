"""The findings docs quote numbers from the analysis reports; these tests keep the two together.

The reports are produced by the scripts in analysis/ from the frozen hold-outs. These tests read the committed reports
(no data/ needed), recompute the derived figures from them, and check the figures the docs state.

Run:
    pytest tests/test_analysis_reports.py
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
LIMITS = (ROOT / "docs" / "limitations_evidence.md").read_text(encoding="utf-8")
LABELS = (ROOT / "docs" / "label_provenance.md").read_text(encoding="utf-8")


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def test_the_second_holdout_figures_reproduce_the_reported_headline():
    s = load("analysis_error_analysis_v2_stage2.json")["summary"]["headline"]
    reported = load("screening_holdout_v2_evaluation.json")["overall"]
    assert (s["true_alerts"], s["wrong_entity_alerts_on_positives"], s["false_alerts_on_negatives"]) == (1474, 132, 187)
    assert round(s["f1"], 4) == reported["f1"]["value"] == 0.7772
    assert s["true_alerts"] == reported["recall"]["k"] and s["true_alerts"] + 132 + 187 == reported["precision"]["n"]


def test_the_shipped_engine_figures_reproduce_the_stage_4_provenance():
    s = load("analysis_error_analysis_v3_current.json")["summary"]["headline"]
    assert round(s["f1"], 4) == round(load("stage4_f1_provenance.json")["third_holdout"]["current_engine"]["f1"], 4)


def test_the_false_negative_tables_add_up_and_match_the_doc():
    f = load("analysis_error_analysis_v2_stage2.json")["summary"]["false_negatives"]
    assert f["count"] == sum(f["by_cause"].values()) == sum(f["by_severity"].values()) == 526
    assert f["by_severity"] == {"borderline": 100, "dangerous": 158, "no signal": 122, "weak": 146}
    assert f["by_cause"] == {
        "missing_attribute": 74,
        "org_translation": 128,
        "other": 210,
        "unvowelled_cross_script": 114,
    }
    cells = f["by_cause_and_severity"]
    for severity, total in f["by_severity"].items():
        assert sum(v for k, v in cells.items() if k.endswith("/" + severity)) == total
    assert "| All | 526 | 100% | 158 | 100 | 146 | 122 |" in LIMITS
    dangerous = load("analysis_error_analysis_v2_stage2.json")["dangerous_false_negatives"]
    assert len(dangerous) == f["dangerous_total"] == 158
    assert all(d["severity"] == "dangerous" and d["dangerous_reasons"] for d in dangerous)


def test_a_dangerous_miss_is_one_of_the_three_stated_things():
    for d in load("analysis_error_analysis_v2_stage2.json")["dangerous_false_negatives"]:
        score = d["final_score"]
        reasons = set(d["dangerous_reasons"])
        assert reasons <= {"final score 0.80 or more", "near-identical text", "query tokens all appear in an alias"}
        assert ("final score 0.80 or more" in reasons) == (score is not None and score >= 0.80)
        if score is not None:
            assert score < 0.86  # a miss


def test_false_positive_counts_add_up():
    p = load("analysis_error_analysis_v2_stage2.json")["summary"]["false_positives"]
    assert p["count"] == sum(p["by_kind"].values()) == sum(p["by_cause"].values()) == 319
    assert p["by_kind"] == {"near_miss": 15, "synthetic": 172, "wrong_entity_on_positive": 132}
    assert p["near_miss_alerts_on_the_imitated_entity"] == 0


def test_the_attribute_and_scope_figures_in_the_label_doc():
    d = load("analysis_label_dependence.json")
    assert round(d["A_attribute_modes"]["both"]["f1"], 4) == 0.7772  # the harness agrees with the engine
    assert round(d["A_attribute_modes"]["neither"]["f1"], 4) == 0.5865
    assert (
        d["A_attribute_modes"]["neither"]["fpr_near_miss"] == 1.0
        and d["A_attribute_modes"]["both"]["fpr_near_miss"] == 0.03
    )
    assert round(d["B_scope"]["every_query"]["f1"], 4) == 0.7109 and d["B_scope"]["headline"]["positives"] == 2000
    e = d["E_what_a_hit_is"]
    assert (e["hits"], e["hits_where_other_entities_also_alerted"], e["extra_entity_alerts_on_hits"]) == (
        1474,
        414,
        1291,
    )
    assert round(e["precision_if_every_extra_entity_alert_counted_as_false"], 4) == 0.4780
    for text in ("0.5865", "0.7109", "0.4780", "0.5799", "1,291", "414"):
        assert text in LABELS or text in LIMITS


def test_the_query_provenance_counts_in_the_label_doc():
    second = load("analysis_query_provenance.json")["second"]
    assert second["positives"] == 2400 and second["positives_whose_query_is_also_a_weak_alias"] == 33
    assert second["by_properties_the_text_appears_in"]["name"] == 1565
    assert second["by_properties_the_text_appears_in"]["alias"] == 701


@pytest.mark.parametrize("engine", ["stage2_engine_on_second_holdout", "shipped_engine_on_third_holdout"])
def test_the_prevalence_table_follows_the_formula(engine):
    t = load("analysis_prevalence.json")[engine]
    r, w, f = t["rates"]["r"], t["rates"]["w"], t["rates"]["f"]
    for row in t["by_prevalence"]:
        p = row["prevalence"]
        assert row["precision"] == pytest.approx(p * r / (p * r + p * w + (1 - p) * f), rel=1e-12)
        assert row["alert_rate"] == pytest.approx(p * (r + w) + (1 - p) * f, rel=1e-12)
    assert t["by_prevalence"][0]["precision"] == pytest.approx(
        load(
            "analysis_error_analysis_v2_stage2.json"
            if "stage2" in engine
            else "analysis_error_analysis_v3_current.json"
        )["summary"]["headline"]["precision"],
        rel=1e-9,
    )
    assert [r["prevalence"] for r in t["by_prevalence"]] == [0.5, 0.01, 0.001, 0.0001, 1e-05]
    precisions = [r["precision"] for r in t["by_prevalence"]]
    assert precisions == sorted(precisions, reverse=True)


def test_the_prevalence_figures_in_the_doc():
    t = load("analysis_prevalence.json")["stage2_engine_on_second_holdout"]["by_prevalence"]
    assert f"| 1 in 1,000 | {t[2]['precision']:.4f} |" in LIMITS
    assert f"| 1 in 100 | {t[1]['precision']:.4f} |" in LIMITS


def test_the_tuning_transfer_figures_in_the_doc():
    d = load("analysis_tuning_transfer.json")
    g = d["grid_on_slice"]
    assert (
        g["configurations"] == 3750
        and g["rank_of_frozen_choice"] == 5
        and g["configurations_within_0.005_of_best"] == 18
    )
    assert g["frozen_choice"]["f1_on_slice"] == 0.7772 and g["regret_f1"] == 0.0003
    v = d["variants_on_slice"]
    assert round(v["stage2_skeleton_max_df_1000"]["f1"], 4) == 0.7772
    assert round(v["rarity_skeleton_max_df_1000"]["f1"], 4) == 0.8085
    assert round(v["rarity_skeleton_max_df_250"]["f1"], 4) == 0.8166
    sizes = [250, 500, 1000, 2000, 5000]
    f1s = [v[f"rarity_skeleton_max_df_{n}"]["f1"] for n in sizes]
    assert f1s == sorted(f1s, reverse=True)  # F1 falls steadily as max_df rises
    assert d["tuned_numbers"]["dev_f1_at_frozen_config"] == 0.7695 and d["tuned_numbers"]["test_f1"] == 0.7645


def test_the_adversarial_table_adds_up_and_controls_lose_nothing():
    a = load("analysis_adversarial_v3_current.json")
    assert a["base_positives_caught_unperturbed"] == 1489
    for name, f in a["families"].items():
        assert f["robust"] + f["slip_through"] == f["variants"], name
    assert a["families"]["control_uppercase"]["slip_through"] == 0
    assert a["families"]["translit_cyrillic_bgn"]["slip_through"] == 0  # the matcher's own scheme
    assert len(a["all_slip_cases"]) == sum(f["slip_through"] for f in a["families"].values())
    s = load("analysis_adversarial_summary_v3_current.json")
    assert (s["character_edits"]["variants"], s["character_edits"]["slip_throughs"]) == (11594, 1549)
    assert s["character_edit_slips"]["slips_where_the_true_entity_was_not_scored_at_all"] == 178
    assert a["bases_with_a_slip_in_at_least_one_family"] == 1260
    for text in ("| consonant_delete | 1,014 | 2,617 | 1,990 | 627 | 24.0% |", "11,594 variants, 1,549 slips (13.4%)"):
        assert text in LIMITS


def test_the_retrieval_gap_figures():
    g = load("analysis_retrieval_gap.json")
    assert g["adversarial_character_edit_slips_with_no_score_at_all"]["cases"] == 178
    assert g["adversarial_character_edit_slips_with_no_score_at_all"]["would_have_alerted"] == 128
    assert g["second_holdout_stage2_engine"]["false_negatives_with_no_score_at_all"]["would_have_alerted"] == 17


def test_the_pairs_alignment_counts_add_up_and_match_the_doc():
    a = load("pairs_external_alignment.json")
    assert a["pairs_total"] == 755540 and a["judgements"] == {"negative": 174391, "positive": 581149}
    assert a["scope"] == {"entity_pairs": 474950, "relationship_or_role_records": 280590}
    assert sum(a["not_scored_by_schema_pair"].values()) == 280590
    assert a["file"]["sha256"] == "4044cf5f6edb0d2cdccf11326063f905408224c3b2fe261baa0d779d80a0edd1"
    s = load("pairs_external_scores.json")
    assert s["file_sha256"] == a["file"]["sha256"]
    assert (
        s["pairs_scored"] + sum(s["pairs_not_scored"].values()) - s["pairs_not_scored"]["not_an_entity_pair"] == 474950
    )
    assert s["pairs_not_scored"] == {
        "a_side_has_no_name_property": 5531, "conflicting_labels_left_out": 14,
        "duplicate_line": 29592, "not_an_entity_pair": 280590,
    }  # fmt: skip
    assert s["self_check_against_screen"] == {"entities_checked_against_screen": 300, "mismatches": 0}
    for text in (
        "439,813",
        "755,540",
        "280,590",
        "5,531",
        "29,592",
        "4044cf5f6edb0d2cdccf11326063f905408224c3b2fe261baa0d779d80a0edd1",
    ):
        assert text in LIMITS


def test_the_pairs_scores_are_consistent_and_match_the_doc():
    s = load("pairs_external_scores.json")["left_names_as_queries_against_right"]
    o = s["overall"]
    assert o["tp"] + o["fp"] + o["fn"] + o["tn"] == o["pairs"] == 439813 and o["positives"] == 268490
    assert o["precision"] == pytest.approx(o["tp"] / (o["tp"] + o["fp"]), rel=1e-12)
    assert o["recall"] == pytest.approx(o["tp"] / o["positives"], rel=1e-12)
    assert o["f1"] == pytest.approx(2 * o["precision"] * o["recall"] / (o["precision"] + o["recall"]), rel=1e-12)
    assert o["auc"] > 0.9 and o["f1_95_interval"][0] < o["f1"] < o["f1_95_interval"][1]
    for name in ("by_entity_type", "by_script"):
        assert sum(v["pairs"] for v in s[name].values()) == o["pairs"]
        assert sum(v["tp"] for v in s[name].values()) == o["tp"]
    same, none = s["shares_an_identical_name"], s["shares_no_identical_name"]
    assert same["pairs"] + none["pairs"] == o["pairs"] and same["positives"] + none["positives"] == o["positives"]
    assert (
        same["auc"] < 0.6 < none["auc"]
    )  # an identical name carries no ranking information among the pairs it contains
    for text in (
        f"{o['f1']:.4f}",
        f"{none['f1']:.4f}",
        f"{none['recall']:.4f}",
        f"{same['false_positive_rate'] * 100:.2f}%",
    ):
        assert text in LIMITS, text
    vessels = s["shares_no_identical_name_by_schema_pair"]["Vessel/Vessel"]
    assert vessels["auc"] < 0.5 and f"{vessels['f1']:.4f}" in LIMITS and f"{vessels['recall']:.4f}" in LIMITS


def test_the_pairs_direction_sensitivity_is_close_to_the_main_run():
    both = load("pairs_external_scores.json")
    assert (
        abs(
            both["left_names_as_queries_against_right"]["overall"]["f1"]
            - both["right_names_as_queries_against_left"]["overall"]["f1"]
        )
        < 0.01
    )


def test_the_licence_and_attribution_are_stated_in_the_readme_and_the_docs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in (readme, LIMITS, LABELS):
        assert "CC BY-NC 4.0" in text or "CC-BY-NC-4.0" in text
        assert "OpenSanctions" in text and "non-commercial" in text.lower()
    assert "arXiv:2603.11051" in LIMITS and "arXiv:2603.11051" in LABELS and "arXiv:2603.11051" in readme
    assert "This project is non-commercial" in LIMITS and "not stored in the repository" in LIMITS
    assert "data/external/" in (ROOT / ".gitignore").read_text()


def test_the_stage_5_dev_verdict_follows_the_criteria_written_before_the_run():
    d = load("stage5_dev.json")
    v = d["verdict"]
    one, two, three, four = (
        v["1_single_letter_slip_rate"],
        v["2_f1"],
        v["3_false_alert_rate"],
        v["4_candidates_and_latency"],
    )
    assert one["pass"] == (one["after"] <= 0.5 * one["before"] and not one["families_whose_rate_rose"])
    assert two["pass"] == (two["after"] >= two["before"] - 0.005)
    assert three["pass"] == (three["overall_rise_points"] <= 0.5 and three["arabic_rise_points"] <= 1.0)
    assert four["pass"] == (four["candidates_ratio"] <= 2 and four["latency_ratio"] <= 2)
    assert v["all_pass"] == (one["pass"] and two["pass"] and three["pass"] and four["pass"])
    assert v["all_pass"] is False and not (REPORTS / "stage5_v4.json").exists()  # rejected on dev: v4 is not scored
    assert d["max_df_250_vs_1000_with_the_new_index"]["adopt"] is False
    for text in ("13.90%", "12.92%", "+0.95", "+2.36", "1.35x", "1.43x"):
        assert text in LIMITS, text


def test_the_threshold_sweep_table_in_the_doc_matches_the_report():
    sweep = load("pairs_external_threshold_sweep.json")
    rows = {"All scored pairs": "all_pairs", "No shared name": "no_shared_name", "Person to person": "person_person"}
    for label, key in rows.items():
        line = next(
            ln for ln in LIMITS.splitlines() if ln.startswith(f"| {label} | ") and "Best" not in ln and "0.85" in ln
        )
        cells = [c.strip() for c in line.strip("|").split("|")]
        at, best = sweep[key]["at_086"], sweep[key]["best"]
        assert sweep[key]["reproduces_stored_086"] is True
        assert cells[1] == f"{sweep[key]['pairs']:,}"
        assert cells[2] == f"{at['precision']:.4f} / {at['recall']:.4f} / {at['f1']:.4f}"
        assert cells[3] == f"{best['threshold']:.2f}" == "0.85"
        assert cells[4] == f"{best['precision']:.4f} / {best['recall']:.4f} / {best['f1']:.4f}"
        assert cells[5] == f"{best['f1'] - at['f1']:+.4f}"
        assert best["f1"] == max(r["f1"] for r in sweep[key]["sweep"])
        assert [r["threshold"] for r in sweep[key]["sweep"]] == [round(0.5 + i / 100, 2) for i in range(50)]


def test_the_docs_are_plain_writing():
    for text in (LIMITS, LABELS):
        assert "—" not in text and "–" not in text
        for line in text.splitlines():
            assert not re.fullmatch(r"\s*([-*_=~])(\s*\1){2,}\s*", line), line

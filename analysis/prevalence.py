"""Precision and alert rate at realistic prevalence, with recall and the false-alert rate held at the measured values.

Run (after analysis/error_analysis.py has written its reports):
    python analysis/prevalence.py

The hold-out has as many sanctioned-name queries as clean-name queries (2,000 headline positives, 2,000 headline
negatives), so its precision describes a world where half of all customers are listed. In a real book of business the
share is tiny. With prevalence p (the share of customers who really are a listed party) and three measured rates:

    r  the share of listed parties that alert on their own entity            (recall)
    w  the share of listed parties that miss their own entity but alert on another one
    f  the share of clean customers that alert                               (false-alert rate)

the expected alerts per customer screened are
    true alerts      p * r
    wrong alerts     p * w         (a listed party's alert names the wrong entity: counted as false, as the headline
                                   does)
    false alerts     (1 - p) * f
    alert rate       p * (r + w) + (1 - p) * f
    precision        p * r / (p * r + p * w + (1 - p) * f)
which is the headline definition of precision with the two groups re-weighted. At p = 0.5 it must give the measured
precision; that is checked. Nothing here is a new measurement: r, w and f are counted from the hold-out, and only the
weights change.
What is NOT known: the false-alert rate on real customers. The hold-out's clean names are made from real name tokens
of listed people, and a quarter are real listed names with a conflicting birth date and country, so f may be higher or
lower than for a real customer book.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PREVALENCES = (0.5, 0.01, 1e-3, 1e-4, 1e-5)
CUSTOMERS = 100_000


def rates_at(r: float, w: float, f: float, p: float) -> dict:
    true_alerts, wrong, false = p * r, p * w, (1 - p) * f
    alerts = true_alerts + wrong + false
    return {
        "prevalence": p,
        "precision": true_alerts / alerts,
        "alert_rate": p * (r + w) + (1 - p) * f,
        "per_100000_customers": {
            "listed_parties": p * CUSTOMERS,
            "true_alerts": true_alerts * CUSTOMERS,
            "wrong_entity_alerts": wrong * CUSTOMERS,
            "false_alerts_on_clean_customers": false * CUSTOMERS,
            "alerts_to_review": alerts * CUSTOMERS,
        },
        "alerts_reviewed_per_true_hit": alerts / true_alerts,
    }


def fpr_needed_for_precision(r: float, w: float, p: float, target: float) -> float:
    """The false-alert rate at which precision equals `target` at prevalence p."""
    return (p * r / target - p * r - p * w) / (1 - p)


def measured(report: dict, holdout: Path) -> dict:
    s = report["summary"]
    h = s["headline"]
    queries = json.loads(holdout.read_text())["queries"]
    head_pos = [q for q in queries if q["kind"] == "positive" and q["group"] in common.ev.HEADLINE_GROUPS]
    head_neg = [q for q in queries if q["kind"] != "positive" and q["group"] in common.ev.HEADLINE_NEG_SCRIPTS]
    from screening import query_script

    scripts = {}
    for script in common.ev.HEADLINE_NEG_SCRIPTS:
        n_pos = sum(query_script(q["name"]) == script for q in head_pos)
        n_neg = sum(query_script(q["name"]) == script for q in head_neg)
        miss = [f for f in report["false_negatives"] if f["query_script"] == script]
        wrong = sum(f["also_alerted_on_another_entity"] for f in miss)
        false = sum(
            1
            for f in report["false_positives"]
            if f["kind"] != "wrong_entity_on_positive" and f["query_script"] == script
        )
        scripts[script] = {
            "positives": n_pos,
            "negatives": n_neg,
            "r": (n_pos - len(miss)) / n_pos,
            "w": wrong / n_pos,
            "f": false / n_neg,
        }
    return {
        "r": h["true_alerts"] / h["headline_positives"],
        "w": h["wrong_entity_alerts_on_positives"] / h["headline_positives"],
        "f": h["false_alerts_on_negatives"] / h["headline_negatives"],
        "headline_precision": h["precision"],
        "headline_f1": h["f1"],
        "by_query_script": scripts,
    }


def table(m: dict) -> dict:
    rows = [rates_at(m["r"], m["w"], m["f"], p) for p in PREVALENCES]
    assert abs(rows[0]["precision"] - m["headline_precision"]) < 1e-9, (
        "the formula must reproduce the measured precision at 50%"
    )
    return {
        "rates": {k: m[k] for k in ("r", "w", "f")},
        "by_prevalence": rows,
        "false_alert_rate_that_would_give_precision_50_percent": {
            str(p): fpr_needed_for_precision(m["r"], m["w"], p, 0.5) for p in PREVALENCES[1:]
        },
        "one_false_alert_per_n_clean_customers_needed_for_50_percent_precision": {
            str(p): 1 / fpr_needed_for_precision(m["r"], m["w"], p, 0.5) for p in PREVALENCES[1:]
        },
        "by_query_script": {
            script: [rates_at(v["r"], v["w"], v["f"], p) for p in PREVALENCES]
            for script, v in m["by_query_script"].items()
        },
        "by_query_script_rates": m["by_query_script"],
    }


def main() -> None:
    out_path = common.OUT / "analysis_prevalence.json"
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists; it is never overwritten")
    result = {}
    for label, name, holdout in (
        ("stage2_engine_on_second_holdout", "v2_stage2", common.HOLDOUT_V2),
        ("shipped_engine_on_third_holdout", "v3_current", common.HOLDOUT_V3),
    ):
        report = json.loads((common.OUT / f"analysis_error_analysis_{name}.json").read_text())
        result[label] = table(measured(report, holdout))
    common.write_json(out_path, result)
    for label, t in result.items():
        print(label, {k: round(v, 4) for k, v in t["rates"].items()})
        for row in t["by_prevalence"]:
            per = row["per_100000_customers"]
            print(
                f"  prevalence {row['prevalence']:<8g} precision {row['precision']:.5f}"
                f"  alert rate {row['alert_rate']:.5f}  per 100k: {per['alerts_to_review']:.0f} alerts,"
                f" {per['true_alerts']:.2f} true"
            )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

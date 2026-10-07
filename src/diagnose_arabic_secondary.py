"""Stage 3 step 5, diagnosis: what do birth date and country do to Arabic false alerts? Dev queries only.

For each dev clean name in Arabic script that alerts under the current engine, take the alerting entity and
report the status of birth date and country: agree, close (a year apart, birth date only), conflict, missing on
the entity side, or missing on the query side. Then whether an agreeing country, or an agreeing birth date,
pushed the score over the threshold (it was under without that adjustment, and at or over with it).

Run:
    python src/diagnose_arabic_secondary.py
Writes reports/stage3_arabic_secondary.json.
"""

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

from entities import load_named_entities
from evaluate_screening import build_query_sets
from profile_data import DEFAULT_INPUT
from scoring import Components, Screener, combine, country_comparison, date_comparison
from screening import CONFIG_PATH, ScreeningConfig

OUTPUT = Path(__file__).resolve().parent.parent / "reports" / "stage3_arabic_secondary.json"


def status(query: str | None, entity_has: bool, compared: str) -> str:
    if not query:
        return "missing_query"
    if not entity_has:
        return "missing_entity"
    return compared


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-df", type=int, default=1000)
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}", file=sys.stderr)

    config = ScreeningConfig.load(CONFIG_PATH)
    t = config.global_threshold
    entities = load_named_entities(args.input)
    targets = [e for e in entities if e["target"]]
    _, negatives, _ = build_query_sets(entities, targets)
    dev = [q for q in negatives if q["split"] in ("dev", "dev_extra") and q["script"] == "Arabic"]
    full = Screener.build(targets, max_df=args.max_df, blend=config.blend)
    adj = full.adjustments

    birth, country, joint = Counter(), Counter(), Counter()
    crossed = Counter()
    alerts = 0
    for q in dev:
        comps = full.components(q["name"], q["dob"], q["country"], blends=(config.blend,))
        if not comps:
            continue
        scores = [(combine(c, config.skeleton_floor, config.skeleton_lift, 0), c) for c in comps]
        score, c = max(scores, key=lambda x: x[0])
        if score < t:
            continue
        alerts += 1
        record = full.records[c.entity_id]
        b = status(q["dob"], bool(record.birth_dates), date_comparison(q["dob"], record.birth_dates, adj)[0])
        k = status(q["country"], bool(record.countries), country_comparison(q["country"], record.countries, adj)[0])
        birth[b] += 1
        country[k] += 1
        joint[f"birth {b} / country {k}"] += 1

        def without(use_dob: bool, use_country: bool, c: Components = c) -> float:
            return combine(c, config.skeleton_floor, config.skeleton_lift, 0, use_dob, use_country)

        crossed["country_agreement_pushed_over"] += k == "agree" and without(True, False) < t
        crossed["birth_agreement_pushed_over"] += b == "agree" and without(False, True) < t
        crossed["either_agreement_pushed_over"] += (k == "agree" or b == "agree") and without(False, False) < t
        crossed["over_even_with_conflicts_present"] += b == "conflict" or k == "conflict"
        crossed["over_with_neither_adjustment"] += without(False, False) >= t
    report = {
        "dev_only": True,
        "arabic_script_clean_names": len(dev),
        "alerts": alerts,
        "birth_date_status": dict(birth),
        "country_status": dict(country),
        "joint": dict(joint.most_common()),
        "pushed_over_the_threshold": dict(crossed),
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

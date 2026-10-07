"""Stage 3 step 2, diagnosis: what drives false alerts on clean Arabic-script names? Dev queries only.

For every dev clean name (synthetic or near-miss, including the extra dev negatives) that raises an alert at the
frozen engine's threshold, find the alerting entity and ask:
  - was the skeleton needed? (the name score alone, with birth date and country, is under the threshold)
  - which skeleton tokens carried the match? Query and entity skeleton tokens are paired by character
    similarity (at least 0.7, as the weighted score does), and each pair is checked for how common the
    entity-side token is across the list (document frequency over listed entities).
  - does the alert rest only on common tokens? Every token that matched is common (document frequency of at
    least COMMON_DF entities) and no rare token matched.
Reported for Arabic, Latin and Cyrillic queries so the Arabic figures have a reference. The test set, and every
hold-out including the third, are not read.

Run:
    python src/diagnose_arabic_fp.py
Writes reports/stage3_arabic_diagnosis.json.
"""

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz, process

from entities import load_named_entities
from evaluate_screening import build_query_sets
from profile_data import DEFAULT_INPUT
from scoring import Screener, combine, scoring_form, skeleton_form
from screening import CONFIG_PATH, ScreeningConfig

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reports" / "stage3_arabic_diagnosis.json"
COMMON_DF = 50  # a skeleton token shared by at least this many listed entities counts as common
TOKEN_CUTOFF = 70


def skeleton_document_frequency(screener: Screener) -> Counter:
    df: Counter = Counter()
    for record in screener.records.values():
        df.update({t for form in record.skeletons for t in form.split()})
    return df


def best_skeleton_form(query_skeleton: str, forms: tuple[str, ...]) -> str:
    return process.extractOne(query_skeleton, forms, scorer=fuzz.token_sort_ratio)[0]


def matched_tokens(query_skeleton: str, form: str) -> list[tuple[str, str, float]]:
    """(query token, entity token, similarity) for each query token with a partner of at least the cutoff."""
    pairs = []
    entity_tokens = form.split()
    for token in query_skeleton.split():
        best = process.extractOne(token, entity_tokens, scorer=fuzz.ratio, score_cutoff=TOKEN_CUTOFF)
        if best:
            pairs.append((token, best[0], best[1] / 100))
    return pairs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-df", type=int, default=1000)
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}", file=sys.stderr)

    config = ScreeningConfig.load(CONFIG_PATH)
    threshold = config.global_threshold
    entities = load_named_entities(args.input)
    targets = [e for e in entities if e["target"]]
    _, negatives, _ = build_query_sets(entities, targets)
    dev = [q for q in negatives if q["split"] in ("dev", "dev_extra")]
    full = Screener.build(targets, max_df=args.max_df, blend=config.blend, skeleton_idf=False)
    sk_df = skeleton_document_frequency(full)

    by_script: dict[str, dict] = {}
    token_counter: dict[str, Counter] = {}
    examples: dict[str, list] = {}
    for q in dev:
        script = q["script"]
        if script not in ("Arabic", "Latin", "Cyrillic"):
            continue
        slot = by_script.setdefault(
            script,
            {"names": 0, "alerts": 0, "needed_skeleton": 0, "arabic_involved_pair": 0, "only_common_tokens": 0,
             "has_rare_token": 0, "no_token_pair": 0},
        )  # fmt: skip
        slot["names"] += 1
        comps = full.components(q["name"], q["dob"], q["country"], blends=(config.blend,))
        scored = sorted(((combine(c, config.skeleton_floor, config.skeleton_lift, 0), c) for c in comps),
                        key=lambda t: -t[0])  # fmt: skip
        if not scored or scored[0][0] < threshold:
            continue
        score, c = scored[0]
        slot["alerts"] += 1
        no_skeleton = min(1.0, max(0.0, c.names[0] + c.dob + c.country))
        slot["needed_skeleton"] += no_skeleton < threshold
        slot["arabic_involved_pair"] += c.arabic
        record = full.records[c.entity_id]
        query_sk = skeleton_form(scoring_form(q["name"]))
        form = best_skeleton_form(query_sk, record.skeletons)
        pairs = matched_tokens(query_sk, form)
        counter = token_counter.setdefault(script, Counter())
        for _, entity_token, _ in pairs:
            counter[entity_token] += 1
        if not pairs:
            slot["no_token_pair"] += 1
        else:
            common = [sk_df[e] >= COMMON_DF for _, e, _ in pairs]
            slot["only_common_tokens"] += all(common)
            slot["has_rare_token"] += not all(common)
        if len(examples.setdefault(script, [])) < 25:
            examples[script].append(
                {"query": q["name"], "entity_name": record.raw_names[0] if record.raw_names else "",
                 "final_score": round(score, 3), "name_score": round(c.names[0], 3),
                 "skeleton_similarity": round(c.skeleton, 3), "needed_skeleton": no_skeleton < threshold,
                 "pairs": [(a, b, round(s, 2), sk_df[b]) for a, b, s in pairs]}
            )  # fmt: skip

    for slot in by_script.values():
        n, a = slot["names"], slot["alerts"]
        slot["alert_rate"] = round(a / n, 4)
        slot["share_of_alerts_needing_skeleton"] = round(slot["needed_skeleton"] / a, 3) if a else None
        slot["share_of_alerts_on_only_common_tokens"] = round(slot["only_common_tokens"] / a, 3) if a else None
    top_tokens = {s: [(t, n, sk_df[t]) for t, n in c.most_common(25)] for s, c in token_counter.items()}
    df_values = sorted(sk_df.values(), reverse=True)
    report = {
        "dev_only": True,
        "threshold": threshold,
        "common_df_cutoff": COMMON_DF,
        "skeleton_tokens_at_or_above_cutoff": sum(v >= COMMON_DF for v in df_values),
        "skeleton_token_types": len(df_values),
        "most_common_skeleton_tokens": [(t, n) for t, n in sk_df.most_common(30)],
        "by_query_script": by_script,
        "tokens_in_alerts_most_frequent_first": top_tokens,
        "examples": examples,
    }
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(by_script, indent=1))


if __name__ == "__main__":
    main()

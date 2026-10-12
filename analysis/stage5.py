"""Stage 5 acceptance run: deletion-neighbourhood keys in the candidate index, before and after.

Run (dev first; v4 only if dev passed, and only once):
    python analysis/stage5.py --set dev
    python analysis/stage5.py --set v4

The criteria are the ones written into the project notes before any code changed, and they are coded here as written:
  1. Single-letter adversarial slip rate falls clearly: the pooled rate over vowel_swap, vowel_delete,
     consonant_double and consonant_delete at least halves (after <= 0.5 x before) and no single-letter family's
     rate rises. Base set: the headline positives the BEFORE engine catches. Same variants for both engines
     (analysis/perturb.py, seed 11, at most 3 per base per family).
  2. Headline F1 does not fall by more than 0.005.
  3. False-alert rate on headline negatives does not rise by more than 0.5 points overall, or 1 point for Arabic script.
  4. Median candidates per query (len of CandidateIndex.query) rise by no more than 2x, and p95 latency (per query,
     Screener.components; median of three runs of the p95, after a warm-up pass) by no more than 2x.
Also reported: index keys, postings, tracemalloc MB, build time; recall by group; retrieval misses; the token-edit and
romanisation families; and, on dev, max_df 250 against 1000 with the new index, under the adoption rule in
the project notes.

"Before" is the shipped engine (rarity-weighted skeleton, max_df 1000, no deletion keys); "after" is the same engine
with deletion_keys=True. Nothing is tuned here. Dev is rebuilt as evaluate_screening.py builds it; v4 is read from its
frozen file after its checksum is checked.
"""

import argparse
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adversarial  # noqa: E402
import common  # noqa: E402
import perturb  # noqa: E402
import tuning_transfer as tt  # noqa: E402

import evaluate_screening as es  # noqa: E402
from candidates import CandidateIndex  # noqa: E402
from entities import load_named_entities  # noqa: E402
from profile_data import DEFAULT_INPUT  # noqa: E402
from screening import CANDIDATE_CAP, ScreeningConfig, ScreeningEngine  # noqa: E402

ev = common.ev
SINGLE_LETTER = ("vowel_swap", "vowel_delete", "consonant_double", "consonant_delete")
TOKEN_EDITS = ("drop_middle_token", "drop_one_token", "reorder_tokens")
ROMANISATION = (
    "translit_cyrillic_gost",
    "translit_cyrillic_german",
    "respell_cyrillic_conventions",
    "respell_arabic_conventions",
)
LIMIT, SEED = adversarial.LIMIT, adversarial.SEED
LATENCY_RUNS = 3

# ---------- criteria, as written in the project notes ----------
SLIP_FACTOR = 0.5
F1_DROP = 0.005
FPR_RISE_OVERALL = 0.005
FPR_RISE_ARABIC = 0.01
CANDIDATE_FACTOR = 2.0
LATENCY_FACTOR = 2.0
MAXDF_F1_GAIN = 0.005
MAXDF_RECALL_FALL = 0.01


def load_dev() -> tuple[list[dict], list[dict], dict, list[dict], str]:
    """(positives, negatives, kept names, entity dicts, label) of the dev split, as evaluate_screening builds it."""
    entities = load_named_entities(DEFAULT_INPUT)
    targets = [e for e in entities if e["target"]]
    all_pos, negs, held = es.build_query_sets(entities, targets)
    pos = [q for q in all_pos if q["split"] == "dev" and q["group"] in ev.HEADLINE_GROUPS]
    neg = [q for q in negs if q["split"] in ("dev", "dev_extra") and q["group"] in ev.HEADLINE_NEG_SCRIPTS]
    return pos, neg, held, targets, "dev split of the existing generator"


def load_v4() -> tuple[list[dict], list[dict], dict, list[dict], str]:
    snapshot = common.read_snapshot(common.CURRENT_SNAPSHOT)
    queries, kept, digest = common.load_holdout(common.ROOT / "holdout" / "screening_holdout_v4.json")
    pos = [q for q in queries if q["kind"] == "positive" and q["group"] in ev.HEADLINE_GROUPS]
    neg = [q for q in queries if q["kind"] != "positive" and q["group"] in ev.HEADLINE_NEG_SCRIPTS]
    return pos, neg, kept, list(snapshot.screened.values()), f"screening_holdout_v4.json sha256 {digest}"


def build_pair(
    entities: list[dict], held: dict, config: ScreeningConfig, max_df: int, deletion_keys: bool
) -> tuple[ScreeningEngine, ScreeningEngine, float]:
    """(engine for positives, engine for negatives, seconds to build both)."""
    started = time.perf_counter()
    pos_entities = [{**e, "names": held.get(e["id"], e["names"])} for e in entities]
    pos = ScreeningEngine.from_entities(pos_entities, config, max_df=max_df, deletion_keys=deletion_keys)
    neg = ScreeningEngine.from_entities(entities, config, max_df=max_df, deletion_keys=deletion_keys)
    return pos, neg, time.perf_counter() - started


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(max(int(fraction * len(ordered) + 0.999999) - 1, 0), len(ordered) - 1)]


def index_stats(entities: list[dict], max_df: int, deletion_keys: bool) -> dict:
    """Keys, postings, build time and tracemalloc memory of the candidate index over the full list."""
    items = [(e["id"], e["names"]) for e in entities]
    started = time.perf_counter()
    plain = CandidateIndex.build(items, max_df=max_df, deletion_keys=deletion_keys)
    seconds = time.perf_counter() - started
    del plain
    tracemalloc.start()
    traced = CandidateIndex.build(items, max_df=max_df, deletion_keys=deletion_keys)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    postings = traced.postings
    over = sum(1 for p in postings.values() if len(p) > max_df)
    return {
        "entities": len(items),
        "keys": len(postings),
        "postings": sum(len(p) for p in postings.values()),
        "keys_over_max_df": over,
        "d_keys": sum(1 for k in postings if k.startswith("d:")),
        "d_postings": sum(len(p) for k, p in postings.items() if k.startswith("d:")),
        "build_seconds_index_only": round(seconds, 2),
        "tracemalloc_mb_current": round(current / 1e6, 1),
        "tracemalloc_mb_peak": round(peak / 1e6, 1),
    }


def measure(
    label: str, queries: list[dict], pos_engine: ScreeningEngine, neg_engine: ScreeningEngine, build_seconds: float
) -> dict:
    """Score every query once, then time Screener.components three more times."""
    rows, candidates = [], []
    for i, q in enumerate(queries):
        engine = pos_engine if q["kind"] == "positive" else neg_engine
        candidates.append(len(engine.screener.index.query(q["name"])))
        got = common.score_query(engine, q)
        rows.append(
            {**{k: q[k] for k in ("kind", "group", "script", "name", "dob", "country", "true_id")}, "index": i, **got}
        )
    p95s, p50s = [], []
    for _ in range(LATENCY_RUNS):
        seconds = []
        for q in queries:
            engine = pos_engine if q["kind"] == "positive" else neg_engine
            started = time.perf_counter()
            engine.screener.components(
                q["name"], q["dob"], q["country"], keep=CANDIDATE_CAP, blends=(engine.config.blend,),
                always=q.get("true_id"),
            )  # fmt: skip
            seconds.append(time.perf_counter() - started)
        p95s.append(percentile(seconds, 0.95))
        p50s.append(percentile(seconds, 0.50))
    m = common.headline_metrics(rows)
    pos = [r for r in rows if r["kind"] == "positive" and r["group"] in ev.HEADLINE_GROUPS]
    by_group = {}
    for group in sorted({r["group"] for r in pos}):
        g = [r for r in pos if r["group"] == group]
        by_group[group] = sum(bool(r["true"]) and r["true"]["score"] >= r["threshold"] for r in g) / len(g)
    out = {
        "label": label,
        **m,
        "per_script": tt.per_script(rows),
        "recall_by_group": by_group,
        "positives_whose_true_entity_was_not_scored": sum(r["true"] is None for r in pos),
        "candidates_per_query": {
            "median": statistics.median(candidates),
            "mean": statistics.fmean(candidates),
            "p95": percentile(candidates, 0.95),
            "max": max(candidates),
            "queries": len(candidates),
        },
        "latency_ms": {
            "p95_runs": [round(x * 1000, 3) for x in p95s],
            "p95_median_of_runs": round(statistics.median(p95s) * 1000, 3),
            "p50_median_of_runs": round(statistics.median(p50s) * 1000, 3),
        },
        "build_seconds_both_engines": round(build_seconds, 2),
    }
    return out


def adversarial_pair(
    pos: list[dict], before: ScreeningEngine, after: ScreeningEngine, names_full: dict[str, list[str]]
) -> dict:
    """The adversarial test on the same variants for both engines. Base set: headline positives `before` catches."""
    families = perturb.FAMILIES
    stats = {
        f: {"bases": 0, "variants": 0, "before": 0, "after": 0, "bases_before": 0, "bases_after": 0} for f in families
    }
    bases = 0
    for n, q in enumerate(pos):
        if not adversarial.hits(common.score_query(before, q)):
            continue
        bases += 1
        for family, make in families.items():
            if not adversarial.applicable(family, q["name"], names_full.get(q["true_id"], [])):
                continue
            variants = make(q["name"], LIMIT, SEED)
            if not variants:
                continue
            s = stats[family]
            s["bases"] += 1
            slip_b = slip_a = False
            for v in variants:
                s["variants"] += 1
                lost_b = not adversarial.hits(common.score_query(before, q, name=v))
                lost_a = not adversarial.hits(common.score_query(after, q, name=v))
                s["before"] += lost_b
                s["after"] += lost_a
                slip_b |= lost_b
                slip_a |= lost_a
            s["bases_before"] += slip_b
            s["bases_after"] += slip_a
        if n % 200 == 0:
            print(f"adversarial {n}/{len(pos)}", file=sys.stderr, flush=True)

    def rate(a: int, b: int) -> float | None:
        return a / b if b else None

    out = {"caught_by_before_engine": bases, "families": {}}
    for family, s in stats.items():
        out["families"][family] = {
            **s,
            "slip_rate_before": rate(s["before"], s["variants"]),
            "slip_rate_after": rate(s["after"], s["variants"]),
        }
    for kind, members in (
        ("single_letter", SINGLE_LETTER),
        ("token_edits", TOKEN_EDITS),
        ("romanisation", ROMANISATION),
    ):
        v = sum(stats[m]["variants"] for m in members)
        b, a = sum(stats[m]["before"] for m in members), sum(stats[m]["after"] for m in members)
        out[kind] = {
            "variants": v,
            "slips_before": b,
            "slips_after": a,
            "rate_before": rate(b, v),
            "rate_after": rate(a, v),
        }
    return out


def verdict(before: dict, after: dict, adv: dict) -> dict:
    """Each criterion as written in the project notes. All must hold."""
    sl = adv["single_letter"]
    fam_rise = []
    for f in SINGLE_LETTER:
        fam = adv["families"][f]
        if fam["variants"] and fam["slip_rate_after"] > fam["slip_rate_before"]:
            fam_rise.append(f)
    c1 = sl["rate_after"] <= SLIP_FACTOR * sl["rate_before"] and not fam_rise
    c2 = after["f1"] >= before["f1"] - F1_DROP
    fpr_all = after["false_positive_rate"] - before["false_positive_rate"]
    fpr_ar = (
        after["per_script"]["Arabic"]["alert_rate_on_clean_names"]
        - before["per_script"]["Arabic"]["alert_rate_on_clean_names"]
    )
    c3 = fpr_all <= FPR_RISE_OVERALL and fpr_ar <= FPR_RISE_ARABIC
    cand = after["candidates_per_query"]["median"] / before["candidates_per_query"]["median"]
    lat = after["latency_ms"]["p95_median_of_runs"] / before["latency_ms"]["p95_median_of_runs"]
    c4 = cand <= CANDIDATE_FACTOR and lat <= LATENCY_FACTOR
    return {
        "1_single_letter_slip_rate": {
            "pass": c1, "before": sl["rate_before"], "after": sl["rate_after"],
            "ratio": sl["rate_after"] / sl["rate_before"] if sl["rate_before"] else None,
            "required_ratio_at_most": SLIP_FACTOR, "families_whose_rate_rose": fam_rise,
        },
        "2_f1": {"pass": c2, "before": before["f1"], "after": after["f1"], "change": after["f1"] - before["f1"],
                 "must_not_fall_by_more_than": F1_DROP},
        "3_false_alert_rate": {
            "pass": c3, "overall_before": before["false_positive_rate"], "overall_after": after["false_positive_rate"],
            "overall_rise_points": 100 * fpr_all, "overall_limit_points": 100 * FPR_RISE_OVERALL,
            "arabic_before": before["per_script"]["Arabic"]["alert_rate_on_clean_names"],
            "arabic_after": after["per_script"]["Arabic"]["alert_rate_on_clean_names"],
            "arabic_rise_points": 100 * fpr_ar, "arabic_limit_points": 100 * FPR_RISE_ARABIC,
        },
        "4_candidates_and_latency": {
            "pass": c4,
            "median_candidates_before": before["candidates_per_query"]["median"],
            "median_candidates_after": after["candidates_per_query"]["median"],
            "candidates_ratio": cand, "candidates_limit": CANDIDATE_FACTOR,
            "p95_ms_before": before["latency_ms"]["p95_median_of_runs"],
            "p95_ms_after": after["latency_ms"]["p95_median_of_runs"],
            "latency_ratio": lat, "latency_limit": LATENCY_FACTOR,
        },
        "all_pass": bool(c1 and c2 and c3 and c4),
    }  # fmt: skip


def max_df_rule(base: dict, other: dict) -> dict:
    """The adoption rule for max_df 250 over 1000 (both with the new index), on dev."""
    scripts_rise = [
        s for s in base["per_script"]
        if other["per_script"][s]["alert_rate_on_clean_names"] > base["per_script"][s]["alert_rate_on_clean_names"]
    ]  # fmt: skip
    gain = other["f1"] - base["f1"]
    recall_change = other["recall"] - base["recall"]
    ok = gain >= MAXDF_F1_GAIN and not scripts_rise and recall_change >= -MAXDF_RECALL_FALL
    return {
        "adopt": bool(ok),
        "f1_change": gain,
        "f1_gain_required": MAXDF_F1_GAIN,
        "scripts_whose_false_alert_rate_rose": scripts_rise,
        "recall_change": recall_change,
        "recall_fall_allowed": MAXDF_RECALL_FALL,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=("dev", "v4"), required=True)
    args = parser.parse_args()
    out_path = common.OUT / f"stage5_{args.set}.json"
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists; it is never overwritten")
    if args.set == "v4":
        dev = out_path.with_name("stage5_dev.json")
        if not dev.exists() or not common.json.loads(dev.read_text())["verdict"]["all_pass"]:  # type: ignore[attr-defined]
            sys.exit("v4 is scored only after dev has been run and passed every criterion. Not scoring it.")
    pos, neg, held, entities, label = load_dev() if args.set == "dev" else load_v4()
    queries = pos + neg
    config = ScreeningConfig.load()
    names_full = {e["id"]: e["names"] for e in entities}
    print(f"{args.set}: {len(pos)} headline positives, {len(neg)} headline negatives ({label})", file=sys.stderr)

    engines: dict[str, tuple[ScreeningEngine, ScreeningEngine, float]] = {
        "before": build_pair(entities, held, config, 1000, False),
        "after": build_pair(entities, held, config, 1000, True),
    }
    if args.set == "dev":
        engines["after_max_df_250"] = build_pair(entities, held, config, 250, True)
    measured = {}
    for name, (pe, ne, secs) in engines.items():
        measured[name] = measure(name, queries, pe, ne, secs)
        print(name, round(measured[name]["f1"], 4), measured[name]["candidates_per_query"], file=sys.stderr, flush=True)
    adv = adversarial_pair(pos, engines["before"][0], engines["after"][0], names_full)
    result = {
        "set": args.set,
        "description": label,
        "snapshot": common.read_snapshot(common.CURRENT_SNAPSHOT).version,
        "index": {
            "before": index_stats(entities, 1000, False),
            "after": index_stats(entities, 1000, True),
            **({"after_max_df_250": index_stats(entities, 250, True)} if args.set == "dev" else {}),
        },
        "measured": measured,
        "adversarial": adv,
        "verdict": verdict(measured["before"], measured["after"], adv),
    }
    if args.set == "dev":
        result["max_df_250_vs_1000_with_the_new_index"] = max_df_rule(measured["after"], measured["after_max_df_250"])
    common.write_json(out_path, result)
    print(common.json.dumps(result["verdict"], indent=1))  # type: ignore[attr-defined]
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

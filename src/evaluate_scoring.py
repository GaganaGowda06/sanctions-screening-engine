"""Evaluate scoring on real data: precision, recall, top-1, false positives and alert rate.

The list screened against is the named target entities. Queries are built so the engine cannot
simply find its own input:

Positives (the query is a real alias of a listed entity, and that name is removed from the list):
  variant        a Latin spelling of an entity that still has another Latin spelling
  cyr_to_latin   Cyrillic query, only the entity's purely Latin names remain
  latin_to_cyr   Latin query, only the purely Cyrillic names remain
  ar_to_latin, latin_to_ar   the same for Arabic
  other_to_latin Han, kana and other scripts against Latin. Reported, but left out of the headline
                 numbers: transliteration cannot reach these (known limitation).
Negatives (the query is not a listed person, and any alert is a false positive):
  synthetic      a first-name token and a last-name token taken from different real names, in Latin,
                 Cyrillic or Arabic, matching no entity name exactly, with random birth date and country
  near_miss      a real listed name with a birth date and country that conflict with that entity's

Positive queries carry the entity's own birth date and country, each dropped with probability 0.3.
Dev and test samples are disjoint. The alert threshold, skeleton floor and lift, and the blend of
character ratio and rarity-weighted token score are tuned on dev for maximum F1, and everything is
reported on test. Birth date and country magnitudes are fixed, not tuned. The length penalty is off.

A third, final hold-out set is generated once into holdout/final_holdout.json, disjoint from dev and
test, and never scored here. It is for the final re-audit only.

Run:
    python src/evaluate_scoring.py
Writes reports/stage2_scoring.json.
"""

import argparse
import hashlib
import itertools
import json
import platform
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from evaluate_candidates import load_named_entities, percentile, script_class
from normalise import normalise
from profile_data import DEFAULT_INPUT
from scoring import Adjustments, Screener, combine

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "stage2_scoring.json"
FINAL_HOLDOUT = ROOT / "holdout" / "final_holdout.json"
SEED = 11
MISSING_P = 0.3
SKELETON_FLOORS = (0.2, 0.3, 0.4, 0.5, 0.6)
SKELETON_LIFTS = (0.1, 0.2, 0.3, 0.4, 0.5)
BLENDS = (0.0, 0.5, 1.0)  # 0 is the character ratio alone, 1 the rarity-weighted token score alone


class Params(NamedTuple):
    skeleton_floor: float
    skeleton_lift: float
    blend_index: int

    @property
    def blend(self) -> float:
        return BLENDS[self.blend_index]


GRID = [
    Params(f, lift, i) for f, lift, i in itertools.product(SKELETON_FLOORS, SKELETON_LIFTS, range(len(BLENDS)))
]
THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50 to 0.99
# Positive groups: (dev count, test count).
POSITIVE_GROUPS = {
    "variant": (400, 800),
    "cyr_to_latin": (150, 300),
    "latin_to_cyr": (150, 300),
    "ar_to_latin": (150, 300),
    "latin_to_ar": (150, 300),
    "other_to_latin": (100, 200),
}
HEADLINE_GROUPS = {g for g in POSITIVE_GROUPS if g != "other_to_latin"}
SYNTHETIC = {
    "dev": {"Latin": 400, "Cyrillic": 175, "Arabic": 175},
    "test": {"Latin": 800, "Cyrillic": 350, "Arabic": 350},
    "final": {"Latin": 800, "Cyrillic": 350, "Arabic": 350},
}
NEAR_MISS = {"dev": 250, "test": 500, "final": 500}
MODES = {
    "neither": (False, False),
    "birth_date_only": (True, False),
    "country_only": (False, True),
    "both": (True, True),
}
GROUP_SPEC = {  # group -> (query script class, script class kept in the list, direction for held_out)
    "cyr_to_latin": ("Cyrillic", "Latin"),
    "latin_to_cyr": ("Latin", "Cyrillic"),
    "ar_to_latin": ("Arabic", "Latin"),
    "latin_to_ar": ("Latin", "Arabic"),
    "other_to_latin": ("other", "Latin"),
}


# ---------- building the query sets ----------


def pick(rng: random.Random, values: list[str]) -> str | None:
    return rng.choice(values) if values else None


def keep_or_drop(rng: random.Random, value: str | None) -> str | None:
    """Missing attributes: drop with probability MISSING_P (always draws, so the stream is stable)."""
    return None if rng.random() < MISSING_P else value


def positive_candidates(entities: list[dict], group: str) -> list[tuple[dict, str, list[str]]]:
    """(entity, query name, names kept in the list) for every entity eligible for the group, query chosen later."""
    out = []
    for e in entities:
        names = e["names"]
        if group == "variant":
            latin = [n for n in names if script_class(n) == "Latin"]
            if len({normalise(n) for n in latin}) >= 2:
                out.append((e, latin, names))
            continue
        query_cls, kept_cls = GROUP_SPEC[group]
        pool = [n for n in names if script_class(n) == query_cls]
        kept = [n for n in names if script_class(n) == kept_cls]
        if pool and kept:
            out.append((e, pool, kept))
    return out


def build_positives(
    targets: list[dict],
    rng: random.Random,
    plan: dict[str, list[tuple[str, int]]] | None = None,
    exclude: set[str] | None = None,
    key: str = "{seed}-{group}",
) -> tuple[list[dict], dict[str, list[str]]]:
    """Positive queries, plus the names each sampled entity keeps in the list.

    `plan` gives, per group, the splits to fill in order with their sizes (default dev then test).
    Entities in `exclude` are not used, and `key` sets the shuffle, so another set can be carved
    out later without disturbing this one.
    """
    plan = plan or {g: [("dev", n_dev), ("test", n_test)] for g, (n_dev, n_test) in POSITIVE_GROUPS.items()}
    queries, kept_names, used = [], {}, set(exclude or ())
    for group, splits in plan.items():
        eligible = [c for c in positive_candidates(targets, group) if c[0]["id"] not in used]
        random.Random(key.format(seed=SEED, group=group)).shuffle(eligible)
        labels = [name for name, count in splits for _ in range(count)]
        for split, (entity, pool, kept) in zip(labels, eligible, strict=False):
            query = rng.choice(pool)
            if group == "variant":
                same = normalise(query)
                kept = [n for n in entity["names"] if normalise(n) != same]
            used.add(entity["id"])
            kept_names[entity["id"]] = kept
            queries.append(
                {
                    "split": split,
                    "kind": "positive",
                    "group": group,
                    "true_id": entity["id"],
                    "name": query,
                    "dob": keep_or_drop(rng, pick(rng, entity["birth_dates"])),
                    "country": keep_or_drop(rng, pick(rng, entity["countries"])),
                }
            )
    return queries, kept_names


def conflicting_birth_date(rng: random.Random, entity_dates: list[str]) -> str | None:
    years = [int(d[:4]) for d in entity_dates if d[:4].isdigit()]
    if not years:
        return None
    for _ in range(50):
        year = rng.choice(years) + rng.choice((-1, 1)) * rng.randint(8, 25)
        if all(abs(year - y) >= 2 for y in years):
            return f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    return None


def build_near_misses(
    targets: list[dict],
    used: set[str],
    country_pool: list[str],
    rng: random.Random,
    splits: tuple[str, ...] = ("dev", "test"),
    key: str = "near",
) -> list[dict]:
    """A real listed name, with a birth date and country that both conflict with the entity's."""
    eligible = [
        e
        for e in targets
        if e["schema"] == "Person" and e["id"] not in used and e["birth_dates"] and e["countries"]
        and any(script_class(n) in ("Latin", "Cyrillic", "Arabic") for n in e["names"])
    ]
    random.Random(f"{SEED}-{key}").shuffle(eligible)
    out = []
    for e in eligible:
        dob = conflicting_birth_date(rng, e["birth_dates"])
        own = {c.lower() for c in e["countries"]}
        country = next((c for c in (rng.choice(country_pool) for _ in range(50)) if c.lower() not in own), None)
        if dob is None or country is None:
            continue
        name = rng.choice([n for n in e["names"] if script_class(n) in ("Latin", "Cyrillic", "Arabic")])
        out.append({"kind": "near_miss", "group": script_class(name), "true_id": None, "name": name,
                    "dob": dob, "country": country, "entity": e["id"]})  # fmt: skip
        if len(out) >= sum(NEAR_MISS[x] for x in splits):
            break
    used.update(q["entity"] for q in out)
    labels = [name for name in splits for _ in range(NEAR_MISS[name])]
    for q, label in zip(out, labels, strict=False):
        q["split"] = label
    return out


def build_synthetic(
    entities: list[dict],
    targets: list[dict],
    dob_pool: list[str],
    country_pool: list[str],
    rng: random.Random,
    splits: tuple[str, ...] = ("dev", "test"),
) -> list[dict]:
    """First-name token and last-name token from different real names; matches no entity name exactly."""
    forbidden = set()
    for e in entities:
        for n in e["names"]:
            norm = normalise(n)
            forbidden.add(norm)
            forbidden.add(" ".join(sorted(norm.split())))
    pools: dict[str, tuple[list[str], list[str]]] = {}
    for script in ("Latin", "Cyrillic", "Arabic"):
        firsts, lasts = [], []
        for e in targets:
            if e["schema"] != "Person":
                continue
            for n in e["names"]:
                tokens = [t for t in re.split(r"[\s,]+", n.strip()) if len(t) > 1]
                if script_class(n) == script and 2 <= len(tokens) <= 3:
                    firsts.append(tokens[0])
                    lasts.append(tokens[-1])
        pools[script] = (firsts, lasts)
    out = []
    for split in splits:
        for script, count in SYNTHETIC[split].items():
            firsts, lasts = pools[script]
            made = 0
            while made < count:
                name = f"{rng.choice(firsts)} {rng.choice(lasts)}"
                norm = normalise(name)
                if not norm or norm in forbidden or " ".join(sorted(norm.split())) in forbidden:
                    continue
                out.append({"split": split, "kind": "synthetic", "group": script, "true_id": None, "name": name,
                            "dob": keep_or_drop(rng, rng.choice(dob_pool)),
                            "country": keep_or_drop(rng, rng.choice(country_pool))})  # fmt: skip
                made += 1
    return out


# ---------- running the engine ----------


def run_queries(screener: Screener, queries: list[dict]) -> list[float]:
    """Attach score components to each query. Returns per-query seconds."""
    latencies = []
    for q in queries:
        start = time.perf_counter()
        q["comps"] = screener.components(q["name"], q["dob"], q["country"], always=q["true_id"], blends=BLENDS)
        latencies.append(time.perf_counter() - start)
    return latencies


# ---------- metrics ----------


def final_scores(
    queries: list[dict], params: Params, use_dob: bool, use_country: bool, arabic_full: bool = True
) -> list[tuple]:
    """Per query: (group or kind, true entity score or None, best other score, true entity is top-1)."""
    rows = []
    for q in queries:
        s_true, s_other = None, 0.0
        for c in q["comps"]:
            score = combine(
                c, params.skeleton_floor, params.skeleton_lift, params.blend_index, use_dob, use_country, arabic_full
            )
            if c.entity_id == q["true_id"]:
                s_true = score
            elif score > s_other:
                s_other = score
        top1 = s_true is not None and s_true > 0 and s_true >= s_other
        rows.append((q["group"] if q["kind"] == "positive" else q["kind"], q["group"], s_true, s_other, top1))
    return rows


def metrics_at(pos: list[tuple], neg: list[tuple], threshold: float) -> dict:
    """Headline metrics. pos rows are from positives, neg rows from negatives (see final_scores)."""
    head = [p for p in pos if p[0] in HEADLINE_GROUPS]
    alerted_true = [p[2] is not None and p[2] >= threshold for p in head]
    wrong_only = [(not t) and p[3] >= threshold for t, p in zip(alerted_true, head, strict=True)]
    neg_alert = [n[3] >= threshold for n in neg]  # for negatives s_other holds the best score
    tp, fn_wrong, fp_neg = sum(alerted_true), sum(wrong_only), sum(neg_alert)
    fp = fp_neg + fn_wrong
    recall = tp / len(head) if head else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    kinds = Counter(n[0] for n in neg)
    by_kind = {k: sum(a for a, n in zip(neg_alert, neg, strict=True) if n[0] == k) / kinds[k] for k in kinds}
    synthetic_by_script = {}
    for script in ("Latin", "Cyrillic", "Arabic"):
        rows = [a for a, n in zip(neg_alert, neg, strict=True) if n[0] == "synthetic" and n[1] == script]
        synthetic_by_script[script] = round(sum(rows) / len(rows), 4) if rows else None
    group_recall = {}
    for group in POSITIVE_GROUPS:
        rows = [p[2] is not None and p[2] >= threshold for p in pos if p[0] == group]
        group_recall[group] = round(sum(rows) / len(rows), 4) if rows else None
    any_alert = sum(a or p[3] >= threshold for a, p in zip(alerted_true, head, strict=True)) + fp_neg
    return {
        "threshold": threshold,
        "queries_positive": len(head),
        "queries_negative": len(neg),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "top1_accuracy": round(sum(p[4] for p in head) / len(head), 4) if head else 0.0,
        "false_positive_rate": round(fp_neg / len(neg), 4) if neg else 0.0,
        "fpr_synthetic": round(by_kind.get("synthetic", 0.0), 4),
        "fpr_near_miss": round(by_kind.get("near_miss", 0.0), 4),
        "fpr_synthetic_by_script": synthetic_by_script,
        "alert_rate_overall": round(any_alert / (len(head) + len(neg)), 4),
        "recall_by_group": group_recall,
    }


def tune(
    dev_pos: list[dict],
    dev_neg: list[dict],
    use_dob: bool,
    use_country: bool,
    grid: list[Params] | None = None,
    arabic_full: bool = True,
) -> tuple[Params, float, float, dict[str, dict[str, float]]]:
    """(params, threshold, F1, best F1 per value of each parameter) maximising F1 on dev.

    Ties go to the higher threshold.
    """
    grid = grid or GRID
    best: tuple[Params, float, float] = (grid[0], 0.0, -1.0)
    best_f1 = {}
    for params in grid:
        pos = final_scores(dev_pos, params, use_dob, use_country, arabic_full)
        neg = final_scores(dev_neg, params, use_dob, use_country, arabic_full)
        top = -1.0
        for t in THRESHOLDS:
            f1 = metrics_at(pos, neg, t)["f1"]
            top = max(top, f1)
            if f1 >= best[2]:
                best = (params, t, f1)
        best_f1[params] = top
    marginals: dict[str, dict[str, float]] = {"skeleton_floor": {}, "skeleton_lift": {}, "blend": {}}
    for params, f1 in best_f1.items():
        for name, value in (
            ("skeleton_floor", params.skeleton_floor),
            ("skeleton_lift", params.skeleton_lift),
            ("blend", params.blend),
        ):
            marginals[name][str(value)] = max(marginals[name].get(str(value), -1.0), f1)
    return (*best, marginals)


def threshold_for_fpr(dev_pos: list[dict], dev_neg: list[dict], params: Params, target: float) -> float:
    pos = final_scores(dev_pos, params, True, True)
    neg = final_scores(dev_neg, params, True, True)
    for t in THRESHOLDS:
        if metrics_at(pos, neg, t)["false_positive_rate"] <= target:
            return t
    return THRESHOLDS[-1]


def evaluate_at(
    test_pos: list[dict],
    test_neg: list[dict],
    params: Params,
    threshold: float,
    mode: str,
    arabic_full: bool = True,
) -> dict:
    use_dob, use_country = MODES[mode]
    pos = final_scores(test_pos, params, use_dob, use_country, arabic_full)
    neg = final_scores(test_neg, params, use_dob, use_country, arabic_full)
    return metrics_at(pos, neg, threshold)


def write_final_holdout(path: Path, queries: list[dict], kept_names: dict[str, list[str]]) -> dict:
    """Freeze the final hold-out query set. An existing file is never overwritten."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{k: v for k, v in q.items() if k != "comps"} for q in queries]
        payload = {"seed": SEED, "kept_names": kept_names, "queries": rows}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    data = json.loads(path.read_text())
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "queries": dict(Counter(f"{q['kind']}/{q['group']}" for q in data["queries"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-df", type=int, default=1000)
    args = parser.parse_args()
    print(f"Python {platform.python_version()}", file=sys.stderr)
    rng = random.Random(SEED)

    entities = load_named_entities(args.input)
    targets = [e for e in entities if e["target"]]
    persons = [e for e in targets if e["schema"] == "Person"]
    dob_pool = [d for e in persons for d in e["birth_dates"][:1]]
    country_pool = [c for e in persons for c in e["countries"][:1]]

    positives, kept_names = build_positives(targets, rng)
    used = {q["true_id"] for q in positives}
    near = build_near_misses(targets, used, country_pool, rng)  # adds its entities to `used`
    synthetic = build_synthetic(entities, targets, dob_pool, country_pool, rng)
    print(f"queries: {len(positives)} positive, {len(synthetic)} synthetic, {len(near)} near-miss", file=sys.stderr)

    # Final hold-out: its own random streams, built after dev and test, entities disjoint from both.
    final_rng = random.Random(f"{SEED}-final")
    final_plan = {g: [("final", n_test)] for g, (_, n_test) in POSITIVE_GROUPS.items()}
    final_pos, final_kept = build_positives(targets, final_rng, final_plan, exclude=used, key="{seed}-final-{group}")
    used_final = used | {q["true_id"] for q in final_pos}
    final_near = build_near_misses(targets, used_final, country_pool, final_rng, ("final",), "near-final")
    final_syn = build_synthetic(entities, targets, dob_pool, country_pool, final_rng, ("final",))
    holdout = write_final_holdout(FINAL_HOLDOUT, final_pos + final_near + final_syn, final_kept)
    print(f"final hold-out frozen: {holdout['sha256'][:12]}", file=sys.stderr)

    adjustments = Adjustments()
    t0 = time.perf_counter()
    held_list = [{**e, "names": kept_names.get(e["id"], e["names"])} for e in targets]
    screener = Screener.build(held_list, max_df=args.max_df, adjustments=adjustments)
    build_seconds = time.perf_counter() - t0
    latencies = run_queries(screener, positives)
    del screener, held_list
    full = Screener.build(targets, max_df=args.max_df, adjustments=adjustments)
    negatives = synthetic + near
    latencies += run_queries(full, negatives)
    del full

    def split(rows: list[dict], name: str) -> list[dict]:
        return [q for q in rows if q["split"] == name]

    dev_pos, test_pos = split(positives, "dev"), split(positives, "test")
    dev_neg, test_neg = split(negatives, "dev"), split(negatives, "test")

    tuned = {}
    for mode, (use_dob, use_country) in MODES.items():
        params, threshold, dev_f1, marginals = tune(dev_pos, dev_neg, use_dob, use_country)
        tuned[mode] = {
            "skeleton_floor": params.skeleton_floor,
            "skeleton_lift": params.skeleton_lift,
            "blend": params.blend,
            "threshold": threshold,
            "dev_f1": dev_f1,
            "dev_best_f1_by_parameter_value": marginals,
            "test": evaluate_at(test_pos, test_neg, params, threshold, mode),
        }
        tuned[mode]["_params"] = params
    both_params, both_t = tuned["both"]["_params"], tuned["both"]["threshold"]
    for mode in MODES:  # same operating point as "both", to isolate what the attributes do
        tuned[mode]["test_at_both_operating_point"] = evaluate_at(test_pos, test_neg, both_params, both_t, mode)
        del tuned[mode]["_params"]
    low_fpr_t = threshold_for_fpr(dev_pos, dev_neg, both_params, 0.01)

    # Each new piece switched off in turn, everything else re-tuned on dev, reported on test.
    ablations = {}
    for label, grid, arabic_full in (
        ("full", GRID, True),
        ("without_rarity_weighting", [p for p in GRID if p.blend_index == 0], True),
        ("without_arabic_full_lift", GRID, False),
        ("step_3b_scoring", [p for p in GRID if p.blend_index == 0], False),
    ):
        params, threshold, dev_f1, _ = tune(dev_pos, dev_neg, True, True, grid, arabic_full)
        test = evaluate_at(test_pos, test_neg, params, threshold, "both", arabic_full)
        ablations[label] = {
            "skeleton_floor": params.skeleton_floor,
            "skeleton_lift": params.skeleton_lift,
            "blend": params.blend,
            "threshold": threshold,
            "dev_f1": dev_f1,
            **{k: test[k] for k in ("recall", "precision", "f1", "top1_accuracy", "false_positive_rate")},
            "fpr_synthetic": test["fpr_synthetic"],
            "fpr_near_miss": test["fpr_near_miss"],
            "recall_by_group": test["recall_by_group"],
            "fpr_synthetic_by_script": test["fpr_synthetic_by_script"],
        }
    sweep = {
        str(t): {k: m[k] for k in ("recall", "precision", "false_positive_rate", "alert_rate_overall", "top1_accuracy")}
        for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
        for m in [evaluate_at(test_pos, test_neg, both_params, t, "both")]
    }
    report = {
        "python_version": platform.python_version(),
        "list_size_named_targets": len(targets),
        "adjustments": vars(adjustments),
        "queries": {
            "positive": dict(Counter(f"{q['split']}/{q['group']}" for q in positives)),
            "negative": dict(Counter(f"{q['split']}/{q['kind']}/{q['group']}" for q in negatives)),
        },
        "tuned_on_dev": tuned,
        "fpr_at_most_1_percent_operating_point": {
            "threshold_from_dev": low_fpr_t,
            "skeleton_floor": both_params.skeleton_floor,
            "skeleton_lift": both_params.skeleton_lift,
            "blend": both_params.blend,
            "test": evaluate_at(test_pos, test_neg, both_params, low_fpr_t, "both"),
        },
        "ablations_both": ablations,
        "final_holdout_frozen_not_scored": holdout,
        "test_threshold_sweep_both": sweep,
        "timing": {
            "screener_build_seconds": round(build_seconds, 2),
            "screen_ms_median": round(percentile(latencies, 0.5) * 1000, 2),
            "screen_ms_p95": round(percentile(latencies, 0.95) * 1000, 2),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

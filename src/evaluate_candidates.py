"""Measure candidate generation on the real data.

Cross-script recall: take entities that have a purely Latin name and a name in another script,
query with one, and check the true entity is among the candidates. For each sampled entity the
names on the query's side are removed from the index, so the entity can only be found through
the other script. Otherwise the query would match itself.

  to_latin    query with a Cyrillic / Arabic / other-script name, index keeps only the entity's
              purely Latin names
  from_latin  query with a Latin name, index keeps only the entity's names in that one script

max_df is chosen on a dev sample and reported on a separate test sample.

Run:
    python src/evaluate_candidates.py
    python src/evaluate_candidates.py --max-df 500   # try another threshold
Writes reports/stage2_candidates.json.
"""

import argparse
import json
import platform
import random
import resource
import sys
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from candidates import CandidateIndex
from normalise import is_clean, normalise
from profile_data import DEFAULT_INPUT, NAME_PROPS, name_scripts

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "stage2_candidates.json"
SEED = 7
# Chosen from the dev sweep: Arabic any-candidate recall rises from 0.75 (300) to 0.91 (1000) and then
# only to 0.94 for sets three times larger; Cyrillic is flat beyond this point.
DEFAULT_MAX_DF = 1000
DEV_SIZE, TEST_SIZE = 1000, 2000
SWEEP = (100, 300, 1000, 3000, 10000)
SCRIPT_CLASSES = ("Cyrillic", "Arabic", "other")
CHEAP_SCRIPTS = {"Greek", "Georgian", "Armenian", "Hangul", "Hebrew"}

_scripts_cache: dict[str, frozenset[str]] = {}


def scripts_of(name: str) -> frozenset[str]:
    if name not in _scripts_cache:
        _scripts_cache[name] = frozenset(name_scripts(name))
    return _scripts_cache[name]


def script_class(name: str) -> str | None:
    """'Latin', 'Cyrillic', 'Arabic', 'other' (non-Latin, no Latin letters), or None for mixed or empty."""
    scripts = scripts_of(name)
    if scripts == {"Latin"}:
        return "Latin"
    if not scripts or "Latin" in scripts:
        return None
    if scripts == {"Cyrillic"}:
        return "Cyrillic"
    if scripts == {"Arabic"}:
        return "Arabic"
    return "other"


def held_out(name: str, cls: str, direction: str) -> bool:
    """True if a sampled entity's name must be removed from the index (the index keeps only the opposite side)."""
    keep = cls if direction == "from_latin" else "Latin"
    return script_class(name) != keep


def load_named_entities(path: Path) -> list[dict]:
    entities = []
    with open(path, "rb") as fh:
        for line in fh:
            try:
                ent = json.loads(line)
                props = ent.get("properties") or {}
                names = list(
                    dict.fromkeys(
                        v.strip() for p in NAME_PROPS for v in props.get(p) or () if isinstance(v, str) and v.strip()
                    )
                )
            except (ValueError, AttributeError):
                continue
            if names:
                entities.append(
                    {"id": ent["id"], "schema": ent.get("schema"), "target": ent.get("target") is True, "names": names}
                )
    return entities


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))] if ordered else 0.0


def unresolvable_entities(entities: list[dict]) -> dict:
    """Entities with names, none of which normalises cleanly."""
    result = {"named_entities": len(entities), "targets_named": sum(e["target"] for e in entities)}
    stuck = [e for e in entities if not any(is_clean(normalise(n)) for n in e["names"])]
    stuck_targets = [e for e in stuck if e["target"]]
    by_script: Counter = Counter()
    cheap = 0
    for e in stuck:
        scripts = set().union(*(scripts_of(n) for n in e["names"])) - {"Latin"}
        by_script.update(scripts)
        cheap += bool(scripts) and scripts <= CHEAP_SCRIPTS
    result.update(
        {
            "no_clean_name": len(stuck),
            "no_clean_name_share": round(len(stuck) / len(entities), 4),
            "no_clean_name_targets": len(stuck_targets),
            "no_clean_name_targets_share": round(len(stuck_targets) / result["targets_named"], 4),
            "by_schema": dict(Counter(e["schema"] for e in stuck).most_common()),
            "by_script_entity_counts": dict(by_script.most_common()),
            "only_cheap_to_add_scripts_greek_georgian_armenian_hangul_hebrew": cheap,
        }
    )
    return result


def build_condition(entities: list[dict], by_id: dict[str, dict], cls: str, direction: str, seed: int) -> dict | None:
    """Pick sample entities for a condition and decide which of their names are held out."""
    side = "Latin" if direction == "from_latin" else cls
    eligible = [
        e
        for e in entities
        if any(script_class(n) == "Latin" for n in e["names"]) and any(script_class(n) == cls for n in e["names"])
    ]
    rng = random.Random(f"{seed}-{cls}-{direction}")
    rng.shuffle(eligible)
    dev, test = eligible[:DEV_SIZE], eligible[DEV_SIZE : DEV_SIZE + TEST_SIZE]
    queries = {}
    for e in dev + test:
        pool = [n for n in e["names"] if script_class(n) == side]
        queries[e["id"]] = rng.choice(pool)
    return {"eligible": len(eligible), "dev": dev, "test": test, "queries": queries, "cls": cls, "direction": direction}


def index_entities(entities: list[dict], condition: dict | None) -> Iterator[tuple[str, list[str]]]:
    """(id, names) pairs; sampled entities lose the names on the query's side."""
    sampled = {e["id"] for e in condition["dev"] + condition["test"]} if condition else set()
    for e in entities:
        names = e["names"]
        if e["id"] in sampled:
            names = [n for n in names if not held_out(n, condition["cls"], condition["direction"])]
        yield e["id"], names


def evaluate(index: CandidateIndex, sample: list[dict], queries: dict[str, str], examples: int = 0) -> dict:
    """Candidate-set recall: is the true entity anywhere in the set a query returns."""
    hits: list[bool] = []
    sizes: list[int] = []
    latencies: list[float] = []
    persons: list[bool] = []
    misses = []
    for e in sample:
        query = queries[e["id"]]
        start = time.perf_counter()
        candidates = index.query(query)
        latencies.append(time.perf_counter() - start)
        hit = any(cid == e["id"] for cid, _ in candidates)
        hits.append(hit)
        sizes.append(len(candidates))
        persons.append(e["schema"] == "Person")
        if not hit and len(misses) < examples:
            misses.append({"query": query, "normalised": normalise(query), "entity_names": e["names"][:4]})

    def recall(flags: list[bool]) -> float | None:
        return round(sum(flags) / len(flags), 4) if flags else None

    return {
        "queries": len(sample),
        "candidate_recall": recall(hits),
        "candidate_recall_person": recall([h for h, p in zip(hits, persons, strict=True) if p]),
        "candidate_recall_non_person": recall([h for h, p in zip(hits, persons, strict=True) if not p]),
        "queries_person": sum(persons),
        "candidate_set_size": {"median": percentile(sizes, 0.5), "p95": percentile(sizes, 0.95)},
        "query_ms": {
            "median": round(percentile(latencies, 0.5) * 1000, 3),
            "p95": round(percentile(latencies, 0.95) * 1000, 3),
        },
        "missed_examples": misses,
    }


def index_stats(index: CandidateIndex) -> dict:
    sizes = sorted(((len(v), k) for k, v in index.postings.items()), reverse=True)
    total = sum(s for s, _ in sizes)
    return {
        "entities": len(index.ids),
        "keys": len(index.postings),
        "postings": total,
        "mean_posting_length": round(total / len(index.postings), 2),
        f"keys_over_max_df_{index.max_df}": sum(s > index.max_df for s, _ in sizes),
        "largest_keys": {k: s for s, k in sizes[:10]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-df", type=int, default=DEFAULT_MAX_DF)
    parser.add_argument("--split", choices=("dev", "test"), default="test", help="sample to report on")
    parser.add_argument("--only", nargs="*", help="run only conditions whose name contains one of these")
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()
    print(f"Python {platform.python_version()}", file=sys.stderr)

    t0 = time.perf_counter()
    entities = load_named_entities(args.input)
    by_id = {e["id"]: e for e in entities}
    t_load = time.perf_counter() - t0

    # Full index: build time without tracing, then memory with tracing.
    t0 = time.perf_counter()
    full_index = CandidateIndex.build(index_entities(entities, None), max_df=args.max_df)
    t_build = time.perf_counter() - t0
    stats = index_stats(full_index)
    del full_index
    tracemalloc.start()
    traced = CandidateIndex.build(index_entities(entities, None), max_df=args.max_df)
    index_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del traced

    conditions: dict[str, dict] = {}
    for cls in SCRIPT_CLASSES:
        for direction in ("to_latin", "from_latin"):
            name = f"{cls}_{direction}"
            if args.only and not any(o in name for o in args.only):
                continue
            cond = build_condition(entities, by_id, cls, direction, SEED)
            sample = cond[args.split]
            index = CandidateIndex.build(index_entities(entities, cond), max_df=args.max_df)
            sweep = {}
            if args.split == "test":
                for max_df in SWEEP:
                    index.max_df = max_df
                    dev = evaluate(index, cond["dev"], cond["queries"])
                    sweep[max_df] = {
                        "candidate_recall": dev["candidate_recall"],
                        "median_set_size": dev["candidate_set_size"]["median"],
                        "p95_set_size": dev["candidate_set_size"]["p95"],
                    }
                index.max_df = args.max_df
            result = evaluate(index, sample, cond["queries"], examples=8)
            result["eligible_entities"] = cond["eligible"]
            if sweep:
                result["dev_sweep_by_max_df"] = sweep
            if direction == "from_latin" and cls in ("Cyrillic", "Arabic") and not args.no_ablation:
                result["ablation"] = {}
                for label, flags in (("folds_only", {"join": False}), ("joins_only", {"do_fold": False})):
                    variant = CandidateIndex.build(index_entities(entities, cond), max_df=args.max_df, **flags)
                    ablated = evaluate(variant, sample, cond["queries"])
                    result["ablation"][label] = {
                        "candidate_recall": ablated["candidate_recall"],
                        "median_set_size": ablated["candidate_set_size"]["median"],
                    }
            conditions[name] = result
            print(
                f"{name}: n={result['queries']} recall={result['candidate_recall']} "
                f"set median={result['candidate_set_size']['median']} p95={result['candidate_set_size']['p95']}",
                file=sys.stderr,
            )

    report = {
        "python_version": platform.python_version(),
        "max_df": args.max_df,
        "split": args.split,
        "unresolvable": unresolvable_entities(entities),
        "full_index": {
            **stats,
            "load_entities_seconds": round(t_load, 2),
            "build_seconds": round(t_build, 2),
            "index_memory_mb_tracemalloc": round(index_bytes / 1e6, 1),
            "build_peak_memory_mb_tracemalloc": round(peak_bytes / 1e6, 1),
            "process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
        },
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

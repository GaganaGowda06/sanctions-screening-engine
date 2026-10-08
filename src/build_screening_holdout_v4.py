"""Build and freeze the fourth screening hold-out, for Stage 5 (typo-tolerant retrieval).

The earlier ones are scored and spent. This
one measures screening: given a customer name, does the engine alert on the right listed entity, and how
often does it alert on innocent names. It is a different question from the Pairs data, which measures
whether two records are the same entity. Both get reported.

Same design as before, built from our own list (the named sanctions targets):
  positives   real aliases of listed entities with that name removed from the list: Latin spelling
              variants and cross-script (Cyrillic and Arabic, both ways), plus Han, kana and other
              scripts against Latin, plus same-script "other" aliases
  negatives   synthetic names made by recombining real first and last name tokens that match no entity,
              in Latin, Cyrillic, Arabic and "other" scripts, and near-miss namesakes: a real listed name
              with a birth date and country that both conflict with the entity's
Entities are disjoint from everything used so far, the second and third hold-outs included (read from their
frozen files and checked against their checksums): the dev and test queries, the same-script "other" queries
added for step 4, and the first final hold-out (rebuilt in memory and checked against its checksum). Nothing is
scored here.

Freezing: written once to holdout/screening_holdout_v4.json with its SHA-256 beside it. An existing file is
never overwritten.

Run:
    python src/build_screening_holdout_v4.py
"""

import argparse
import hashlib
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path

import evaluate_scoring as ev
from entities import load_named_entities
from profile_data import DEFAULT_INPUT

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "holdout" / "screening_holdout_v4.json"
TAG = "holdout4"
EXTRA_GROUPS = {"other_same_script": 200}
OLD_FINAL_SHA = ROOT / "holdout" / "final_holdout.sha256"
EARLIER = (
    ROOT / "holdout" / "screening_holdout_v2.json",
    ROOT / "holdout" / "screening_holdout_v3.json",
)  # each has its checksum beside it


def entities_of(queries: list[dict]) -> set[str]:
    return {q.get("true_id") or q.get("entity") for q in queries if q.get("true_id") or q.get("entity")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=HOLDOUT)
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}", file=sys.stderr)

    entities = load_named_entities(args.input)
    targets = [e for e in entities if e["target"]]
    persons = [e for e in targets if e["schema"] == "Person"]
    dob_pool = [d for e in persons for d in e["birth_dates"][:1]]
    country_pool = [c for e in persons for c in e["countries"][:1]]

    # Everything used so far, rebuilt in memory exactly as the earlier steps built it.
    rng = random.Random(ev.SEED)
    positives, kept, near, synthetic, used = ev.build_dev_test_queries(entities, targets, dob_pool, country_pool, rng)
    old_final, old_kept = ev.build_final_holdout(entities, targets, dob_pool, country_pool, used)
    if hashlib.sha256(ev.holdout_text(old_final, old_kept).encode()).hexdigest() != OLD_FINAL_SHA.read_text().strip():
        sys.exit("The first final hold-out no longer rebuilds to its recorded checksum; refusing to continue.")
    reserved = used | entities_of(old_final)
    for path in EARLIER:
        if hashlib.sha256(path.read_bytes()).hexdigest() != path.with_suffix(".sha256").read_text().strip():
            sys.exit(f"{path.name} does not match its recorded checksum; refusing to continue.")
        reserved |= entities_of(json.loads(path.read_text())["queries"])
    extra_rng = random.Random(f"{ev.SEED}-extra")
    extra_pos, _ = ev.build_positives(
        targets, extra_rng, {"other_same_script": [("dev", 100), ("test", 200)]},
        exclude=reserved, key="{seed}-extra-{group}",
    )  # fmt: skip
    reserved |= entities_of(extra_pos)

    queries, kept_names = ev.build_final_holdout(
        entities, targets, dob_pool, country_pool, reserved, tag=TAG, split=TAG, extra_groups=EXTRA_GROUPS
    )
    mine = entities_of(queries)
    wanted = {**{g: n for g, (_, n) in ev.POSITIVE_GROUPS.items()}, **EXTRA_GROUPS}
    got = Counter(q["group"] for q in queries if q["kind"] == "positive")
    if dict(got) != wanted:
        sys.exit(f"The entity pool ran short: positives per group {dict(got)}, wanted {wanted}. Not freezing.")
    near = sum(q["kind"] == "near_miss" for q in queries)
    if near != ev.NEAR_MISS[TAG]:
        sys.exit(f"Only {near} near-miss queries could be built, wanted {ev.NEAR_MISS[TAG]}. Not freezing.")
    if mine & reserved:
        sys.exit(f"{len(mine & reserved)} hold-out entities were used in an earlier set. Not freezing.")

    text = ev.holdout_text(queries, kept_names, tag=TAG)
    if not args.output.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        args.output.with_suffix(".sha256").write_text(hashlib.sha256(args.output.read_bytes()).hexdigest() + "\n")
        print("Frozen.", file=sys.stderr)
    elif args.output.read_text() != text:
        sys.exit(f"{args.output} exists and differs from what this code builds. It is not overwritten.")
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "entities_disjoint_from_earlier_sets": len(mine),
                "queries": dict(Counter(f"{q['kind']}/{q['group']}" for q in queries)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Adversarial robustness: single realistic edits to known positives, one family at a time, through screen()'s own path.

Run:
    python analysis/adversarial.py --holdout v3 --engine current

Base set: the headline positives that the engine already catches (true entity's score at or over the threshold on the
unperturbed query). A perturbation can only lose a catch the engine had. For each base and each family
(analysis/perturb.py) up to three distinct single-edit variants are generated, with the same birth date and country
as the base query, and screened against the same list (the entity's tested alias held out, as in the hold-out's own
scoring).

Per variant:  robust = the true entity still alerts;  slip-through = it no longer does (the party passes on its
own name, or on another entity's alert). Reported per family, per hold-out group, and for the worst single edits.
Research path only: Screener.components, as screen() uses it. No audit records are written.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import perturb  # noqa: E402

LIMIT = 3
SEED = 11
ARABIC_FAMILY, CYRILLIC_FAMILY = "respell_arabic_conventions", "respell_cyrillic_conventions"


def applicable(family: str, name: str, entity_names: list[str]) -> bool:
    if family.startswith("translit_cyrillic"):
        return perturb.has_cyrillic(name)
    if family == CYRILLIC_FAMILY:
        return not perturb.has_cyrillic(name) and any(perturb.has_cyrillic(n) for n in entity_names)
    if family == ARABIC_FAMILY:
        return not perturb.has_arabic(name) and any(perturb.has_arabic(n) for n in entity_names)
    return True


def hits(scored: dict) -> bool:
    return scored["true"] is not None and scored["true"]["score"] >= scored["threshold"]


def run(holdout_path: Path, engine_kind: str) -> dict:
    snap = common.read_snapshot(common.CURRENT_SNAPSHOT)
    queries, kept, digest = common.load_holdout(holdout_path)
    pos_engine, _ = common.build_engines(engine_kind, snap, kept)
    bases = []
    for q in queries:
        if q["kind"] == "positive" and q["group"] in common.ev.HEADLINE_GROUPS:
            scored = common.score_query(pos_engine, q)
            if hits(scored):
                bases.append((q, scored))
    families = perturb.FAMILIES
    per_family: dict[str, dict] = {
        f: {"bases": 0, "variants": 0, "robust": 0, "slip": 0, "slip_to_wrong_entity": 0, "bases_with_a_slip": 0,
            "score_drop_sum": 0.0, "by_group": defaultdict(lambda: {"variants": 0, "slip": 0})}
        for f in families
    }  # fmt: skip
    slipping_bases: dict[int, set[str]] = defaultdict(set)
    cases = []
    for n, (q, base) in enumerate(bases):
        entity_names = snap.screened[q["true_id"]]["names"]
        for family, make in families.items():
            if not applicable(family, q["name"], entity_names):
                continue
            variants = make(q["name"], LIMIT, SEED)
            if not variants:
                continue
            stats = per_family[family]
            stats["bases"] += 1
            base_slipped = False
            for v in variants:
                got = common.score_query(pos_engine, q, name=v)
                robust = hits(got)
                stats["variants"] += 1
                stats["by_group"][q["group"]]["variants"] += 1
                drop = base["true"]["score"] - (got["true"]["score"] if got["true"] else 0.0)
                stats["score_drop_sum"] += drop
                if robust:
                    stats["robust"] += 1
                    continue
                stats["slip"] += 1
                stats["by_group"][q["group"]]["slip"] += 1
                base_slipped = True
                wrong = got["best_other"] >= got["threshold"]
                stats["slip_to_wrong_entity"] += wrong
                cases.append({
                    "family": family, "group": q["group"], "script": q["script"], "base_query": q["name"],
                    "variant": v, "entity_id": q["true_id"], "schema": snap.screened[q["true_id"]]["schema"],
                    "base_score": base["true"]["score"],
                    "variant_score": None if got["true"] is None else got["true"]["score"],
                    "score_drop": round(drop, 4), "alerted_another_entity": bool(wrong),
                })  # fmt: skip
            if base_slipped:
                stats["bases_with_a_slip"] += 1
                slipping_bases[n].add(family)
        if n % 200 == 0:
            print(f"{n}/{len(bases)} bases", file=sys.stderr, flush=True)

    summary = {}
    for family, s in per_family.items():
        v = s["variants"]
        summary[family] = {
            "bases_tested": s["bases"],
            "variants": v,
            "robust": s["robust"],
            "slip_through": s["slip"],
            "slip_through_rate": s["slip"] / v if v else None,
            "slips_that_alerted_another_entity": s["slip_to_wrong_entity"],
            "bases_with_at_least_one_slip": s["bases_with_a_slip"],
            "base_slip_rate": s["bases_with_a_slip"] / s["bases"] if s["bases"] else None,
            "mean_drop_in_true_entity_score": s["score_drop_sum"] / v if v else None,
            "by_group": {
                g: {**d, "slip_through_rate": d["slip"] / d["variants"]} for g, d in sorted(s["by_group"].items())
            },
        }
    real = [f for f in families if f != "control_uppercase"]
    any_slip = sum(1 for n in slipping_bases if slipping_bases[n] & set(real))
    worst = sorted((c for c in cases if c["family"] != "control_uppercase"),
                   key=lambda c: (-c["base_score"] + (c["variant_score"] or 0), c["base_query"]))  # fmt: skip
    by_rate = sorted(
        ((f, summary[f]["slip_through_rate"]) for f in real if summary[f]["variants"]), key=lambda t: -t[1]
    )
    by_count = sorted(((f, summary[f]["slip_through"]) for f in real), key=lambda t: -t[1])
    return {
        "holdout": holdout_path.name,
        "holdout_sha256": digest,
        "engine": engine_kind,
        "snapshot": snap.version,
        "threshold": common.THRESHOLD,
        "base_positives_caught_unperturbed": len(bases),
        "variants_per_base_per_family_at_most": LIMIT,
        "families": summary,
        "bases_with_a_slip_in_at_least_one_family": any_slip,
        "families_by_slip_through_rate": by_rate,
        "families_by_number_of_slip_throughs": by_count,
        "worst_cases": worst[:60],
        "all_slip_cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--holdout", choices=("v2", "v3"), required=True)
    parser.add_argument("--engine", choices=("stage2", "current"), required=True)
    args = parser.parse_args()
    path = common.HOLDOUT_V2 if args.holdout == "v2" else common.HOLDOUT_V3
    out = common.OUT / f"analysis_adversarial_{args.holdout}_{args.engine}.json"
    if out.exists():
        sys.exit(f"{out.name} already exists; it is never overwritten")
    result = run(path, args.engine)
    common.write_json(out, result)
    keys = (
        "base_positives_caught_unperturbed",
        "bases_with_a_slip_in_at_least_one_family",
        "families_by_slip_through_rate",
        "families_by_number_of_slip_throughs",
    )
    brief = {k: result[k] for k in keys}
    print(common.json.dumps(brief, indent=1))  # type: ignore[attr-defined]
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

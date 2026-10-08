"""A summary of an adversarial run, derived only from the run's own report (nothing is screened again).

Run:
    python analysis/adversarial_summary.py --holdout v3 --engine current

Groups the perturbation families by what kind of edit they are, because "one realistic edit" means very different
things:
  character_edits   one letter changed, dropped or doubled (vowel_swap, vowel_delete, consonant_double,
                    consonant_delete)
  token_edits       a whole token left out or the order changed (drop_middle_token, drop_one_token, reorder_tokens)
  romanisation      the same name spelled under another romanisation scheme or convention
Controls (uppercase, and the BGN scheme the matcher itself uses) are reported apart and should lose nothing.
Positives that fall below the threshold are counted per group, not per variant, and the worst single character edits
are listed: a catch with a score of at least 0.95 that one edit takes under the threshold.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

KINDS = {
    "character_edits": ("vowel_swap", "vowel_delete", "consonant_double", "consonant_delete"),
    "token_edits": ("drop_middle_token", "drop_one_token", "reorder_tokens"),
    "token_edits_without_dropping_a_token": ("reorder_tokens",),
    "romanisation": (
        "translit_cyrillic_gost",
        "translit_cyrillic_german",
        "respell_cyrillic_conventions",
        "respell_arabic_conventions",
    ),
    "controls": ("control_uppercase", "translit_cyrillic_bgn"),
}


def summarise(report: dict) -> dict:
    cases = report["all_slip_cases"]
    bases = report["base_positives_caught_unperturbed"]
    fam = report["families"]
    out: dict = {"base_positives_caught_unperturbed": bases}
    for kind, members in KINDS.items():
        slipped = {(c["entity_id"], c["base_query"]) for c in cases if c["family"] in members}
        variants = sum(fam[m]["variants"] for m in members)
        slips = sum(fam[m]["slip_through"] for m in members)
        out[kind] = {
            "families": list(members),
            "variants": variants,
            "slip_throughs": slips,
            "slip_through_rate": slips / variants if variants else None,
            "positives_with_at_least_one_slip": len(slipped),
            "share_of_caught_positives": len(slipped) / bases,
        }
    by_group: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for family, s in fam.items():
        for kind, members in KINDS.items():
            if family in members:
                for group, d in s["by_group"].items():
                    by_group[kind][group][0] += d["variants"]
                    by_group[kind][group][1] += d["slip"]
    out["slip_through_rate_by_kind_and_hold_out_group"] = {
        kind: {g: {"variants": v, "slip_throughs": s, "rate": s / v} for g, (v, s) in sorted(groups.items())}
        for kind, groups in by_group.items()
    }
    char = [c for c in cases if c["family"] in KINDS["character_edits"]]
    near_certain = [c for c in char if c["base_score"] >= 0.95]
    unseen = [c for c in char if c["variant_score"] is None]
    out["character_edit_slips"] = {
        "slips": len(char),
        "slips_from_a_catch_with_a_score_of_0.95_or_more": len(near_certain),
        "slips_where_the_true_entity_was_not_scored_at_all": len(unseen),
        "slips_that_alerted_another_entity_instead": sum(c["alerted_another_entity"] for c in char),
    }
    worst = sorted(
        near_certain, key=lambda c: (c["variant_score"] is not None, c["variant_score"] or 0, c["base_query"])
    )
    out["worst_character_edits"] = worst[:40]
    out["slips_by_schema_character_edits"] = {
        schema: sum(1 for c in char if c["schema"] == schema) for schema in sorted({c["schema"] for c in char})
    }
    short = [c for c in char if len(c["base_query"].split()) == 1]
    out["character_edit_slips_on_one_word_names"] = len(short)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--holdout", choices=("v2", "v3"), required=True)
    parser.add_argument("--engine", choices=("stage2", "current"), required=True)
    args = parser.parse_args()
    name = f"{args.holdout}_{args.engine}"
    report = common.json.loads((common.OUT / f"analysis_adversarial_{name}.json").read_text())  # type: ignore[attr-defined]
    out_path = common.OUT / f"analysis_adversarial_summary_{name}.json"
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists; it is never overwritten")
    result = summarise(report)
    common.write_json(out_path, result)
    for kind in KINDS:
        print(
            kind, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result[kind].items() if k != "families"}
        )
    print(result["character_edit_slips"], "one-word-name slips:", result["character_edit_slips_on_one_word_names"])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

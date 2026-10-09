"""External human-labelled validation: OpenSanctions Pairs (analyst judgements), loaded from a local cache.

The data is CC BY-NC 4.0 (non-commercial use, with attribution to OpenSanctions; see README). It is read from
data/external/hf_opensanctions_pairs/ (git-ignored) and is never copied into the repository: the reports written here
hold aggregate counts only, with no names, IDs or pair content.

Run:
    python analysis/pairs_external.py profile      # step 0 and 1: size, schema, labels, alignment, script coverage
    python analysis/pairs_external.py score        # step 2: the engine over the labelled pairs, once

Protocol (fixed before any pair was scored):
  A pair is two entities and an analyst judgement (positive: same entity). The engine takes a query (a name, an
  optional birth date and country) and a listed entity. A pair is scored as: each name the LEFT entity is known by
  is used as a query, with the left entity's first birth date and first country, against the RIGHT entity as it
  would be on the list (all its names, its birth dates and its countries). The pair's score is the highest final
  score over the left names. Predicted match = score >= 0.86, the engine's one threshold; nothing is tuned. The
  scoring is the engine's own: the deployment list's Screener (its rarity table and skeleton statistics), with
  retrieval bypassed for the one right-hand entity, so the name, skeleton, birth date and country scoring is exactly
  that of screen(). As a sensitivity, the same is run with the RIGHT entity as the query. Only entity pairs (both
  sides a name-bearing schema) are scored; the relationship and role records in the file (Occupancy, Succession,
  Position and so on) are counted and reported, not scored. Duplicate unordered pairs are scored once; pairs whose
  duplicates disagree on the label are counted and left out of the headline.
"""

import gzip
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from profile_data import COUNTRY_PROPS, NAME_PROPS, name_scripts  # noqa: E402

CACHE = common.ROOT / "data" / "external" / "hf_opensanctions_pairs"
FILE = CACHE / "pairs.json.gz"
SHA256 = "4044cf5f6edb0d2cdccf11326063f905408224c3b2fe261baa0d779d80a0edd1"
ENTITY_SCHEMAS = frozenset(
    {"Person", "Organization", "Company", "LegalEntity", "PublicBody", "Vessel", "Airplane", "Security", "CryptoWallet"}
)
ORG_LIKE = frozenset({"Organization", "Company", "LegalEntity", "PublicBody"})
CONSUMED = frozenset({*NAME_PROPS, "birthDate", *COUNTRY_PROPS})
THRESHOLD = 0.86


def names_of(entity: dict) -> list[str]:
    props = entity.get("properties") or {}
    return list(
        dict.fromkeys(v.strip() for p in NAME_PROPS for v in props.get(p) or () if isinstance(v, str) and v.strip())
    )


def values(entity: dict, keys: tuple[str, ...]) -> list[str]:
    props = entity.get("properties") or {}
    return [v for k in keys for v in props.get(k) or () if isinstance(v, str) and v]


def entity_type(left: dict, right: dict) -> str:
    a, b = left.get("schema"), right.get("schema")
    if a == b == "Person":
        return "person_person"
    if a in ORG_LIKE and b in ORG_LIKE:
        return "organisation_organisation"
    if {a, b} <= {"Person", *ORG_LIKE} and (a == "Person" or b == "Person"):
        return "person_organisation"
    return "other_entity_types"  # vessel, airplane, security, crypto wallet, or mixed with one of them


def script_bucket(names: list[str]) -> str:
    """The engine's own ordering (screening.query_script): Arabic wins over Cyrillic, which wins over the rest."""
    scripts: set[str] = set()
    for n in names:
        scripts |= name_scripts(n)
    if "Arabic" in scripts:
        return "Arabic"
    if "Cyrillic" in scripts:
        return "Cyrillic"
    if scripts - {"Latin"}:
        return "other non-Latin"
    return "Latin" if scripts else "no letters"


def side_scripts(names: list[str]) -> frozenset[str]:
    out: set[str] = set()
    for n in names:
        out |= name_scripts(n)
    return frozenset(out)


def pair_script_bucket(left_names: list[str], right_names: list[str]) -> str:
    return script_bucket(left_names + right_names)


def cross_script(left_names: list[str], right_names: list[str]) -> bool:
    a, b = side_scripts(left_names), side_scripts(right_names)
    return bool(a and b) and not (a & b)


def iter_pairs(path: Path = FILE) -> Iterator[tuple[int, dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            yield i, json.loads(line)


def pair_key(rec: dict) -> tuple[str, str]:
    a, b = rec["left"]["id"], rec["right"]["id"]
    return (a, b) if a <= b else (b, a)


def profile() -> dict:
    import hashlib

    digest = hashlib.sha256()
    with open(FILE, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        sys.exit("the cached file does not match the recorded SHA-256; refusing to read it")
    total, labels, schema_pairs = 0, Counter(), Counter()
    scope = Counter()
    types: Counter = Counter()
    by_type_label: Counter = Counter()
    by_script_label: Counter = Counter()
    by_type_script: Counter = Counter()
    cross: Counter = Counter()
    nameless: Counter = Counter()
    keys_any: Counter = Counter()
    keys_both: Counter = Counter()
    attr: Counter = Counter()
    seen: dict[tuple[str, str], str] = {}
    duplicate_lines = conflicting = 0
    conflict_keys: set = set()
    record_keys: Counter = Counter()
    for _, rec in iter_pairs():
        total += 1
        record_keys[tuple(sorted(rec))] += 1
        labels[rec["judgement"]] += 1
        a, b = rec["left"].get("schema"), rec["right"].get("schema")
        schema_pairs[f"{a}/{b}"] += 1
        in_scope = a in ENTITY_SCHEMAS and b in ENTITY_SCHEMAS
        scope["entity_pairs" if in_scope else "relationship_or_role_records"] += 1
        if not in_scope:
            scope[f"not_scored:{a}/{b}"] += 1
            continue
        key = pair_key(rec)
        if key in seen:
            duplicate_lines += 1
            if seen[key] != rec["judgement"]:
                conflicting += 1
                conflict_keys.add(key)
            continue
        seen[key] = rec["judgement"]
        ln, rn = names_of(rec["left"]), names_of(rec["right"])
        t = entity_type(rec["left"], rec["right"])
        label = rec["judgement"]
        types[t] += 1
        by_type_label[f"{t}/{label}"] += 1
        if not ln or not rn:
            nameless["left has no name property" if not ln else "right has no name property"] += 1
            nameless["either side has no name property"] += 1
            if (not ln and rec["left"].get("caption")) or (not rn and rec["right"].get("caption")):
                nameless["...of which the side has a caption"] += 1
            continue
        s = pair_script_bucket(ln, rn)
        by_script_label[f"{s}/{label}"] += 1
        by_type_script[f"{t}/{s}"] += 1
        if cross_script(ln, rn):
            cross[f"cross-script/{label}"] += 1
        lk, rk = set(rec["left"].get("properties") or {}), set(rec["right"].get("properties") or {})
        for k in lk | rk:
            keys_any[k] += 1
        for k in lk & rk:
            keys_both[k] += 1
        for name, keys in (("birth_date", ("birthDate",)), ("country", COUNTRY_PROPS)):
            l_has, r_has = bool(values(rec["left"], keys)), bool(values(rec["right"], keys))
            attr[f"{name}:both"] += l_has and r_has
            attr[f"{name}:left_only"] += l_has and not r_has
            attr[f"{name}:right_only"] += r_has and not l_has
            attr[f"{name}:neither"] += not l_has and not r_has
        attr["pairs"] += 1
    ignored = {k: {"pairs_with_it_on_either_side": keys_any[k], "pairs_with_it_on_both": keys_both[k]}
               for k, _ in keys_any.most_common() if k not in CONSUMED and keys_any[k] > 0}  # fmt: skip
    consumed = {k: {"pairs_with_it_on_either_side": keys_any[k], "pairs_with_it_on_both": keys_both[k]}
                for k in sorted(CONSUMED) if keys_any[k] > 0}  # fmt: skip
    return {
        "file": {
            "name": "pairs.json.gz",
            "bytes": FILE.stat().st_size,
            "sha256": digest.hexdigest(),
            "source": "huggingface.co/datasets/sanctions-er-anon/opensanctions_pairs",
            "licence": "CC BY-NC 4.0",
        },
        "record_keys": {"+".join(k): v for k, v in record_keys.items()},
        "pairs_total": total,
        "judgements": dict(labels),
        "positive_share": labels["positive"] / total,
        "schema_pairs_left_right": dict(schema_pairs.most_common()),
        "scope": {k: v for k, v in scope.items() if not k.startswith("not_scored:")},
        "not_scored_by_schema_pair": {k.split(":", 1)[1]: v for k, v in scope.items() if k.startswith("not_scored:")},
        "entity_pairs_unique": sum(types.values()),
        "duplicate_lines_of_an_entity_pair": duplicate_lines,
        "duplicates_with_a_different_label": conflicting,
        "entity_pairs_with_conflicting_labels": len(conflict_keys),
        "unique_entity_pairs_by_type": dict(types),
        "unique_entity_pairs_by_type_and_label": dict(by_type_label),
        "pairs_whose_two_sides_have_no_name_property": dict(nameless),
        "unique_entity_pairs_by_script_bucket_and_label": dict(by_script_label),
        "unique_entity_pairs_by_type_and_script_bucket": dict(by_type_script),
        "cross_script_pairs_by_label": dict(cross),
        "attribute_coverage_entity_pairs": {k: v for k, v in attr.items()},
        "properties_the_engine_consumes": consumed,
        "properties_the_engine_ignores": dict(list(ignored.items())[:40]),
        "n_ignored_property_keys": len(ignored),
    }


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "profile":
        out = common.OUT / "pairs_external_alignment.json"
        if out.exists():
            sys.exit(f"{out.name} already exists; it is never overwritten")
        result = profile()
        common.write_json(out, result)
        print(json.dumps(result, indent=1)[:6000])
        print(f"wrote {out}")
    elif mode == "score":
        import pairs_external_score

        pairs_external_score.main()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

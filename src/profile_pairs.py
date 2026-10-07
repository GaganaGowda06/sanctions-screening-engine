"""Profile the OpenSanctions Pairs dataset: size, labels, scripts, attribute coverage, entity overlap.

Streams the file once. Writes reports/pairs_profile.json.

Script groups per entity are taken from every name it carries (caption, name, alias, previousName,
weakAlias): Latin, Cyrillic, Arabic, and "other" for any other script. Attribute coverage:
  birth date   any birthDate value
  country      any of country, nationality, citizenship, jurisdiction, birthCountry, mainCountry, flag
  identifier   any identifier-typed property of the entity's schema, leaving out wikidataId and zipCode
               (and, in the second figure, programId, which names a sanctions programme, not the entity)

Every table is given for all pairs and for the name-bearing entity pairs alone (pairs.ENTITY_SCHEMAS),
because 37% of the file labels relationship and role records (Occupancy, Succession, Position, ...).

Run:
    python src/profile_pairs.py
"""

import argparse
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path

from pairs import PAIRS_PATH, in_scope, iter_pairs
from profile_data import COUNTRY_PROPS, NAME_PROPS, PROPS_PATH, name_scripts

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "pairs_profile.json"
EXCLUDED_IDENTIFIERS = {"zipCode", "wikidataId"}
GROUPS = ("Latin", "Cyrillic", "Arabic", "other")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


class Entity:
    """What the profile needs from one side of a pair."""

    __slots__ = ("scripts", "years", "countries", "ids", "ids_no_programme")

    def __init__(self, entity: dict, identifier_props: dict[str, list[str]]) -> None:
        props = entity.get("properties") or {}
        names = {entity.get("caption") or ""}
        names.update(v for p in NAME_PROPS for v in props.get(p) or () if isinstance(v, str))
        scripts: set[str] = set()
        for name in names:
            for script in name_scripts(name):
                scripts.add(script if script in GROUPS else "other")
        self.scripts = frozenset(scripts)
        self.years = frozenset(v[:4] for v in props.get("birthDate") or () if isinstance(v, str) and v[:4].isdigit())
        self.countries = frozenset(
            v.lower() for p in COUNTRY_PROPS for v in props.get(p) or () if isinstance(v, str) and v
        )
        ids, ids_no_programme = set(), set()
        for prop in identifier_props.get(entity.get("schema"), ()):
            for value in props.get(prop) or ():
                if isinstance(value, str) and (key := _NON_ALNUM.sub("", value.lower())):
                    ids.add(key)
                    if prop != "programId":
                        ids_no_programme.add(key)
        self.ids = frozenset(ids)
        self.ids_no_programme = frozenset(ids_no_programme)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def share(n: int, total: int) -> float:
    return round(n / total, 4) if total else 0.0


def pair_class(left: frozenset[str], right: frozenset[str]) -> str:
    left_non, right_non = bool(left - {"Latin"}), bool(right - {"Latin"})
    if not left_non and not right_non:
        return "Latin on both sides only"
    if left_non != right_non:
        return "non-Latin on one side only"
    return "non-Latin on both sides"


def profile(path: Path) -> dict:
    ftm = json.loads(PROPS_PATH.read_text())["schemas"]
    identifier_props = {s: [p for p in v["identifier"] if p not in EXCLUDED_IDENTIFIERS] for s, v in ftm.items()}
    cache: dict[str, Entity] = {}

    def entity(raw: dict) -> Entity:
        eid = raw["id"]
        if eid not in cache:
            cache[eid] = Entity(raw, identifier_props)
        return cache[eid]

    scopes = ("all", "entity_schemas")
    judgements = {sc: Counter() for sc in scopes}
    schemas: Counter = Counter()
    classes: dict[str, dict[str, Counter]] = {sc: {} for sc in scopes}
    presence = {sc: {g: Counter() for g in GROUPS} for sc in scopes}
    no_shared_script = {sc: Counter() for sc in scopes}
    coverage: dict[str, dict[str, Counter]] = {sc: {} for sc in scopes}
    agreement: dict[str, dict[str, Counter]] = {sc: {} for sc in scopes}
    totals = dict.fromkeys(scopes, 0)
    positive_flags = bytearray()  # in file order, for the share of positives along the file
    uf = UnionFind()
    pair_counts: Counter = Counter()
    pair_labels: dict[tuple[str, str], set[str]] = {}
    degree: Counter = Counter()
    keys = []

    def bump(table: dict[str, Counter], key: str, judgement: str) -> None:
        table.setdefault(key, Counter())[judgement] += 1

    for _, rec in iter_pairs(path):
        judgement = rec["judgement"]
        left, right = rec["left"], rec["right"]
        keys.append((left["id"], right["id"]))
        schemas[(left.get("schema"), right.get("schema"))] += 1
        a, b = entity(left), entity(right)
        uf.union(left["id"], right["id"])
        degree[left["id"]] += 1
        degree[right["id"]] += 1
        unordered = tuple(sorted((left["id"], right["id"])))
        pair_counts[unordered] += 1
        pair_labels.setdefault(unordered, set()).add(judgement)
        positive_flags.append(judgement == "positive")

        facts = {
            "birth_date": bool(a.years and b.years),
            "country": bool(a.countries and b.countries),
            "identifier": bool(a.ids and b.ids),
            "identifier_without_programId": bool(a.ids_no_programme and b.ids_no_programme),
        }
        all_three = facts["birth_date"] and facts["country"] and facts["identifier_without_programId"]
        shared = a.ids_no_programme & b.ids_no_programme
        for sc in scopes if in_scope(rec) else scopes[:1]:
            totals[sc] += 1
            judgements[sc][judgement] += 1
            bump(classes[sc], pair_class(a.scripts, b.scripts), judgement)
            for g in GROUPS:
                if g in a.scripts or g in b.scripts:
                    presence[sc][g][judgement] += 1
            if not (a.scripts & b.scripts):
                no_shared_script[sc][judgement] += 1
            for name, both in facts.items():
                if both:
                    bump(coverage[sc], name, judgement)
            if all_three:
                bump(coverage[sc], "birth_date_country_and_identifier_all_present", judgement)
            if facts["birth_date"]:
                bump(agreement[sc], "birth_year_shared" if a.years & b.years else "birth_year_conflict", judgement)
            if facts["country"]:
                bump(agreement[sc], "country_shared" if a.countries & b.countries else "country_conflict", judgement)
            if facts["identifier_without_programId"]:
                bump(agreement[sc], "identifier_shared" if shared else "identifier_differs", judgement)

    total = len(positive_flags)
    tenth = max(1, total // 10)

    def positive_share_of_tenth(i: int) -> float:
        chunk = positive_flags[i * tenth : (i + 1) * tenth if i < 9 else total]
        return share(sum(chunk), len(chunk))

    by_tenth = {f"{i * 10} to {i * 10 + 10}%": positive_share_of_tenth(i) for i in range(10)}
    components = Counter(uf.find(e) for e in cache)
    sizes = sorted(components.values(), reverse=True)
    largest_root = max(components, key=components.get)
    pairs_in_largest = sum(1 for left_id, _ in keys if uf.find(left_id) == largest_root)

    def table(counter: Counter) -> dict:
        n = sum(counter.values())
        return {"pairs": n, "positive": counter["positive"], "negative": counter["negative"],
                "positive_share": share(counter["positive"], n)}  # fmt: skip

    def per_scope(sc: str) -> dict:
        return {
            "pairs": totals[sc],
            "judgements": dict(judgements[sc]),
            "positive_share": share(judgements[sc]["positive"], totals[sc]),
            "script_class": {k: table(v) for k, v in sorted(classes[sc].items())},
            "pairs_with_script_on_either_side": {g: table(presence[sc][g]) for g in GROUPS},
            "pairs_with_no_script_in_common": table(no_shared_script[sc]),
            "attribute_coverage_both_sides": {
                k: {**table(v), "share_of_pairs": share(sum(v.values()), totals[sc])}
                for k, v in coverage[sc].items()
            },
            "attribute_agreement_when_both_present": {k: table(v) for k, v in sorted(agreement[sc].items())},
        }

    return {
        "file": str(path),
        "python_version": platform.python_version(),
        "pairs": total,
        "positive_share_by_tenth_of_file": by_tenth,
        "schemas_left_right": {f"{left}/{right}": n for (left, right), n in schemas.most_common()},
        "unique_entities": len(cache),
        "entities_in_more_than_one_pair": sum(1 for n in degree.values() if n > 1),
        "most_pairs_for_one_entity": max(degree.values()),
        "duplicate_unordered_pairs": sum(1 for n in pair_counts.values() if n > 1),
        "pairs_with_conflicting_labels": sum(1 for v in pair_labels.values() if len(v) > 1),
        "connected_components": {
            "count": len(components),
            "largest": sizes[0],
            "five_largest": sizes[:5],
            "pairs_in_largest": pairs_in_largest,
            "pairs_in_largest_share": share(pairs_in_largest, total),
            "singletons": sum(1 for s in sizes if s == 1),
            "size_2_to_10": sum(1 for s in sizes if 2 <= s <= 10),
            "size_over_10": sum(1 for s in sizes if s > 10),
        },
        "all_pairs": per_scope("all"),
        "entity_pairs": per_scope("entity_schemas"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=PAIRS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"Python {platform.python_version()}", file=sys.stderr)
    result = profile(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

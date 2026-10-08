"""Which kind of name is each positive query? Reads the entities file and the frozen hold-outs; writes only a report.

The matcher's entity loader pools four OpenSanctions name properties (name, alias, previousName, weakAlias) into one
list, so a positive query can be a primary name, an alias, a former name or a weak alias. This counts, for each
hold-out, which property each positive query's text comes from.

Run:
    python analysis/query_provenance.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PROPS = ("name", "alias", "previousName", "weakAlias")


def main() -> None:
    out_path = common.OUT / "analysis_query_provenance.json"
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists; it is never overwritten")
    holdouts = {}
    for label, path in (("second", common.HOLDOUT_V2), ("third", common.HOLDOUT_V3)):
        queries, _, digest = common.load_holdout(path)
        holdouts[label] = (digest, [q for q in queries if q["kind"] == "positive"])
    wanted = {q["true_id"] for _, pos in holdouts.values() for q in pos}
    props_of: dict[str, dict[str, set[str]]] = {}
    with open(common.CURRENT_SNAPSHOT, "rb") as fh:
        for line in fh:
            try:
                ent = json.loads(line)
            except ValueError:
                continue
            if ent.get("id") in wanted:
                p = ent.get("properties") or {}
                props_of[ent["id"]] = {k: {v.strip() for v in p.get(k) or () if isinstance(v, str)} for k in PROPS}
    result = {}
    for label, (digest, pos) in holdouts.items():
        only, any_weak, missing = Counter(), 0, 0
        by_group: dict[str, Counter] = {}
        for q in pos:
            where = tuple(k for k in PROPS if q["name"].strip() in props_of.get(q["true_id"], {}).get(k, set()))
            if not where:
                missing += 1
                continue
            key = "+".join(where)
            only[key] += 1
            any_weak += "weakAlias" in where
            by_group.setdefault(q["group"], Counter())[key] += 1
        result[label] = {
            "holdout_sha256": digest,
            "positives": len(pos),
            "query_text_found_in_the_entity_properties": len(pos) - missing,
            "not_found_in_the_current_snapshot": missing,
            "positives_whose_query_is_also_a_weak_alias": any_weak,
            "by_properties_the_text_appears_in": dict(sorted(only.items(), key=lambda t: -t[1])),
            "by_group": {g: dict(sorted(c.items(), key=lambda t: -t[1])) for g, c in sorted(by_group.items())},
        }
    result["snapshot"] = common.read_snapshot(common.CURRENT_SNAPSHOT).version
    common.write_json(out_path, result)
    print(json.dumps(result, indent=1)[:3500])


if __name__ == "__main__":
    main()

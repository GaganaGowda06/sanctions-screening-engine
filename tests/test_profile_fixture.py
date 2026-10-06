"""Check profile_data.py against a hand-built fixture with known answers.

Run:
    python tests/test_profile_fixture.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from profile_data import profile  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini.ftm.json"

EXPECTED = {
    "total_entities": 11,
    "target_entities": 4,
    "unparseable_lines": 1,
    "schema_counts": {
        "all": {"Person": 7, "Company": 2, "Passport": 1, "Sanction": 1},
        "targets": {"Person": 3, "Company": 1},
    },
    "names": {
        "all_entities": {
            "entities_with_names": 8, "entities_without_names": 3, "total_names": 11,
            "names_with_non_latin_letter": 7, "names_without_letters": 0,
            "names_containing_script": {"Latin": 5, "Cyrillic": 2, "Japanese kana": 2,
                                        "Arabic": 1, "Han": 1, "Hangul": 1},
            "entities_with_any_non_latin_name": 6, "entities_with_no_purely_latin_name": 4,
        },
        "targets": {
            "entities_with_names": 4, "entities_without_names": 0, "total_names": 5,
            "names_with_non_latin_letter": 4, "names_without_letters": 0,
            "names_containing_script": {"Cyrillic": 2, "Latin": 2, "Arabic": 1, "Hangul": 1},
            "entities_with_any_non_latin_name": 4, "entities_with_no_purely_latin_name": 3,
        },
    },
    # fields: n, birthDate, country, own_identifier, any_identifier, own_no_prog, any_no_prog
    "coverage": {
        "named_entities": {
            "ALL": (8, 3, 3, 3, 4, 2, 3),
            "Person": (6, 3, 2, 2, 3, 1, 2),
            "Company": (2, 0, 1, 1, 1, 1, 1),
        },
        "targets": {
            "ALL": (4, 2, 2, 2, 3, 2, 3),
            "Person": (3, 2, 1, 1, 2, 1, 2),
            "Company": (1, 0, 1, 1, 1, 1, 1),
        },
    },
    "top_identifier_properties": [
        ("idNumber", 1, 1), ("leiCode", 1, 1), ("number", 1, 0),
        ("passportNumber", 1, 0), ("programId", 1, 0),
    ],
}
FIELDS = ("n", "birthDate", "country", "own_identifier", "any_identifier",
          "own_identifier_no_programId", "any_identifier_no_programId")


def flatten(r: dict) -> dict:
    out = {k: r[k] for k in ("total_entities", "target_entities", "unparseable_lines", "schema_counts", "names")}
    out["coverage"] = {pop: {s: tuple(row[f] for f in FIELDS) for s, row in rows.items()}
                       for pop, rows in r["coverage"].items()}
    out["top_identifier_properties"] = [(t["property"], t["entities"], t["target_entities"])
                                        for t in r["top_identifier_properties"]]
    return out


def main() -> None:
    actual = flatten(profile(FIXTURE))
    failures = 0
    for key, exp in EXPECTED.items():
        got = actual[key]
        # Compare as JSON so dict key order and tuple/list differences do not matter.
        same = json.loads(json.dumps(exp, sort_keys=True)) == json.loads(json.dumps(got, sort_keys=True))
        failures += not same
        print(f"[{'OK' if same else 'FAIL'}] {key}")
        if not same:
            print("  expected:", json.dumps(exp, ensure_ascii=False, sort_keys=True))
            print("  actual:  ", json.dumps(got, ensure_ascii=False, sort_keys=True))
    print("\nALL MATCH" if not failures else f"\n{failures} MISMATCH(ES)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

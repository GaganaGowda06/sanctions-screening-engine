"""Profile the OpenSanctions entities file (FollowTheMoney, one JSON entity per line).

Streams the file line by line and never loads it whole. Writes reports/stage1_profile.json.

Identifier and country-typed properties come from src/ftm_props.json, a static map
generated from the followthemoney 4.11.0 schema (inheritance resolved), so this script
needs only the standard library.

Run:
    python src/profile_data.py
    python src/profile_data.py --input path/to/entities.ftm.json --output path/to/profile.json
"""
import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "sanctions" / "entities.ftm.json"
DEFAULT_OUTPUT = ROOT / "reports" / "stage1_profile.json"
PROPS_PATH = Path(__file__).resolve().parent / "ftm_props.json"

NAME_PROPS = ("name", "alias", "previousName", "weakAlias")
COUNTRY_PROPS = ("country", "nationality", "citizenship", "jurisdiction",
                 "birthCountry", "mainCountry", "flag")
EXCLUDED_IDENTIFIERS = {"zipCode", "wikidataId"}
# programId is identifier-typed in FtM but names a sanctions programme, not the entity.
# It is kept in the main definition (as specified) and stripped in the *_no_programId variants.
PROGRAMME_PROP = "programId"
HOLDER_SCHEMAS = {"Identification", "Passport"}

LATIN = "Latin"
KANA = "Japanese kana"
# Letter characters whose Unicode names do not start with a script name but are Latin-derived.
LATIN_LIKE_FIRST_WORDS = {"MODIFIER", "FEMININE", "MASCULINE", "SCRIPT", "DOUBLE-STRUCK",
                          "BLACK-LETTER", "MATHEMATICAL", "ANGSTROM", "PLANCK", "MICRO"}

_script_cache: dict[str, str | None] = {}


def char_script(ch: str) -> str | None:
    """Script label for a letter, None for non-letters."""
    try:
        return _script_cache[ch]
    except KeyError:
        pass
    script = None
    if ch.isalpha():
        name = unicodedata.name(ch, "")
        for prefix in ("FULLWIDTH ", "HALFWIDTH "):
            if name.startswith(prefix):
                name = name[len(prefix):]
        first = name.split(" ", 1)[0]
        if first == "LATIN" or first in LATIN_LIKE_FIRST_WORDS:
            script = LATIN
        elif first in ("CJK", "IDEOGRAPHIC"):
            script = "Han"
        elif first.startswith(("HIRAGANA", "KATAKANA")):
            script = KANA
        elif first:
            script = first.title()
        else:
            script = "Unknown"
    _script_cache[ch] = script
    return script


def name_scripts(name: str) -> set[str]:
    if name.isascii():
        return {LATIN} if any(c.isalpha() for c in name) else set()
    return {s for s in map(char_script, name) if s}


def has_value(props: dict, key: str) -> bool:
    return any(isinstance(v, str) and v.strip() for v in props.get(key) or ())


class NameStats:
    def __init__(self) -> None:
        self.entities_with_names = 0
        self.total_names = 0
        self.names_non_latin = 0
        self.names_no_letters = 0
        self.per_script: Counter = Counter()
        self.entities_non_latin = 0
        self.entities_no_pure_latin = 0

    def add(self, names: set[str]) -> None:
        if not names:
            return
        self.entities_with_names += 1
        any_non_latin = False
        any_pure_latin = False
        for n in names:
            scripts = name_scripts(n)
            self.total_names += 1
            if not scripts:
                self.names_no_letters += 1
                continue
            for s in scripts:
                self.per_script[s] += 1
            if scripts - {LATIN}:
                self.names_non_latin += 1
                any_non_latin = True
            else:
                any_pure_latin = True
        self.entities_non_latin += any_non_latin
        self.entities_no_pure_latin += not any_pure_latin

    def report(self, entities_total: int) -> dict:
        return {
            "entities_with_names": self.entities_with_names,
            "entities_without_names": entities_total - self.entities_with_names,
            "total_names": self.total_names,
            "names_with_non_latin_letter": self.names_non_latin,
            "names_without_letters": self.names_no_letters,
            "names_containing_script": dict(self.per_script.most_common()),
            "entities_with_any_non_latin_name": self.entities_non_latin,
            "entities_with_no_purely_latin_name": self.entities_no_pure_latin,
        }


class Coverage:
    """Per-schema coverage counts for one population."""
    FIELDS = ("n", "birthDate", "country", "own_identifier", "any_identifier",
              "own_identifier_no_programId", "any_identifier_no_programId")

    def __init__(self) -> None:
        self.rows: dict[str, Counter] = defaultdict(Counter)

    def add(self, schema: str, flags: dict[str, bool]) -> None:
        for key in ("ALL", schema):
            row = self.rows[key]
            row["n"] += 1
            for f, v in flags.items():
                row[f] += v

    def report(self) -> dict:
        return {s: {f: self.rows[s][f] for f in self.FIELDS}
                for s in sorted(self.rows, key=lambda k: (k != "ALL", -self.rows[k]["n"], k))}


def profile(path: Path) -> dict:
    ftm = json.loads(PROPS_PATH.read_text())["schemas"]
    id_props = {s: [p for p in v["identifier"] if p not in EXCLUDED_IDENTIFIERS] for s, v in ftm.items()}

    total = targets = unparseable = 0
    schema_all: Counter = Counter()
    schema_targets: Counter = Counter()
    names_all, names_targets = NameStats(), NameStats()
    cov_named, cov_targets = Coverage(), Coverage()
    id_prop_all: Counter = Counter()
    id_prop_targets: Counter = Counter()
    holders: set[str] = set()
    # Entities with no own non-programme identifier, waiting to see if an Identification/Passport points at them.
    pending: dict[str, tuple[str, bool, bool, bool]] = {}

    with open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                ent = json.loads(line)
                schema = ent["schema"]
                eid = ent["id"]
                props = ent.get("properties") or {}
                if not isinstance(ent, dict) or not isinstance(props, dict):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                unparseable += 1
                continue

            is_target = ent.get("target") is True
            total += 1
            targets += is_target
            schema_all[schema] += 1
            if is_target:
                schema_targets[schema] += 1

            if schema in HOLDER_SCHEMAS:
                holders.update(h for h in props.get("holder") or () if isinstance(h, str))

            own_props = [p for p in id_props.get(schema, ()) if has_value(props, p)]
            for p in own_props:
                id_prop_all[p] += 1
                if is_target:
                    id_prop_targets[p] += 1

            names = {v.strip() for p in NAME_PROPS for v in props.get(p) or ()
                     if isinstance(v, str) and v.strip()}
            names_all.add(names)
            if is_target:
                names_targets.add(names)

            own = bool(own_props)
            own_np = any(p != PROGRAMME_PROP for p in own_props)
            flags = {
                "birthDate": schema == "Person" and has_value(props, "birthDate"),
                "country": any(has_value(props, p) for p in COUNTRY_PROPS),
                "own_identifier": own,
                "any_identifier": own,
                "own_identifier_no_programId": own_np,
                "any_identifier_no_programId": own_np,
            }
            if names:
                cov_named.add(schema, flags)
            if is_target:
                cov_targets.add(schema, flags)
            if not own_np:
                pending[eid] = (schema, is_target, bool(names), own)

    # Entities referenced as holder by an Identification/Passport gain "any identifier".
    for eid in holders:
        item = pending.get(eid)
        if not item:
            continue
        schema, is_target, named, own = item
        for cov, applies in ((cov_named, named), (cov_targets, is_target)):
            if not applies:
                continue
            for key in ("ALL", schema):
                row = cov.rows[key]
                if not own:
                    row["any_identifier"] += 1
                row["any_identifier_no_programId"] += 1

    index_path = path.with_name("index.json")
    version = None
    if index_path.exists():
        try:
            version = json.loads(index_path.read_text()).get("version")
        except ValueError:
            pass

    top = [{"property": p, "entities": n, "target_entities": id_prop_targets[p]}
           for p, n in id_prop_all.most_common()]
    top.sort(key=lambda r: (-r["entities"], r["property"]))

    size = path.stat().st_size
    return {
        "input": str(path),
        "dataset_version": version,
        "file_size_bytes": size,
        "file_size_mib": round(size / 1024 / 1024, 1),
        "total_entities": total,
        "target_entities": targets,
        "unparseable_lines": unparseable,
        "schema_counts": {
            "all": dict(schema_all.most_common()),
            "targets": dict(schema_targets.most_common()),
        },
        "names": {
            "all_entities": names_all.report(total),
            "targets": names_targets.report(targets),
        },
        "coverage": {
            "named_entities": cov_named.report(),
            "targets": cov_targets.report(),
        },
        "top_identifier_properties": top[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = profile(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

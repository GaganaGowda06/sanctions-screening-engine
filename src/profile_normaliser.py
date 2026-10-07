"""Run the name normaliser over the real data and report how it behaves.

Quality is measured two ways:
- Output shape: clean (non-empty, only a-z, digits, spaces), empty, or unresolved
  (letters left untransliterated).
- Agreement with the data itself: for entities that have both a purely Cyrillic or purely
  Arabic name and a purely Latin name, how close is the transliterated form to the
  best-matching Latin name of the same entity. This is a proxy: a Latin alias is not always
  a transliteration of the other name, so treat low scores as "worth looking at".

Run:
    python src/profile_normaliser.py
Writes reports/stage2_normaliser.json.
"""

import argparse
import json
import platform
import random
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from normalise import is_clean, normalise
from profile_data import DEFAULT_INPUT, NAME_PROPS, name_scripts, read_version_file

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "stage2_normaliser.json"
SEED = 42
VOWELS = re.compile("[aeiouyw]")


def classify(scripts: set[str]) -> str:
    """Group a name by the scripts it contains."""
    if not scripts:
        return "no letters"
    if scripts == {"Latin"}:
        return "Latin only"
    rest = scripts - {"Latin"}
    if rest == {"Cyrillic"}:
        return "Cyrillic (with or without Latin)"
    if rest == {"Arabic"}:
        return "Arabic (with or without Latin)"
    return "other non-Latin"


def collect_names(path: Path) -> list[tuple[bool, list[str]]]:
    """One (is_target, deduplicated names) item per entity that has names."""
    entities = []
    with open(path, "rb") as fh:
        for line in fh:
            try:
                ent = json.loads(line)
                props = ent.get("properties") or {}
            except (ValueError, AttributeError):
                continue
            names = list(
                dict.fromkeys(
                    v.strip() for p in NAME_PROPS for v in props.get(p) or () if isinstance(v, str) and v.strip()
                )
            )
            if names:
                entities.append((ent.get("target") is True, names))
    return entities


def similarity(a: str, b: str) -> tuple[float, float]:
    """(full, consonant-skeleton) similarity with word order ignored."""

    def key(s: str, skeleton: bool) -> str:
        tokens = (VOWELS.sub("", t) if skeleton else t for t in s.split())
        return " ".join(sorted(t for t in tokens if t))

    full = SequenceMatcher(None, key(a, False), key(b, False)).ratio()
    skel_a, skel_b = key(a, True), key(b, True)
    skel = SequenceMatcher(None, skel_a, skel_b).ratio() if skel_a and skel_b else 0.0
    return full, skel


def summarise(scores: list[float]) -> dict:
    scores = sorted(scores)
    if not scores:
        return {"pairs": 0}
    return {
        "pairs": len(scores),
        "median": round(scores[len(scores) // 2], 3),
        "share_at_least_0.8": round(sum(s >= 0.8 for s in scores) / len(scores), 3),
        "share_below_0.5": round(sum(s < 0.5 for s in scores) / len(scores), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"Python {platform.python_version()}", file=sys.stderr)

    t0 = time.perf_counter()
    entities = collect_names(args.input)
    t_read = time.perf_counter() - t0

    flat = [(target, name) for target, names in entities for name in names]
    t0 = time.perf_counter()
    normalised = [normalise(name) for _, name in flat]
    t_norm = time.perf_counter() - t0

    # Output shape per group, for all names and for target names only.
    shape = {pop: {} for pop in ("all", "targets")}
    residual_chars: dict[str, Counter] = {"Cyrillic": Counter(), "Arabic": Counter()}
    fallthrough: dict[str, dict] = {}
    scripts_by_name: dict[str, set[str]] = {}
    for (target, name), norm in zip(flat, normalised, strict=True):
        scripts = scripts_by_name.get(name) or scripts_by_name.setdefault(name, name_scripts(name))
        group = classify(scripts)
        state = "clean" if is_clean(norm) else ("empty" if not norm else "unresolved")
        for pop in ("all", "targets") if target else ("all",):
            row = shape[pop].setdefault(group, Counter())
            row["names"] += 1
            row[state] += 1
        if state == "unresolved":
            for script in name_scripts(norm) - {"Latin"}:
                entry = fallthrough.setdefault(script, {"names": 0, "examples": []})
                entry["names"] += 1
                if len(entry["examples"]) < 3:
                    entry["examples"].append([name, norm])
            if group.startswith(("Cyrillic", "Arabic")):
                key = "Cyrillic" if group.startswith("Cyrillic") else "Arabic"
                residual_chars[key].update(c for c in norm if not c.isascii())

    # Agreement with the same entity's Latin names.
    pairs: dict[str, list[tuple[float, float, str, str, str]]] = {"Cyrillic": [], "Arabic": []}
    cursor = 0
    for _, names in entities:
        chunk = list(zip(names, normalised[cursor : cursor + len(names)], strict=True))
        cursor += len(names)
        latin = [(n, v) for n, v in chunk if name_scripts(n) == {"Latin"} and v]
        if not latin:
            continue
        for raw, norm in chunk:
            scripts = name_scripts(raw)
            if scripts not in ({"Cyrillic"}, {"Arabic"}) or not norm:
                continue
            best = max((similarity(norm, v) + (n,) for n, v in latin), key=lambda r: r[0])
            full, skel, latin_raw = best
            pairs[next(iter(scripts))].append((full, skel, raw, norm, latin_raw))

    # Names of four tokens or fewer are mostly people, so legal-form translations matter less.
    agreement = {
        script: {
            "all_names": {
                "full_string": summarise([p[0] for p in rows]),
                "consonant_skeleton": summarise([p[1] for p in rows]),
            },
            "names_up_to_4_tokens": {
                "full_string": summarise([p[0] for p in rows if len(p[2].split()) <= 4]),
                "consonant_skeleton": summarise([p[1] for p in rows if len(p[2].split()) <= 4]),
            },
        }
        for script, rows in pairs.items()
    }

    # Samples: 6 Latin names that change in a non-trivial way, 7 Cyrillic, 7 Arabic.
    rng = random.Random(SEED)
    unique = list(dict.fromkeys(flat))
    latin_changed = [
        n for _, n in unique if name_scripts(n) == {"Latin"} and (not n.isascii() or not n.replace(" ", "").isalnum())
    ]
    samples = [(n, normalise(n), "") for n in rng.sample(latin_changed, 6)]
    for script in ("Cyrillic", "Arabic"):
        for full, _, raw, norm, latin_raw in rng.sample(pairs[script], 7):
            samples.append((raw, norm, f"{latin_raw}  (similarity {full:.2f})"))
    worst = {
        script: [[raw, norm, latin_raw, round(full, 2)] for full, _, raw, norm, latin_raw in sorted(rows)[:5]]
        for script, rows in pairs.items()
    }

    version_file = args.input.with_name("VERSION")
    version = read_version_file(version_file)
    result = {
        "python_version": platform.python_version(),
        "dataset_version": version.get("version"),
        "names_normalised": len(flat),
        "timing_seconds": {
            "read_and_extract_names": round(t_read, 2),
            "normalise_only": round(t_norm, 2),
            "total": round(t_read + t_norm, 2),
            "normalise_names_per_second": round(len(flat) / t_norm),
        },
        "output_shape": {pop: {g: dict(c) for g, c in rows.items()} for pop, rows in shape.items()},
        "fallthrough_scripts": dict(sorted(fallthrough.items(), key=lambda kv: -kv[1]["names"])),
        "unmapped_chars_in_cyrillic_and_arabic_names": {k: dict(c.most_common(20)) for k, c in residual_chars.items()},
        "agreement_with_same_entity_latin_name": agreement,
        "samples": [{"before": b, "after": a, "same_entity_latin_name": s} for b, a, s in samples],
        "lowest_agreement_pairs": worst,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

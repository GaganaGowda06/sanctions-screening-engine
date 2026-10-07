"""Command line wrapper over screening.screen().

Run:
    python src/screen_cli.py "Vladimir Putin" --birth-date 1952-10-07 --country ru
    python src/screen_cli.py "محمد علي" --json --limit 5

Building the list takes a few seconds on every run. Exit status is 0 whether or not there are matches.
"""

import argparse
import json
import sys
from pathlib import Path

from profile_data import DEFAULT_INPUT
from screening import ScreeningEngine, ScreenMatch


def format_match(rank: int, match: ScreenMatch) -> str:
    e = match.explanation
    others = ", ".join(n for n in match.names if n != e.matched_name)
    skeleton = "used" if e.skeleton_used else "not used"
    return (
        f"{rank:>2}. {match.score:.3f}  {match.entity_id}\n"
        f"      matched name: {e.matched_name}   other names: {others}\n"
        f"      name score {e.name_score:.3f}, skeleton {skeleton} (similarity {e.skeleton_score:.3f}), "
        f"birth date {e.birth_date} ({e.birth_date_adjustment:+.2f}), country {e.country} ({e.country_adjustment:+.2f})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the name to screen")
    parser.add_argument("--birth-date", help="YYYY, YYYY-MM or YYYY-MM-DD")
    parser.add_argument("--country", help="ISO 3166 two-letter country code, for example ru")
    parser.add_argument("--limit", type=int, help="show at most this many matches")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="entities file to screen against")
    args = parser.parse_args(argv)

    print("Building the list...", file=sys.stderr)
    engine = ScreeningEngine.from_file(args.input)
    matches = engine.screen(args.name, args.birth_date, args.country, limit=args.limit)

    if args.json:
        print(json.dumps([m.as_dict() for m in matches], indent=2, ensure_ascii=False))
    elif not matches:
        print(f"No alert for {args.name!r}.")
    else:
        script = matches[0].explanation.script
        threshold = matches[0].explanation.threshold
        print(f"{len(matches)} alert(s) for {args.name!r} (script {script}, threshold {threshold})")
        print("\n".join(format_match(i, m) for i, m in enumerate(matches, 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

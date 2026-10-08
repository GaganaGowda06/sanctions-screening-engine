"""Command line screening, audited. Production entry point: every decision is written to the audit log first.

Run:
    python src/screen_cli.py init --audit-dir /var/lib/screening/audit          (once, to create the audit store)
    python src/screen_cli.py "Vladimir Putin" --birth-date 1952-10-07 --country ru --audit-dir /var/lib/screening/audit
    python src/screen_cli.py "محمد علي" --json --limit 5 --customer-ref cust-123
    python src/screen_cli.py --batch names.jsonl --audit-dir /var/lib/screening/audit

The audit directory comes from --audit-dir or $SCREENING_AUDIT_DIR and has no default. With neither set the command
refuses to start (exit status 2) before it builds the list. Screening never creates the store: "init" does, once, and
refuses if the directory already exists and is not empty. If the store is missing or wiped, screening exits 1,
prints no decision, and says log_missing and how to run init. A wiped trail and a new deployment look the same from
inside, so a person decides which it is. The record of each name (header and payload, see
src/audit.py) is written before the decision is printed. If a record cannot be written the command prints no
decision for that name, says so on stderr, and exits non-zero (exit status 1).

--batch reads a JSON-lines file, or standard input for "-": one object per line with "name" and optionally
"birth_date", "country" and "customer_ref". The whole file is checked before anything is screened, so a malformed
line stops the run with exit status 2 and no records. Names are then screened in order, one record each. If a record
cannot be written the run stops there: earlier names keep their records and decisions, later ones are not screened.

--customer-ref links the record to a customer (opaque to the matcher; stored in the payload only). Exit status is 0
whether or not there are matches. Building the list takes a few seconds on every run.
"""

import argparse
import json
import sys
from pathlib import Path

from audit import AuditConfigError, AuditError, AuditLog, LogMissingError
from audited_screening import AuditedEngine, AuditedResult, resolve_audit_dir
from profile_data import DEFAULT_INPUT
from screening import ScreenMatch

EXIT_AUDIT_FAILURE = 1
EXIT_USAGE = 2


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


def render(name: str, result: AuditedResult, limit: int | None, as_json: bool, compact: bool = False) -> str:
    matches = result.matches[:limit]
    if as_json:
        body = {"request_id": result.request_id, "matches": [m.as_dict() for m in matches]}
        return json.dumps(body, indent=None if compact else 2, ensure_ascii=False)
    if not matches:
        return f"No alert for {name!r}. (record {result.request_id})"
    script, threshold = matches[0].explanation.script, matches[0].explanation.threshold
    lines = [
        f"{len(matches)} alert(s) for {name!r} (script {script}, threshold {threshold}, record {result.request_id})"
    ]
    lines += [format_match(i, m) for i, m in enumerate(matches, 1)]
    return "\n".join(lines)


def read_batch(source: str) -> list[dict]:
    """Parse and check every line of a batch file before anything is screened."""
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    items = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"line {number}: not valid JSON ({exc})") from exc
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise ValueError(f'line {number}: needs an object with a non-empty string "name"')
        for key in ("birth_date", "country", "customer_ref"):
            if item.get(key) is not None and not isinstance(item[key], str):
                raise ValueError(f"line {number}: {key!r} must be a string")
        unknown = set(item) - {"name", "birth_date", "country", "customer_ref"}
        if unknown:
            raise ValueError(f"line {number}: unknown keys {sorted(unknown)}")
        items.append(item)
    if not items:
        raise ValueError("the batch file has no names")
    return items


def init_main(argv: list[str]) -> int:
    """Create the audit store: the directory, an empty header log and the genesis checkpoint."""
    parser = argparse.ArgumentParser(prog="screen_cli.py init", description="create a new, empty audit store")
    parser.add_argument("--audit-dir", help="audit directory (or set $SCREENING_AUDIT_DIR); there is no default")
    args = parser.parse_args(argv)
    try:
        directory = resolve_audit_dir(args.audit_dir)
    except AuditConfigError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        AuditLog.create_new(directory)
    except AuditError as exc:
        print(f"not initialised: {exc}", file=sys.stderr)
        return EXIT_AUDIT_FAILURE
    print(f"initialised an empty audit store at {directory} (genesis checkpoint written)")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "init":
        return init_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", help="the name to screen (not with --batch)")
    parser.add_argument("--batch", metavar="FILE", help='JSON-lines file of names to screen, or "-" for stdin')
    parser.add_argument("--birth-date", help="YYYY, YYYY-MM or YYYY-MM-DD")
    parser.add_argument("--country", help="ISO 3166 two-letter country code, for example ru")
    parser.add_argument("--customer-ref", help="opaque customer reference, stored in the payload only")
    parser.add_argument("--limit", type=int, help="show at most this many matches per name (all are recorded)")
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON (one line per name with --batch)"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="entities file to screen against")
    parser.add_argument("--audit-dir", help="audit directory (or set $SCREENING_AUDIT_DIR); there is no default")
    args = parser.parse_args(argv)

    if bool(args.name) == bool(args.batch):
        parser.error("give a name, or --batch FILE, but not both")
    if args.batch and (args.birth_date or args.country or args.customer_ref):
        parser.error("--birth-date, --country and --customer-ref belong in the batch file lines, not on the command")

    try:
        audit_dir = resolve_audit_dir(args.audit_dir)
    except AuditConfigError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        items = (
            read_batch(args.batch)
            if args.batch
            else [{"name": args.name, "birth_date": args.birth_date, "country": args.country,
                   "customer_ref": args.customer_ref}]
        )  # fmt: skip
    except (ValueError, OSError) as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print("Building the list...", file=sys.stderr)
    try:
        engine = AuditedEngine.from_file(args.input, audit_dir)
    except LogMissingError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_AUDIT_FAILURE
    except AuditError as exc:
        print(f"refusing to start: the audit store is not usable: {exc}", file=sys.stderr)
        return EXIT_AUDIT_FAILURE

    for done, item in enumerate(items):
        try:
            result = engine.audited_screen(
                item["name"], item.get("birth_date"), item.get("country"), item.get("customer_ref")
            )
        except AuditError as exc:
            print(f"audit record could not be written, no decision produced: {exc}", file=sys.stderr)
            if args.batch:
                print(
                    f"batch stopped: {done} of {len(items)} names were screened and recorded before the failure; "
                    f"{len(items) - done} were not screened",
                    file=sys.stderr,
                )
            return EXIT_AUDIT_FAILURE
        print(
            render(item["name"], result, args.limit, args.json, compact=bool(args.batch)), flush=True
        )  # only after the record exists
    return 0


if __name__ == "__main__":
    sys.exit(main())

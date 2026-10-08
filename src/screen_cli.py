"""Command line screening, audited. Production entry point: every decision is written to the audit log first.

Every mode is an explicit subcommand. A bare name is not a command (exit status 2), so a customer called "init",
"replay" or "screen" is screened like anyone else.

Run:
    python src/screen_cli.py init --audit-dir /var/lib/screening/audit          (once, to create the audit store)
    python src/screen_cli.py screen "Vladimir Putin" --birth-date 1952-10-07 --country ru --audit-dir /var/lib/audit
    python src/screen_cli.py screen "محمد علي" --json --limit 5 --customer-ref cust-123
    python src/screen_cli.py screen --batch names.jsonl --audit-dir /var/lib/screening/audit
    python src/screen_cli.py replay REQUEST_ID --audit-dir /var/lib/screening/audit
    python src/screen_cli.py recover --audit-dir /var/lib/screening/audit                 (operator, after a crash)
    python src/screen_cli.py purge --mapping ended.jsonl --dry-run --audit-dir /var/lib/screening/audit
    python src/screen_cli.py metrics --from 0 --to 999 --audit-dir /var/lib/screening/audit
    python src/screen_cli.py drift --report /var/lib/screening/audit/monitoring/0-999.json
    python src/screen_cli.py gate --new-snapshot data/snapshots/<version>/entities.ftm.json

The audit directory comes from --audit-dir or $SCREENING_AUDIT_DIR and has no default. With neither set the command
refuses to start (exit status 2) before it builds the list. Screening never creates the store: "init" does, once, and
refuses if the directory already exists and is not empty. If the store is missing or wiped, screening exits 1,
prints no decision, and says log_missing and how to run init. A wiped trail and a new deployment look the same from
inside, so a person decides which it is. The record of each name (header and payload, see
src/audit.py) is written before the decision is printed. If a record cannot be written the command prints no
decision for that name, says so on stderr, and exits non-zero (exit status 1).

`screen --batch` reads a JSON-lines file, or standard input for "-": one object per line with "name" and optionally
"birth_date", "country" and "customer_ref". The whole file is checked before anything is screened, so a malformed
line stops the run with exit status 2 and no records. Names are then screened in order, one record each. If a record
cannot be written the run stops there: earlier names keep their records and decisions, later ones are not screened.

`replay REQUEST_ID` rebuilds one recorded decision from its header, payload and run context and exits 0 only if the
result is identical. It refuses, with a field-level diff, if the engine's run context differs from the record's, and
it writes nothing. It prints no names.

`recover` is for one situation: a crash during a write left a partial last line and/or one payload with no header, and
the log refuses to append. It moves the torn bytes to quarantine/, cuts the log back to its last complete line,
deletes the orphan payload (no decision was made, so there is no basis to keep it), and appends one recovery record
holding hashes and sizes only. It refuses, changing nothing, for a clean log, more than one orphan, or any sign of
tampering.

`purge --mapping FILE [--dry-run]` deletes the payloads whose retention period is over. The mapping is JSON lines of
{"customer_ref": ..., "relationship_end": "YYYY-MM-DD"} from the caller, because only the caller knows when a
relationship ended. A purge record (request IDs only, no customer_ref, no dates) is chained before any file is deleted.
Records with no customer_ref, a customer_ref not in the mapping, or a date after their customer's end date are never
deleted. `--hold FILE` (JSON lines, each a "customer_ref" or a "request_id") puts records under legal hold: never
purged, whatever their end date, checked first and shown by a dry run. Deleting records relevant to an investigation is
an offence (MLR 2017 regulation 87) and the hold list is the caller's responsibility. This deletes files only;
backups and snapshots are outside its reach.

`metrics --from SEQ --to SEQ` writes an aggregate report for a window of records to <audit_dir>/monitoring/: per run
context, the alert rate, a fixed 20-bin score histogram, near-threshold counts, per-script rates (small cells
suppressed) and duration percentiles. It holds no request ID, name or customer reference.

`drift --report FILE` compares a metrics report with baseline/<context_hash>.json (src/make_baseline.py), PSI on scores
and on the script mix, per-script alert rates, the near-threshold rate. Exit status: 0 ok, 3 insufficient data, 10 warn,
20 alert, 2 refused (a baseline only applies to the run context it was measured under).

`gate --new-snapshot PATH` is the list-update regression gate (src/list_gate.py): it runs the spent hold-out on the new
snapshot and on the current one, resolves each positive's entity through referents, and fails if the F1 loss, or the
increase in false alerts on Arabic names, is beyond tolerance with 95% confidence. It uses screen() on the research
path and writes no audit records. Exit status: 0 pass, 1 fail, 2 refused.

`screen --customer-ref` links the record to a customer (opaque to the matcher; stored in the payload only). Exit
status is 0 whether or not there are matches. Building the list takes a few seconds on every run.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from audit import AuditConfigError, AuditError, AuditLog, LogMissingError
from audit_drift import BASELINE_DIR, BaselineMismatch, compare_to_baseline, load_baseline
from audit_metrics import compute_metrics
from audit_purge import PurgeRefused, purge
from audit_recover import RecoveryRefused, recover
from audited_screening import AuditedEngine, AuditedResult, resolve_audit_dir
from list_gate import SnapshotRefused, default_report_path, run_gate, summary_lines, write_report
from profile_data import DEFAULT_INPUT
from replay import replay
from run_context import load_context
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


def cmd_init(args: argparse.Namespace) -> int:
    """Create the audit store: the directory, an empty header log and the genesis checkpoint."""
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


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay one recorded decision. Exit 0 if the replay is identical, 1 for anything else."""
    try:
        audit_dir = resolve_audit_dir(args.audit_dir)
    except AuditConfigError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print("Building the list...", file=sys.stderr)
    try:
        engine = AuditedEngine.from_file(args.input, audit_dir, for_screening=False)
    except AuditError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_AUDIT_FAILURE
    result = replay(engine, args.request_id)
    print(f"{result.status}: {result.reason}")
    for line in result.differences:
        print(f"  {line}")
    if result.reproduce:
        r = result.reproduce
        dirty = (
            " (the tree was dirty at record time, so git alone may not recover the exact source)"
            if r["git_dirty_at_record_time"]
            else ""
        )
        print(
            f"  to reproduce: check out git {r['git_sha']}{dirty}, restore snapshot {r['snapshot_id']} "
            f"(sha256 {r['snapshot_sha256'][:16]}...), rebuild, then replay"
        )
    return 0 if result.identical else 1


def read_mapping(source: str) -> dict[str, str]:
    """A JSON-lines file of {"customer_ref": ..., "relationship_end": "YYYY-MM-DD"}, checked line by line."""
    mapping: dict[str, str] = {}
    for number, line in enumerate(Path(source).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"line {number}: not valid JSON ({exc})") from exc
        if not isinstance(item, dict) or set(item) != {"customer_ref", "relationship_end"}:
            raise ValueError(f'line {number}: needs exactly "customer_ref" and "relationship_end"')
        ref, end = item["customer_ref"], item["relationship_end"]
        if not isinstance(ref, str) or not ref or not isinstance(end, str):
            raise ValueError(f"line {number}: customer_ref and relationship_end must be strings")
        if ref in mapping:
            raise ValueError(f"line {number}: customer_ref appears twice")
        try:
            date.fromisoformat(end)
        except ValueError as exc:
            raise ValueError(f"line {number}: relationship_end must be YYYY-MM-DD") from exc
        mapping[ref] = end
    if not mapping:
        raise ValueError("the mapping file has no customers")
    return mapping


def read_holds(source: str) -> tuple[set[str], set[str]]:
    """A JSON-lines file of legal holds: each line has either a "customer_ref" or a "request_id"."""
    refs: set[str] = set()
    ids: set[str] = set()
    for number, line in enumerate(Path(source).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"hold file line {number}: not valid JSON ({exc})") from exc
        if not isinstance(item, dict) or len(item) != 1 or next(iter(item)) not in ("customer_ref", "request_id"):
            raise ValueError(f'hold file line {number}: needs exactly one of "customer_ref" or "request_id"')
        key, value = next(iter(item.items()))
        if not isinstance(value, str) or not value:
            raise ValueError(f"hold file line {number}: {key} must be a non-empty string")
        if key == "request_id":
            if len(value) != 32 or set(value) - set("0123456789abcdef"):
                raise ValueError(f"hold file line {number}: request_id must be 32 lowercase hex characters")
            ids.add(value)
        else:
            refs.add(value)
    if not refs and not ids:
        raise ValueError("the hold file has no holds")
    return refs, ids


def cmd_metrics(args: argparse.Namespace) -> int:
    """Write the aggregate monitoring report for a window of records (by seq). Prints the path and counts only."""
    try:
        directory = resolve_audit_dir(args.audit_dir)
    except AuditConfigError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        report, path = compute_metrics(directory, args.from_seq, args.to_seq)
    except ValueError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except AuditError as exc:
        print(f"metrics failed: {exc}", file=sys.stderr)
        return EXIT_AUDIT_FAILURE
    w = report["window"]
    print(f"wrote {path}")
    print(
        f"window {w['from_seq']}-{w['to_seq']}: {w['decisions']} decision(s) of {w['records']} record(s), "
        f"{len(report['groups'])} run context group(s)"
    )
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """The list-update regression gate. Exit 0 pass, 1 fail, 2 refused. Uses the research path (no audit records)."""
    from list_gate import CURRENT_SNAPSHOT, HOLDOUT

    try:
        report = run_gate(
            Path(args.old_snapshot) if args.old_snapshot else CURRENT_SNAPSHOT,
            Path(args.new_snapshot),
            Path(args.holdout) if args.holdout else HOLDOUT,
        )
        out = Path(args.out) if args.out else default_report_path(report)
        path = write_report(report, out)
    except (SnapshotRefused, OSError, ValueError) as exc:
        print(f"refusing to run the gate: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print("\n".join(summary_lines(report)))
    print(f"wrote {path}")
    return 0 if report["gate"]["passed"] else 1


EXIT_DRIFT = {"ok": 0, "insufficient_data": 3, "warn": 10, "alert": 20}


def describe_drift(result: dict) -> list[str]:
    """Plain lines for a drift result: numbers and statuses, no names."""

    def num(value: object) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        f"decision context {result['decision_context_hash'][:12]}...: overall {result['overall']} "
        f"({result.get('count_live', '?')} live, {result.get('count_baseline', '?')} baseline decisions)"
    ]
    if "reason" in result:
        return [*lines, f"  {result['reason']}"]

    def psi_line(label: str, item: dict) -> str:
        if item["value"] is None:
            return f"  {label}: n/a [{item['status']}] {item.get('reason', '')}"
        return (
            f"  {label}: {num(item['value'])} [{item['status']}] (n {item['n']}; limits warn "
            f"{num(item['warn_limit'])}, alert {num(item['alert_limit'])}; sampling null p95 "
            f"{num(item['null_p95'])}, p99 {num(item['null_p99'])})"
        )

    for name, key in (("score PSI", "score_psi"), ("input-mix PSI", "input_mix_psi")):
        lines.append(psi_line(name, result[key]))
    for script, entry in result["per_script"].items():
        a = entry["alert_rate"]
        lines.append(psi_line(f"{script} score PSI", entry["score_psi"]))
        lines.append(
            f"  {script} alert rate {a['baseline']:.3f} -> {a['live']:.3f} "
            f"(diff {a['abs_diff']:+.3f}, ratio {num(a['ratio'])}) [{a['status']}]"
        )
    n = result["near_threshold"]
    lines.append(
        f"  near-threshold rate {n['baseline_rate']:.3f} -> {n['live_rate']:.3f} "
        f"(diff {n['abs_diff']:+.3f}) [{n['status']}]"
    )
    for script, why in result["skipped_scripts"].items():
        lines.append(f"  {script}: not compared ({why})")
    d = result["duration_ms"]
    if d["baseline"] and d["live"]:
        lines.append(
            f"  duration_ms p50/p95/p99 (informational, machine-dependent, not compared): baseline "
            f"{d['baseline']['p50']}/{d['baseline']['p95']}/{d['baseline']['p99']}, live "
            f"{d['live']['p50']}/{d['live']['p95']}/{d['live']['p99']}"
        )
    return lines


def cmd_drift(args: argparse.Namespace) -> int:
    """Compare a monitoring report with the baseline for its run context. Exit 0 ok, 3 insufficient data, 10 warn,
    20 alert, 2 refused (no baseline for the report's context, or a bad file)."""
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        groups = [g["run_context_hash"] for g in report["groups"]]
        decision_hashes = [g.get("decision_context_hash") for g in report["groups"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"refusing to start: {args.report} is not a monitoring report ({exc})", file=sys.stderr)
        return EXIT_USAGE
    directory = Path(args.baseline_dir) if args.baseline_dir else BASELINE_DIR
    try:
        candidates = (
            [Path(args.baseline)]
            if args.baseline
            else [directory / f"{h}.json" for h in dict.fromkeys(decision_hashes) if h]
        )
        baselines = [load_baseline(p) for p in candidates if p.exists()]
        live_contexts = {}
        if args.audit_dir:
            for h in groups:
                stored = load_context(Path(args.audit_dir) / "contexts", h)
                if stored:
                    live_contexts[h] = stored.as_dict()
    except (OSError, ValueError, AuditError) as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not baselines:
        shown = ", ".join(str(h)[:12] + "..." for h in decision_hashes) or "none"
        print(
            f"refusing to compare: no baseline for this report's decision context(s) ({shown}) in {directory}. A "
            "baseline only applies to the decision context it was measured under: make one with src/make_baseline.py.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    worst_exit, shown_results = 0, []
    for baseline in baselines:
        try:
            result = compare_to_baseline(report, baseline, live_contexts=live_contexts)
        except BaselineMismatch as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_USAGE
        shown_results.append(result)
        worst_exit = max(worst_exit, EXIT_DRIFT[result["overall"]])
    if args.json:
        print(json.dumps(shown_results, indent=2, sort_keys=True))
    else:
        print("\n".join(line for r in shown_results for line in describe_drift(r)))
    return worst_exit


def cmd_purge(args: argparse.Namespace) -> int:
    """Delete payloads whose retention period (relationship end plus N years) is over. Exit 0 on success."""
    try:
        directory = resolve_audit_dir(args.audit_dir)
        mapping = read_mapping(args.mapping)
        hold_refs, hold_ids = read_holds(args.hold) if args.hold else (set(), set())
    except AuditConfigError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (ValueError, OSError) as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        result = purge(
            AuditLog.in_directory(directory),
            mapping,
            retention_years=args.retention_years,
            dry_run=args.dry_run,
            hold_customer_refs=hold_refs,
            hold_request_ids=hold_ids,
        )
    except ValueError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except PurgeRefused as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_AUDIT_FAILURE
    except AuditError as exc:
        print(f"purge failed: {exc}", file=sys.stderr)
        return EXIT_AUDIT_FAILURE
    verb = "would purge" if result.dry_run else "purged"
    print(
        f"{verb} {result.eligible} payload(s)"
        + ("" if result.dry_run else f" (purge record {result.record['request_id']})" if result.record else "")
    )
    if result.completed:
        print(
            f"{'would finish' if result.dry_run else 'finished'} {result.completed} deletion(s) "
            "left over from an earlier purge"
        )
    print(
        f"skipped: {result.skipped_no_customer_ref} with no customer_ref, "
        f"{result.skipped_not_in_mapping} not in the mapping (live customers), "
        f"{result.skipped_later_relationship} dated after their customer's end date (a later relationship), "
        f"{result.skipped_retention_not_elapsed} whose retention period is not over"
    )
    print(f"held (under legal hold, never purged): {result.held}")
    if result.hold_already_purged:
        print(
            f"warning: {result.hold_already_purged} held request(s) had already been purged; "
            "a hold cannot bring a deleted payload back"
        )
    if not result.dry_run:
        print("note: this deleted files only; copies in backups and snapshots are not touched")
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    """Recover from a crash during a write. Exit 0 if recovered, 1 if refused or failed."""
    try:
        directory = resolve_audit_dir(args.audit_dir)
    except AuditConfigError as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        outcome = recover(AuditLog.in_directory(directory))
    except RecoveryRefused as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_AUDIT_FAILURE
    except AuditError as exc:
        print(f"recovery failed: {exc}", file=sys.stderr)
        return EXIT_AUDIT_FAILURE
    parts = []
    if outcome.quarantine_file:
        parts.append(
            f"{outcome.quarantine_size} torn bytes moved to quarantine/{outcome.quarantine_file} "
            f"(sha256 {outcome.quarantine_sha256[:16]}...)"
        )
    if outcome.orphan_request_id:
        parts.append(
            f"orphan payload {outcome.orphan_request_id} deleted (file sha256 {outcome.orphan_payload_sha256[:16]}...)"
        )
    note = "" if outcome.record else " (the recovery record was already in the chain; only the deletion was finished)"
    print("recovered: " + "; ".join(parts) + note)
    return 0


def cmd_screen(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
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
        print(render(item["name"], result, args.limit, args.json, compact=bool(args.batch)), flush=True)
    return 0


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(prog="screen_cli.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="{screen,init,replay,recover,purge,metrics,drift,gate}"
    )
    audit_dir_help = "audit directory (or set $SCREENING_AUDIT_DIR); there is no default"

    screen_cmd = commands.add_parser("screen", help="screen a name, or a batch of names")
    screen_cmd.add_argument("name", nargs="?", help="the name to screen (not with --batch)")
    screen_cmd.add_argument("--batch", metavar="FILE", help='JSON-lines file of names to screen, or "-" for stdin')
    screen_cmd.add_argument("--birth-date", help="YYYY, YYYY-MM or YYYY-MM-DD")
    screen_cmd.add_argument("--country", help="ISO 3166 two-letter country code, for example ru")
    screen_cmd.add_argument("--customer-ref", help="opaque customer reference, stored in the payload only")
    screen_cmd.add_argument("--limit", type=int, help="show at most this many matches per name (all are recorded)")
    screen_cmd.add_argument(
        "--json", action="store_true", help="print machine-readable JSON (one line per name with --batch)"
    )
    screen_cmd.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="entities file to screen against")
    screen_cmd.add_argument("--audit-dir", help=audit_dir_help)

    init = commands.add_parser("init", help="create a new, empty audit store")
    init.add_argument("--audit-dir", help=audit_dir_help)

    rep = commands.add_parser("replay", help="replay a recorded decision")
    rep.add_argument("request_id", help="the request ID printed when the name was screened")
    rep.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="entities file the engine is built from")
    rep.add_argument("--audit-dir", help=audit_dir_help)
    rec = commands.add_parser(
        "recover", help="after a crash during a write: quarantine a torn final line and/or delete one orphan payload"
    )
    rec.add_argument("--audit-dir", help=audit_dir_help)

    pur = commands.add_parser(
        "purge", help="delete payloads whose retention period (relationship end + 5 years) is over"
    )
    pur.add_argument("--mapping", required=True, metavar="FILE", help="JSON lines: customer_ref and relationship_end")
    pur.add_argument("--dry-run", action="store_true", help="report what would be purged; change nothing")
    pur.add_argument(
        "--hold",
        metavar="FILE",
        help='JSON lines of legal holds, each a "customer_ref" or a "request_id"; held records are never purged',
    )
    pur.add_argument("--retention-years", type=int, default=5, help="retention period in years (at least 5)")
    pur.add_argument("--audit-dir", help=audit_dir_help)

    met = commands.add_parser("metrics", help="write an aggregate monitoring report for a window of records")
    met.add_argument("--from", dest="from_seq", type=int, required=True, metavar="SEQ", help="first record (its seq)")
    met.add_argument("--to", dest="to_seq", type=int, required=True, metavar="SEQ", help="last record, inclusive")
    met.add_argument("--audit-dir", help=audit_dir_help)

    drift = commands.add_parser("drift", help="compare a monitoring report with the baseline for its run context")
    drift.add_argument("--report", required=True, metavar="FILE", help="a monitoring report (from the metrics command)")
    drift.add_argument("--baseline", metavar="FILE", help="a baseline file (default: baseline/<context_hash>.json)")
    drift.add_argument("--baseline-dir", metavar="DIR", help="where to look for baseline/<context_hash>.json")
    drift.add_argument("--audit-dir", help="optional: lets a refusal say which run context fields differ")
    drift.add_argument("--json", action="store_true", help="print the full result as JSON")

    gate = commands.add_parser("gate", help="list-update regression gate: compare the hold-out on a newer snapshot")
    gate.add_argument("--new-snapshot", required=True, metavar="PATH", help="entities.ftm.json of the snapshot to try")
    gate.add_argument("--old-snapshot", metavar="PATH", help="the current snapshot (default: data/sanctions)")
    gate.add_argument("--holdout", metavar="FILE", help="hold-out file (default: screening_holdout_v3.json)")
    gate.add_argument("--out", metavar="FILE", help="where to write the aggregate report (default: reports/)")
    return parser, screen_cmd


def main(argv: list[str] | None = None) -> int:
    parser, screen_parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "init":
        return cmd_init(args)
    if args.command == "replay":
        return cmd_replay(args)
    if args.command == "recover":
        return cmd_recover(args)
    if args.command == "purge":
        return cmd_purge(args)
    if args.command == "metrics":
        return cmd_metrics(args)
    if args.command == "drift":
        return cmd_drift(args)
    if args.command == "gate":
        return cmd_gate(args)
    return cmd_screen(args, screen_parser)


if __name__ == "__main__":
    sys.exit(main())

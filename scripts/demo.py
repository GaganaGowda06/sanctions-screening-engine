"""Terminal demo of the whole audit pipeline, against a throwaway store.

It builds the list once, then: creates a new audit store in a temporary directory, screens a name that is on the
list and a clean name, prints both audit records, replays the alert, runs the retention purge, and verifies the chain.
The store is deleted at the end. It never touches the real audit store and does not change the engine.

Run:
    venv/bin/python scripts/demo.py
    venv/bin/python scripts/demo.py --input path/to/entities.ftm.json     (a list with a VERSION file beside it)

Exit status 0 only if every step succeeded.
"""

import argparse
import json
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit import AuditLog  # noqa: E402
from audit_purge import add_years, purge  # noqa: E402
from audit_schema import decision_summary  # noqa: E402
from audit_verify import verify_chain  # noqa: E402
from audited_screening import AuditedEngine  # noqa: E402
from entities import load_named_entities  # noqa: E402
from profile_data import DEFAULT_INPUT  # noqa: E402
from replay import replay  # noqa: E402
from screening import query_script  # noqa: E402

CLEAN_NAME = "Margaret Thornbury"
CLEAN_BIRTH_DATE = "1975-03-12"
CLEAN_COUNTRY = "gb"
ON_LIST_REF = "demo-customer-001"
CLEAN_REF = "demo-customer-002"
MAX_TRIES = 20000
VOWELS = set("aeiouy")


class DemoFailure(Exception):
    pass


def heading(number: int, text: str) -> None:
    print(f"\n=== Step {number}: {text} ===", flush=True)


def check(ok: bool, message: str) -> None:
    print(f"  [{'ok' if ok else 'FAILED'}] {message}", flush=True)
    if not ok:
        raise DemoFailure(message)


def variants(name: str) -> list[tuple[str, str]]:
    """Realistic variants of a name, in the order they are tried: a misspelling of one letter in the longest word (a
    consonant dropped, then a consonant doubled, middle letter first and then outwards), then, as the fallback, the
    tokens reordered."""
    tokens = name.replace(",", " ").split()
    longest = max(range(len(tokens)), key=lambda i: (len(tokens[i]), -i))
    word = tokens[longest]
    middle = len(word) // 2
    spots = sorted((i for i in range(1, len(word)) if word[i].lower() not in VOWELS and word[i].isalpha()),
                   key=lambda i: (abs(i - middle), i))  # fmt: skip

    def with_word(new: str) -> str:
        return " ".join(tokens[:longest] + [new] + tokens[longest + 1 :])

    found = [
        (with_word(word[:i] + word[i + 1 :]), f"one consonant dropped ({word[i]!r} removed from {word!r})")
        for i in spots
    ]
    found += [
        (with_word(word[:i] + word[i] + word[i:]), f"one consonant doubled ({word[i]!r} in {word!r})") for i in spots
    ]
    reordered = " ".join(reversed(tokens))
    if reordered != name:
        found.append((reordered, "token reorder (fallback)"))
    return found


def alerts_on(engine: AuditedEngine, name: str, entity: dict, birth_date: str, country: str) -> float | None:
    """The score of the listed entity if the engine alerts on it as the best match for this name, else None."""
    matches = engine.evaluate(name, birth_date, country).matches
    return matches[0].score if matches and matches[0].entity_id == entity["id"] else None


def pick_list_name(engine: AuditedEngine, entities: list[dict]) -> dict:
    """The first sanctions target (by entity id) that has a Latin-script name of two or more words, a full birth date
    and a country, that the engine alerts on when its own name is screened, and for which at least one variant
    (see variants) also alerts with a score below 1.0, so the demo shows a fuzzy match. The first such variant is
    used; if none qualifies, the next entity is tried."""
    for entity in sorted((e for e in entities if e["target"] and e["schema"] == "Person"), key=lambda e: e["id"])[
        :MAX_TRIES
    ]:
        name = entity["names"][0]
        dates = [d for d in entity["birth_dates"] if len(d) == 10]
        if query_script(name) != "Latin" or len(name.split()) < 2 or not dates or not entity["countries"]:
            continue
        birth_date, country = dates[0], entity["countries"][0]
        if alerts_on(engine, name, entity, birth_date, country) is None:
            continue
        for variant, how in variants(name):
            score = alerts_on(engine, variant, entity, birth_date, country)
            if score is not None and score < 1.0:
                return {
                    "entity": entity["id"], "name": name, "birth_date": birth_date, "country": country,
                    "variant": variant, "how": how,
                }  # fmt: skip
    raise DemoFailure("no suitable listed person found in this list")


def show(value: object, indent: int = 4) -> None:
    pad = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                print(f"{pad}{key}:")
                show(item, indent + 2)
            else:
                print(f"{pad}{key}: {json.dumps(item, ensure_ascii=False)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                print(f"{pad}-", ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in item.items()))
            else:
                print(f"{pad}- {json.dumps(item, ensure_ascii=False)}")


def show_record(engine: AuditedEngine, request_id: str, label: str) -> tuple[dict, dict]:
    header = next(h for h in engine.log.headers.read_all() if h["request_id"] == request_id)
    payload = engine.log.payloads.read(request_id)
    print(f"  {label}")
    print("  HEADER (no names or personal data; this part is kept):")
    shown = dict(header)
    shown["candidates"] = header["candidates"][:3]
    show(shown)
    if len(header["candidates"]) > 3:
        print(f"    (candidates: {len(header['candidates'])} kept in the header, first 3 shown)")
    print("  PAYLOAD (raw input and name-derived data; this part is purged at the end of retention):")
    data = dict(payload.data)
    keys = data.pop("retrieval_keys")
    data["retrieval_keys"] = f"{len(keys)} keys"
    show(data)
    return header, payload.data


def run(input_path: Path) -> dict:
    started = time.perf_counter()
    summary: dict = {}
    entities = load_named_entities(input_path)
    with tempfile.TemporaryDirectory(prefix="screening-demo-") as tmp:
        audit_dir = Path(tmp) / "audit"
        print(f"Throwaway audit store: {audit_dir} (deleted when the demo ends; the real store is not used)")

        heading(1, "Create a new, empty audit store")
        log = AuditLog.create_new(audit_dir)
        check(
            log.headers.path.exists() and verify_chain(log).ok,
            "store created, genesis checkpoint written, chain verifies",
        )

        print("\nBuilding the list from the current snapshot (a few seconds)...", flush=True)
        # The purge rule never deletes a record dated after its customer's relationship ended, so a record made today
        # cannot be shown purged. The demo clock is set to 7 years ago for this throwaway store only.
        then = datetime.combine(add_years(date.today(), -7), dtime(12, 0), tzinfo=timezone.utc)
        print(
            f"  Audit clock simulated to {then.date()} so the five-year retention purge can be shown; "
            "real deployments never backdate."
        )
        engine = AuditedEngine.from_file(input_path, audit_dir, clock=lambda: then)
        listed = pick_list_name(engine, entities)
        end = add_years(date.today(), -6).replace(day=1)  # relationship ended 6 years ago: over five

        heading(2, "Screen a name that is on the list")
        print(f"  Query: {listed['name']!r}, born {listed['birth_date']}, country {listed['country']}")
        print(f"  Customer reference {ON_LIST_REF}; relationship ended {end} (more than five years ago)")
        hit = engine.audited_screen(listed["name"], listed["birth_date"], listed["country"], ON_LIST_REF)
        top = hit.matches[0]
        print(f"  Decision: ALERT, {len(hit.matches)} match(es), best {top.entity_id} score {top.score:.4f}")
        check(top.entity_id == listed["entity"], f"the alert is on the listed entity {listed['entity']}")

        heading(3, "Screen a misspelt variant of the same listed name")
        print(f"  Variant: {listed['variant']!r} (from {listed['name']!r}) -- edit applied: {listed['how']}")
        print(f"  Same birth date and country, same customer reference {ON_LIST_REF}")
        variant = engine.audited_screen(listed["variant"], listed["birth_date"], listed["country"], ON_LIST_REF)
        vtop = variant.matches[0]
        print(f"  Decision: ALERT, best {vtop.entity_id} score {vtop.score:.4f}")
        print(f"  Score: exact name {top.score:.4f}, variant {vtop.score:.4f}")
        check(
            vtop.entity_id == listed["entity"] and vtop.score < top.score,
            "the variant alerts on the same listed entity with a lower score",
        )

        heading(4, "Screen a clean name that should not alert")
        print(f"  Query: {CLEAN_NAME!r}, born {CLEAN_BIRTH_DATE}, country {CLEAN_COUNTRY}")
        print(f"  Customer reference {CLEAN_REF}; still a live customer (no end date)")
        clear = engine.audited_screen(CLEAN_NAME, CLEAN_BIRTH_DATE, CLEAN_COUNTRY, CLEAN_REF)
        print(f"  Decision: {'ALERT' if clear.matches else 'CLEAR'}")
        check(not clear.matches, "no alert on the clean name")

        heading(5, "Print both audit records")
        header_hit, data_hit = show_record(
            engine, hit.request_id, f"Record for the listed name, request {hit.request_id}"
        )
        print()
        header_clear, _ = show_record(
            engine, clear.request_id, f"Record for the clean name, request {clear.request_id}"
        )
        check(
            header_hit["decision"] == "alert" and header_clear["decision"] == "clear",
            "header decisions are alert and clear",
        )

        heading(6, "Replay the alert from its record")
        result = replay(engine, hit.request_id)
        raw = data_hit["raw_input"]
        again = engine.evaluate(raw["name"], raw["birth_date"], raw["country"])
        expected = decision_summary(again.entries, again.threshold, engine.top_candidates)
        print(
            f"  Recorded: score {header_hit['final_score']}, threshold {header_hit['threshold']}, "
            f"decision {header_hit['decision']}"
        )
        print(
            f"  Replayed: score {expected['final_score']}, threshold {expected['threshold']}, "
            f"decision {expected['decision']}"
        )
        print(f"  Replay status: {result.status} ({result.reason})")
        check(result.identical, "replay is identical to the record")
        check(
            (expected["final_score"], expected["threshold"], expected["decision"])
            == (header_hit["final_score"], header_hit["threshold"], header_hit["decision"]),
            "same score, same threshold, same decision",
        )

        heading(7, "Run the retention purge")
        payload_path = engine.log.payloads.path_for(hit.request_id)
        clear_path = engine.log.payloads.path_for(clear.request_id)
        print(
            f"  Retention: relationship end plus 5 years. Mapping: {ON_LIST_REF} ended {end}; "
            f"{CLEAN_REF} is not in it (live)."
        )
        variant_path = engine.log.payloads.path_for(variant.request_id)
        check(payload_path.exists() and variant_path.exists(), "before: both listed-name payload files exist")
        outcome = purge(log, {ON_LIST_REF: end})
        print(f"  Purged {outcome.eligible} payload(s); skipped {outcome.skipped_not_in_mapping} live customer(s)")
        check(
            outcome.eligible == 2 and set(outcome.eligible_ids) == {hit.request_id, variant.request_id},
            "exactly the two records of the ended relationship were purged",
        )
        check(not payload_path.exists() and not variant_path.exists(), "after: both payload files are gone")
        check(clear_path.exists(), "the live customer's payload is untouched")
        tombstone = outcome.record
        print("  Tombstone appended to the chained log (request IDs only, no names, no customer reference):")
        show(tombstone)
        check(tombstone is not None and tombstone["kind"] == "purge", "a purge tombstone was written")
        header_after = next(h for h in engine.log.headers.read_all() if h["request_id"] == hit.request_id)
        check(header_after == header_hit, "the decision header is unchanged and still in the log")
        gone = replay(engine, hit.request_id)
        print(f"  Replay after purge: {gone.status} ({gone.reason})")
        check(gone.status == "purged", "replay now reports the payload as purged by design")

        heading(8, "Verify the hash chain")
        verified = verify_chain(log)
        print(
            f"  Records checked: {verified.records}; payloads checked: {verified.payloads_checked}; "
            f"payloads gone under a tombstone: {verified.payloads_purged}; failures: {len(verified.failures)}"
        )
        check(verified.ok, "the chain still verifies after the purge")
        summary = {"steps": 8, "records": verified.records, "seconds": time.perf_counter() - started}
    print(f"\nAll 8 steps succeeded in {summary['seconds']:.1f} s. Throwaway store deleted.")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="entities file (default: the current snapshot)"
    )
    args = parser.parse_args(argv)
    try:
        run(args.input)
    except DemoFailure as exc:
        print(f"\nDEMO FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

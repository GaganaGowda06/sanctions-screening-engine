"""Replay: rebuild a recorded decision from its header, payload and run context, and say whether it is identical.

    result = replay(engine, request_id)
    result.identical            True only if the recomputed decision matches the record exactly
    result.status               identical | different | refused_context | refused_tampered | not_replayable |
                                unknown_request
    result.differences          field-level differences, one line each (no names, no payload content)
    result.reproduce            for refused_context: the git SHA and snapshot the record was made under

What it does, in order:
  1. finds the header (and checks it still matches its own hash) and loads the payload;
  2. a missing payload is "not replayable", classified as verify_chain classifies it;
  3. the payload must match the digest in the header (salt plus canonical JSON), or the record is refused as tampered;
  4. the record's run context hash must equal the live engine's. If it does not, replay refuses and prints a
     field-level diff between the stored context file and the live context, naming the git SHA and snapshot needed.
     It never produces an answer under a different context;
  5. with the same context, the inputs from the payload go through the engine and the candidate IDs, per-feature
     scores, final score, threshold, decision and matched IDs must be identical, as must the name-derived payload
     fields (script, normalised forms, tokens, retrieval keys, match explanations). Scores are compared exactly.

Replay writes nothing: it reads the header log and the payload store, never calls the logging path, and the engine it
runs on should be built with record_context=False so it does not even write a context file.

Replay of a purged record is impossible by design: the raw input is gone.

Reproducing a record made under an older context is a manual procedure (the project notes): check out the recorded SHA,
restore the recorded snapshot, rebuild, then replay.
"""

from dataclasses import dataclass, field

from audit import AuditError, compute_record_hash
from audit_schema import DECISION_FIELDS, decision_summary
from audit_verify import classify_missing_payload
from audited_screening import AuditedEngine
from run_context import diff_contexts, load_context

IDENTICAL = "identical"
DIFFERENT = "different"
REFUSED_CONTEXT = "refused_context"
REFUSED_TAMPERED = "refused_tampered"
NOT_REPLAYABLE = "not_replayable"
UNKNOWN_REQUEST = "unknown_request"


@dataclass(frozen=True)
class ReplayResult:
    request_id: str
    status: str
    reason: str
    differences: tuple[str, ...] = field(default_factory=tuple)
    reproduce: dict | None = None

    @property
    def identical(self) -> bool:
        return self.status == IDENTICAL


def find_header(engine: AuditedEngine, request_id: str) -> dict | None:
    for header in engine.log.headers.read_all():
        if header.get("request_id") == request_id:
            return header
    return None


def replay(engine: AuditedEngine, request_id: str) -> ReplayResult:
    def result(
        status: str, reason: str, differences: tuple[str, ...] | list[str] = (), reproduce: dict | None = None
    ) -> ReplayResult:
        return ReplayResult(request_id, status, reason, tuple(differences), reproduce)

    try:
        header = find_header(engine, request_id)
    except (OSError, ValueError) as exc:
        return result(REFUSED_TAMPERED, f"the header log cannot be read ({type(exc).__name__})")
    if header is None:
        return result(UNKNOWN_REQUEST, "no header with this request ID in the log")
    if compute_record_hash(header) != header.get("record_hash"):
        return result(REFUSED_TAMPERED, "the header does not match its own record hash: it has been edited")

    try:
        payload = engine.log.payloads.read(request_id)
    except (AuditError, ValueError, KeyError, OSError) as exc:
        return result(REFUSED_TAMPERED, f"the payload cannot be read ({type(exc).__name__})")
    if payload is None:
        kind = classify_missing_payload(engine.log, request_id)
        why = "covered by a purge tombstone" if kind == "purged" else "no purge tombstone covers it"
        return result(
            NOT_REPLAYABLE,
            f"the payload is gone ({kind}: {why}); the raw input cannot be recovered, so the decision cannot "
            "be replayed",
        )
    if payload.request_id != request_id or payload.digest() != header.get("payload_digest"):
        return result(REFUSED_TAMPERED, "the payload does not match the digest in its header: it has been changed")

    recorded_hash = header.get("run_context_hash")
    if recorded_hash != engine.context_hash:
        return refuse_context(engine, request_id, recorded_hash)

    raw = payload.data["raw_input"]
    evaluation = engine.evaluate(raw["name"], raw["birth_date"], raw["country"])
    expected = decision_summary(evaluation.entries, evaluation.threshold, engine.top_candidates)
    differences = []
    for key in DECISION_FIELDS:
        if key == "candidates":
            differences += diff_candidates(header.get(key), expected[key])
        elif header.get(key) != expected[key]:
            differences.append(f"{key}: recorded {header.get(key)!r}, replayed {expected[key]!r}")
    stored_derived = {k: v for k, v in payload.data.items() if k not in ("customer_ref", "raw_input")}
    for key in sorted(set(stored_derived) | set(evaluation.derived)):
        if stored_derived.get(key) != evaluation.derived.get(key):
            differences.append(f"payload.{key}: recorded and replayed values differ")
    if differences:
        return result(DIFFERENT, "same run context, different answer", differences)
    return result(IDENTICAL, "the replayed decision is identical to the record")


def diff_candidates(recorded: object, replayed: list[dict]) -> list[str]:
    """Candidate-level differences: IDs added or missing, and any changed per-feature score."""
    if not isinstance(recorded, list):
        return ["candidates: recorded value is not a list"]
    a = {c.get("entity_id"): c for c in recorded}
    b = {c["entity_id"]: c for c in replayed}
    lines = [f"candidates: {i} recorded but not replayed" for i in sorted(a.keys() - b.keys(), key=str)]
    lines += [f"candidates: {i} replayed but not recorded" for i in sorted(b.keys() - a.keys())]
    for i in sorted(a.keys() & b.keys()):
        for key in sorted(set(a[i]) | set(b[i])):
            if a[i].get(key) != b[i].get(key):
                lines.append(f"candidates[{i}].{key}: recorded {a[i].get(key)!r}, replayed {b[i].get(key)!r}")
    if not lines and [c.get("entity_id") for c in recorded] != [c["entity_id"] for c in replayed]:
        lines.append("candidates: same IDs in a different order")
    return lines


def refuse_context(engine: AuditedEngine, request_id: str, recorded_hash: object) -> ReplayResult:
    live = engine.run_context
    short = str(recorded_hash)[:12]
    try:
        stored = load_context(engine.contexts_dir, recorded_hash) if isinstance(recorded_hash, str) else None
    except AuditError as exc:
        return ReplayResult(request_id, REFUSED_CONTEXT, f"the record's run context {short}... cannot be used: {exc}")
    if stored is None:
        return ReplayResult(
            request_id,
            REFUSED_CONTEXT,
            f"the record was made under run context {short}..., which is not this engine's "
            f"({engine.context_hash[:12]}...), and no context file for it was found, so what differs cannot be listed",
        )
    reproduce = {
        "git_sha": stored.git_sha,
        "git_dirty_at_record_time": stored.git_dirty,
        "snapshot_id": stored.snapshot_id,
        "snapshot_sha256": stored.snapshot_sha256,
    }
    return ReplayResult(
        request_id,
        REFUSED_CONTEXT,
        f"the record was made under run context {short}..., not this engine's ({engine.context_hash[:12]}...): "
        "replay will not produce an answer under a different context",
        tuple(diff_contexts(stored.as_dict(), live.as_dict())),
        reproduce,
    )

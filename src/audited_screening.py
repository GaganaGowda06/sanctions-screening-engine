"""Audited screening: the same decision as screen(), with an audit record written before the answer is returned.

    from audited_screening import audited_screen
    result = audited_screen("Vladimir Putin", "1952-10-07", "ru")
    result.matches       # the same ScreenMatch list that screening.screen() returns
    result.request_id    # key of the header (audit/headers.jsonl) and of the payload (audit/payloads/<id>.json)

screening.py is a Stage 2 file and does not change, so this lives beside it. AuditedEngine.audited_screen() calls the
screener's components() exactly once, applies config.threshold_for(script) itself, builds the matches the way
ScreeningEngine.screen() does, and builds the audit header and payload from that same result. An equivalence test
runs both on every query of the three hold-outs and compares entity IDs, scores and decisions.

The run context is computed once, when the AuditedEngine is built, and held. Records take its hash from there; the
files are never read again at log time, so a record cannot disagree with the engine that answered it.

If the audit record cannot be written, the call raises AuditError and returns nothing: a name is never screened
without a record. (The wider wiring, and what a caller does on that error, is step 4.)

The payload holds, for the hand-tuned engine: the optional customer_ref; the raw name, birth date and country;
the query script; the normalised form, the scoring form and the skeleton form; the name tokens; the retrieval
keys; and, for each alert, the matched name and the explanation. This engine uses no phonetic keys (Double
Metaphone belongs to the deduplication model, which is outside the decision path).
"""

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from audit import (
    AuditConfigError,
    AuditError,
    AuditLog,
    PayloadRecord,
    new_request_id,
    new_salt,
    payload_digest,
)
from audit_schema import KIND_ENGINE_START, build_engine_start_header, build_header, candidate_entry
from candidates import name_keys
from normalise import normalise
from profile_data import DEFAULT_INPUT
from run_context import RunContext, build_run_context, changed_field_names, load_context, store_context
from scoring import scoring_form, skeleton_form
from screening import (
    CANDIDATE_CAP,
    ScreeningConfig,
    ScreeningEngine,
    ScreenMatch,
    build_matches,
    query_script,
    score_components,
)

AUDIT_DIR_ENV = "SCREENING_AUDIT_DIR"


@dataclass(frozen=True)
class AuditedResult:
    request_id: str
    matches: list[ScreenMatch]


@dataclass(frozen=True)
class Evaluation:
    """The matcher's answer for one input, before anything is logged: used by audited_screen and by replay."""

    script: str
    threshold: float
    matches: list[ScreenMatch]
    entries: list[dict]  # one candidate_entry per scored candidate
    derived: dict  # the name-derived payload fields (everything but the raw input and the customer ref)


def resolve_audit_dir(setting: str | Path | None = None) -> Path:
    """The audit directory: the explicit setting (a CLI flag), else $SCREENING_AUDIT_DIR. There is no default,
    because the payloads hold personal data and a silent default would put them somewhere nobody chose. Raises
    AuditConfigError if neither is set."""
    chosen = setting or os.environ.get(AUDIT_DIR_ENV)
    if not chosen:
        raise AuditConfigError(f"no audit directory: pass one explicitly or set ${AUDIT_DIR_ENV}")
    return Path(chosen)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AuditedEngine:
    """A ScreeningEngine, the run context it was built under, and the audit log it writes to."""

    def __init__(
        self,
        engine: ScreeningEngine,
        run_context: RunContext,
        log: AuditLog,
        *,
        clock: Callable[[], datetime] = utc_now,
        for_screening: bool = True,
    ) -> None:
        """With for_screening True the run context is written to <audit_dir>/contexts/<hash>.json (see
        run_context.store_context). Replay builds its engine with False: it writes nothing."""
        self.engine = engine
        self.run_context = run_context
        self.context_hash = run_context.context_hash  # fixed here; never recomputed from files
        self.log = log
        self.clock = clock
        self.top_candidates = run_context.top_candidates  # from the context, so replay uses the record's own value
        self.contexts_dir = log.headers.path.parent / "contexts"
        if for_screening:
            store_context(self.contexts_dir, run_context)
            self._record_engine_start()

    def _record_engine_start(self) -> None:
        """Append an engine_start record unless the most recent one already names this run context. Done under the
        log's lock, so two engines starting together do not both append. A failure stops the engine from starting:
        a name is never screened under a configuration the chain does not show."""
        headers = self.log.headers
        with headers.locked(create_directory=False):
            headers.require_initialised()
            last = headers.last_of_kind(KIND_ENGINE_START)
            if last is not None and last["context_hash"] == self.context_hash:
                return
            previous = last["context_hash"] if last else None
            changed: list[str] = []
            if previous is not None:
                before = load_context(self.contexts_dir, previous)
                changed = (
                    ["context_file_missing"]
                    if before is None
                    else changed_field_names(before.as_dict(), self.run_context.as_dict())
                )
            headers.append_locked(
                build_engine_start_header(
                    request_id=new_request_id(),
                    moment=self.clock(),
                    context_hash=self.context_hash,
                    previous_context_hash=previous,
                    changed_fields=changed,
                )
            )

    @classmethod
    def from_file(
        cls,
        path: Path = DEFAULT_INPUT,
        audit_dir: Path | str | None = None,
        config: ScreeningConfig | None = None,
        fsync: bool = True,
        top_candidates: int | None = None,
        **kwargs: object,
    ) -> "AuditedEngine":
        """Build from a list file. `audit_dir` is resolved by resolve_audit_dir() before the list is read, so a
        missing setting fails at once. The store must already exist (AuditLog.create_new): none is made here."""
        log = AuditLog.in_directory(resolve_audit_dir(audit_dir), fsync=fsync)
        log.headers.require_initialised()  # before the slow list build: a missing store fails at once
        engine = ScreeningEngine.from_file(path, config)
        context = build_run_context(engine, path, top_candidates=top_candidates)
        return cls(engine, context, log, **kwargs)  # type: ignore[arg-type]

    def evaluate(self, name: str, birth_date: str | None = None, country: str | None = None) -> Evaluation:
        """Screen one input and return everything the record is made from. Calls components() exactly once, applies
        config.threshold_for(script) itself, and logs nothing."""
        config, screener = self.engine.config, self.engine.screener
        script = query_script(name)
        threshold = config.threshold_for(script)
        query = scoring_form(name)
        components = screener.components(name, birth_date, country, keep=CANDIDATE_CAP, blends=(config.blend,))
        scored = score_components(components, config)
        matches = build_matches(screener, config, name, birth_date, country, scored, script, threshold)
        entries = [
            candidate_entry(c.entity_id, score, c.names[0], c.skeleton, c.dob, c.country, c.arabic, used, threshold)
            for c, score, used in scored
        ]
        index = screener.index
        derived = {
            "query_script": script,
            "normalised": {"normalise": normalise(name), "scoring_form": query, "skeleton": skeleton_form(query)},
            "name_tokens": query.split(),
            "retrieval_keys": sorted(name_keys(name, index.do_fold, index.join, index.strip_article)),
            "matches": [{"entity_id": m.entity_id, **asdict(m.explanation)} for m in matches],
        }
        return Evaluation(script, threshold, matches, entries, derived)

    def audited_screen(
        self,
        name: str,
        birth_date: str | None = None,
        country: str | None = None,
        customer_ref: str | None = None,
    ) -> AuditedResult:
        if customer_ref is not None and not isinstance(customer_ref, str):
            raise ValueError("customer_ref must be a string or None")
        result = self.evaluate(name, birth_date, country)
        data = {
            "customer_ref": customer_ref,
            "raw_input": {"name": name, "birth_date": birth_date, "country": country},
            **result.derived,
        }
        salt, request_id = new_salt(), new_request_id()
        try:
            header = build_header(
                request_id=request_id,
                moment=self.clock(),
                run_context_hash=self.context_hash,
                entries=result.entries,
                threshold=result.threshold,
                digest=payload_digest(salt, data),
                top=self.top_candidates,
            )
            self.log.write(header, PayloadRecord(request_id, salt, data))
        except AuditError:
            raise
        except (ValueError, TypeError, OSError) as exc:
            raise AuditError(f"could not build or write the audit record: {exc}") from exc
        return AuditedResult(request_id, result.matches)


_audited: AuditedEngine | None = None


def get_audited_engine() -> AuditedEngine:
    """The shared audited engine over the downloaded list, built on first use (a few seconds). The audit directory
    is $SCREENING_AUDIT_DIR and must be set: there is no default."""
    global _audited
    if _audited is None:
        _audited = AuditedEngine.from_file()
    return _audited


def audited_screen(
    name: str, birth_date: str | None = None, country: str | None = None, customer_ref: str | None = None
) -> AuditedResult:
    """Screen a name as screening.screen() does, and write the audit record before returning.

    customer_ref is an optional opaque string from the caller that links the record to a customer. It goes in the
    payload only. A later purge needs it, because the retention period runs from the end of the relationship."""
    return get_audited_engine().audited_screen(name, birth_date, country, customer_ref)

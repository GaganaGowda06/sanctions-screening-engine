"""Local web page for screening one name. Not a deployment and not for production decisions.

Run (from the repository root):
    venv/bin/streamlit run app/streamlit_app.py

The engine is built once (a few seconds) and cached. Screening here writes no audit record at all: the page uses the
research path (ScreeningEngine, the same scoring the audited path uses) and imports nothing from the audit modules, so
it cannot write to any audit store, real or throwaway. A decision that has to be kept and replayable goes through
src/screen_cli.py, which writes the audit record first.

Standard library, streamlit and the project's src/ modules only.
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from profile_data import DEFAULT_INPUT  # noqa: E402
from screening import ScreeningEngine, build_matches, query_script, score_components  # noqa: E402

TOP = 5
DATE_FORMAT = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
COUNTRY_FORMAT = re.compile(r"^[A-Za-z]{2}$")

# Measured on the second hold-out, Stage 2 engine (docs/limitations_evidence.md, part 2 and "What it adds up to"):
# recall with both attributes supplied 0.737, with neither 0.654; F1 0.7772 with both, 0.7384 with birth date only,
# 0.6885 with country only, 0.5865 with neither.
WEAKER_WITHOUT = (
    "Recall is weaker without a birth date and a country. On the second hold-out, withholding both lowered recall "
    "from 0.737 to 0.654 and F1 from 0.7772 to 0.5865; supplying only a birth date gave F1 0.7384 and only a country "
    "0.6885. Precision falls too, because the birth date and country are what rule out namesakes "
    "(docs/limitations_evidence.md)."
)


@dataclass(frozen=True)
class Candidate:
    rank: int
    entity_id: str
    score: float
    name: str  # the listed name that matched best
    entity_type: str
    birth_dates: str
    countries: str
    source_lists: str
    name_score: float
    skeleton_score: float
    skeleton_used: bool
    birth_date_status: str
    birth_date_adjustment: float
    country_status: str
    country_adjustment: float


@dataclass(frozen=True)
class Screening:
    name: str
    script: str
    threshold: float
    decision: str  # "alert" or "clear"
    top_score: float | None
    candidates: list[Candidate]
    missing: list[str]  # which of "birth date" and "country" were not supplied


def read_metadata(path: Path) -> dict[str, tuple[str, str]]:
    """entity id -> (schema, source lists) for the sanctions targets, from the entities file."""
    found = {}
    with open(path, "rb") as fh:
        for line in fh:
            try:
                ent = json.loads(line)
            except ValueError:
                continue
            if ent.get("target") is True:
                found[ent["id"]] = (ent.get("schema") or "", ", ".join(ent.get("datasets") or ()))
    return found


@st.cache_resource(show_spinner="Building the list (a few seconds, once)...")
def load_engine() -> ScreeningEngine:
    return ScreeningEngine.from_file(DEFAULT_INPUT)


@st.cache_resource(show_spinner="Reading entity types and source lists...")
def load_metadata() -> dict[str, tuple[str, str]]:
    return read_metadata(DEFAULT_INPUT)


def check_inputs(name: str, birth_date: str, country: str) -> str | None:
    """A reason the inputs cannot be screened, or None."""
    if not name.strip():
        return "Enter a name."
    if birth_date and not DATE_FORMAT.match(birth_date):
        return "Birth date must be YYYY, YYYY-MM or YYYY-MM-DD."
    if country and not COUNTRY_FORMAT.match(country):
        return "Country must be a two-letter ISO code, for example ru."
    return None


def screen_name(
    engine: ScreeningEngine,
    metadata: dict[str, tuple[str, str]],
    name: str,
    birth_date: str = "",
    country: str = "",
) -> Screening:
    """Screen one name and keep the five best candidates, alert or not. Uses the engine's own components, scoring and
    explanation code; the only difference from ScreeningEngine.screen is that candidates under the threshold are kept
    for display. Nothing is written anywhere."""
    name, birth, ctry = name.strip(), birth_date.strip() or None, country.strip().lower() or None
    config, screener = engine.config, engine.screener
    script = query_script(name)
    threshold = config.threshold_for(script)
    components = screener.components(name, birth, ctry, keep=10_000, blends=(config.blend,))
    scored = score_components(components, config)
    best = build_matches(screener, config, name, birth, ctry, scored, script, threshold=-1.0)[:TOP]
    candidates = []
    for rank, match in enumerate(best, start=1):
        record = screener.records[match.entity_id]
        e = match.explanation
        schema, sources = metadata.get(match.entity_id, ("unknown", "unknown"))
        candidates.append(
            Candidate(
                rank, match.entity_id, match.score, e.matched_name, schema,
                ", ".join(record.birth_dates[:3]) or "none listed",
                ", ".join(sorted(record.countries)) or "none listed",
                sources or "unknown", e.name_score, e.skeleton_score, e.skeleton_used,
                e.birth_date, e.birth_date_adjustment, e.country, e.country_adjustment,
            )
        )  # fmt: skip
    top = candidates[0].score if candidates else None
    return Screening(
        name, script, threshold, "alert" if top is not None and top >= threshold else "clear", top, candidates,
        [label for label, value in (("birth date", birth), ("country", ctry)) if value is None],
    )  # fmt: skip


def shorten(sources: str, keep: int = 3) -> str:
    parts = sources.split(", ")
    return sources if len(parts) <= keep else ", ".join(parts[:keep]) + f" (+{len(parts) - keep} more)"


def adjustment_words(what: str, status: str, adjustment: float) -> str:
    if status == "agree":
        return f"The {what} agrees with the listed one, which adds {adjustment:+.2f}."
    if status == "conflict":
        return f"The {what} conflicts with every listed one, which subtracts {-adjustment:.2f}."
    if status == "close":
        return f"The {what} is a year away from a listed one, which changes nothing."
    return f"No {what} to compare (not supplied, or none listed), so it changes nothing."


def explain(c: Candidate) -> list[str]:
    """The score components of one candidate, in plain words."""
    lines = [
        f"The closest listed name is “{c.name}”. "
        f"The name similarity is {c.name_score:.3f} (1.0 is the same name, ignoring case, accents and word order)."
    ]
    if c.skeleton_used:
        lines.append(
            f"The consonant-skeleton similarity, {c.skeleton_score:.3f}, was higher and was used in place of the "
            "name similarity. The skeleton ignores vowels, which catches different spellings of the same name."
        )
    else:
        lines.append(f"The consonant-skeleton similarity is {c.skeleton_score:.3f}; it did not raise the name score.")
    lines.append(adjustment_words("birth date", c.birth_date_status, c.birth_date_adjustment))
    lines.append(adjustment_words("country", c.country_status, c.country_adjustment))
    lines.append(f"Final score {c.score:.4f}: the name part plus the two adjustments, kept between 0 and 1.")
    return lines


def render(result: Screening) -> None:
    alert = result.decision == "alert"
    (st.error if alert else st.success)(f"**{result.decision.upper()}**")
    left, right = st.columns(2)
    left.metric("Top score", "none" if result.top_score is None else f"{result.top_score:.4f}")
    right.metric(f"Threshold ({result.script} script)", f"{result.threshold:.2f}")
    if result.top_score is not None:
        st.caption(
            f"The top score is {'at or above' if alert else 'below'} the threshold of {result.threshold:.2f}, "
            f"so the decision is {result.decision}."
        )
    if result.missing:
        st.warning(f"No {' or '.join(result.missing)} was given. {WEAKER_WITHOUT}")
    if not result.candidates:
        st.info("The list produced no candidates for this name.")
        return
    st.subheader(f"Top {len(result.candidates)} candidates")
    st.dataframe(
        [
            {
                "rank": c.rank,
                "score": round(c.score, 4),
                "over threshold": c.score >= result.threshold,
                "listed name": c.name,
                "entity type": c.entity_type,
                "birth date": c.birth_dates,
                "country": c.countries,
                "source list": shorten(c.source_lists),
            }
            for c in result.candidates
        ],
        hide_index=True,
        width="stretch",
    )
    top = result.candidates[0]
    st.subheader("What drove the top match")
    for line in explain(top):
        st.markdown(f"- {line}")


def main() -> None:
    st.set_page_config(page_title="Sanctions name screening", layout="wide")
    st.title("Sanctions name screening")
    st.caption(
        "Local exploration page. It writes no audit record and is not for production decisions; "
        "use src/screen_cli.py for a decision that must be kept and replayed."
    )
    with st.form("screen"):
        name = st.text_input("Name (required)")
        c1, c2 = st.columns(2)
        birth_date = c1.text_input("Birth date (optional)", placeholder="YYYY, YYYY-MM or YYYY-MM-DD")
        country = c2.text_input("Country (optional)", placeholder="two-letter code, for example ru")
        submitted = st.form_submit_button("Screen")
    if not submitted:
        return
    problem = check_inputs(name, birth_date.strip(), country.strip())
    if problem:
        st.error(problem)
        return
    render(screen_name(load_engine(), load_metadata(), name, birth_date, country))


if __name__ == "__main__":
    main()

"""Scoring: rank the candidates from the index by name similarity, adjusted by birth date and country.

Name score
  Each name is held in two forms: the normalised string and its skeleton string (every token
  folded and vowel-stripped, see candidates.py). A query is compared to each form with a
  token-order-insensitive fuzzy ratio (rapidfuzz token_sort_ratio, 0 to 1). The name score of an
  entity is the best over all its names of  max(normalised, skeleton_weight * skeleton).
  The skeleton is lossy, so it is discounted.

Birth date and country adjust the name score, then the result is clipped to 0..1:
  birth date   same full date  +dob_full        same year  +dob_year
               years differ by 2 or more  -dob_conflict     one year apart  no effect
  country      query country among the entity's countries  +country_agree
               entity has countries and the query country is not one of them  -country_conflict
  Missing on either side: no effect. The magnitudes are set by hand, not tuned.

An alert is an entity whose final score reaches the alert threshold.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from rapidfuzz import fuzz, process

from candidates import CandidateIndex, fold, skeleton
from normalise import normalise

MIN_RAW_SCORE = 0.35  # candidates below this on both forms cannot reach an alert threshold of 0.5 even with every boost


@dataclass(frozen=True)
class Adjustments:
    dob_full: float = 0.10
    dob_year: float = 0.05
    dob_conflict: float = 0.30
    country_agree: float = 0.03
    country_conflict: float = 0.15


class Components(NamedTuple):
    """Everything needed to rebuild an entity's score for any skeleton weight or adjustment switch."""

    entity_id: str
    name: float
    skeleton: float
    dob: float
    country: float


class Match(NamedTuple):
    entity_id: str
    score: float
    components: Components


@dataclass(frozen=True)
class EntityRecord:
    norms: tuple[str, ...]
    skeletons: tuple[str, ...]
    birth_dates: tuple[str, ...]
    countries: frozenset[str]


def skeleton_form(normalised: str) -> str:
    """Per-token skeleton of a normalised name, tokens kept in order."""
    return " ".join(sk for token in normalised.split() if (sk := skeleton(fold(token))))


def make_record(names: Iterable[str], birth_dates: Iterable[str] = (), countries: Iterable[str] = ()) -> EntityRecord:
    norms = tuple(dict.fromkeys(n for n in map(normalise, names) if n))
    skeletons = tuple(dict.fromkeys(s for s in map(skeleton_form, norms) if s))
    return EntityRecord(
        norms=norms,
        skeletons=skeletons,
        birth_dates=tuple(dict.fromkeys(birth_dates)),
        countries=frozenset(c.lower() for c in countries),
    )


def best_similarity(query: str, forms: Sequence[str]) -> float:
    """Best token-sort ratio (0 to 1) of the query against any of the forms."""
    if not query or not forms:
        return 0.0
    best = process.extractOne(query, forms, scorer=fuzz.token_sort_ratio)
    return best[1] / 100 if best else 0.0


def _year(date: str) -> int | None:
    return int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else None


def date_adjustment(query: str | None, entity_dates: Sequence[str], adj: Adjustments) -> float:
    query_year = _year(query) if query else None
    if query_year is None or not entity_dates:
        return 0.0
    years = [(_year(d), d) for d in entity_dates]
    years = [(y, d) for y, d in years if y is not None]
    if not years:
        return 0.0
    if any(y == query_year for y, _ in years):
        full = len(query) >= 10 and any(len(d) >= 10 and d[:10] == query[:10] for _, d in years)
        return adj.dob_full if full else adj.dob_year
    if min(abs(y - query_year) for y, _ in years) <= 1:
        return 0.0
    return -adj.dob_conflict


def country_adjustment(query: str | None, entity_countries: frozenset[str], adj: Adjustments) -> float:
    if not query or not entity_countries:
        return 0.0
    return adj.country_agree if query.lower() in entity_countries else -adj.country_conflict


def combine(c: Components, skeleton_weight: float, use_dob: bool = True, use_country: bool = True) -> float:
    score = max(c.name, skeleton_weight * c.skeleton)
    if use_dob:
        score += c.dob
    if use_country:
        score += c.country
    return min(1.0, max(0.0, score))


class Screener:
    """Candidate index plus the entity records needed to score candidates."""

    def __init__(self, index: CandidateIndex, records: dict[str, EntityRecord], adjustments: Adjustments) -> None:
        self.index = index
        self.records = records
        self.adjustments = adjustments

    @classmethod
    def build(
        cls, entities: Iterable[dict], max_df: int = 1000, adjustments: Adjustments | None = None
    ) -> "Screener":
        """Build from dicts with id, names and optionally birth_dates and countries."""
        entities = list(entities)
        index = CandidateIndex.build(((e["id"], e["names"]) for e in entities), max_df=max_df)
        records = {
            e["id"]: make_record(e["names"], e.get("birth_dates", ()), e.get("countries", ())) for e in entities
        }
        return cls(index, records, adjustments or Adjustments())

    def components(
        self,
        name: str,
        birth_date: str | None = None,
        country: str | None = None,
        keep: int = 200,
        always: str | None = None,
    ) -> list[Components]:
        """Score components for the best `keep` candidates, ranked by the highest score they could reach.

        `always` names an entity that is kept even if it falls outside the best `keep`.
        """
        query = normalise(name)
        query_skeleton = skeleton_form(query)
        scored = []
        for entity_id, _ in self.index.query(name):
            record = self.records[entity_id]
            name_sim = best_similarity(query, record.norms)
            skel_sim = best_similarity(query_skeleton, record.skeletons)
            if max(name_sim, skel_sim) < MIN_RAW_SCORE:
                continue
            scored.append(
                Components(
                    entity_id,
                    name_sim,
                    skel_sim,
                    date_adjustment(birth_date, record.birth_dates, self.adjustments),
                    country_adjustment(country, record.countries, self.adjustments),
                )
            )
        scored.sort(key=lambda c: -(max(c.name, c.skeleton) + max(c.dob, 0.0) + max(c.country, 0.0)))
        kept = scored[:keep]
        if always is not None and all(c.entity_id != always for c in kept):
            kept += [c for c in scored[keep:] if c.entity_id == always]
        return kept

    def screen(
        self,
        name: str,
        birth_date: str | None = None,
        country: str | None = None,
        threshold: float = 0.8,
        skeleton_weight: float = 0.9,
    ) -> list[Match]:
        """Entities whose final score reaches the threshold, best first."""
        matches = []
        for c in self.components(name, birth_date, country, keep=10_000):
            score = combine(c, skeleton_weight)
            if score >= threshold:
                matches.append(Match(c.entity_id, score, c))
        matches.sort(key=lambda m: (-m.score, m.entity_id))
        return matches

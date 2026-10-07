"""Scoring: rank the candidates from the index by name similarity, adjusted by birth date and country.

Name score
  Names are normalised and company legal-form words are stripped (ltd, llc, gmbh, "obshchestvo s
  ogranichennoy otvetstvennostyu" ...), so an organisation matches its aliases on the part that
  identifies it. The query is compared to each name of an entity with a token-order-insensitive fuzzy
  ratio (rapidfuzz token_sort_ratio, 0 to 1).
  Length mismatch: a name pair loses  length_penalty * (1 - shorter / longer)  in token count, unless
  the tokens of the shorter name are all in the longer one (a clean subset).
  The name score of an entity is the best penalised score over its names.
Skeleton (vowel-stripped, folded tokens, see candidates.py) only confirms. If the name score is at
  least skeleton_floor, it may be raised towards the skeleton similarity by at most skeleton_lift.
  Below the floor the skeleton has no effect, so it cannot produce a high score on its own.

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

MIN_NAME_SCORE = 0.35  # below this a candidate cannot reach an alert threshold of 0.5 even with every boost and lift
MAX_SKELETON_LIFT = 0.3  # largest lift the tuning grid allows, used to rank candidates by their best reachable score
MIN_SKELETON_FLOOR = 0.4  # smallest floor the tuning grid allows
# Defaults are the values tuned on dev in src/evaluate_scoring.py (floor and lift sit at the edge of the grid searched).
DEFAULT_SKELETON_FLOOR = 0.4
DEFAULT_SKELETON_LIFT = 0.3
DEFAULT_LENGTH_PENALTY = 0.0  # the tuner found any penalty lowers dev F1; the mechanism stays available
DEFAULT_THRESHOLD = 0.89

# Tokens that mark a company legal form. Only stripped from names that contain at least one of them,
# and never down to nothing. Ambiguous short words (as, ab, de, ad) are left out on purpose.
LEGAL_FORMS = frozenset(
    """ltd limited co company corp corporation inc incorporated llc llp lp plc gmbh ag sa sarl sas srl spa nv bv kg
    pty pvt bhd jsc ojsc pjsc cjsc psc fze fzco fzc fz liability joint stock ooo zao oao pao ao tov obshchestvo
    ogranichennoy ogranichennoi ogranichennoj otvetstvennostyu otvetstvennostiu aktsionernoe aktsionerne
    aktsionernoye tovaristvo tovarystvo obmezhenoyu obmezhenoiu vidpovidalnistyu vidpovidalnistiu""".split()
)
LEGAL_FILLERS = frozenset({"s", "z"})  # the "с" and "з" in "Общество с ограниченной ответственностью"


@dataclass(frozen=True)
class Adjustments:
    dob_full: float = 0.10
    dob_year: float = 0.05
    dob_conflict: float = 0.30
    country_agree: float = 0.03
    country_conflict: float = 0.15


class Components(NamedTuple):
    """Everything needed to rebuild an entity's score for any skeleton setting or adjustment switch."""

    entity_id: str
    names: tuple[float, ...]  # best penalised name score, one per length penalty asked for
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
    token_sets: tuple[frozenset[str], ...]  # tokens of each name in norms
    skeletons: tuple[str, ...]
    birth_dates: tuple[str, ...]
    countries: frozenset[str]


def strip_legal_forms(normalised: str) -> str:
    """Remove company legal-form words. Names without any are returned as they are, and never emptied."""
    tokens = normalised.split()
    if not any(t in LEGAL_FORMS for t in tokens):
        return normalised
    kept = [t for t in tokens if t not in LEGAL_FORMS and t not in LEGAL_FILLERS]
    return " ".join(kept) if kept else normalised


def scoring_form(name: str) -> str:
    return strip_legal_forms(normalise(name))


def skeleton_form(normalised: str) -> str:
    """Per-token skeleton of a normalised name, tokens kept in order."""
    return " ".join(sk for token in normalised.split() if (sk := skeleton(fold(token))))


def make_record(names: Iterable[str], birth_dates: Iterable[str] = (), countries: Iterable[str] = ()) -> EntityRecord:
    norms = tuple(dict.fromkeys(n for n in map(scoring_form, names) if n))
    skeletons = tuple(dict.fromkeys(s for s in map(skeleton_form, norms) if s))
    return EntityRecord(
        norms=norms,
        token_sets=tuple(frozenset(n.split()) for n in norms),
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


def name_scores(query: str, record: EntityRecord, penalties: Sequence[float]) -> tuple[float, ...]:
    """Best length-penalised name score of the query against the entity's names, per penalty."""
    best = [0.0] * len(penalties)
    if not query or not record.norms:
        return tuple(best)
    query_tokens = query.split()
    query_set = frozenset(query_tokens)
    for _, ratio, index in process.extract(query, record.norms, scorer=fuzz.token_sort_ratio, limit=None):
        entity_set = record.token_sets[index]
        if query_set <= entity_set or entity_set <= query_set:
            mismatch = 0.0  # clean subset
        else:
            shorter, longer = sorted((len(query_tokens), len(record.norms[index].split())))
            mismatch = 1 - shorter / longer
        for i, penalty in enumerate(penalties):
            best[i] = max(best[i], ratio / 100 - penalty * mismatch)
    return tuple(max(0.0, b) for b in best)


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


def combine(
    c: Components,
    skeleton_floor: float = DEFAULT_SKELETON_FLOOR,
    skeleton_lift: float = DEFAULT_SKELETON_LIFT,
    penalty_index: int = 0,
    use_dob: bool = True,
    use_country: bool = True,
) -> float:
    score = c.names[penalty_index]
    if score >= skeleton_floor:
        score = max(score, min(c.skeleton, score + skeleton_lift))
    if use_dob:
        score += c.dob
    if use_country:
        score += c.country
    return min(1.0, max(0.0, score))


class Screener:
    """Candidate index plus the entity records needed to score candidates."""

    def __init__(
        self,
        index: CandidateIndex,
        records: dict[str, EntityRecord],
        adjustments: Adjustments,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
    ) -> None:
        self.index = index
        self.records = records
        self.adjustments = adjustments
        self.length_penalty = length_penalty

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
        penalties: Sequence[float] | None = None,
    ) -> list[Components]:
        """Score components for the best `keep` candidates, ranked by the highest score they could reach.

        `penalties` lists the length penalties to score for (smallest first); default is the screener's own.
        `always` names an entity that is kept even if it falls outside the best `keep`.
        """
        penalties = tuple(penalties) if penalties is not None else (self.length_penalty,)
        query = scoring_form(name)
        query_skeleton = skeleton_form(query)
        scored = []
        for entity_id, _ in self.index.query(name):
            record = self.records[entity_id]
            names = name_scores(query, record, penalties)
            if names[0] < MIN_NAME_SCORE:
                continue
            scored.append(
                Components(
                    entity_id,
                    names,
                    best_similarity(query_skeleton, record.skeletons),
                    date_adjustment(birth_date, record.birth_dates, self.adjustments),
                    country_adjustment(country, record.countries, self.adjustments),
                )
            )

        def reachable(c: Components) -> float:
            best = c.names[0]
            if best >= MIN_SKELETON_FLOOR:
                best = max(best, min(c.skeleton, best + MAX_SKELETON_LIFT))
            return best + max(c.dob, 0.0) + max(c.country, 0.0)

        scored.sort(key=lambda c: -reachable(c))
        kept = scored[:keep]
        if always is not None and all(c.entity_id != always for c in kept):
            kept += [c for c in scored[keep:] if c.entity_id == always]
        return kept

    def screen(
        self,
        name: str,
        birth_date: str | None = None,
        country: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        skeleton_floor: float = DEFAULT_SKELETON_FLOOR,
        skeleton_lift: float = DEFAULT_SKELETON_LIFT,
    ) -> list[Match]:
        """Entities whose final score reaches the threshold, best first."""
        matches = []
        for c in self.components(name, birth_date, country, keep=10_000):
            score = combine(c, skeleton_floor, skeleton_lift)
            if score >= threshold:
                matches.append(Match(c.entity_id, score, c))
        matches.sort(key=lambda m: (-m.score, m.entity_id))
        return matches

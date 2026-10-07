"""Scoring: rank the candidates from the index by name similarity, adjusted by birth date and country.

Name score
  Names are normalised and company legal-form words are stripped (ltd, llc, gmbh, "obshchestvo s
  ogranichennoy otvetstvennostyu" ...), so an organisation matches its aliases on the part that
  identifies it. The query is compared to each name of an entity two ways:
    ratio     rapidfuzz token_sort_ratio, order-insensitive character similarity, 0 to 1
    weighted  rarity-weighted token overlap: every token is weighted by its inverse document
              frequency over the entities in the index, and matched to the best token on the other
              side (similarity of at least 0.7, exact matches score 1). A rare surname counts for
              much more than "mohammed" or "ivan", and a missing rare token costs more than a missing
              common one.
  name = (1 - blend) * ratio + blend * weighted. The entity's name score is the best over its names.
  An optional length penalty (default 0, off) takes length_penalty * (1 - shorter / longer) in token
  count off a name pair unless one side's tokens are all in the other.

Skeleton (vowel-stripped, folded tokens, see candidates.py)
  Arabic-involved pairs (the query is in Arabic script, or the entity has an Arabic-script name, so its
  Latin names are Arabic-derived): the skeleton may stand in for the name score in full, because Arabic
  is written without short vowels and Latin spellings of it vary.
  Everything else: the skeleton only confirms. If the name score is at least skeleton_floor it may be
  raised towards the skeleton similarity by at most skeleton_lift, so it cannot create a match alone.

Birth date and country adjust the name score, then the result is clipped to 0..1:
  birth date   same full date  +dob_full        same year  +dob_year
               years differ by 2 or more  -dob_conflict     one year apart  no effect
  country      query country among the entity's countries  +country_agree
               entity has countries and the query country is not one of them  -country_conflict
  Missing on either side: no effect. The magnitudes are set by hand, not tuned.

An alert is an entity whose final score reaches the alert threshold.
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from rapidfuzz import fuzz, process

from candidates import CandidateIndex, fold, skeleton
from normalise import normalise

MIN_NAME_SCORE = 0.2  # below this on every blend a candidate cannot reach an alert threshold of 0.5
MAX_SKELETON_LIFT = 0.5  # largest lift and smallest floor the tuning grid allows, used to rank candidates
MIN_SKELETON_FLOOR = 0.2
TOKEN_MATCH_CUTOFF = 70  # fuzz.ratio a token pair needs, out of 100, to count as the same token
WEIGHTED_MIN_RATIO = 0.2  # the weighted score is only worked out for names at least this similar by ratio
# Defaults are the values tuned on dev on the deployed candidate pool in src/evaluate_screening.py.
DEFAULT_SKELETON_FLOOR = 0.6  # dev F1 is flat across 0.2 to 0.6; the tie goes to the top
DEFAULT_SKELETON_LIFT = 0.2
DEFAULT_BLEND = 0.5
DEFAULT_LENGTH_PENALTY = 0.0
DEFAULT_THRESHOLD = 0.86
_ARABIC = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")

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
    names: tuple[float, ...]  # best name score, one per blend asked for
    skeleton: float
    dob: float
    country: float
    arabic: bool = False  # Arabic-involved pair: the skeleton may stand in for the name score


class Match(NamedTuple):
    entity_id: str
    score: float
    components: Components


@dataclass(frozen=True)
class EntityRecord:
    raw_names: tuple[str, ...]  # the first original spelling of each name in norms
    norms: tuple[str, ...]
    token_sets: tuple[frozenset[str], ...]  # tokens of each name in norms
    token_lists: tuple[tuple[str, ...], ...]
    skeletons: tuple[str, ...]
    birth_dates: tuple[str, ...]
    countries: frozenset[str]
    arabic: bool = False  # the entity has a name in Arabic script


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


def has_arabic(text: str) -> bool:
    return _ARABIC.search(text) is not None


def make_record(names: Iterable[str], birth_dates: Iterable[str] = (), countries: Iterable[str] = ()) -> EntityRecord:
    names = list(names)
    by_form: dict[str, str] = {}
    for name in names:
        form = scoring_form(name)
        if form and form not in by_form:
            by_form[form] = name
    norms = tuple(by_form)
    skeletons = tuple(dict.fromkeys(s for s in map(skeleton_form, norms) if s))
    return EntityRecord(
        raw_names=tuple(by_form.values()),
        norms=norms,
        token_sets=tuple(frozenset(n.split()) for n in norms),
        token_lists=tuple(tuple(n.split()) for n in norms),
        skeletons=skeletons,
        birth_dates=tuple(dict.fromkeys(birth_dates)),
        countries=frozenset(c.lower() for c in countries),
        arabic=any(has_arabic(n) for n in names),
    )


def best_similarity(query: str, forms: Sequence[str]) -> float:
    """Best token-sort ratio (0 to 1) of the query against any of the forms."""
    if not query or not forms:
        return 0.0
    best = process.extractOne(query, forms, scorer=fuzz.token_sort_ratio)
    return best[1] / 100 if best else 0.0


def _token_match(token: str, tokens: Sequence[str], token_set: frozenset[str]) -> float:
    if token in token_set:
        return 1.0
    best = process.extractOne(token, tokens, scorer=fuzz.ratio, score_cutoff=TOKEN_MATCH_CUTOFF)
    return best[1] / 100 if best else 0.0


def weighted_token_score(
    query_tokens: Sequence[str],
    entity_tokens: Sequence[str],
    entity_set: frozenset[str],
    idf: Callable[[str], float],
) -> float:
    """Rarity-weighted soft token overlap, symmetric: unmatched rare tokens on either side cost the most."""
    query_set = frozenset(query_tokens)
    matched = total = 0.0
    for token in query_tokens:
        weight = idf(token)
        total += weight
        matched += weight * _token_match(token, entity_tokens, entity_set)
    for token in entity_tokens:
        weight = idf(token)
        total += weight
        matched += weight * _token_match(token, query_tokens, query_set)
    return matched / total if total else 0.0


def name_scores(
    query: str,
    record: EntityRecord,
    idf: Callable[[str], float],
    blends: Sequence[float],
    length_penalty: float = 0.0,
) -> tuple[float, ...]:
    """Best name score of the query against the entity's names, one per blend of ratio and weighted score."""
    best = [0.0] * len(blends)
    if not query or not record.norms:
        return tuple(best)
    query_tokens = query.split()
    query_set = frozenset(query_tokens)
    need_weighted = any(b > 0 for b in blends)
    for _, score, index in process.extract(query, record.norms, scorer=fuzz.token_sort_ratio, limit=None):
        ratio = score / 100
        entity_set = record.token_sets[index]
        entity_tokens = record.token_lists[index]
        if length_penalty and not (query_set <= entity_set or entity_set <= query_set):
            shorter, longer = sorted((len(query_tokens), len(entity_tokens)))
            ratio = max(0.0, ratio - length_penalty * (1 - shorter / longer))
        weighted = ratio
        if need_weighted and ratio >= WEIGHTED_MIN_RATIO:
            weighted = weighted_token_score(query_tokens, entity_tokens, entity_set, idf)
        for i, blend in enumerate(blends):
            best[i] = max(best[i], (1 - blend) * ratio + blend * weighted)
    return tuple(best)


def _year(date: str) -> int | None:
    return int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else None


def date_comparison(query: str | None, entity_dates: Sequence[str], adj: Adjustments) -> tuple[str, float]:
    """(status, adjustment). Status is agree, conflict, close (a year apart, no effect) or missing."""
    query_year = _year(query) if query else None
    if query_year is None or not entity_dates:
        return "missing", 0.0
    years = [(_year(d), d) for d in entity_dates]
    years = [(y, d) for y, d in years if y is not None]
    if not years:
        return "missing", 0.0
    if any(y == query_year for y, _ in years):
        full = len(query) >= 10 and any(len(d) >= 10 and d[:10] == query[:10] for _, d in years)
        return "agree", adj.dob_full if full else adj.dob_year
    if min(abs(y - query_year) for y, _ in years) <= 1:
        return "close", 0.0
    return "conflict", -adj.dob_conflict


def country_comparison(
    query: str | None, entity_countries: frozenset[str], adj: Adjustments
) -> tuple[str, float]:
    """(status, adjustment). Status is agree, conflict or missing."""
    if not query or not entity_countries:
        return "missing", 0.0
    if query.lower() in entity_countries:
        return "agree", adj.country_agree
    return "conflict", -adj.country_conflict


def date_adjustment(query: str | None, entity_dates: Sequence[str], adj: Adjustments) -> float:
    return date_comparison(query, entity_dates, adj)[1]


def country_adjustment(query: str | None, entity_countries: frozenset[str], adj: Adjustments) -> float:
    return country_comparison(query, entity_countries, adj)[1]


def name_with_skeleton(
    c: Components,
    skeleton_floor: float = DEFAULT_SKELETON_FLOOR,
    skeleton_lift: float = DEFAULT_SKELETON_LIFT,
    blend_index: int = 0,
    arabic_full_lift: bool = True,
) -> tuple[float, bool]:
    """(name score after the skeleton, whether the skeleton raised it)."""
    score = c.names[blend_index]
    if c.arabic and arabic_full_lift:
        lifted = max(score, c.skeleton)
    elif score >= skeleton_floor:
        lifted = max(score, min(c.skeleton, score + skeleton_lift))
    else:
        lifted = score
    return lifted, lifted > score


def best_matching_name(query: str, record: EntityRecord, idf: Callable[[str], float], blend: float) -> str:
    """The entity's own spelling that the query matches best, for explaining a match."""
    query_tokens = query.split()
    best_score, best_index = -1.0, 0
    for _, score, index in process.extract(query, record.norms, scorer=fuzz.token_sort_ratio, limit=None):
        weighted = score / 100
        if blend > 0 and weighted >= WEIGHTED_MIN_RATIO:
            weighted = weighted_token_score(query_tokens, record.token_lists[index], record.token_sets[index], idf)
        combined = (1 - blend) * score / 100 + blend * weighted
        if combined > best_score:
            best_score, best_index = combined, index
    return record.raw_names[best_index] if record.raw_names else ""


def combine(
    c: Components,
    skeleton_floor: float = DEFAULT_SKELETON_FLOOR,
    skeleton_lift: float = DEFAULT_SKELETON_LIFT,
    blend_index: int = 0,
    use_dob: bool = True,
    use_country: bool = True,
    arabic_full_lift: bool = True,
) -> float:
    score, _ = name_with_skeleton(c, skeleton_floor, skeleton_lift, blend_index, arabic_full_lift)
    if use_dob:
        score += c.dob
    if use_country:
        score += c.country
    return min(1.0, max(0.0, score))


class Screener:
    """Candidate index plus the entity records and token statistics needed to score candidates."""

    def __init__(
        self,
        index: CandidateIndex,
        records: dict[str, EntityRecord],
        adjustments: Adjustments,
        blend: float = DEFAULT_BLEND,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        idf_table: tuple[Mapping[str, int], int] | None = None,
    ) -> None:
        """`idf_table` is (document frequency, documents) to use instead of counting the records here, so training
        and serving can share one table (see idf_table.py)."""
        self.index = index
        self.records = records
        self.adjustments = adjustments
        self.blend = blend
        self.length_penalty = length_penalty
        if idf_table is not None:
            self._df, self._n = idf_table
        else:
            document_frequency: Counter[str] = Counter()
            for record in records.values():
                document_frequency.update(set().union(*record.token_sets))
            self._df = document_frequency
            self._n = len(records)
        self._idf_cache: dict[str, float] = {}

    def idf(self, token: str) -> float:
        """Inverse document frequency of a name token over the entities in the index (never below 0.1)."""
        if token not in self._idf_cache:
            self._idf_cache[token] = max(0.1, math.log((self._n + 1) / (self._df.get(token, 0) + 1)))
        return self._idf_cache[token]

    @classmethod
    def build(
        cls,
        entities: Iterable[dict],
        max_df: int = 1000,
        adjustments: Adjustments | None = None,
        blend: float = DEFAULT_BLEND,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        idf_table: tuple[Mapping[str, int], int] | None = None,
    ) -> "Screener":
        """Build from dicts with id, names and optionally birth_dates and countries."""
        entities = list(entities)
        index = CandidateIndex.build(((e["id"], e["names"]) for e in entities), max_df=max_df)
        records = {
            e["id"]: make_record(e["names"], e.get("birth_dates", ()), e.get("countries", ())) for e in entities
        }
        return cls(index, records, adjustments or Adjustments(), blend, length_penalty, idf_table)

    def components(
        self,
        name: str,
        birth_date: str | None = None,
        country: str | None = None,
        keep: int = 200,
        always: str | None = None,
        blends: Sequence[float] | None = None,
    ) -> list[Components]:
        """Score components for the best `keep` candidates, ranked by the highest score they could reach.

        `blends` lists the ratio/weighted blends to score for; default is the screener's own.
        `always` names an entity that is kept even if it falls outside the best `keep`.
        """
        blends = tuple(blends) if blends is not None else (self.blend,)
        query = scoring_form(name)
        query_skeleton = skeleton_form(query)
        query_arabic = has_arabic(name)
        scored = []
        for entity_id, _ in self.index.query(name):
            record = self.records[entity_id]
            names = name_scores(query, record, self.idf, blends, self.length_penalty)
            if max(names) < MIN_NAME_SCORE:
                continue
            scored.append(
                Components(
                    entity_id,
                    names,
                    best_similarity(query_skeleton, record.skeletons),
                    date_adjustment(birth_date, record.birth_dates, self.adjustments),
                    country_adjustment(country, record.countries, self.adjustments),
                    query_arabic or record.arabic,
                )
            )

        def reachable(c: Components) -> float:
            best = max(c.names)
            if c.arabic:
                best = max(best, c.skeleton)
            elif best >= MIN_SKELETON_FLOOR:
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

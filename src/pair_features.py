"""Pairwise features for two entities: how alike are their names, and do birth date, country and
identifiers agree.

Eight features per pair, nothing learned here:

  name_score       the current engine's name score: half character ratio, half rarity-weighted token score,
                   for the best-matching pair of names (legal-form words stripped first)
  skeleton_score   best similarity of vowel-stripped, folded names (the consonant skeleton)
  weighted_score   the rarity-weighted token score alone: tokens weighted by inverse document frequency, so a
                   shared rare surname counts for far more than a shared "mohammed"
  phonetic_score   best similarity of Double Metaphone codes, token by token (primary codes, and secondary
                   codes where there are any)
  arabic_score     best similarity after Arabic-name folding (articles, abd- compounds, bin/ibn, Mohammed
                   and other variant families, see arabic_names.py)
  birth_state      agree, close (a year apart), conflict, or missing
  country_state    agree, conflict, or missing
  identifier_state agree, conflict, or missing. Compared property by property: two different property types
                   (a passport number and a tax number) are not comparable and count as missing.
                   wikidataId, zipCode and programId are not used: a customer has none of them, and programId
                   names a sanctions programme, not the entity.

The five scores are in [0, 1]. An entity has up to MAX_NAMES names (caption plus name, alias, previousName,
weakAlias, deduplicated after normalisation). Names are compared by taking each name on one side, finding
its best partner on the other by character ratio, and scoring the TOP_PAIRS best pairs in full.

Standard library plus rapidfuzz and Metaphone.
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache

from metaphone import doublemetaphone
from rapidfuzz import fuzz, process

from arabic_names import arabic_normalise
from profile_data import COUNTRY_PROPS, NAME_PROPS, name_scripts
from scoring import scoring_form, skeleton_form, weighted_token_score

MAX_NAMES = 30
TOP_PAIRS = 3
BLEND = 0.5
SCORES = ("name_score", "skeleton_score", "weighted_score", "phonetic_score", "arabic_score")
STATES = ("birth_state", "country_state", "identifier_state")
EXCLUDED_IDENTIFIERS = {"zipCode", "wikidataId", "programId"}
SCRIPT_GROUPS = ("Latin", "Cyrillic", "Arabic")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


@lru_cache(maxsize=2_000_000)
def phonetic_token(token: str) -> tuple[str, str]:
    """(primary, secondary) Double Metaphone codes. Tokens that are not plain letters stand for themselves."""
    if token.isascii() and token.isalpha():
        primary, secondary = doublemetaphone(token)
        primary = primary or token.upper()
        return primary, secondary or primary
    return token, token


def phonetic_strings(tokens: Sequence[str]) -> tuple[str, str]:
    codes = [phonetic_token(t) for t in tokens]
    return " ".join(c[0] for c in codes), " ".join(c[1] for c in codes)


def entity_norms(entity: dict) -> list[str]:
    """The entity's distinct normalised names, legal forms stripped, at most MAX_NAMES."""
    props = entity.get("properties") or {}
    raw = [entity.get("caption") or ""]
    raw += [v for p in NAME_PROPS for v in props.get(p) or () if isinstance(v, str)]
    norms: list[str] = []
    seen: set[str] = set()
    for name in raw:
        form = scoring_form(name)
        if form and form not in seen:
            seen.add(form)
            norms.append(form)
        if len(norms) == MAX_NAMES:
            break
    return norms


def name_tokens(entity: dict) -> frozenset[str]:
    """Every token of every name of the entity, for counting document frequencies."""
    return frozenset(token for norm in entity_norms(entity) for token in norm.split())


class Forms:
    """One entity as the features need it: name forms and attributes."""

    __slots__ = (
        "norms", "tokens", "token_sets", "skeletons", "phonetic_primary", "phonetic_secondary", "arabic",
        "scripts", "years", "countries", "identifiers",
    )  # fmt: skip

    def __init__(self, entity: dict, identifier_props: Mapping[str, Sequence[str]]) -> None:
        props = entity.get("properties") or {}
        raw = [entity.get("caption") or ""]
        raw += [v for p in NAME_PROPS for v in props.get(p) or () if isinstance(v, str)]
        norms = entity_norms(entity)
        self.norms = norms
        self.tokens = [tuple(n.split()) for n in norms]
        self.token_sets = [frozenset(t) for t in self.tokens]
        self.skeletons = list(dict.fromkeys(s for s in map(skeleton_form, norms) if s))
        codes = [phonetic_strings(t) for t in self.tokens]
        self.phonetic_primary = list(dict.fromkeys(c[0] for c in codes))
        self.phonetic_secondary = list(dict.fromkeys(c[1] for c in codes))
        self.arabic = list(dict.fromkeys(a for a in map(arabic_normalise, norms) if a))
        scripts: set[str] = set()
        for name in raw:
            for script in name_scripts(name):
                scripts.add(script if script in SCRIPT_GROUPS else "other")
        self.scripts = frozenset(scripts)
        self.years = frozenset(
            int(v[:4]) for v in props.get("birthDate") or () if isinstance(v, str) and v[:4].isdigit()
        )
        self.countries = frozenset(
            v.lower() for p in COUNTRY_PROPS for v in props.get(p) or () if isinstance(v, str) and v
        )
        identifiers: dict[str, frozenset[str]] = {}
        for prop in identifier_props.get(entity.get("schema"), ()):
            if prop in EXCLUDED_IDENTIFIERS:
                continue
            values = {_NON_ALNUM.sub("", v.lower()) for v in props.get(prop) or () if isinstance(v, str)}
            values.discard("")
            if values:
                identifiers[prop] = frozenset(values)
        self.identifiers = identifiers


def make_idf(document_frequency: Mapping[str, int], documents: int) -> Callable[[str], float]:
    """Inverse document frequency of a name token, never below 0.1, for tokens never seen as well."""

    def idf(token: str) -> float:
        return max(0.1, math.log((documents + 1) / (document_frequency.get(token, 0) + 1)))

    return idf


def count_tokens(forms: Forms, counter: Counter) -> None:
    """Add one entity's name tokens to a document-frequency counter (each token once per entity)."""
    counter.update(set().union(*forms.token_sets))


def _best(query: str, choices: Sequence[str]) -> tuple[float, int]:
    found = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
    return (found[1] / 100, found[2]) if found else (0.0, -1)


def best_ratio(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    return max(_best(s, right)[0] for s in left)


def name_features(a: Forms, b: Forms, idf: Callable[[str], float]) -> dict[str, float]:
    zero = dict.fromkeys(SCORES, 0.0)
    if not a.norms or not b.norms:
        return zero
    pairs = []
    for i, name in enumerate(a.norms):
        ratio, j = _best(name, b.norms)
        pairs.append((ratio, i, j))
    pairs.sort(reverse=True)
    weighted = [weighted_token_score(a.tokens[i], b.tokens[j], b.token_sets[j], idf) for _, i, j in pairs[:TOP_PAIRS]]
    blended = [(1 - BLEND) * ratio + BLEND * w for (ratio, _, _), w in zip(pairs[:TOP_PAIRS], weighted, strict=True)]
    return {
        "name_score": max(blended),
        "skeleton_score": best_ratio(a.skeletons, b.skeletons),
        "weighted_score": max(weighted),
        "phonetic_score": max(
            best_ratio(a.phonetic_primary, b.phonetic_primary), best_ratio(a.phonetic_secondary, b.phonetic_secondary)
        ),
        "arabic_score": best_ratio(a.arabic, b.arabic),
    }


def birth_state(a: frozenset[int], b: frozenset[int]) -> str:
    if not a or not b:
        return "missing"
    gap = min(abs(x - y) for x in a for y in b)
    return "agree" if gap == 0 else "close" if gap == 1 else "conflict"


def country_state(a: frozenset[str], b: frozenset[str]) -> str:
    if not a or not b:
        return "missing"
    return "agree" if a & b else "conflict"


def identifier_state(a: Mapping[str, frozenset[str]], b: Mapping[str, frozenset[str]]) -> str:
    common = a.keys() & b.keys()
    if not common:
        return "missing"
    return "agree" if any(a[p] & b[p] for p in common) else "conflict"


def pair_features(a: Forms, b: Forms, idf: Callable[[str], float]) -> dict[str, float | str]:
    features: dict[str, float | str] = dict(name_features(a, b, idf))
    features["birth_state"] = birth_state(a.years, b.years)
    features["country_state"] = country_state(a.countries, b.countries)
    features["identifier_state"] = identifier_state(a.identifiers, b.identifiers)
    return features


def script_class(a: Forms, b: Forms) -> str:
    left, right = bool(a.scripts - {"Latin"}), bool(b.scripts - {"Latin"})
    if not left and not right:
        return "Latin on both sides only"
    if left != right:
        return "non-Latin on one side only"
    return "non-Latin on both sides"


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Area under the ROC curve (Mann-Whitney, ties count half). None if one class is missing."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum, i = 0.0, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        rank_sum += average_rank * sum(labels[order[k]] for k in range(i, j + 1))
        i = j + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def state_rates(states: Iterable[str], labels: Iterable[int]) -> dict[str, float]:
    """Share of positives in each state, for scoring a categorical feature as a number."""
    totals, positives = Counter(), Counter()
    for state, label in zip(states, labels, strict=True):
        totals[state] += 1
        positives[state] += label
    return {state: positives[state] / totals[state] for state in totals}

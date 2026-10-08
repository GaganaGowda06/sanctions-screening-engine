"""Name similarity that does not use the matcher: standard library only (unicodedata and difflib).

Used by the analysis scripts as an independent check on the matcher's own scores. It does not import anything from src/
and does not share the matcher's normalisation, transliteration, skeleton or rarity weighting.
"""

import re
import unicodedata
from difflib import SequenceMatcher


def tokens(name: str) -> list[str]:
    """Case-folded tokens of a name: accents stripped, punctuation treated as a space. Scripts are left as they are."""
    text = unicodedata.normalize("NFKD", name.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in re.split(r"[^\w]+", text) if t]


def similarity(a: str, b: str) -> float:
    """Character-level ratio (difflib) of the two names with their tokens sorted, so a reordering does not count."""
    ta, tb = sorted(tokens(a)), sorted(tokens(b))
    return SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio() if ta and tb else 0.0


def best_alias(query: str, aliases: list[str]) -> tuple[str | None, float]:
    """The alias most similar to the query, and the similarity."""
    scored = [(similarity(query, a), a) for a in aliases]
    if not scored:
        return None, 0.0
    best = max(scored, key=lambda t: (t[0], t[1]))
    return best[1], best[0]


def token_relation(query: str, alias: str) -> str:
    """How the token sets of the query and an alias relate: the same tokens (a reordering counts), a token
    missing from the query, a token added in the query, the same number of tokens spelled differently, or
    different counts both ways."""
    q, a = tokens(query), tokens(alias)
    if sorted(q) == sorted(a):
        return "same tokens"
    qs, as_ = set(q), set(a)
    if qs < as_:
        return "query is missing a token the alias has"
    if as_ < qs:
        return "query has a token the alias lacks"
    if len(q) == len(a):
        return "same number of tokens, some spelled differently"
    return "different number of tokens and spellings"

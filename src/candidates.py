"""Candidate generation: an inverted index over folded name tokens and consonant skeletons.

Each name is normalised (src/normalise.py) and split into tokens. Every token is folded so that
spelling variants of one sound meet (see FOLDS), and yields two keys:
  t:<folded token>
  s:<skeleton>   the folded token with vowels (a e i o u y w) removed and repeated letters collapsed

The skeleton lets an Arabic-derived form line up with its Latin spellings:
  "mhmd" (from محمد), "mohammed", "muhammad" and "mohamed" all give "s:mhmd".

Joined tokens: adjacent pairs and triples of tokens are also joined into one word and keyed the
same way, so "abd allah" and "abdallah", or "al hamdani" and "alhamdani", share keys.

A query is turned into keys the same way. To keep candidate sets small, keys that occur in
more than max_df entities (common names, legal forms such as "limited" or "llc") do not
generate candidates. They still add to the score of entities that are already candidates.
If every key of a query is that common, the two rarest keys are used so the query still
returns something. All candidates are returned, ranked by the summed inverse document frequency
of the keys they share with the query. There is no cut: the scoring step sees the whole set.

Known limits: silent letters ("John" and "Jon"), Han and kana names (they only match themselves).

Standard library only.
"""

import math
import re
from collections.abc import Iterable, Sequence

from normalise import normalise

_VOWELS = re.compile("[aeiouyw]")
_REPEATS = re.compile(r"(.)\1+")
_ARTICLE_MIN_LENGTH = 5  # "alhmdany" gets an extra key without "al", "alan" does not
_MAX_JOIN = 3  # longest run of adjacent tokens that is also keyed as one word

# Applied in order, to ASCII tokens only. Each rule folds spellings of one sound together:
#   dzh zh dj -> j     Russian ж/дж, Balkan dj
#   sh ch -> s         English sh, French ch
#   th dh -> t         Arabic ث ذ, both written "th" in English ("Thanoon", "Dhanun")
#   ph -> f, ck -> k
#   gh q -> k          Arabic غ ق, "Ghasempour", "Qasim", "Kasim"
#   kh -> h, g -> h    Arabic خ, Russian х, Ukrainian г ("Havrylov", "Gavrilov")
#   c -> k, z -> s, v -> w, x -> ks
FOLDS = (
    ("dzh", "j"), ("zh", "j"), ("dj", "j"), ("sh", "s"), ("ch", "s"), ("th", "t"), ("dh", "t"),
    ("ph", "f"), ("ck", "k"), ("gh", "k"), ("kh", "h"), ("q", "k"), ("g", "h"), ("c", "k"),
    ("z", "s"), ("v", "w"), ("x", "ks"),
)  # fmt: skip


def fold(token: str) -> str:
    """Fold spelling variants of the same sound. Non-ASCII tokens are returned unchanged."""
    if not token.isascii():
        return token
    for old, new in FOLDS:
        token = token.replace(old, new)
    return token


def skeleton(token: str) -> str:
    """Vowels removed and repeated letters collapsed. Non-ASCII tokens are returned unchanged."""
    if not token.isascii():
        return token
    return _REPEATS.sub(r"\1", _VOWELS.sub("", token))


def name_keys(name: str, do_fold: bool = True, join: bool = True, strip_article: bool = True) -> set[str]:
    """All index keys for a name. Used for both indexing and querying."""
    tokens = [t for t in normalise(name).split() if not (len(t) == 1 and t.isascii())]
    if do_fold:
        tokens = [fold(t) for t in tokens]
    words = set(tokens)
    if join:
        for size in range(2, _MAX_JOIN + 1):
            for start in range(len(tokens) - size + 1):
                words.add("".join(tokens[start : start + size]))
    keys: set[str] = set()
    for word in words:
        keys.add(f"t:{word}")
        if sk := skeleton(word):
            keys.add(f"s:{sk}")
        if strip_article and word.isascii() and word.startswith("al") and len(word) >= _ARTICLE_MIN_LENGTH:
            # Arabic joins the article to the name ("alhmdany"); Latin spellings often separate it.
            if sk := skeleton(word[2:]):
                keys.add(f"s:{sk}")
    return keys


def _rank_key(item: tuple[int, float]) -> tuple[float, int]:
    """Highest score first, ties broken by insertion order."""
    return (-item[1], item[0])


class CandidateIndex:
    def __init__(
        self, max_df: int = 1000, do_fold: bool = True, join: bool = True, strip_article: bool = True
    ) -> None:
        self.max_df = max_df
        self.do_fold = do_fold
        self.join = join
        self.strip_article = strip_article
        self.ids: list[str] = []
        self.postings: dict[str, list[int]] = {}

    @classmethod
    def build(
        cls,
        entities: Iterable[tuple[str, Sequence[str]]],
        max_df: int = 1000,
        do_fold: bool = True,
        join: bool = True,
        strip_article: bool = True,
    ) -> "CandidateIndex":
        """Build from (entity id, names) pairs."""
        index = cls(max_df=max_df, do_fold=do_fold, join=join, strip_article=strip_article)
        for entity_id, names in entities:
            index.add(entity_id, names)
        return index

    def _keys(self, name: str) -> set[str]:
        return name_keys(name, self.do_fold, self.join, self.strip_article)

    def add(self, entity_id: str, names: Sequence[str]) -> None:
        keys: set[str] = set()
        for name in names:
            keys |= self._keys(name)
        position = len(self.ids)
        self.ids.append(entity_id)
        for key in keys:
            self.postings.setdefault(key, []).append(position)

    def _idf(self, key: str) -> float:
        return math.log((len(self.ids) + 1) / (len(self.postings[key]) + 1))

    def query(self, name: str) -> list[tuple[str, float]]:
        """Every candidate for a name as (entity id, score), best first."""
        keys = [k for k in self._keys(name) if k in self.postings]
        if not keys:
            return []
        generating = {k for k in keys if len(self.postings[k]) <= self.max_df}
        if not generating:
            generating = set(sorted(keys, key=lambda k: len(self.postings[k]))[:2])

        scores: dict[int, float] = {}
        for key in generating:
            weight = self._idf(key)
            for position in self.postings[key]:
                scores[position] = scores.get(position, 0.0) + weight
        for key in keys:
            if key in generating:
                continue
            weight = self._idf(key)
            for position in self.postings[key]:
                if position in scores:
                    scores[position] += weight

        ranked = sorted(scores.items(), key=_rank_key)
        return [(self.ids[p], score) for p, score in ranked]

"""Name perturbations that do not come from the matcher: standard library only, no import from src/.

Each family makes realistic single edits of a name, one family at a time. The rules are written from conventions that
exist outside this project, not from the matcher's normalisation, transliteration or skeleton code:

  vowel_swap, vowel_delete, consonant_double, consonant_delete
        the usual single-character spelling slips: a substituted, omitted, doubled or collapsed letter (the edit types
        of classic spelling-error studies, the Damerau-Levenshtein operations). The vowel confusion pairs are the ones
        that occur between spellings of the same name in practice (a/e, e/i, i/y, o/u).
  drop_middle_token, drop_one_token
        a middle name or another token left out, as on forms with a short name field.
  reorder_tokens
        surname-first and given-name-first orders, and a rotated order.
  translit_cyrillic_gost, translit_cyrillic_german
        a Cyrillic-script name written out in Latin letters under a published scheme that differs from the matcher's:
        GOST 7.79-2000 system B, and the German (Duden) convention. Ukrainian letters follow the same pattern.
  translit_cyrillic_bgn
        the BGN/PCGN 1947 romanisation of Russian. The matcher's own Cyrillic table is documented as close to BGN/PCGN
        and is identical to this one on every Russian letter, so this family is NOT an alternate transliteration. It is
        kept as a control: re-typing the matcher's own scheme should lose nothing.
  respell_cyrillic_conventions, respell_arabic_conventions
        a Latin-script name rewritten under another romanisation convention (for example ya/ia/ja, kh/h/ch,
        Mohammed/Muhammad/Mohamed, Abdul/Abd al, ou/u). These apply only to a name whose entity has an alias in that
        script (the caller passes that fact), because the rules would make unrealistic edits to other names.

The tables were typed out from those schemes. Nothing here is imported from, or copied out of, normalise.py,
candidates.py or scoring.py; tests/test_analysis_perturb.py checks that this module imports nothing from src/. Every
generator is deterministic for a given name, family and seed, and returns at most `limit` distinct variants that differ
from the input.
"""

import random
import re

VOWELS = "aeiouy"
VOWEL_CONFUSION = {"a": "e", "e": "a", "i": "e", "o": "u", "u": "o", "y": "i"}
DOUBLEABLE = "bcdfglmnprst"
CONSONANTS = "bcdfghjklmnpqrstvwxz"

# --- published romanisation schemes for Russian (and the Ukrainian letters that differ) ---
BGN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i", "й": "y",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya", "і": "i", "ї": "yi", "є": "ye", "ґ": "g",
}  # fmt: skip
GOST = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i", "й": "j",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "x", "ц": "c", "ч": "ch", "ш": "sh", "щ": "shh", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya", "і": "i", "ї": "ji", "є": "e", "ґ": "g",
}  # fmt: skip
GERMAN = {
    "а": "a", "б": "b", "в": "w", "г": "g", "д": "d", "е": "e", "ё": "jo", "ж": "sh", "з": "s", "и": "i", "й": "j",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "ch", "ц": "z", "ч": "tsch", "ш": "sch", "щ": "schtsch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "ju",
    "я": "ja", "і": "i", "ї": "ji", "є": "je", "ґ": "g",
}  # fmt: skip
CYRILLIC_SCHEMES = {"translit_cyrillic_bgn": BGN, "translit_cyrillic_gost": GOST, "translit_cyrillic_german": GERMAN}

# --- Latin-script respelling rules, (pattern, replacement), each applied once, case-insensitively ---
CYRILLIC_CONVENTIONS = [
    (r"ya", "ia"), (r"ya", "ja"), (r"yu", "iu"), (r"yu", "ju"), (r"yo", "io"), (r"ye", "e"),
    (r"kh", "h"), (r"kh", "ch"), (r"zh", "j"), (r"zh", "sh"), (r"ts", "c"), (r"ts", "tz"),
    (r"shch", "sch"), (r"shch", "shh"), (r"sh", "ch"), (r"ch", "tch"),
    (r"iy\b", "y"), (r"iy\b", "ii"), (r"iy\b", "ij"), (r"yy\b", "y"), (r"sky\b", "ski"), (r"skiy\b", "sky"),
    (r"ev\b", "eff"), (r"ov\b", "off"), (r"ko\b", "ka"), (r"v", "w"), (r"w", "v"), (r"ii\b", "i"),
]  # fmt: skip
ARABIC_CONVENTIONS = [
    (r"mohammed", "muhammad"), (r"mohammad", "muhammed"), (r"muhammad", "mohamed"), (r"mohamed", "mohammed"),
    (r"abdul\b", "abd al"), (r"abd al\b", "abdul"), (r"abdel\b", "abdul"), (r"abdul\b", "abdel"),
    (r"\bal-", "el-"), (r"\bel-", "al-"), (r"\bal ", "el "), (r"ou", "u"), (r"u", "ou"),
    (r"ee", "i"), (r"i", "ee"), (r"aa", "a"), (r"a", "aa"), (r"q", "k"), (r"k", "q"),
    (r"kh", "h"), (r"gh", "g"), (r"dj", "j"), (r"j", "dj"), (r"sh", "ch"), (r"ch", "sh"), (r"th", "t"), (r"dh", "d"),
    (r"sayyid", "said"), (r"saleh", "salih"), (r"hussein", "husayn"), (r"yusuf", "youssef"), (r"khaled", "khalid"),
]  # fmt: skip


def _rng(name: str, family: str, seed: int) -> random.Random:
    return random.Random(f"{seed}|{family}|{name}")


def _distinct(name: str, variants: list[str], limit: int, rng: random.Random) -> list[str]:
    seen, out = {name}, []
    pool = list(dict.fromkeys(variants))
    rng.shuffle(pool)
    for v in pool:
        v = re.sub(r"\s+", " ", v).strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
        if len(out) >= limit:
            break
    return out


def _match_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _interior(name: str, chars: str) -> list[int]:
    """Positions of letters in `chars` that are inside a token (not its first letter)."""
    out, start = [], True
    for i, ch in enumerate(name):
        if ch.isalpha() and not start and ch.lower() in chars:
            out.append(i)
        start = not ch.isalnum() and ch not in "'"
    return out


def vowel_swap(name: str, limit: int = 3, seed: int = 0) -> list[str]:
    positions = _interior(name, VOWELS)
    swapped = [name[:i] + _match_case(name[i], VOWEL_CONFUSION[name[i].lower()]) + name[i + 1 :] for i in positions]
    return _distinct(name, swapped, limit, _rng(name, "vowel_swap", seed))


def vowel_delete(name: str, limit: int = 3, seed: int = 0) -> list[str]:
    positions = _interior(name, VOWELS)
    return _distinct(name, [name[:i] + name[i + 1 :] for i in positions], limit, _rng(name, "vowel_delete", seed))


def consonant_double(name: str, limit: int = 3, seed: int = 0) -> list[str]:
    """Repeat one interior consonant that is not already part of a doubled pair (no triple letters)."""
    positions = [
        i
        for i in _interior(name, DOUBLEABLE)
        if (i + 1 >= len(name) or name[i + 1].lower() != name[i].lower()) and name[i - 1].lower() != name[i].lower()
    ]
    return _distinct(name, [name[: i + 1] + name[i] + name[i + 1 :] for i in positions], limit, _rng(name, "cd", seed))


def consonant_delete(name: str, limit: int = 3, seed: int = 0) -> list[str]:
    """Collapse a doubled consonant if the name has one; otherwise drop one interior consonant."""
    doubled = [
        i
        for i in range(len(name) - 1)
        if name[i].isalpha() and name[i].lower() == name[i + 1].lower() and name[i].lower() in CONSONANTS
    ]
    if doubled:
        variants = [name[:i] + name[i + 1 :] for i in doubled]
    else:
        variants = [name[:i] + name[i + 1 :] for i in _interior(name, CONSONANTS)]
    return _distinct(name, variants, limit, _rng(name, "consonant_delete", seed))


def _tokens(name: str) -> list[str]:
    return name.split()


def drop_middle_token(name: str, limit: int = 3, seed: int = 0) -> list[str]:
    t = _tokens(name)
    if len(t) < 3:
        return []
    return _distinct(name, [" ".join(t[:i] + t[i + 1 :]) for i in range(1, len(t) - 1)], limit, _rng(name, "dm", seed))


def drop_one_token(name: str, limit: int = 3, seed: int = 0) -> list[str]:
    """Leave the first, or the last, token out (names with at least two tokens)."""
    t = _tokens(name)
    if len(t) < 2:
        return []
    return _distinct(name, [" ".join(t[1:]), " ".join(t[:-1])], limit, _rng(name, "d1", seed))


def reorder_tokens(name: str, limit: int = 3, seed: int = 0) -> list[str]:
    t = _tokens(name)
    if len(t) < 2:
        return []
    variants = [" ".join([t[-1], *t[:-1]]), " ".join([*t[1:], t[0]])]
    if len(t) >= 3:
        variants.append(" ".join([t[-1], *t[1:-1], t[0]]))
    return _distinct(name, variants, limit, _rng(name, "ro", seed))


def has_cyrillic(name: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in name)


def has_arabic(name: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f" for ch in name)


def _transliterate(name: str, scheme: dict[str, str]) -> str:
    out = []
    for ch in name:
        low = ch.lower()
        if low in scheme:
            out.append(_match_case(ch, scheme[low]) if scheme[low] else "")
        else:
            out.append(ch)
    return "".join(out)


def translit_name(scheme_name: str, name: str) -> str:
    return _transliterate(name, CYRILLIC_SCHEMES[scheme_name])


def make_translit(scheme_name: str):  # noqa: ANN201
    scheme = CYRILLIC_SCHEMES[scheme_name]

    def generate(name: str, limit: int = 1, seed: int = 0) -> list[str]:
        if not has_cyrillic(name):
            return []
        out = _transliterate(name, scheme)
        return [out] if out != name else []

    generate.__name__ = scheme_name
    return generate


def _respell(name: str, rules: list[tuple[str, str]], family: str, limit: int, seed: int) -> list[str]:
    if not any(ch.isalpha() and ch.isascii() for ch in name):
        return []
    variants = []
    for pattern, replacement in rules:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            new = _match_case(match.group(0), replacement)
            variants.append(name[: match.start()] + new + name[match.end() :])
    return _distinct(name, variants, limit, _rng(name, family, seed))


def respell_cyrillic_conventions(name: str, limit: int = 3, seed: int = 0) -> list[str]:
    return _respell(name, CYRILLIC_CONVENTIONS, "respell_cyrillic_conventions", limit, seed)


def respell_arabic_conventions(name: str, limit: int = 3, seed: int = 0) -> list[str]:
    return _respell(name, ARABIC_CONVENTIONS, "respell_arabic_conventions", limit, seed)


def control_uppercase(name: str, limit: int = 1, seed: int = 0) -> list[str]:
    """A control, not a perturbation of the name: the same letters in capitals. It should never change a result."""
    return [name.upper()] if name.upper() != name else []


FAMILIES = {
    "vowel_swap": vowel_swap,
    "vowel_delete": vowel_delete,
    "consonant_double": consonant_double,
    "consonant_delete": consonant_delete,
    "drop_middle_token": drop_middle_token,
    "drop_one_token": drop_one_token,
    "reorder_tokens": reorder_tokens,
    "translit_cyrillic_bgn": make_translit("translit_cyrillic_bgn"),
    "translit_cyrillic_gost": make_translit("translit_cyrillic_gost"),
    "translit_cyrillic_german": make_translit("translit_cyrillic_german"),
    "respell_cyrillic_conventions": respell_cyrillic_conventions,
    "respell_arabic_conventions": respell_arabic_conventions,
    "control_uppercase": control_uppercase,
}

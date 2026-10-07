"""Name normalisation and transliteration to a comparable Latin form.

normalise("Юрий Щербак")  -> "yuriy shcherbak"
normalise("محمد علي")      -> "mhmd ly"

Steps: NFKC (folds fullwidth forms and Arabic presentation forms), casefold, transliterate
Cyrillic and Arabic, strip accents from Latin letters, drop apostrophes, turn other
punctuation into spaces, collapse whitespace.

Design choices that matter for matching:
- Cyrillic uses a fixed letter-for-letter scheme close to BGN/PCGN, with no language
  detection and no positional rules (so Russian "Евгений" gives "evgeniy", not "yevgeny").
- Arabic is written without short vowels, so the output is a consonant skeleton plus long
  vowels ("محمد" gives "mhmd", not "muhammad"). Short vowels cannot be recovered by rules.
- Scripts without a table (Han, kana, Greek, Hebrew, Georgian, ...) are lowercased and
  cleaned of punctuation but otherwise left as they are. is_clean() flags them.

Standard library only.
"""

import re
import unicodedata

_CYRILLIC = {
    # Russian
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
    # Ukrainian, Belarusian, Serbian, Macedonian
    "і": "i", "ї": "yi", "є": "ye", "ґ": "g", "ў": "u", "ј": "j", "љ": "lj", "њ": "nj",
    "ћ": "c", "ђ": "dj", "џ": "dz", "ѕ": "dz", "ќ": "k", "ѓ": "g",
    # Central Asian and other Cyrillic-script languages
    "ә": "a", "ғ": "gh", "қ": "q", "ң": "ng", "ө": "o", "ү": "u", "ұ": "u", "һ": "h",
    "ҳ": "h", "ҷ": "j", "ӣ": "i", "ӯ": "u", "ӏ": "", "ҡ": "q", "ҙ": "z", "ҫ": "s",
}  # fmt: skip

_ARABIC = {
    "ا": "a", "أ": "a", "إ": "a", "آ": "a", "ٱ": "a", "ء": "", "ؤ": "u", "ئ": "i",
    "ب": "b", "ة": "a", "ت": "t", "ث": "th", "ج": "j", "ح": "h", "خ": "kh", "د": "d",
    "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t",
    "ظ": "z", "ع": "", "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m",
    "ن": "n", "ه": "h", "و": "w", "ى": "a", "ي": "y",
    # Persian and Urdu additions
    "پ": "p", "چ": "ch", "ژ": "zh", "گ": "g", "ک": "k", "ی": "y", "ٹ": "t", "ڈ": "d",
    "ڑ": "r", "ں": "n", "ھ": "h", "ہ": "h", "ے": "y", "ڤ": "v", "ۃ": "a",
    # Pashto and Kurdish additions
    "ټ": "t", "ډ": "d", "ړ": "r", "ږ": "zh", "ښ": "sh", "ځ": "dz", "څ": "ts", "ڼ": "n",
    "ۏ": "w", "ۆ": "o", "ۅ": "o", "ە": "e", "ێ": "e", "ڕ": "r", "ڵ": "l",
}  # fmt: skip

# Latin letters that do not decompose into base letter plus accent.
_LATIN_SPECIAL = {
    "ø": "o", "đ": "d", "ł": "l", "æ": "ae", "œ": "oe", "þ": "th", "ð": "d", "ı": "i",
    "ŋ": "ng", "ħ": "h", "ĸ": "k", "ə": "a",
}  # fmt: skip

_APOSTROPHES = "'`´‘’‛ʹʻʼʽʾʿˮ"
# Zero-width and directional marks, soft hyphen, BOM.
_INVISIBLE = "​‌‍‎‏؜­﻿"


def _build_table() -> dict[int, str]:
    table: dict[int, str] = {}
    for source in (_CYRILLIC, _ARABIC, _LATIN_SPECIAL):
        table.update({ord(k): v for k, v in source.items()})
    for ch in _APOSTROPHES + _INVISIBLE + "ـ":  # last one is the Arabic tatweel
        table[ord(ch)] = ""
    # Standalone combining marks, Arabic vowel signs and Quranic annotation marks.
    for lo, hi in ((0x0300, 0x036F), (0x064B, 0x065F), (0x06D6, 0x06DC),
                   (0x06DF, 0x06E4), (0x06E7, 0x06E8), (0x06EA, 0x06ED)):  # fmt: skip
        for cp in range(lo, hi + 1):
            table[cp] = ""
    table[0x0670] = ""
    # Arabic-Indic and extended Arabic-Indic digits.
    for i in range(10):
        table[0x0660 + i] = str(i)
        table[0x06F0 + i] = str(i)
    return table


_TABLE = _build_table()
_LATIN_ACCENTED = re.compile("[À-ɏḀ-ỿ]")
_NON_ALNUM_ASCII = re.compile("[^a-z0-9]+")
_CLEAN = re.compile("[a-z0-9]+(?: [a-z0-9]+)*")


def _strip_accent(match: re.Match[str]) -> str:
    decomposed = unicodedata.normalize("NFKD", match.group())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def normalise(name: str) -> str:
    """Return the comparable Latin form of a name (may be empty)."""
    s = unicodedata.normalize("NFKC", name).casefold().translate(_TABLE)
    if not s.isascii():
        s = _LATIN_ACCENTED.sub(_strip_accent, s)
    if s.isascii():
        return " ".join(_NON_ALNUM_ASCII.sub(" ", s).split())
    # Untransliterated scripts: keep letters, digits and combining marks, split on the rest.
    kept = (c if c.isalnum() or unicodedata.category(c)[0] == "M" else " " for c in s)
    return " ".join("".join(kept).split())


def is_clean(normalised: str) -> bool:
    """True if the output is non-empty and only lowercase a-z, digits and single spaces."""
    return _CLEAN.fullmatch(normalised) is not None

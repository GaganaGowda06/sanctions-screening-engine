"""Arabic-name normalisation for matching, applied to a name already normalised by normalise.py.

Latin spellings of Arabic names vary in ways ordinary string similarity punishes:
  articles           al-Hamdani, el-Hamdani, alhamdani; with sun letters ar-Rashid, ash-Shami, an-Nasser
  abd- compounds     Abd al-Rahman, Abdul Rahman, Abdelrahman, Abdurrahman; Abdullah, Abdallah
  bin / ibn          dropped: Osama bin Laden, Osama Laden
  Mohammed           Mohammed, Muhammad, Mohamad, Mohd, and the consonant skeleton "mhmd" that Arabic script gives
  common names       Hussein / Husayn, Youssef / Yusuf, Khaled / Khalid, ...
The output folds all of those to one form, so a fuzzy comparison of two outputs sees them as the same.
Letter-level variation that is left (Qaddafi against Gaddafi) is for the fuzzy comparison and the
phonetic and skeleton features to absorb.

Standard library only.
"""

import re

# Articles written as separate tokens. Sun-letter forms only count before a token starting with that letter.
_ALWAYS_ARTICLES = {"al", "el", "ul", "ol", "il"}
_SUN_ARTICLES = {
    "ar": "r", "as": "s", "ash": "sh", "at": "t", "ad": "d", "an": "n", "az": "z", "adh": "dh", "ath": "th",
    "ed": "d", "es": "s", "esh": "sh", "et": "t", "en": "n", "er": "r", "ez": "z",
}  # fmt: skip
# Names that merely start with the letters of an article.
_NOT_ARTICLE_PREFIXED = {
    "albert", "alberto", "alfred", "alfredo", "alfonso", "allen", "alvin", "alton", "almond", "alpha", "alder",
}  # fmt: skip
_ABD_PREFIXES = {"abd", "abdul", "abdel", "abdal", "abdu", "abdur", "abdus", "bd"}
_ABD_JOINED = ("abdul", "abdel", "abdal", "abdur", "abdus", "abdu")
_ALLAH = {"allah", "allh", "alah", "ullah", "ulla", "alla", "lah"}
_DROPPED = {"bin", "ibn", "bint", "binti", "bn"}
_ABU = {"abu", "abou", "abo", "aboo", "abuu"}
_MOHAMMED = re.compile(r"^(mo|mu|mou)h?a?m+[ae]d$")
_MOHAMMED_ABBREVIATIONS = {"mohd", "muhd", "mhd", "md", "mohamd", "mhmd", "mhammad", "mhamad", "mhmmd"}

_FAMILIES = {
    "muhammad": [],
    "ahmad": ["ahmed", "ahmet", "ahmat", "ahmd"],
    "husayn": ["hussein", "hussain", "husain", "hosein", "hossein", "huseyn", "hsyn", "hosain", "hussin"],
    "hasan": ["hassan", "hasen", "hsn"],
    "ali": ["aly", "ally", "ly"],
    "umar": ["omar", "omer", "oumar", "umer"],
    "yusuf": ["yousef", "youssef", "yousuf", "yusif", "yousif", "youssouf", "ysf"],
    "ibrahim": ["ebrahim", "ibrahem", "ibraheem", "brhym"],
    "khalid": ["khaled", "khalad", "khlyd"],
    "mahmud": ["mahmoud", "mahmood", "mehmood", "mahmud"],
    "mustafa": ["mostafa", "moustafa", "mustapha", "mustofa", "mstfa"],
    "salih": ["salah", "saleh", "salih", "slh"],
    "sulayman": ["suleiman", "sulaiman", "soliman", "suleyman", "sulaimon", "slymn"],
    "ismail": ["ismael", "ismayil", "ismaeel", "ismaiel"],
    "usama": ["osama", "oussama", "usamah", "osamah"],
    "hamza": ["hamzah", "hamzeh", "hmzh"],
    "zaynab": ["zainab", "zeinab", "zaineb", "zaynb"],
    "fatima": ["fatma", "fatimah", "fatema", "ftmh"],
    "aisha": ["ayesha", "aysha", "aiysha", "aishah", "aysh"],
    "rahman": ["rehman", "rahmen", "rhmn"],
    "rashid": ["rasheed", "rashed", "rshyd"],
    "karim": ["kareem", "kerim", "krym"],
    "nasir": ["nasser", "nassir", "naser", "nasr", "nasser"],
    "sayyid": ["sayed", "seyed", "sayid", "syed", "sayyed", "seyyed", "sayyid", "syd"],
    "said": ["saeed", "saied", "sa'id", "said"],
}
_CANONICAL = {variant: canonical for canonical, variants in _FAMILIES.items() for variant in variants}


def _is_article_before(token: str, following: str) -> bool:
    if token in _ALWAYS_ARTICLES:
        return len(following) >= 2
    letter = _SUN_ARTICLES.get(token)
    return letter is not None and following.startswith(letter) and len(following) >= 2


def _strip_joined_article(token: str) -> str:
    """alhamdani -> hamdani, arrashid -> rashid. Left alone: alexander, albert, assad."""
    if token in _NOT_ARTICLE_PREFIXED or len(token) < 6:
        return token
    if token[:2] in ("al", "el", "ul") and token[2] not in "aeiouy":
        return token[2:]
    for article, letter in _SUN_ARTICLES.items():
        if token.startswith(article + letter) and len(token) >= len(article) + len(letter) + 3:
            return token[len(article) :]
    return token


def _fold_letters(token: str) -> str:
    token = re.sub(r"(.)\1+", r"\1", token)  # ll -> l, aa -> a
    return token.replace("ou", "u").replace("oo", "u").replace("ee", "i")


def _canonical(token: str) -> str:
    token = _CANONICAL.get(token, token)
    if token in _MOHAMMED_ABBREVIATIONS or _MOHAMMED.match(token):
        return "muhammad"
    return token


def arabic_normalise(name: str) -> str:
    """Fold article, abd- and bin/ibn forms and common name variants. `name` is already lower-case ASCII words."""
    tokens = name.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        if token in _DROPPED:
            i += 1
        elif following and _is_article_before(token, following):
            i += 1
        elif token in _ABD_PREFIXES:
            j = i + 1
            if j + 1 < len(tokens) and _is_article_before(tokens[j], tokens[j + 1]):
                j += 1
            if j < len(tokens):
                rest = tokens[j]
                rest = "lah" if rest in _ALLAH else _strip_joined_article(rest)
                out.append(_fold_letters("abd" + rest))
                i = j + 1
            else:
                out.append("abd")
                i += 1
        else:
            joined = next((p for p in _ABD_JOINED if token.startswith(p) and len(token) >= len(p) + 2), None)
            if joined:
                rest = token[len(joined) :]
                rest = "lah" if rest in _ALLAH else _strip_joined_article(rest)
                out.append(_fold_letters("abd" + rest))
            elif token in _ABU:
                out.append("abu")
            else:
                token = _strip_joined_article(token)
                out.append(_fold_letters(_canonical(_fold_letters(token) if token not in _CANONICAL else token)))
            i += 1
    return " ".join(t for t in out if t)

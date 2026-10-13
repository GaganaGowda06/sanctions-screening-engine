"""Tests for src/normalise.py. Expected outputs were worked out by hand from the tables.

Run:
    pytest tests/test_normalise.py
"""

import pytest

from normalise import is_clean, normalise


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  José   O'Brien-Smith ", "jose obrien smith"),  # accent, apostrophe dropped, hyphen to space
        ("Ørsted Æble", "orsted aeble"),  # letters that do not decompose
        ("Straße MÜLLER", "strasse muller"),  # casefold ß, strip umlaut
        ("Ｊｏｈｎ Ｓｍｉｔｈ", "john smith"),  # fullwidth Latin folded by NFKC
        ("Đoković", "dokovic"),
        ("“Reaviz\u02ee", "reaviz"),  # modifier letter double apostrophe used as a closing quote
        ("İstanbul", "istanbul"),  # dotted capital I leaves a combining dot, which is removed
        ("Jean-Pierre  de   la Fontaine", "jean pierre de la fontaine"),
        ("ACME Corp. (Holdings) Ltd.", "acme corp holdings ltd"),
        ("Unit 42", "unit 42"),
    ],
)
def test_latin(raw, expected):
    assert normalise(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Владимир Путин", "vladimir putin"),
        ("Юрий Щербак", "yuriy shcherbak"),  # ю=yu, й=y, щ=shch
        ("Ёлкин", "yolkin"),  # ё=yo, and й/ё survive because we transliterate before stripping accents
        ("Жуков Хрущёв", "zhukov khrushchyov"),  # ж=zh, х=kh, щ=shch, ё=yo
        ("Олексій Їжак", "oleksiy yizhak"),  # Ukrainian і, ї
        ("Нұрсұлтан Назарбаев", "nursultan nazarbaev"),  # Kazakh ұ
        ("Ђоковић", "djokovic"),  # Serbian ђ, ћ
        ("Объект Ельцин", "obekt eltsin"),  # hard sign dropped
        ("Борис Ельцин-Мл.", "boris eltsin ml"),
        ("ВЛАДИМИР", "vladimir"),  # uppercase
    ],
)
def test_cyrillic(raw, expected):
    assert normalise(raw) == expected


def test_cyrillic_decomposed_input_matches_composed():
    composed = "\u0439\u0451"  # й ё as single code points
    decomposed = "\u0438\u0306\u0435\u0308"  # и + breve, е + diaeresis
    assert composed != decomposed
    assert normalise(decomposed) == normalise(composed) == "yyo"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("محمد علي", "mhmd ly"),  # consonant skeleton; ع dropped
        ("مُحَمَّد", "mhmd"),  # fatha, damma, shadda stripped
        ("محـمد", "mhmd"),  # tatweel removed
        ("عبد الله", "bd allh"),
        ("فاطمة", "fatma"),  # alef kept as a, taa marbuta as a
        ("عائشة", "aisha"),  # ئ as i
        ("پوتین", "pwtyn"),  # Persian پ and ی
        ("٢٠٢٤", "2024"),  # Arabic-Indic digits
        ("محمد\u200cعلي", "mhmdly"),  # zero-width non-joiner removed
        ("ځدران", "dzdran"),  # Pashto ځ
        ("ﷲ", "allh"),  # presentation ligature expanded by NFKC
    ],
)
def test_arabic(raw, expected):
    assert normalise(raw) == expected


def test_mixed_script_name():
    assert normalise("Gazprom Нефть") == "gazprom neft"


@pytest.mark.parametrize("raw", ["张伟", "たなか", "კონსტანტინე", "Αλέξανδρος", "משה"])
def test_untransliterated_scripts_are_kept_and_flagged(raw):
    out = normalise(raw)
    assert out == raw.casefold()
    assert not is_clean(out)


def test_kana_voiced_and_unvoiced_stay_distinct():
    assert normalise("が") != normalise("か")


@pytest.mark.parametrize("raw", ["", "   ", "—-", "()", "'''"])
def test_empty_outputs(raw):
    assert normalise(raw) == ""
    assert not is_clean("")


def test_is_clean():
    assert is_clean("vladimir putin")
    assert is_clean("unit 42")
    assert not is_clean("张伟")
    assert not is_clean("gazprom 张")
    assert not is_clean(" leading space")
    assert not is_clean("double  space")


@pytest.mark.parametrize(
    "raw",
    ["Владимир Путин", "محمد علي", "José O'Brien", "Ørsted", "张伟 Wei", "Gazprom Нефть", "  ", "Ｊｏｈｎ"],
)
def test_idempotent(raw):
    once = normalise(raw)
    assert normalise(once) == once

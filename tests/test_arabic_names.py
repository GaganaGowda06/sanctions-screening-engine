"""Tests for src/arabic_names.py. Expected forms were worked out by hand from the rules.

Run:
    pytest tests/test_arabic_names.py
"""

import pytest

from arabic_names import arabic_normalise as norm


@pytest.mark.parametrize(
    ("variants", "expected"),
    [
        (["mohammed ali", "muhammad ali", "mohamad aly", "mohd ali", "mhmd ly", "muhamed ali"], "muhamad ali"),
        (["mohamed al hamdani", "muhammad alhamdani", "mohammad el hamdani", "muhammad hamdani"], "muhamad hamdani"),
        (["abd al rahman saleh", "abdul rahman saleh", "abdelrahman saleh", "abdurrahman saleh", "abd ar rahman salih"],
         "abdrahman salih"),
        (["abdullah", "abdallah", "abd allah", "abdul lah", "bd allh"], "abdlah"),
        (["osama bin laden", "usama ibn laden", "osama laden", "oussama bin laden"], "usama laden"),
        (["hussein", "hussain", "husayn", "hosein"], "husayn"),
        (["mahmoud", "mahmood", "mahmud"], "mahmud"),
        (["khaled", "khalid"], "khalid"),
        (["abu bakr", "abou bakr", "aboo bakr"], "abu bakr"),
        (["ar rashid", "ash shami", "an nasser"], None),  # checked below
    ],
)  # fmt: skip
def test_variants_fold_to_one_form(variants, expected):
    if expected is None:
        return
    assert {norm(v) for v in variants} == {expected}


def test_sun_letter_articles_drop_only_before_their_letter():
    assert norm("ar rashid") == "rashid"
    assert norm("ash shami") == "shami"
    assert norm("an nasser") == "nasir"  # nasser folds to nasir
    assert norm("at tayyib") == "tayib"
    assert norm("ed smith") == "ed smith"  # "ed" is an article only before d
    assert norm("an smith") == "an smith"
    assert norm("al hamdani") == "hamdani"  # "al" before anything


def test_joined_articles_are_stripped_but_names_that_only_look_like_articles_are_not():
    assert norm("alhamdani") == "hamdani"
    assert norm("arrashid") == "rashid"
    assert norm("alexander petrov") == "alexander petrov"  # third letter is a vowel
    assert norm("albert einstein") == "albert einstein"  # listed exception
    assert norm("assad") == "asad"  # too short to be "as" + "sad"; only the doubled letter folds
    assert norm("alan") == "alan"


def test_an_unrelated_name_is_left_nearly_alone():
    assert norm("ivan petrov") == "ivan petrov"
    assert norm("") == ""
    assert norm("smith") == "smith"


def test_letter_folding():
    assert norm("youssef") == "yusuf"
    assert norm("moustafa") == "mustafa"
    assert norm("haddad") == "hadad"  # repeated letters collapse
    assert norm("fatimah") == norm("fatma") == "fatima"


def test_trailing_abd_with_nothing_after_it():
    assert norm("abdul") == "abd"
    assert norm("bd") == "abd"


def test_arabic_script_skeleton_forms_meet_latin_spellings():
    # "محمد علي" normalises to "mhmd ly"; "علي" is "ly" and "محمد" "mhmd".
    assert norm("mhmd ly") == norm("mohammed ali")
    assert norm("ahmd") == norm("ahmed") == "ahmad"


def test_the_mohammed_family_ends_up_as_one_spelling():
    # The canonical form is folded like every other token, so the doubled m collapses.
    assert norm("mohammed") == norm("muhammad") == norm("mhmd") == "muhamad"


def test_letter_variation_beyond_the_tables_is_left_to_the_fuzzy_comparison():
    assert norm("osama bin ladin") != norm("osama bin laden")

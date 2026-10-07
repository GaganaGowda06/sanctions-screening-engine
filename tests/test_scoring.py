"""Tests for src/scoring.py. Expected values were worked out by hand.

Run:
    pytest tests/test_scoring.py
"""

import pytest

from scoring import (
    Adjustments,
    Components,
    Screener,
    best_similarity,
    combine,
    country_adjustment,
    date_adjustment,
    make_record,
    skeleton_form,
)

ADJ = Adjustments()  # dob_full 0.10, dob_year 0.05, dob_conflict 0.30, country_agree 0.03, country_conflict 0.15


@pytest.mark.parametrize(
    ("normalised", "expected"),
    [
        ("mohammed ali", "mhmd l"),  # mhmd, ali -> l
        ("mhmd ly", "mhmd l"),  # the Arabic-derived form gives the same skeleton
        ("dzhon smith", "jn smt"),  # dzh -> j, th -> t
        ("a e", ""),  # tokens that are all vowels disappear
        ("", ""),
    ],
)
def test_skeleton_form(normalised, expected):
    assert skeleton_form(normalised) == expected


@pytest.mark.parametrize(
    ("query", "entity_dates", "expected"),
    [
        ("1970-05-12", ["1970-05-12"], 0.10),  # same full date
        ("1970-05-12", ["1970"], 0.05),  # entity has only a year
        ("1970-05-12", ["1970-01-01"], 0.05),  # same year, different day
        ("1970", ["1970-05-12"], 0.05),  # query has only a year
        ("1970", ["1971"], 0.0),  # one year apart: no effect
        ("1970", ["1969-03-01"], 0.0),
        ("1970", ["1975"], -0.30),  # clear conflict
        ("1970", ["1975", "1970"], 0.05),  # any entity date may agree
        ("1970", ["1950", "1990"], -0.30),
        (None, ["1970"], 0.0),  # missing query date
        ("1970", [], 0.0),  # missing entity date
        ("1970", ["unknown"], 0.0),  # unparseable entity date
        ("abcd", ["1970"], 0.0),  # unparseable query date
    ],
)
def test_date_adjustment(query, entity_dates, expected):
    assert date_adjustment(query, entity_dates, ADJ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("query", "countries", "expected"),
    [
        ("ru", frozenset({"ru", "ua"}), 0.03),
        ("RU", frozenset({"ru"}), 0.03),  # case-insensitive
        ("us", frozenset({"ru"}), -0.15),
        (None, frozenset({"ru"}), 0.0),
        ("us", frozenset(), 0.0),
    ],
)
def test_country_adjustment(query, countries, expected):
    assert country_adjustment(query, countries, ADJ) == pytest.approx(expected)


def test_combine():
    c = Components("e", name=0.8, skeleton=1.0, dob=0.05, country=-0.15)
    # max(0.8, 0.9 * 1.0) = 0.9, plus 0.05, minus 0.15
    assert combine(c, 0.9) == pytest.approx(0.80)
    assert combine(c, 0.9, use_dob=False) == pytest.approx(0.75)
    assert combine(c, 0.9, use_country=False) == pytest.approx(0.95)
    assert combine(c, 0.9, use_dob=False, use_country=False) == pytest.approx(0.90)
    assert combine(c, 0.7) == pytest.approx(0.70)  # 0.7 * 1.0 < 0.8, so the name form wins: 0.8 + 0.05 - 0.15


def test_combine_clips_to_unit_interval():
    assert combine(Components("e", 1.0, 1.0, 0.10, 0.03), 0.9) == 1.0
    assert combine(Components("e", 0.4, 0.0, -0.30, -0.15), 0.9) == 0.0


def test_best_similarity():
    assert best_similarity("mohammed ali", ["ali mohammed"]) == pytest.approx(1.0)  # word order ignored
    assert best_similarity("mohammed ali", ["ivan petrov", "muhammad ali"]) == pytest.approx(0.8333, abs=1e-3)
    assert best_similarity("x", []) == 0.0
    assert best_similarity("", ["a"]) == 0.0


def test_make_record_deduplicates_and_lowercases():
    record = make_record(["Mohammed Ali", "MOHAMMED  ALI", "محمد علي", "()"], ["1970", "1970"], ["SA", "ae"])
    assert record.norms == ("mohammed ali", "mhmd ly")
    assert record.skeletons == ("mhmd l",)  # both names share one skeleton
    assert record.birth_dates == ("1970",)
    assert record.countries == frozenset({"sa", "ae"})


def screener():
    return Screener.build(
        [
            {"id": "e1", "names": ["Mohammed Ali", "محمد علي"], "birth_dates": ["1970-05-12"], "countries": ["sa"]},
            {"id": "e2", "names": ["Ivan Petrov"], "birth_dates": ["1980"], "countries": ["ru"]},
            {"id": "e3", "names": ["Ivan Petrov Sidorov"], "countries": ["ru"]},
        ]
    )


def scores(matches):
    return {m.entity_id: m.score for m in matches}


def test_name_only_match_uses_the_skeleton_form():
    # Normalised forms differ (ratio 0.8333) but skeletons are identical, so 0.9 * 1.0 = 0.9 wins.
    assert scores(screener().screen("Muhammad Ali", threshold=0.5)) == pytest.approx({"e1": 0.9})


def test_agreeing_birth_date_and_country_lift_the_score_to_the_cap():
    result = screener().screen("Muhammad Ali", "1970-05-12", "sa", threshold=0.5)
    assert scores(result) == {"e1": 1.0}  # 0.9 + 0.10 + 0.03, clipped


def test_conflicting_birth_date_and_country_remove_the_match():
    # 0.9 - 0.30 - 0.15 = 0.45, below even a 0.5 threshold.
    assert screener().screen("Muhammad Ali", "1999-01-01", "us", threshold=0.5) == []


def test_missing_attributes_have_no_effect():
    assert scores(screener().screen("Muhammad Ali", None, "sa", threshold=0.5))["e1"] == pytest.approx(0.93)
    assert scores(screener().screen("Muhammad Ali", "1970", None, threshold=0.5))["e1"] == pytest.approx(0.95)


def test_entity_without_attributes_is_not_affected_by_query_attributes():
    # e3 has no birth date, so only the country (agreeing, +0.03) applies to it.
    plain = scores(screener().screen("Ivan Petrov Sidorov", threshold=0.5))["e3"]
    with_query = scores(screener().screen("Ivan Petrov Sidorov", "1950", "ru", threshold=0.5))["e3"]
    assert with_query == pytest.approx(min(1.0, plain + 0.03))


def test_arabic_query_matches_through_its_own_name_form():
    assert scores(screener().screen("محمد علي", threshold=0.9))["e1"] == pytest.approx(1.0)


def test_word_order_does_not_matter_and_extra_tokens_score_lower():
    result = screener().screen("Petrov Ivan", threshold=0.5)
    assert result[0].entity_id == "e2" and result[0].score == pytest.approx(1.0)
    assert scores(result)["e3"] < 1.0


def test_threshold_filters_matches():
    assert [m.entity_id for m in screener().screen("Ivan Petrov", threshold=0.99)] == ["e2"]


def test_unknown_name_returns_nothing():
    assert screener().screen("Zzyzx Qwrt") == []


def test_components_keep_limits_the_number_returned():
    assert len(screener().components("Ivan Petrov", keep=1)) == 1


def test_components_always_keeps_the_named_entity():
    s = screener()
    assert [c.entity_id for c in s.components("Ivan Petrov", keep=1)] == ["e2"]
    assert {c.entity_id for c in s.components("Ivan Petrov", keep=1, always="e3")} == {"e2", "e3"}


def test_components_rank_by_best_reachable_score_so_a_boost_can_lift_an_entity_into_the_kept_set():
    entities = [
        {"id": "a", "names": ["Karim Nasser"], "birth_dates": ["1950"]},
        {"id": "b", "names": ["Karim Nasser"], "birth_dates": ["1970-05-12"]},
    ]
    kept = Screener.build(entities).components("Karim Nasser", "1970-05-12", None, keep=1)
    assert [c.entity_id for c in kept] == ["b"]  # name scores tie, "b" can reach 1.0 with the full-date boost

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
    has_arabic,
    make_record,
    name_scores,
    skeleton_form,
    strip_legal_forms,
    weighted_token_score,
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


def test_combine_skeleton_confirms_only_above_the_floor_and_only_up_to_the_lift():
    # name 0.8 is above the floor 0.5, so it can be raised towards the skeleton (1.0) by at most 0.1: 0.9.
    c = Components("e", names=(0.8,), skeleton=1.0, dob=0.05, country=-0.15)
    assert combine(c, 0.5, 0.1) == pytest.approx(0.80)  # 0.9 + 0.05 - 0.15
    assert combine(c, 0.5, 0.1, use_dob=False) == pytest.approx(0.75)
    assert combine(c, 0.5, 0.1, use_country=False) == pytest.approx(0.95)
    assert combine(c, 0.5, 0.1, use_dob=False, use_country=False) == pytest.approx(0.90)


def test_combine_skeleton_cannot_produce_a_high_score_alone():
    # Name 0.3 is below the floor, so a perfect skeleton does nothing.
    assert combine(Components("e", (0.3,), 1.0, 0, 0), 0.5, 0.3) == pytest.approx(0.3)
    # Exactly at the floor it can lift.
    assert combine(Components("e", (0.5,), 1.0, 0, 0), 0.5, 0.2) == pytest.approx(0.7)
    # The lift stops at the skeleton similarity itself.
    assert combine(Components("e", (0.6,), 0.65, 0, 0), 0.5, 0.3) == pytest.approx(0.65)
    # A skeleton below the name score never lowers it.
    assert combine(Components("e", (0.8,), 0.2, 0, 0), 0.5, 0.3) == pytest.approx(0.8)


def test_combine_picks_the_name_score_for_the_requested_blend():
    c = Components("e", names=(0.9, 0.7), skeleton=0.0, dob=0.0, country=0.0)
    assert combine(c, 0.5, 0.1, blend_index=0) == pytest.approx(0.9)
    assert combine(c, 0.5, 0.1, blend_index=1) == pytest.approx(0.7)


def test_combine_arabic_involved_pairs_get_the_full_skeleton():
    # Below the floor and far above the lift cap, but the pair is Arabic-involved: the skeleton stands in.
    c = Components("e", names=(0.3,), skeleton=0.9, dob=0.0, country=0.0, arabic=True)
    assert combine(c, 0.5, 0.1) == pytest.approx(0.9)
    assert combine(c, 0.5, 0.1, arabic_full_lift=False) == pytest.approx(0.3)  # the gate applies instead
    # The skeleton never lowers a better name score.
    assert combine(Components("e", (0.8,), 0.2, 0, 0, arabic=True), 0.5, 0.1) == pytest.approx(0.8)


def test_combine_clips_to_unit_interval():
    assert combine(Components("e", (1.0,), 1.0, 0.10, 0.03)) == 1.0
    assert combine(Components("e", (0.4,), 0.0, -0.30, -0.15)) == 0.0


@pytest.mark.parametrize(
    ("normalised", "expected"),
    [
        ("limited liability company nextstep", "nextstep"),
        ("obshchestvo s ogranichennoy otvetstvennostyu nekststep", "nekststep"),  # the lone "s" goes too
        ("beijing cbt optics equipment co ltd", "beijing cbt optics equipment"),
        ("joint stock company gazprom neft", "gazprom neft"),
        ("anjan s sahni", "anjan s sahni"),  # no legal form, so the initial stays
        ("ivan petrov", "ivan petrov"),
        ("ooo", "ooo"),  # never stripped to nothing
        ("co ltd", "co ltd"),
    ],
)
def test_strip_legal_forms(normalised, expected):
    assert strip_legal_forms(normalised) == expected


def one(_token):
    return 1.0


def test_name_scores_clean_subset_is_not_penalised():
    record = make_record(["Ivan Petrov Sidorov"])
    # "ivan petrov" is a subset of the listed name: ratio 2 * 11 / (11 + 19) = 0.7333 even with a penalty.
    assert name_scores("ivan petrov", record, one, (0.0,), length_penalty=0.6) == pytest.approx((0.7333,), abs=1e-3)


def test_name_scores_length_mismatch_is_penalised_only_when_asked():
    record = make_record(["Ivan Petrov Sidorov"])
    (plain,) = name_scores("ivan sidorova", record, one, (0.0,))
    (penalised,) = name_scores("ivan sidorova", record, one, (0.0,), length_penalty=0.6)
    # Not a subset, 2 tokens against 3: mismatch 1 - 2/3, penalty 0.6 * 1/3 = 0.2.
    assert plain > 0.2
    assert penalised == pytest.approx(plain - 0.2)


def test_name_scores_equal_length_is_not_penalised():
    record = make_record(["Ivan Petrov"])
    assert name_scores("ivan petrova", record, one, (0.0,), 0.6) == name_scores("ivan petrova", record, one, (0.0,))


def test_name_scores_take_the_best_name():
    record = make_record(["Anna Sidorova Ivanovna Petrova", "Ivan Petrov"])
    assert name_scores("petrov ivan", record, one, (0.0, 1.0)) == pytest.approx((1.0, 1.0))


def test_name_scores_blend_between_ratio_and_weighted_score():
    record = make_record(["Muhammad Ali"])
    ratio, weighted = 0.8333, 0.75 * 0.5 + 0.5  # see test_weighted below: tokens weigh equally
    # query "mohammed ali": ratio 10/12 = 0.8333. Weighted: "ali" exact (1), "mohammed"~"muhammad" (0.75), both sides.
    scores = name_scores("mohammed ali", record, one, (0.0, 1.0, 0.5))
    assert scores[0] == pytest.approx(ratio, abs=1e-3)
    assert scores[1] == pytest.approx((0.75 + 1.0) / 2, abs=1e-3)  # (0.75 + 1 + 0.75 + 1) / 4
    assert scores[2] == pytest.approx((scores[0] + scores[1]) / 2)
    assert weighted == pytest.approx(0.875)


def test_weighted_token_score_counts_rare_tokens_more():
    idf = {"ahmad": 0.1, "zorblax": 5.0, "name": 2.0}.get
    # Only the common token matches: matched weight 0.1 + 0.1, total 0.1 + 5 + 0.1 + 2.
    assert weighted_token_score(["ahmad", "zorblax"], ("ahmad", "name"), frozenset({"ahmad", "name"}), idf) == (
        pytest.approx(0.2 / 7.2)
    )
    # The rare token matches and the common one does not: far higher.
    weights = {"ahmad": 0.1, "zorblax": 5.0, "karim": 0.1}.get
    entity_tokens = ("karim", "zorblax")
    rare_match = weighted_token_score(["ahmad", "zorblax"], entity_tokens, frozenset(entity_tokens), weights)
    assert rare_match == pytest.approx(10.0 / 10.2)  # matched 5 + 5 of total 0.1 + 5 + 0.1 + 5


def test_weighted_token_score_fuzzy_tokens():
    assert weighted_token_score(["muhammad"], ("mohammed",), frozenset({"mohammed"}), one) == pytest.approx(0.75)
    # "aly" against "ali" is 0.667, under the 0.7 cut-off, so it counts as a different token.
    assert weighted_token_score(["aly"], ("ali",), frozenset({"ali"}), one) == 0.0
    assert weighted_token_score([], (), frozenset(), one) == 0.0


def test_idf_comes_from_the_index():
    s = Screener.build(
        [
            {"id": "e1", "names": ["Ahmad Hassan"]},
            {"id": "e2", "names": ["Ahmad Zorblax"]},
            {"id": "e3", "names": ["Ahmad Karim"]},
        ]
    )
    assert s.idf("ahmad") == pytest.approx(0.1)  # in every entity: ln(4/4) = 0, floored at 0.1
    assert s.idf("hassan") == pytest.approx(0.6931, abs=1e-3)  # ln(4/2)
    assert s.idf("never seen") == pytest.approx(1.3863, abs=1e-3)  # ln(4/1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("محمد", True), ("Mohammed محمد", True), ("Mohammed", False), ("Мухаммед", False), ("张伟", False), ("", False)],
)
def test_has_arabic(text, expected):
    assert has_arabic(text) is expected


def test_record_knows_when_an_entity_has_an_arabic_name():
    assert make_record(["Mohammed Ali"]).arabic is False
    assert make_record(["Mohammed Ali", "محمد علي"]).arabic is True


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
    # blend 0 so these tests exercise the ratio and the skeleton gate on their own.
    return Screener.build(
        [
            {"id": "e1", "names": ["Mohammed Ali"], "birth_dates": ["1970-05-12"], "countries": ["sa"]},
            {"id": "e2", "names": ["Ivan Petrov"], "birth_dates": ["1980"], "countries": ["ru"]},
            {"id": "e3", "names": ["Ivan Petrov Sidorov"], "countries": ["ru"]},
        ],
        blend=0.0,
    )


def scores(matches):
    return {m.entity_id: m.score for m in matches}


FLOOR, LIFT = 0.5, 0.1


def screen(query, dob=None, country=None, threshold=0.5):
    return screener().screen(query, dob, country, threshold, skeleton_floor=FLOOR, skeleton_lift=LIFT)


def test_name_only_match_is_confirmed_by_the_skeleton():
    # Normalised forms give 0.8333, the skeletons are identical, so the score rises by the lift cap to 0.9333.
    assert scores(screen("Muhammad Ali")) == pytest.approx({"e1": 0.9333}, abs=1e-3)


def test_skeleton_alone_cannot_create_a_match():
    entities = [{"id": "n", "names": ["Natallia Holub"]}]
    result = Screener.build(entities, blend=0.0).screen("Gleb Anatoliyivna", threshold=0.5)
    assert result == []


def test_company_legal_forms_do_not_get_in_the_way():
    entities = [{"id": "c", "names": ["Beijing CBT Optics Equipment Co., Ltd."]}]
    s = Screener.build(entities, blend=0.0)
    assert scores(s.screen("Beijing CBT Optics Equipment", threshold=0.99)) == {"c": 1.0}
    assert scores(s.screen("Beijing CBT Optics Equipment Limited", threshold=0.99)) == {"c": 1.0}


def test_agreeing_birth_date_and_country_lift_the_score_to_the_cap():
    assert scores(screen("Muhammad Ali", "1970-05-12", "sa")) == {"e1": 1.0}  # 0.9333 + 0.10 + 0.03, clipped


def test_conflicting_birth_date_and_country_remove_the_match():
    # 0.9333 - 0.30 - 0.15 = 0.4833, below a 0.5 threshold.
    assert screen("Muhammad Ali", "1999-01-01", "us") == []


def test_missing_attributes_have_no_effect():
    assert scores(screen("Muhammad Ali", None, "sa"))["e1"] == pytest.approx(0.9633, abs=1e-3)
    assert scores(screen("Muhammad Ali", "1970", None))["e1"] == pytest.approx(0.9833, abs=1e-3)


def test_entity_without_attributes_is_not_affected_by_query_attributes():
    # e3 has no birth date, so only the agreeing country (+0.03) applies to it. "ivan petrov" is a clean subset: 0.7333.
    plain = scores(screen("Ivan Petrov"))["e3"]
    with_query = scores(screen("Ivan Petrov", "1950", "ru"))["e3"]
    assert plain == pytest.approx(0.7333, abs=1e-3)
    assert with_query == pytest.approx(plain + 0.03)


def test_arabic_query_matches_a_latin_spelling_through_the_skeleton():
    # "mhmd ly" against "mohammed ali" is a weak name match, but the query is Arabic, so the skeleton stands in.
    assert scores(screen("محمد علي", threshold=0.9))["e1"] == pytest.approx(1.0)


def test_latin_query_against_an_entity_with_an_arabic_alias_gets_the_full_skeleton():
    plain = Screener.build([{"id": "y", "names": ["Mohammed Ali"]}], blend=0.0)
    arabic = Screener.build([{"id": "y", "names": ["Mohammed Ali", "محمد علي"]}], blend=0.0)
    kwargs = {"threshold": 0.5, "skeleton_floor": 0.5, "skeleton_lift": 0.1}
    assert scores(plain.screen("Muhammad Ali", **kwargs)) == pytest.approx({"y": 0.9333}, abs=1e-3)  # gated lift
    assert scores(arabic.screen("Muhammad Ali", **kwargs)) == pytest.approx({"y": 1.0})  # full lift


def test_word_order_does_not_matter_and_extra_tokens_score_lower():
    result = screen("Petrov Ivan")
    assert result[0].entity_id == "e2" and result[0].score == pytest.approx(1.0)
    assert scores(result)["e3"] < 1.0


def test_threshold_filters_matches():
    assert [m.entity_id for m in screen("Ivan Petrov", threshold=0.99)] == ["e2"]


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
    kept = Screener.build(entities, blend=0.0).components("Karim Nasser", "1970-05-12", None, keep=1)
    assert [c.entity_id for c in kept] == ["b"]  # name scores tie, "b" can reach 1.0 with the full-date boost


def test_rarity_weighted_skeleton_is_on_by_default_and_discounts_common_tokens():
    from scoring import Screener

    common = [{"id": f"c{i}", "names": [f"Mohammed Zxq{i}rt"]} for i in range(30)]
    target = [{"id": "t", "names": ["Mohammed Khalid"]}]
    plain = Screener.build(common + target, skeleton_idf=False)
    weighted = Screener.build(common + target)
    assert weighted.skeleton_idf is True
    query = "Mohammed Qqqq"
    sk = lambda s: next(c.skeleton for c in s.components(query) if c.entity_id == "t")  # noqa: E731
    assert sk(weighted) < sk(plain)

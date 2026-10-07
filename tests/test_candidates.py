"""Tests for src/candidates.py. Expected values were worked out by hand.

Run:
    pytest tests/test_candidates.py
"""

import pytest

from candidates import CandidateIndex, fold, name_keys, skeleton


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("mohammed", "mhmd"),  # m h m m d, repeat collapsed
        ("muhammad", "mhmd"),
        ("mhmd", "mhmd"),  # already a skeleton, as produced from Arabic
        ("ahmad", "hmd"),
        ("yusuf", "sf"),  # y and w count as vowels
        ("wameed", "md"),
        ("abdullah", "bdlh"),  # l l collapsed
        ("", ""),
        ("42", "42"),
        ("星龍", "星龍"),  # non-ASCII tokens are left alone
    ],
)
def test_skeleton(token, expected):
    assert skeleton(token) == expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("dzhon", "jon"),  # Russian Джон
        ("zhukov", "jukow"),  # zh -> j, v -> w
        ("khalid", "halid"),
        ("ghasempour", "kasempour"),  # gh -> k
        ("qasm", "kasm"),  # q -> k
        ("george", "heorhe"),  # g -> h
        ("havrylov", "hawrylow"),  # Ukrainian spelling
        ("gavrilov", "hawrilow"),
        ("katherine", "katerine"),  # th -> t
        ("shahadeh", "sahadeh"),  # sh -> s
        ("chahada", "sahada"),  # French ch -> s
        ("sarbeynz", "sarbeyns"),  # z -> s
        ("dhanun", "tanun"),  # Arabic ذ
        ("nextstep", "nekststep"),  # x -> ks
        ("星龍", "星龍"),
    ],
)
def test_fold(token, expected):
    assert fold(token) == expected


def test_folding_makes_variant_spellings_share_a_skeleton():
    # Russian transliteration of Katherine Jolluck: "ketrin dzhollak". Both sides fold to the same key.
    assert skeleton(fold("dzhollak")) == skeleton(fold("jolluck")) == "jlk"
    assert skeleton(fold("sarbeynz")) == skeleton(fold("sarbanes")) == "srbns"
    assert skeleton(fold("ghasempour")) == skeleton(fold("qasempour")) == "ksmpr"
    assert skeleton(fold("havrylov")) == skeleton(fold("gavrilov")) == "hrl"


def test_name_keys_without_joins_or_folds_that_apply():
    assert name_keys("Mohammed Ali", join=False) == {"t:mohammed", "s:mhmd", "t:ali", "s:l"}


def test_name_keys_with_joined_tokens():
    assert name_keys("Mohammed Ali") == {
        "t:mohammed", "s:mhmd", "t:ali", "s:l", "t:mohammedali", "s:mhmdl",
    }  # fmt: skip


def test_arabic_and_latin_share_skeleton_keys():
    arabic = name_keys("محمد علي")  # normalises to "mhmd ly"
    latin = name_keys("Mohammed Ali")
    assert arabic == {"t:mhmd", "s:mhmd", "t:ly", "s:l", "t:mhmdly", "s:mhmdl"}
    assert arabic & latin == {"s:mhmd", "s:l", "s:mhmdl"}


def test_split_and_merged_words_meet():
    assert {"t:abdallah", "s:bdlh"} <= name_keys("Abd Allah") & name_keys("Abdallah")
    assert name_keys("Abd Allah", join=False) & name_keys("Abdallah") == set()  # without joins they do not


def test_three_token_names_join_pairs_and_triples():
    keys = name_keys("Abd Al Rahman", join=True)
    assert "t:abdalrahman" in keys  # triple
    assert "t:abdal" in keys and "t:alrahman" in keys  # pairs


def test_single_letter_ascii_tokens_are_skipped():
    assert name_keys("A B Brown") == {"t:brown", "s:brn"}


def test_article_stripping_adds_a_skeleton_key():
    # "alhmdany" -> skeleton "lhmdn"; without the article "hmdany" -> "hmdn".
    assert name_keys("alhmdany") == {"t:alhmdany", "s:lhmdn", "s:hmdn"}
    assert name_keys("alhmdany", strip_article=False) == {"t:alhmdany", "s:lhmdn"}
    # Latin spelling with a separate article gives the same stripped key.
    assert "s:hmdn" in name_keys("Al-Hamdani")


def test_short_al_words_are_not_stripped():
    assert name_keys("alan") == {"t:alan", "s:ln"}


def people():
    return [
        ("e1", ["Mohammed Ali"]),
        ("e2", ["Ivan Petrov"]),
        ("e3", ["Maria Garcia"]),
        ("e4", ["Vladimir Putin"]),
    ]


def ids(result):
    return [entity_id for entity_id, _ in result]


def test_finds_entity_by_arabic_name():
    result = CandidateIndex.build(people()).query("محمد علي")
    assert ids(result) == ["e1"]


def test_finds_entity_by_cyrillic_name():
    assert ids(CandidateIndex.build(people()).query("Владимир Путин")) == ["e4"]


def test_finds_han_name_by_exact_match():
    index = CandidateIndex.build([("h1", ["星龍興業"]), ("h2", ["Acme Corp"])])
    assert ids(index.query("星龍興業")) == ["h1"]


def test_finds_entity_across_spelling_variants_and_word_splits():
    index = CandidateIndex.build([("a", ["Abdallah Karim"]), ("b", ["Ivan Petrov"]), ("c", ["Maria Garcia"])])
    # Split word and a q/k variant: "abd allah" joins to "abdallah", "qarim" folds to "karim".
    assert ids(index.query("Abd Allah Qarim"))[0] == "a"


def test_unknown_and_empty_queries_return_nothing():
    index = CandidateIndex.build(people())
    for query in ("Zzyzx Qwrt", "", "   ", "()"):
        assert index.query(query) == []


def test_postings_hold_each_entity_once_per_key():
    index = CandidateIndex.build([("e1", ["Ali Hassan", "Ali Hassan Jr"])])
    assert index.postings["t:ali"] == [0]


def common_token_entities():
    # "holdings" is in 5 entities. "zorblax" is in two of them. max_df=3 makes "holdings" a common key.
    entities = [("z2", ["Zorblax Trading"]), ("z1", ["Zorblax Holdings"])]
    entities += [(f"h{i}", [f"Holdings Company{i}x"]) for i in range(4)]
    return entities


def test_common_keys_do_not_generate_candidates():
    index = CandidateIndex.build(common_token_entities(), max_df=3)
    result = index.query("Zorblax Holdings")
    # Only the two "zorblax" entities are candidates, not all five "holdings" entities.
    # The common key still scores: z1 shares it, z2 does not, so z1 ranks first despite being added second.
    assert ids(result) == ["z1", "z2"]


def test_query_of_only_common_keys_falls_back_to_the_rarest():
    index = CandidateIndex.build(common_token_entities(), max_df=3)
    # Both keys of "holdings" occur in 5 entities, above max_df, so the fallback uses them anyway.
    assert len(index.query("Holdings")) == 5


def test_there_is_no_cut_on_the_candidate_set():
    entities = [(f"e{i}", [f"Karim Nasser{i}x"]) for i in range(300)]
    assert len(CandidateIndex.build(entities, max_df=1000).query("Karim")) == 300


def test_ranking_is_deterministic_for_ties():
    index = CandidateIndex.build([("a", ["Karim Nasser"]), ("b", ["Karim Nasser"])])
    assert ids(index.query("Karim Nasser")) == ["a", "b"]

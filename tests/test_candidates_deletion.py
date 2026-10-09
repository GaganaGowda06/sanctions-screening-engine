"""Tests for the deletion-neighbourhood keys in the candidate index (src/candidates.py).

With the keys on, a name that is one letter off an indexed name (a letter missing, added or changed) still retrieves the
entity. With them off (the default) the index is exactly what it was.

Run:
    pytest tests/test_candidates_deletion.py
"""

import run_context as rc
from candidates import CandidateIndex, deletion_neighbourhood, name_keys
from scoring import Screener

ENTITIES = [
    ("e-zorv", ["Zorvandik Quellmark"]),
    ("e-bakr", ["Abu Bakr"]),
    ("e-other", ["Marlenti Hosskopf"]),
    ("e-oy", ["Sukhrob Oimakhmadov"]),
    ("e-solo", ["Brandoval"]),
]


def index(deletion_keys: bool, max_df: int = 1000) -> CandidateIndex:
    return CandidateIndex.build(ENTITIES, max_df=max_df, deletion_keys=deletion_keys)


def found(idx: CandidateIndex, name: str) -> set[str]:
    return {entity for entity, _ in idx.query(name)}


# ---------- the keys ----------


def test_the_neighbourhood_of_a_token():
    assert deletion_neighbourhood("bakr") == {"d:bakr", "d:akr", "d:bkr", "d:bar", "d:bak"}
    assert deletion_neighbourhood("bak") == {"d:bak"}  # long enough to be found, too short to be deleted from
    assert deletion_neighbourhood("ab") == set()
    assert deletion_neighbourhood("") == set()


def test_with_the_flag_off_the_keys_are_exactly_the_old_ones():
    for name in ("Mohammed Ali Hassan", "Abu Bakr", "Юрий Щербак", "Al-Hamdani"):
        assert name_keys(name) == name_keys(name, deletion_keys=False)
        assert not any(k.startswith("d:") for k in name_keys(name))
    assert not any(k.startswith("d:") for k in index(False).postings)


def test_the_flag_adds_only_d_keys():
    for name in ("Mohammed Ali Hassan", "Abu Bakr"):
        extra = name_keys(name, deletion_keys=True) - name_keys(name)
        assert extra and all(k.startswith("d:") for k in extra)
        assert name_keys(name) <= name_keys(name, deletion_keys=True)


def test_the_neighbourhood_is_taken_from_the_folded_token():
    # "kh" folds to "h": the d: keys of "sukhrob" are those of "suhrob"
    assert "d:suhrob" in name_keys("Sukhrob", deletion_keys=True)
    assert "d:sukhrob" not in name_keys("Sukhrob", deletion_keys=True)


def test_joined_words_get_no_d_keys():
    keys = name_keys("Zorvandik Quellmark", deletion_keys=True)
    assert "t:sorwandikkuellmark" in keys or any(k.startswith("t:") and "quellmark"[:3] in k for k in keys)
    assert not any(k.startswith("d:") and len(k) > 20 for k in keys)  # no d: key made from the joined pair


# ---------- retrieval ----------
# The old index already tolerates changed vowels and doubled letters, through the consonant skeleton keys. The typos
# that it loses are the ones that change a consonant: one dropped, added or changed. Those are the cases tested here.


def test_a_consonant_deleted_added_or_changed_in_a_single_token_name_retrieves_the_entity():
    old, new = index(False), index(True)
    for query, kind in (
        ("Bradoval", "a consonant deleted"),
        ("Brandovkal", "a consonant added"),
        ("Brandotal", "a consonant changed"),
        ("Brandoval", "no typo at all"),
    ):
        assert "e-solo" in found(new, query), kind
        assert ("e-solo" in found(old, query)) == (query == "Brandoval"), kind  # the old index misses every typo


def test_a_consonant_typo_in_every_token_of_a_two_token_name_retrieves_the_entity():
    old, new = index(False), index(True)
    for query in ("Zorvadik Quellmak", "Zorvandit Quellmarp", "Zorvandikt Quellmarkt", "Zorvanrik Quellmank"):
        assert "e-zorv" not in found(old, query), query
        assert "e-zorv" in found(new, query), query


def test_a_vowel_typo_was_already_retrieved_by_the_skeleton_and_still_is():
    for query in ("Zorvandok Quellmark", "Zorvandiik Quellmarkk"):
        assert "e-zorv" in found(index(False), query) and "e-zorv" in found(index(True), query)


def test_a_short_token_is_found_from_a_longer_typo_and_the_reverse():
    assert "e-bakr" in found(index(True), "Abu Bak")  # indexed "bakr", the query lost the r
    short = CandidateIndex.build([("e-short", ["Qwxlbrt Bak"])], deletion_keys=True)
    assert "e-short" in found(short, "Qwxlbrt Bakr")  # indexed "bak", the query gained an r


def test_two_edits_in_one_token_do_not_retrieve_it():
    assert "e-solo" not in found(index(True), "Brandotl")  # v changed and a deleted
    assert "e-solo" not in found(index(True), "Brandxxal")


def test_unrelated_names_are_not_retrieved():
    assert found(index(True), "Qwertyuiop Asdfghjkl") == set()
    assert "e-other" not in found(index(True), "Zorvandik Quellmark")


def test_a_d_key_over_max_df_does_not_generate_candidates():
    shared = [(f"e{i}", [f"Mbdelnp{c}"]) for i, c in enumerate("abde")] + [("e-rare", ["Rarename Tlmnpr"])]
    idx = CandidateIndex.build(shared, max_df=2, deletion_keys=True)
    assert len(idx.postings["d:mbdelnp"]) == 4  # shared by all four, over max_df
    assert found(idx, "Mbdelnpr Rarename") == {"e-rare"}  # the common d: key neither generates nor adds the four
    assert found(CandidateIndex.build(shared, max_df=2), "Mbdelnpr Rarename") == {"e-rare"}


def test_a_d_key_within_max_df_generates_candidates():
    idx = CandidateIndex.build([("e1", ["Mbdelnpa"]), ("e2", ["Totally Different"])], max_df=2, deletion_keys=True)
    assert found(idx, "Mbdelnpr") == {"e1"}


def test_candidates_gained_come_only_from_the_new_keys_and_none_are_lost():
    old, new = index(False), index(True)
    for name in ("Zorvandik Quellmark", "Abu Bakr", "Sukhrob Oimakhmadov", "Marlenti Hosskopf", "Hosskopf", "Bradoval"):
        assert found(old, name) <= found(new, name)


# ---------- scoring is unchanged ----------


def entities() -> list[dict]:
    return [{"id": i, "names": n, "birth_dates": [], "countries": []} for i, n in ENTITIES]


def test_scores_of_common_candidates_are_identical_with_and_without_the_keys():
    old, new = Screener.build(entities()), Screener.build(entities(), deletion_keys=True)
    for name in ("Zorvandik Quellmark", "Abu Bakr", "Sukhrob Oimakhmadov", "Marlenti"):
        a = {c.entity_id: c for c in old.components(name)}
        b = {c.entity_id: c for c in new.components(name)}
        assert set(a) <= set(b)
        for entity_id, comp in a.items():
            assert b[entity_id] == comp  # every component, name score, skeleton, dob, country


def test_a_typo_name_is_scored_once_it_is_retrieved():
    new = Screener.build(entities(), deletion_keys=True)
    got = {c.entity_id: c for c in new.components("Zorvadik Quellmak")}
    assert "e-zorv" in got and got["e-zorv"].names[0] > 0.6  # retrieval was missing; scoring takes it from there
    assert "e-zorv" not in {c.entity_id for c in Screener.build(entities()).components("Zorvadik Quellmak")}


def test_the_screener_flag_reaches_the_index_and_the_run_context_records_it():
    off, on = Screener.build(entities()), Screener.build(entities(), deletion_keys=True)
    assert off.index.deletion_keys is False and on.index.deletion_keys is True
    assert rc.engine_settings(off)["deletion_keys"] is False and rc.engine_settings(on)["deletion_keys"] is True
    assert rc.canonical_json(rc.engine_settings(off)) != rc.canonical_json(rc.engine_settings(on))

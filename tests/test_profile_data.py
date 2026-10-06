"""Tests for src/profile_data.py against a small fixture built in code.

Every expected value below was worked out by hand from the fixture definition,
not copied from profiler output. The derivation is in the comment above each assert.

Run:
    pytest
"""
import json

import pytest

from profile_data import profile

ARABIC = "محمد علي"
CYRILLIC = "Владимир Иванов"
HAN = "张伟"
HANGUL = "김민수"
HIRAGANA = "たなか"
KATAKANA = "ヤマダ"
FULLWIDTH = "Ｊｏｈｎ Ｓｍｉｔｈ"
MIXED = "Gazprom Нефть"  # Latin + Cyrillic in one name


def entity(eid, schema, target=False, **props):
    return {"id": eid, "schema": schema, "target": target, "properties": props}


FIXTURE_ENTITIES = [
    # Arabic name + Latin alias listed twice + the same alias again as weakAlias: dedupes to 2 names.
    entity("P1", "Person", True, name=[ARABIC], alias=["Mohammed Ali", "Mohammed Ali"],
           weakAlias=["Mohammed Ali"], birthDate=["1970"], nationality=["sa"]),
    entity("P2", "Person", True, name=[CYRILLIC], idNumber=["123"]),
    # Han name + Latin alias + the same Latin string as previousName: dedupes to 2 names.
    entity("P3", "Person", False, name=[HAN], alias=["Zhang Wei"], previousName=["Zhang Wei"],
           birthDate=["1980-01-01"], citizenship=["cn"]),
    entity("P4", "Person", True, name=[HANGUL], birthDate=["1990"]),
    entity("P5", "Person", False, name=[HIRAGANA], alias=[KATAKANA]),
    entity("P6", "Person", False, name=[FULLWIDTH], programId=["PRG"]),
    entity("P7", "Person", False, birthDate=["1960"]),  # no name at all
    entity("C1", "Company", True, name=[MIXED], jurisdiction=["ru"], leiCode=["LEI1"]),
    entity("C2", "Company", False, name=["Acme Ltd"], wikidataId=["Q2"]),  # wikidataId is excluded
    entity("PS1", "Passport", False, holder=["P1"], passportNumber=["X1"], number=["X1"]),
    entity("S1", "Sanction", False, entity=["P2"]),
]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    lines = [json.dumps(e, ensure_ascii=False) for e in FIXTURE_ENTITIES]
    lines.insert(5, '{"id": "broken", "schema": ')  # malformed, mid-file
    lines.insert(8, "")  # blank line: ignored, not counted as unparseable
    path = tmp_path_factory.mktemp("fixture") / "entities.ftm.json"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return profile(path)


def test_totals(result):
    # 11 valid entities (P1-P7, C1, C2, PS1, S1); targets are P1, P2, P4, C1.
    assert result["total_entities"] == 11
    assert result["target_entities"] == 4
    # Only the truncated JSON line; the blank line is skipped.
    assert result["unparseable_lines"] == 1


def test_schema_counts(result):
    assert result["schema_counts"]["all"] == {"Person": 7, "Company": 2, "Passport": 1, "Sanction": 1}
    # Targets: P1, P2, P4 are Person; C1 is Company.
    assert result["schema_counts"]["targets"] == {"Person": 3, "Company": 1}


def test_names_all_entities(result):
    n = result["names"]["all_entities"]
    # Entities with names: P1-P6, C1, C2 = 8. Without: P7, PS1, S1 = 3.
    assert n["entities_with_names"] == 8
    assert n["entities_without_names"] == 3
    # Unique names: P1 2, P2 1, P3 2, P4 1, P5 2, P6 1, C1 1, C2 1 = 11.
    assert n["total_names"] == 11
    # Non-Latin: Arabic, Cyrillic, Han, Hangul, hiragana, katakana, mixed = 7.
    assert n["names_with_non_latin_letter"] == 7
    assert n["names_without_letters"] == 0
    # Latin: Mohammed Ali, Zhang Wei, fullwidth John Smith, mixed, Acme Ltd = 5.
    # Cyrillic: Владимир Иванов, mixed = 2. Kana: 2 names. Arabic, Han, Hangul: 1 each.
    assert n["names_containing_script"] == {
        "Latin": 5, "Cyrillic": 2, "Japanese kana": 2, "Arabic": 1, "Han": 1, "Hangul": 1,
    }
    # Entities with a non-Latin name: P1, P2, P3, P4, P5, C1 = 6.
    assert n["entities_with_any_non_latin_name"] == 6
    # No purely-Latin name: P2, P4, P5, C1 = 4 (P1 and P3 each have a Latin alias).
    assert n["entities_with_no_purely_latin_name"] == 4


def test_names_targets(result):
    n = result["names"]["targets"]
    # Targets P1 (2 names), P2 (1), P4 (1), C1 (1) = 5 names, all 4 targets named.
    assert n["entities_with_names"] == 4
    assert n["entities_without_names"] == 0
    assert n["total_names"] == 5
    # Non-Latin names: Arabic, Cyrillic, Hangul, mixed = 4 (Mohammed Ali is Latin).
    assert n["names_with_non_latin_letter"] == 4
    assert n["names_containing_script"] == {"Latin": 2, "Cyrillic": 2, "Arabic": 1, "Hangul": 1}
    assert n["entities_with_any_non_latin_name"] == 4  # P1, P2, P4, C1
    assert n["entities_with_no_purely_latin_name"] == 3  # P2, P4, C1


def test_fullwidth_latin_is_latin(result):
    # P6 is the only entity whose sole name is fullwidth Latin. It must not appear as non-Latin:
    # of the 8 named entities, exactly 6 are non-Latin, so P6 and C2 are the other 2.
    n = result["names"]["all_entities"]
    assert n["entities_with_names"] - n["entities_with_any_non_latin_name"] == 2


def cov(result, population, schema):
    return result["coverage"][population][schema]


def test_coverage_named_person(result):
    c = cov(result, "named_entities", "Person")
    assert c["n"] == 6  # P1-P6 (P7 has no name)
    assert c["birthDate"] == 3  # P1, P3, P4
    assert c["country"] == 2  # P1 nationality, P3 citizenship
    assert c["own_identifier"] == 2  # P2 idNumber, P6 programId
    assert c["any_identifier"] == 3  # plus P1 via Passport PS1
    assert c["own_identifier_no_programId"] == 1  # P2 only
    assert c["any_identifier_no_programId"] == 2  # P2, P1


def test_coverage_named_company(result):
    c = cov(result, "named_entities", "Company")
    assert c["n"] == 2
    assert c["birthDate"] == 0  # birthDate counted for Person only
    assert c["country"] == 1  # C1 jurisdiction
    assert c["own_identifier"] == 1  # C1 leiCode; C2 wikidataId is excluded
    assert c["any_identifier"] == 1
    assert c["own_identifier_no_programId"] == 1
    assert c["any_identifier_no_programId"] == 1


def test_coverage_named_all(result):
    c = cov(result, "named_entities", "ALL")
    assert (c["n"], c["birthDate"], c["country"]) == (8, 3, 3)
    assert (c["own_identifier"], c["any_identifier"]) == (3, 4)  # P2, P6, C1 ; + P1
    assert (c["own_identifier_no_programId"], c["any_identifier_no_programId"]) == (2, 3)


def test_coverage_targets(result):
    person = cov(result, "targets", "Person")  # P1, P2, P4
    assert (person["n"], person["birthDate"], person["country"]) == (3, 2, 1)
    assert person["own_identifier"] == 1  # P2
    assert person["any_identifier"] == 2  # P2, P1 via Passport
    company = cov(result, "targets", "Company")  # C1
    assert (company["n"], company["birthDate"], company["country"]) == (1, 0, 1)
    assert company["own_identifier"] == company["any_identifier"] == 1
    total = cov(result, "targets", "ALL")
    assert (total["n"], total["birthDate"], total["country"]) == (4, 2, 2)
    assert (total["own_identifier"], total["any_identifier"]) == (2, 3)  # P2, C1 ; + P1
    # P6 (programId only) is not a target, so the no_programId variants match the main ones.
    assert (total["own_identifier_no_programId"], total["any_identifier_no_programId"]) == (2, 3)


def test_unnamed_entities_excluded_from_named_coverage(result):
    schemas = set(result["coverage"]["named_entities"])
    assert schemas == {"ALL", "Person", "Company"}  # Passport and Sanction have no names


def test_top_identifier_properties(result):
    top = {t["property"]: (t["entities"], t["target_entities"]) for t in result["top_identifier_properties"]}
    # One entity each: P2 idNumber (target), C1 leiCode (target), PS1 number and passportNumber,
    # P6 programId. wikidataId on C2 is excluded.
    assert top == {
        "idNumber": (1, 1), "leiCode": (1, 1), "number": (1, 0),
        "passportNumber": (1, 0), "programId": (1, 0),
    }

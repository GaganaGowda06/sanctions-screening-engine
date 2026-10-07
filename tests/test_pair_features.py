"""Tests for src/pair_features.py, src/clean_pairs.py and src/build_pair_features.py, on hand-built pairs.

Run:
    pytest tests/test_pair_features.py
"""

import gzip
import json

import pytest

import build_pair_features as bpf
import build_pair_splits as bps
import clean_pairs
import pair_features as pf
from profile_data import PROPS_PATH

IDENTIFIER_PROPS = {s: list(v["identifier"]) for s, v in json.loads(PROPS_PATH.read_text())["schemas"].items()}
FLAT = pf.make_idf({}, 100)  # every token equally rare


def entity(eid, names, schema="Person", **props):
    return {"id": eid, "caption": names[0] if names else "", "schema": schema,
            "properties": {"name": names[1:], **{k: v for k, v in props.items()}}}  # fmt: skip


def forms(names, schema="Person", **props):
    return pf.Forms(entity("x", names, schema, **props), IDENTIFIER_PROPS)


def features(a, b, idf=FLAT):
    return pf.name_features(a, b, idf)


def test_identical_names_score_one_on_every_feature():
    f = features(forms(["Ivan Petrov"]), forms(["Ivan Petrov"]))
    assert all(f[s] == pytest.approx(1.0) for s in pf.SCORES)


def test_cyrillic_against_latin_is_the_same_name_after_normalisation():
    f = features(forms(["Иван Петров"]), forms(["Ivan Petrov"]))
    assert f["name_score"] == pytest.approx(1.0) and f["phonetic_score"] == pytest.approx(1.0)


def test_mohammed_variants():
    f = features(forms(["Mohammed Ali"]), forms(["Muhammad Ali"]))
    # Character ratio 10/12 = 0.8333. Rarity-weighted with equal weights: (0.75 + 1 + 0.75 + 1) / 4 = 0.875.
    assert f["name_score"] == pytest.approx(0.5 * 0.8333 + 0.5 * 0.875, abs=0.005)
    assert f["weighted_score"] == pytest.approx(0.875, abs=0.005)
    assert f["phonetic_score"] == pytest.approx(1.0)  # MHMT = MHMT
    assert f["arabic_score"] == pytest.approx(1.0)  # both fold to "muhamad ali"
    assert f["skeleton_score"] == pytest.approx(1.0)  # both are "mhmd l"
    assert f["name_score"] < 1.0


def test_arabic_script_against_latin_spelling():
    f = features(forms(["محمد علي"]), forms(["Mohammed Ali"]))  # normalises to "mhmd ly"
    assert f["arabic_score"] == pytest.approx(1.0)
    assert f["skeleton_score"] == pytest.approx(1.0)
    assert f["name_score"] < 0.7  # the plain name comparison cannot see it


def test_unrelated_names_score_low():
    f = features(forms(["Ivan Petrov"]), forms(["Maria Garcia"]))
    assert f["name_score"] < 0.5 and f["weighted_score"] < 0.2 and f["phonetic_score"] < 0.6


def test_a_shared_rare_token_counts_for_far_more_than_a_shared_common_one():
    idf = pf.make_idf({"ahmad": 9000, "karim": 50, "zorblax": 1}, 10_000)
    common = features(forms(["Ahmad Zorblax"]), forms(["Ahmad Karim"]), idf)["weighted_score"]
    rare = features(forms(["Ahmad Zorblax"]), forms(["Karim Zorblax"]), idf)["weighted_score"]
    assert common < 0.05 and rare > 0.7


def test_the_best_names_are_found_among_many():
    a = forms(["Ivan Petrov", "Someone Entirely Different", "Other Alias"])
    b = forms(["Totally Unrelated", "Petrov Ivan"])
    assert features(a, b)["name_score"] == pytest.approx(1.0)  # word order is ignored


def test_an_entity_with_no_names_scores_zero():
    assert features(forms([]), forms(["Ivan Petrov"])) == dict.fromkeys(pf.SCORES, 0.0)


def test_legal_forms_do_not_count_against_a_match():
    f = features(forms(["Gazprom Neft Ltd"], "Company"), forms(["Gazprom Neft"], "Company"))
    assert f["name_score"] == pytest.approx(1.0)


def test_names_are_capped():
    many = forms([f"Alias Number{i}x Name" for i in range(100)])
    assert len(many.norms) == pf.MAX_NAMES


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ({1970}, {1970}, "agree"),
        ({1970, 1980}, {1980}, "agree"),
        ({1970}, {1971}, "close"),
        ({1970}, {1985}, "conflict"),
        (set(), {1970}, "missing"),
        ({1970}, set(), "missing"),
    ],
)
def test_birth_state(a, b, expected):
    assert pf.birth_state(frozenset(a), frozenset(b)) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ({"ru", "ua"}, {"ua"}, "agree"),
        ({"ru"}, {"us"}, "conflict"),
        (set(), {"us"}, "missing"),
        ({"ru"}, set(), "missing"),
    ],
)
def test_country_state(a, b, expected):
    assert pf.country_state(frozenset(a), frozenset(b)) == expected


def test_identifier_state_compares_like_with_like():
    same = pf.identifier_state({"idNumber": frozenset({"123"})}, {"idNumber": frozenset({"123", "9"})})
    different = pf.identifier_state({"idNumber": frozenset({"123"})}, {"idNumber": frozenset({"456"})})
    not_comparable = pf.identifier_state({"idNumber": frozenset({"123"})}, {"passportNumber": frozenset({"123"})})
    assert (same, different, not_comparable) == ("agree", "conflict", "missing")
    assert pf.identifier_state({}, {"idNumber": frozenset({"1"})}) == "missing"


def test_identifier_values_are_normalised_and_some_properties_are_ignored():
    a = forms(["Ivan"], idNumber=["AB-123"], wikidataId=["Q1"], programId=["EU-X"])
    b = forms(["Ivan"], idNumber=["ab 123"], wikidataId=["Q1"], programId=["EU-X"])
    assert pf.identifier_state(a.identifiers, b.identifiers) == "agree"
    assert set(a.identifiers) == {"idNumber"}  # wikidataId and programId are left out
    c = forms(["Ivan"], wikidataId=["Q1"], programId=["EU-X"])
    assert pf.identifier_state(c.identifiers, c.identifiers) == "missing"


def test_pair_features_has_every_column():
    row = pf.pair_features(forms(["Ivan Petrov"], birthDate=["1970"]), forms(["Ivan Petrov"], birthDate=["1971"]), FLAT)
    assert set(row) == set(pf.SCORES) | set(pf.STATES)
    assert row["birth_state"] == "close" and row["country_state"] == "missing"


def test_script_class():
    assert pf.script_class(forms(["Ivan"]), forms(["Petr"])) == "Latin on both sides only"
    assert pf.script_class(forms(["Иван"]), forms(["Ivan"])) == "non-Latin on one side only"
    assert pf.script_class(forms(["Иван"]), forms(["محمد"])) == "non-Latin on both sides"


def test_phonetic_tokens():
    assert pf.phonetic_token("mohammed") == pf.phonetic_token("muhammad") == ("MHMT", "MHMT")
    assert pf.phonetic_token("smith") == ("SM0", "XMT")
    assert pf.phonetic_token("42") == ("42", "42") and pf.phonetic_token("张伟") == ("张伟", "张伟")


def test_document_frequency_counts_each_token_once_per_entity():
    from collections import Counter

    counter = Counter()
    pf.count_tokens(forms(["Ivan Petrov", "Petrov Ivan"]), counter)
    pf.count_tokens(forms(["Ivan Sidorov"]), counter)
    assert counter["ivan"] == 2 and counter["petrov"] == 1 and counter["sidorov"] == 1
    assert pf.name_tokens(entity("x", ["Ivan Petrov Ltd"], "Company")) == frozenset({"ivan", "petrov"})


def test_idf_has_a_floor_and_unseen_tokens_are_rare():
    idf = pf.make_idf({"common": 1000}, 1000)
    assert idf("common") == pytest.approx(0.1) and idf("never seen") > 5


def test_auc():
    assert pf.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert pf.auc([1, 1, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.5)  # all tied
    assert pf.auc([1, 1, 2], [0, 1, 1]) == pytest.approx(0.75)  # one tie counts half
    assert pf.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert pf.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert pf.auc([0.5, 0.6], [1, 1]) is None


def test_state_rates():
    assert pf.state_rates(["a", "a", "b", "b", "b"], [1, 0, 1, 1, 1]) == {"a": 0.5, "b": 1.0}


# ----- cleaning, and the whole pipeline on a tiny fake dataset -----


def fake_record(left, right, judgement, schema="Person", names=None, **props):
    def side(eid):
        return {"id": eid, "caption": names[0] if names else eid, "schema": schema,
                "properties": {"name": names or [eid], **props}}  # fmt: skip

    return {"judgement": judgement, "left": side(left), "right": side(right)}


def write_pairs(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in records) + "\n")


def test_cleaning_keeps_entity_pairs_drops_conflicts_and_duplicates(tmp_path):
    records = [
        fake_record("a", "b", "positive"),
        fake_record("b", "a", "positive"),  # the same pair, other way round: a duplicate
        fake_record("c", "d", "positive"),
        fake_record("c", "d", "negative"),  # conflicts with the line above: both go
        fake_record("e", "f", "negative", schema="Occupancy"),  # a relationship record
        fake_record("g", "h", "negative"),
        fake_record("g", "h", "negative"),  # a duplicate with the same label
    ]
    path = tmp_path / "pairs.json.gz"
    write_pairs(path, records)
    mask, stats = clean_pairs.clean_mask(path)
    assert mask == "1000010"
    assert stats["conflicting_unordered_pairs"] == 1
    assert stats["dropped"] == {"duplicate_lines": 2, "conflicting_label_lines": 2, "not_an_entity_pair": 1}
    assert stats["kept"] == 2


def tiny_dataset(tmp_path, n=400):
    records = []
    for i in range(n):
        positive = i % 3 != 0
        left_names = [f"Person Number{i}x Alpha"]
        right_names = left_names if positive else [f"Different Name{i}y Beta"]
        records.append(fake_record(f"l{i}", f"r{i}", "positive" if positive else "negative", names=None))
        records[-1]["left"]["properties"]["name"] = left_names
        records[-1]["left"]["caption"] = left_names[0]
        records[-1]["right"]["properties"]["name"] = right_names
        records[-1]["right"]["caption"] = right_names[0]
    path = tmp_path / "pairs-20251209.json.gz"
    write_pairs(path, records)
    (tmp_path / "SOURCE.json").write_text(json.dumps({"file": path.name, "sha256": "x" * 64}))
    return path


def test_the_whole_pipeline_runs_on_a_tiny_dataset(tmp_path):
    path = tiny_dataset(tmp_path)
    split_dir = tmp_path / "splits"
    common = ["--input", str(path), "--split-dir", str(split_dir)]
    bps.main([*common, "--report", str(tmp_path / "splits_report.json")])
    clean_pairs.main(common)
    out, report = tmp_path / "features", tmp_path / "features.json"
    bpf.main([*common, "--out", str(out), "--report", str(report), "--workers", "1"])

    data = json.loads(report.read_text())
    assert set(data["tables"]) == {"train", "dev", "test"}  # the final hold-out is never featurised
    assert sum(t["pairs"] for t in data["tables"].values()) < 400  # the hold-out and nothing else is left out
    dev = bpf.read_table(out / "dev.csv.gz")
    assert dev and list(dev[0]) == bpf.COLUMNS
    assert all(r["split"] == "dev" and r["shares_entity_with_train"] in ("0", "1") for r in dev)
    positives = [float(r["name_score"]) for r in dev if r["label"] == "1"]
    negatives = [float(r["name_score"]) for r in dev if r["label"] == "0"]
    assert min(positives) > max(negatives)  # identical names against clearly different ones
    assert data["separation_on_dev"]["all"]["name_score"]["auc"] == 1.0
    train = bpf.read_table(out / "train.csv.gz")
    assert all(r["shares_entity_with_train"] == "" for r in train)


def test_tables_are_written_reproducibly(tmp_path):
    rows = [[1, "dev", 1, 0, "Person", "Person", "Latin on both sides only", *[1.0] * 5, "agree", "agree", "agree"]]
    first = bpf.write_table(tmp_path / "a.csv.gz", iter(rows))
    second = bpf.write_table(tmp_path / "b.csv.gz", iter(rows))
    assert first == second
    assert bpf.read_table(tmp_path / "a.csv.gz")[0]["birth_state"] == "agree"


def test_only_train_dev_and_test_are_featurised():
    assert bpf.SPLITS == ("train", "dev", "test")


def test_pearson():
    assert bpf.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert bpf.pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    assert bpf.pearson([1, 2, 3], [5, 5, 5]) == 0.0

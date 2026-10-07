"""Tests for the deployment rarity table (src/idf_table.py).

Run:
    pytest tests/test_idf_table.py
"""

import json

import pytest

import idf_table as it


def entity(*names):
    return {"names": list(names)}


def test_a_token_counts_once_per_entity_and_legal_forms_are_left_out():
    frequency, documents = it.build_table(
        [entity("Ivan Petrov", "Petrov Ivan"), entity("Ivan Sidorov"), entity("Gazprom Neft Ltd"), entity("")]
    )
    assert documents == 3  # the entity with no name is not counted
    assert frequency["ivan"] == 2 and frequency["petrov"] == 1 and frequency["sidorov"] == 1
    assert "ltd" not in frequency and frequency["gazprom"] == 1


def test_table_bytes_depend_only_on_content_and_it_round_trips(tmp_path):
    frequency, documents = it.build_table([entity("Ivan Petrov"), entity("Ivan Sidorov")])
    first = it.write_table(tmp_path / "a.json.gz", frequency, documents)
    second = it.write_table(tmp_path / "b.json.gz", frequency, documents)
    assert first == second
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"table_sha256": first}))
    assert it.load_table(tmp_path / "a.json.gz", manifest) == ({"ivan": 2, "petrov": 1, "sidorov": 1}, 2)


def test_a_table_that_does_not_match_its_manifest_is_refused(tmp_path):
    frequency, documents = it.build_table([entity("Ivan Petrov")])
    it.write_table(tmp_path / "t.json.gz", frequency, documents)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"table_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="does not match"):
        it.load_table(tmp_path / "t.json.gz", manifest)


def test_the_committed_manifest_describes_the_deployment_list():
    manifest = json.loads(it.MANIFEST_PATH.read_text())
    assert manifest["entities"] == 69637 and manifest["tokens"] > 100_000
    assert len(manifest["table_sha256"]) == 64

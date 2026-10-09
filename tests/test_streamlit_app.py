"""The local web page (app/streamlit_app.py) imports, screens a name, and cannot write to an audit store.

The list is a few entities built in memory, so no downloaded data is needed and nothing is read from data/.

Run:
    pytest tests/test_streamlit_app.py
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

from screening import ScreeningEngine

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "streamlit_app.py"
LIMITS = (ROOT / "docs" / "limitations_evidence.md").read_text(encoding="utf-8")


def entity(i, name, schema="Person", born=None, country=None):
    return {"id": i, "schema": schema, "target": True, "names": [name],
            "birth_dates": [born] if born else [], "countries": [country] if country else []}  # fmt: skip


LIST = [
    entity("e1", "Zanthor Fictionalis Examplewood", born="1900-01-01", country="ir"),
    entity("e2", "Ivan Petrovich Sokolov", born="1980-05-05", country="ru"),
    entity("e3", "Acme Trading Limited", schema="Company"),
]


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # importing must not build the list or start anything
    return module


@pytest.fixture(scope="module")
def engine():
    return ScreeningEngine.from_entities(LIST)


META = {"e1": ("Person", "us_ofac_sdn"), "e2": ("Person", "eu_fsf"), "e3": ("Company", "gb_hmt_sanctions")}


def test_a_listed_name_alerts_and_shows_the_columns_and_components(app, engine, tmp_path, monkeypatch):
    real_store = tmp_path / "real-audit"
    monkeypatch.setenv("SCREENING_AUDIT_DIR", str(real_store))
    result = app.screen_name(engine, META, "Zanthor Fictionalis Examplewood", "1900-01-01", "IR")
    assert result.decision == "alert" and result.top_score >= result.threshold == 0.86
    assert result.missing == [] and len(result.candidates) <= app.TOP
    top = result.candidates[0]
    assert (top.entity_id, top.entity_type, top.source_lists) == ("e1", "Person", "us_ofac_sdn")
    assert top.birth_dates == "1900-01-01" and top.countries == "ir"
    text = " ".join(app.explain(top))
    assert "closest listed name" in text and "birth date agrees" in text and "country agrees" in text
    assert not real_store.exists()


def test_a_clean_name_clears_and_missing_attributes_are_reported(app, engine):
    result = app.screen_name(engine, META, "Margaret Thornbury")
    assert result.decision == "clear"
    assert result.top_score is None or result.top_score < result.threshold  # none: nothing retrieved at all
    assert result.missing == ["birth date", "country"]
    assert all(c.score < result.threshold for c in result.candidates)


def test_the_page_imports_nothing_from_the_audit_modules():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not {m for m in imported if "audit" in m or m in ("replay", "run_context")}, imported
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "audited_screen"]


def test_inputs_are_checked(app):
    assert app.check_inputs("", "", "") and app.check_inputs("x", "1969/12/20", "") and app.check_inputs("x", "", "rus")
    assert app.check_inputs("x", "1969", "ru") is None and app.check_inputs("x", "", "") is None


def test_the_missing_attribute_note_quotes_figures_found_in_the_limitations_doc(app):
    for figure in re.findall(r"\b0\.\d{3,4}\b", app.WEAKER_WITHOUT):
        assert figure in LIMITS, figure

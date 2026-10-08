"""Guard: production code screens only through the audit, and research code writes no audit records.

Every module in src/ is classified in exactly one of four sets below. A module that is in none of them fails the
test, so a new script cannot appear without someone saying whether it makes decisions.

  PRODUCTION       screens real inputs for a decision. Must not call screen(), get_engine(), ScreeningEngine.screen
                   or any .screen( method (the audited path is audited_screen / AuditedEngine.audited_screen).
  INFRASTRUCTURE   the audit plumbing. Same ban.
  LIBRARY          defines the matcher and its parts. screen() lives here. Must not import the audit modules.
  RESEARCH         evaluation, hold-out work, profiling, fairness analysis, data download, training. May call
                   screen() unchanged. Must not import the audit modules, so evaluation runs write no audit records.

An AST check cannot see every dynamic trick; it catches the ways code normally reaches screen(), including aliases
and getattr with the name as a string. The classification test is what makes a new path visible.

Run:
    pytest tests/test_production_guard.py
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

PRODUCTION = {"screen_cli.py", "audited_screening.py"}
INFRASTRUCTURE = {
    "audit.py", "audit_drift.py", "audit_metrics.py", "audit_purge.py", "audit_recover.py", "audit_report.py",
    "audit_schema.py", "audit_verify.py", "make_baseline.py", "replay.py", "run_drift_batches.py", "run_context.py",
}  # fmt: skip
LIBRARY = {
    "arabic_names.py", "candidates.py", "entities.py", "idf_table.py", "matcher_model.py", "normalise.py",
    "pair_features.py", "pairs.py", "profile_data.py", "scoring.py", "screening.py",
}  # fmt: skip
RESEARCH = {
    "build_idf.py", "build_pair_features.py", "build_pair_splits.py", "build_screening_holdout.py",
    "build_screening_holdout_v3.py", "clean_pairs.py", "compare_matchers.py", "diagnose_arabic_fp.py",
    "diagnose_arabic_secondary.py", "diagnose_screening.py", "download_data.py", "download_pairs.py",
    "download_snapshot.py",
    "evaluate_candidates.py", "evaluate_common_tier.py", "evaluate_final.py", "evaluate_orthographic.py",
    "evaluate_pairs_final.py", "evaluate_scoring.py", "evaluate_screening.py", "evaluate_skeleton_idf.py",
    "evaluate_skeleton_particles.py", "f1_provenance.py", "list_gate.py", "measure_alert_disparity.py",
    "profile_normaliser.py", "profile_pairs.py",
    "train_constrained.py", "train_matcher.py",
}  # fmt: skip
AUDIT_MODULES = {
    "audit",
    "audit_drift",
    "audit_metrics",
    "audit_report",
    "audit_purge",
    "audit_recover",
    "audit_schema",
    "audit_verify",
    "audited_screening",
    "replay",
}
BANNED_NAMES = {"screen", "get_engine"}


def unaudited_screening(source: str) -> list[str]:
    """Where the source reaches screen(), get_engine() or a .screen( method. Empty if it does not."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        where = f"line {getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.ImportFrom) and node.module == "screening":
            found += [f"{where}: imports {a.name} from screening" for a in node.names if a.name in BANNED_NAMES]
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES and isinstance(node.ctx, ast.Load):
            found.append(f"{where}: uses the name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES | {"get_engine"}:
            found.append(f"{where}: uses the attribute .{node.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if any(isinstance(a, ast.Constant) and a.value in BANNED_NAMES for a in node.args[1:2]):
                found.append(f"{where}: getattr with a banned name")
    return found


def audit_imports(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in AUDIT_MODULES]
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in AUDIT_MODULES:
            found.append(node.module)
    return found


# ---------- the checker itself ----------


@pytest.mark.parametrize(
    "source",
    [
        "from screening import screen\nscreen('x')",
        "from screening import screen as s\ns('x')",
        "from screening import get_engine\nget_engine()",
        "import screening\nscreening.screen('x')",
        "import screening\nscreening.get_engine()",
        "from screening import ScreeningEngine\nScreeningEngine.from_file().screen('x')",
        "engine.screen('x')",
        "getattr(engine, 'screen')('x')",
        "x = screen",
    ],
)
def test_the_checker_flags_every_way_of_reaching_screen(source):
    assert unaudited_screening(source), source


@pytest.mark.parametrize(
    "source",
    [
        "from screening import ScreeningConfig, build_matches, query_script",
        "from audited_screening import audited_screen\naudited_screen('x')",
        "engine.audited_screen('x')",
        "screening_config = 1\nscreen_cli = 2",
        "def format_screen_result():\n    pass",
    ],
)
def test_the_checker_passes_audited_code(source):
    assert unaudited_screening(source) == []


def test_the_checker_flags_audit_imports():
    assert audit_imports("import audit") and audit_imports("from audited_screening import AuditedEngine")
    assert audit_imports("from audit_verify import verify_chain")
    assert audit_imports("from screening import ScreeningConfig") == []


# ---------- the real modules ----------


def source_files() -> set[str]:
    return {p.name for p in SRC.glob("*.py")}


def test_every_module_is_classified_exactly_once():
    classes = [PRODUCTION, INFRASTRUCTURE, LIBRARY, RESEARCH]
    everything = set().union(*classes)
    assert sum(len(c) for c in classes) == len(everything), "a module is in two classes"
    unclassified = source_files() - everything
    assert not unclassified, f"classify these as production or research before they land: {sorted(unclassified)}"
    assert not everything - source_files(), f"classified but not in src/: {sorted(everything - source_files())}"


@pytest.mark.parametrize("name", sorted(PRODUCTION | INFRASTRUCTURE))
def test_production_and_audit_modules_never_call_screen_directly(name):
    assert unaudited_screening((SRC / name).read_text()) == [], name


@pytest.mark.parametrize("name", sorted(RESEARCH | LIBRARY))
def test_research_and_library_modules_write_no_audit_records(name):
    assert audit_imports((SRC / name).read_text()) == [], name


def test_the_production_cli_screens_through_the_audited_engine():
    source = (SRC / "screen_cli.py").read_text()
    assert "audited_screen(" in source and "AuditedEngine" in source


# ---------- the deduplication model is outside the decision path ----------

# The trained pair matcher (the "deduplication model"): its model code, its features, its rarity table and the scripts
# that build or train it. The decision path and the Stage 4 monitoring must not reach any of it.
DEDUP_MODULES = {
    "matcher_model", "idf_table", "pair_features", "pairs", "build_idf", "build_pair_features",
    "build_pair_splits", "clean_pairs", "train_matcher", "train_constrained", "compare_matchers",
    "evaluate_pairs_final", "diagnose_screening",
}  # fmt: skip
DEDUP_DATA_MARKERS = ("deployment_idf", "idf_manifest", "pairs_lr", "data/idf", "data/pairs", "models/")
MONITORING_RESEARCH = {"list_gate.py", "download_snapshot.py", "f1_provenance.py"}  # Stage 4 gate modules


def local_imports(source: str) -> set[str]:
    """Names of src/ modules a source imports (a top-level or a local import), by AST."""
    local = {p.stem for p in SRC.glob("*.py")}
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found & local


def import_closure(start: set[str]) -> dict[str, str]:
    """Every src module reachable from `start` through local imports, with the module that first reached it."""
    reached: dict[str, str] = {name: "(start)" for name in start}
    todo = list(start)
    while todo:
        name = todo.pop()
        for dep in local_imports((SRC / f"{name}.py").read_text(encoding="utf-8")):
            if dep not in reached:
                reached[dep] = name
                todo.append(dep)
    return reached


def decision_path_modules() -> set[str]:
    return {m[:-3] for m in PRODUCTION | INFRASTRUCTURE | MONITORING_RESEARCH}


def test_the_import_checker_follows_chains_and_sees_local_imports():
    assert "scoring" in import_closure({"screening"})  # screening imports scoring, a direct import
    assert "candidates" in import_closure({"screening"})  # and scoring imports candidates: two steps
    assert local_imports("def f():\n    import idf_table") == {"idf_table"}  # an import inside a function counts
    assert local_imports("import json, os") == set()


def test_production_and_monitoring_modules_never_reach_the_deduplication_model():
    reached = import_closure(decision_path_modules())
    bad = {m: reached[m] for m in DEDUP_MODULES if m in reached}
    assert not bad, (
        f"the decision path or the monitoring imports the deduplication model: {bad} (module: first importer)"
    )


def test_production_and_monitoring_modules_never_name_the_deduplication_models_data():
    for name in sorted(set(import_closure(decision_path_modules())) - DEDUP_MODULES):
        text = (SRC / f"{name}.py").read_text(encoding="utf-8")
        for marker in DEDUP_DATA_MARKERS:
            assert marker not in text, f"{name}.py names {marker!r}"


def test_the_deduplication_check_bites_on_a_planted_import(tmp_path, monkeypatch):
    import sys

    sys.modules.pop("x", None)
    planted = tmp_path / "src"
    planted.mkdir()
    (planted / "screening.py").write_text("import scoring\n")
    (planted / "scoring.py").write_text("import idf_table\n")
    (planted / "idf_table.py").write_text("")
    monkeypatch.setattr(sys.modules[__name__], "SRC", planted)
    assert "idf_table" in import_closure({"screening"})

"""Tests for the screening interface (src/screening.py) and its command line wrapper (src/screen_cli.py).

Run:
    pytest tests/test_screening.py
"""

import hashlib
import json

import pytest

import screen_cli
import screening
from screening import ScreeningConfig, ScreeningEngine, query_script

LOW = {"Latin": 0.6, "Cyrillic": 0.6, "Arabic": 0.6, "other": 0.6}


def entities():
    return [
        {
            "id": "e1",
            "names": ["Mohammed Ali Hassan", "محمد علي حسن"],
            "birth_dates": ["1970-05-12"],
            "countries": ["sa"],
        },
        {"id": "e2", "names": ["Ivan Petrov"], "birth_dates": ["1980"], "countries": ["ru"]},
        {"id": "e3", "names": ["Zorblax Holdings Ltd"]},
    ]


def engine(thresholds=None):
    config = ScreeningConfig(0.4, 0.3, 1.0, dict(thresholds or LOW), per_script=True)
    return ScreeningEngine.from_entities(entities(), config)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Mohammed Ali", "Latin"),
        ("Иван Петров", "Cyrillic"),
        ("محمد علي", "Arabic"),
        ("张伟", "other"),
        ("Gazprom Нефть", "Cyrillic"),  # mixed: the non-Latin script decides
        ("Mohammed محمد", "Arabic"),  # Arabic wins over everything
        ("Мухаммед محمد", "Arabic"),
        ("עבד אלחי", "other"),
        ("123", "Latin"),  # no letters
        ("", "Latin"),
    ],
)
def test_query_script(name, expected):
    assert query_script(name) == expected


def test_config_defaults_when_the_file_is_missing(tmp_path):
    config = ScreeningConfig.load(tmp_path / "nope.json")
    assert set(config.thresholds) == {"Latin", "Cyrillic", "Arabic", "other"}
    assert len(set(config.thresholds.values())) == 1
    assert config.per_script is False


def test_config_loads_from_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "skeleton_floor": 0.3, "skeleton_lift": 0.2, "blend": 0.5,
        "thresholds": {"Latin": 0.8, "Cyrillic": 0.7, "Arabic": 0.95, "other": 0.6},
    }))  # fmt: skip
    config = ScreeningConfig.load(path)
    assert (config.skeleton_floor, config.skeleton_lift, config.blend) == (0.3, 0.2, 0.5)
    assert config.thresholds == {"Latin": 0.8, "Cyrillic": 0.7, "Arabic": 0.95, "other": 0.6}
    assert config.per_script is False  # off unless the file says otherwise
    assert config.threshold_for("Arabic") == config.global_threshold


def test_the_shipped_config_has_a_threshold_for_every_script_and_one_global_threshold_on():
    config = ScreeningConfig.load()
    assert set(config.thresholds) == {"Latin", "Cyrillic", "Arabic", "other"}
    assert config.per_script is False
    assert {config.threshold_for(s) for s in config.thresholds} == {config.global_threshold}


def test_explanation_for_agreeing_attributes():
    match = engine().screen("Muhammad Ali Hassan", "1970-05-12", "sa")[0]
    e = match.explanation
    assert (match.entity_id, match.score) == ("e1", 1.0)
    assert e.script == "Latin" and e.threshold == 0.6
    assert e.matched_name == "Mohammed Ali Hassan"
    assert e.skeleton_used is True  # the entity has an Arabic alias, so the skeleton stands in for the name score
    assert e.skeleton_score == pytest.approx(1.0)
    assert e.name_score < 1.0
    assert (e.birth_date, e.country) == ("agree", "agree")
    assert (e.birth_date_adjustment, e.country_adjustment) == pytest.approx((0.10, 0.03))
    assert match.names == ("Mohammed Ali Hassan", "محمد علي حسن")


def test_explanation_for_conflicting_attributes():
    low = dict.fromkeys(LOW, 0.5)
    match = engine(low).screen("Mohammed Ali Hassan", "1999-01-01", "us")[0]
    e = match.explanation
    assert (e.birth_date, e.country) == ("conflict", "conflict")
    assert (e.birth_date_adjustment, e.country_adjustment) == pytest.approx((-0.30, -0.15))
    assert match.score == pytest.approx(0.55)  # a perfect name, minus 0.45


def test_explanation_for_missing_attributes_and_for_a_close_birth_year():
    e = engine().screen("Mohammed Ali Hassan")[0].explanation
    assert (e.birth_date, e.country) == ("missing", "missing")
    assert (e.birth_date_adjustment, e.country_adjustment) == (0.0, 0.0)
    close = engine().screen("Mohammed Ali Hassan", "1971")[0].explanation
    assert close.birth_date == "close" and close.birth_date_adjustment == 0.0


def test_skeleton_not_used_when_the_name_already_matches_fully():
    e = engine().screen("Ivan Petrov")[0].explanation
    assert e.name_score == pytest.approx(1.0) and e.skeleton_used is False


def test_threshold_depends_on_the_script_of_the_query():
    strict_latin = dict(LOW, Latin=0.95, Cyrillic=0.5)
    # "Ivan Petrovich" against "Ivan Petrov" scores about 0.88: under the Latin threshold, over the Cyrillic one.
    assert engine(strict_latin).screen("Ivan Petrovich") == []
    cyrillic = engine(strict_latin).screen("Иван Петрович")
    assert [m.entity_id for m in cyrillic] == ["e2"]
    assert 0.5 < cyrillic[0].score < 0.95
    assert cyrillic[0].explanation.script == "Cyrillic" and cyrillic[0].explanation.threshold == 0.5


def test_per_script_thresholds_are_ignored_unless_switched_on():
    # The same query is alerted or not depending on per_script, with one global threshold of 0.95.
    low = dict.fromkeys(("Latin", "Cyrillic", "Arabic", "other"), 0.5)
    off = ScreeningConfig(0.4, 0.3, 1.0, dict(low), global_threshold=0.95)
    on = ScreeningConfig(0.4, 0.3, 1.0, dict(low), global_threshold=0.95, per_script=True)
    assert ScreeningEngine.from_entities(entities(), off).screen("Ivan Petrovich") == []
    assert [m.entity_id for m in ScreeningEngine.from_entities(entities(), on).screen("Ivan Petrovich")] == ["e2"]


def test_legal_forms_do_not_stop_an_organisation_matching():
    assert engine().screen("Zorblax Holdings Limited")[0].entity_id == "e3"


def test_results_are_ranked_and_limit_applies():
    low = dict.fromkeys(LOW, 0.3)
    matches = engine(low).screen("Ivan Petrov Mohammed")
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)
    assert len(matches) >= 2
    assert len(engine(low).screen("Ivan Petrov Mohammed", limit=1)) == 1


def test_unknown_name_returns_nothing():
    assert engine().screen("Zzyzx Qwrt") == []


def test_matches_serialise_to_json():
    match = engine().screen("Muhammad Ali Hassan", "1970-05-12", "sa")[0]
    data = json.loads(json.dumps(match.as_dict(), ensure_ascii=False))
    assert data["entity_id"] == "e1" and data["explanation"]["birth_date"] == "agree"


def test_module_function_uses_the_shared_engine(monkeypatch):
    eng = engine()
    monkeypatch.setattr(screening, "_engine", eng)
    assert screening.screen("Muhammad Ali Hassan", "1970-05-12", "sa") == eng.screen(
        "Muhammad Ali Hassan", "1970-05-12", "sa"
    )


def write_ftm(path):
    rows = [
        {"id": "t1", "schema": "Person", "target": True,
         "properties": {"name": ["Ivan Petrov"], "birthDate": ["1980"], "nationality": ["ru"]}},
        {"id": "n1", "schema": "Person", "target": False, "properties": {"name": ["Maria Garcia"]}},
        {"id": "t2", "schema": "Company", "target": True, "properties": {"name": ["Zorblax Holdings Ltd"]}},
    ]  # fmt: skip
    body = "\n".join(json.dumps(r) for r in rows) + "\n"
    path.write_text(body)
    # the audited CLI identifies the list by its VERSION file
    path.with_name("VERSION").write_text(f"version=test\nsha1={hashlib.sha1(body.encode()).hexdigest()}\n")  # noqa: S324


def test_engine_from_file_screens_only_target_entities(tmp_path):
    path = tmp_path / "entities.ftm.json"
    write_ftm(path)
    eng = ScreeningEngine.from_file(path, ScreeningConfig(0.4, 0.3, 1.0, dict(LOW), per_script=True))
    assert [m.entity_id for m in eng.screen("Ivan Petrov")] == ["t1"]
    assert eng.screen("Maria Garcia") == []  # on the file, but not a target


def test_cli_prints_matches_and_explanation(tmp_path, capsys):
    path = tmp_path / "entities.ftm.json"
    write_ftm(path)
    audit_dir = str(tmp_path / "audit")
    assert screen_cli.main(["init", "--audit-dir", audit_dir]) == 0
    code = screen_cli.main(
        [
            "screen",
            "Ivan Petrov",
            "--birth-date",
            "1980",
            "--country",
            "ru",
            "--input",
            str(path),
            "--audit-dir",
            audit_dir,
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "t1" in out and "birth date agree" in out and "country agree" in out and "skeleton not used" in out


def test_cli_json_output_is_machine_readable(tmp_path, capsys):
    path = tmp_path / "entities.ftm.json"
    write_ftm(path)
    audit_dir = str(tmp_path / "audit")
    assert screen_cli.main(["init", "--audit-dir", audit_dir]) == 0
    capsys.readouterr()  # the init message
    screen_cli.main(["screen", "Ivan Petrov", "--json", "--limit", "1", "--input", str(path), "--audit-dir", audit_dir])
    data = json.loads(capsys.readouterr().out)  # the audited CLI wraps the matches with their record's request ID
    assert data["request_id"] and len(data["matches"]) == 1 and data["matches"][0]["entity_id"] == "t1"
    assert set(data["matches"][0]["explanation"]) >= {"name_score", "skeleton_used", "birth_date", "country"}


def test_cli_says_when_there_is_no_alert(tmp_path, capsys):
    path = tmp_path / "entities.ftm.json"
    write_ftm(path)
    audit_dir = str(tmp_path / "audit")
    assert screen_cli.main(["init", "--audit-dir", audit_dir]) == 0
    assert screen_cli.main(["screen", "Zzyzx Qwrt", "--input", str(path), "--audit-dir", audit_dir]) == 0
    assert "No alert" in capsys.readouterr().out

"""Tests for the adversarial perturbation rules (analysis/perturb.py) and the independent name similarity.

The rules must not come from the matcher: the module imports nothing from src/, and the tests check each family's
edits by hand-worked examples.

Run:
    pytest tests/test_analysis_perturb.py
"""

import ast
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(ANALYSIS))

import perturb  # noqa: E402
import textsim  # noqa: E402


def imported_modules(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_the_perturbation_and_similarity_modules_import_nothing_from_the_matcher():
    local = {p.stem for p in SRC.glob("*.py")}
    for module in ("perturb.py", "textsim.py"):
        imports = imported_modules(ANALYSIS / module)
        assert not imports & local, (module, imports & local)
        assert imports <= {"random", "re", "unicodedata", "difflib"}, imports  # standard library only


def test_the_bgn_family_is_the_matchers_own_scheme_and_gost_and_german_are_real_alternates():
    """The matcher's Cyrillic table is documented as close to BGN/PCGN. Compare letter by letter (the test may read
    the matcher; the analysis module does not)."""
    from normalise import _CYRILLIC

    russian = [ch for ch in perturb.BGN if ch in _CYRILLIC and ch not in "іїєґ"]
    same_as_matcher = {name: sum(_CYRILLIC[ch] == table[ch] for ch in russian) for name, table in
                       (("bgn", perturb.BGN), ("gost", perturb.GOST), ("german", perturb.GERMAN))}  # fmt: skip
    assert same_as_matcher["bgn"] == len(russian) == 33  # identical: so it is a control, not an alternate
    assert same_as_matcher["gost"] <= len(russian) - 4  # differs on at least four Russian letters
    assert same_as_matcher["german"] <= len(russian) - 12
    assert len(perturb.BGN) == len(perturb.GOST) == len(perturb.GERMAN)


def test_vowel_swap_changes_one_interior_vowel_by_the_confusion_pairs():
    variants = perturb.vowel_swap("Ivan Petrov", limit=10)
    assert set(variants) == {"Iven Petrov", "Ivan Patrov", "Ivan Petruv"}  # the first letters of tokens are untouched
    assert all(sum(a != b for a, b in zip(v, "Ivan Petrov", strict=True)) == 1 for v in variants)


def test_vowel_swap_keeps_the_case_of_the_replaced_letter():
    assert "OLEG PETROV".replace("E", "A") in perturb.vowel_swap("OLEG PETROV", limit=10) or any(
        v.isupper() for v in perturb.vowel_swap("OLEG PETROV", limit=10)
    )
    assert all(v.isupper() for v in perturb.vowel_swap("OLEG PETROV", limit=10))


def test_vowel_delete_drops_one_interior_vowel():
    assert set(perturb.vowel_delete("Ivan Petrov", limit=10)) == {"Ivn Petrov", "Ivan Ptrov", "Ivan Petrv"}


def test_consonant_double_repeats_one_interior_letter_that_is_not_already_doubled():
    assert perturb.consonant_double("Ivan", limit=10) == ["Ivann"]  # only the n is an interior doubleable consonant
    assert set(perturb.consonant_double("Ivan Petrov", limit=10)) >= {"Ivann Petrov", "Ivan Pettrov", "Ivan Petrrov"}
    assert "Mohammmed" not in perturb.consonant_double("Mohammed", limit=10)  # already doubled: not tripled


def test_consonant_delete_collapses_a_doubled_consonant_first():
    assert perturb.consonant_delete("Mohammed") == ["Mohamed"]
    assert set(perturb.consonant_delete("Ivan Petrov", limit=10)) == {
        "Ian Petrov", "Iva Petrov", "Ivan Perov", "Ivan Petov", "Ivan Petro",
    }  # fmt: skip


def test_drop_middle_token_needs_three_tokens_and_keeps_the_first_and_last():
    assert perturb.drop_middle_token("Ivan Petrov") == []
    assert perturb.drop_middle_token("Ivan Ivanovich Petrov") == ["Ivan Petrov"]
    assert set(perturb.drop_middle_token("A B C D", limit=5)) == {"A C D", "A B D"}


def test_drop_one_token_leaves_out_the_first_or_the_last():
    assert perturb.drop_one_token("Ivan") == []
    assert set(perturb.drop_one_token("Ivan Ivanovich Petrov")) == {"Ivanovich Petrov", "Ivan Ivanovich"}


def test_reorder_tokens():
    assert perturb.reorder_tokens("Ivan") == []
    assert perturb.reorder_tokens("Ivan Petrov") == ["Petrov Ivan"]
    assert "Petrov Ivan Ivanovich" in perturb.reorder_tokens("Ivan Ivanovich Petrov", limit=5)


def test_the_published_schemes_romanise_a_known_name_differently():
    name = "Щербаков Юрий"
    assert perturb.translit_name("translit_cyrillic_bgn", name) == "Shcherbakov Yuriy"
    assert perturb.translit_name("translit_cyrillic_gost", name) == "Shherbakov Yurij"
    assert perturb.translit_name("translit_cyrillic_german", name) == "Schtscherbakow Jurij"


def test_transliteration_only_applies_to_cyrillic_names_and_gives_one_variant():
    assert perturb.FAMILIES["translit_cyrillic_bgn"]("Ivan Petrov") == []
    assert perturb.FAMILIES["translit_cyrillic_bgn"]("Иван Петров") == ["Ivan Petrov"]


def test_respelling_applies_a_convention_once_and_keeps_the_case():
    assert "Muhammad Ali" in perturb.respell_arabic_conventions("Mohammad Ali", limit=10) or "Muhammed Ali" in (
        perturb.respell_arabic_conventions("Mohammad Ali", limit=10)
    )
    assert "Aleksandr Yurievich".replace("Yu", "Iu") in perturb.respell_cyrillic_conventions(
        "Aleksandr Yurievich", limit=20
    )
    assert perturb.respell_cyrillic_conventions("Хан") == []  # not a Latin name


def test_every_variant_differs_from_the_input_and_the_limit_holds():
    for name in ("Mohammed Abdul Rahman", "Kalinin Aleksei Petrovich", "Иван Петрович Сидоров"):
        for family, make in perturb.FAMILIES.items():
            variants = make(name, 2, 0)
            assert len(variants) <= 2 or family.startswith("translit") and len(variants) == 1
            assert all(v != name and v.strip() == v for v in variants), (family, variants)
            assert len(set(variants)) == len(variants)


def test_generation_is_deterministic_and_the_seed_changes_the_choice():
    name = "Alexander Yuriyevich Khodakovsky"
    assert perturb.vowel_swap(name, 3, 0) == perturb.vowel_swap(name, 3, 0)
    assert perturb.vowel_swap(name, 3, 0) != perturb.vowel_swap(name, 3, 1)


def test_the_control_family_only_changes_the_case():
    assert perturb.control_uppercase("Ivan Petrov") == ["IVAN PETROV"]
    assert perturb.control_uppercase("IVAN") == []


# ---------- the independent similarity ----------


def test_similarity_is_one_for_the_same_tokens_in_any_order_and_ignores_case_and_accents():
    assert textsim.similarity("Ivan Petrov", "PETROV, ivan") == 1.0
    assert textsim.similarity("José García", "jose garcia") == 1.0


def test_similarity_of_unrelated_names_is_low_and_of_near_spellings_high():
    assert textsim.similarity("Tony Montana", "Antonio Oseguera Cervantes") < 0.5
    assert textsim.similarity("Mohammed Ali", "Muhammad Ali") > 0.8


def test_token_relations():
    assert textsim.token_relation("Ivan Petrov", "Petrov Ivan") == "same tokens"
    assert textsim.token_relation("Ivan", "Ivan Petrov") == "query is missing a token the alias has"
    assert textsim.token_relation("Ivan Petrov Jr", "Ivan Petrov") == "query has a token the alias lacks"
    assert textsim.token_relation("Ivan Petrov", "Ivan Petroff") == "same number of tokens, some spelled differently"


def test_best_alias_picks_the_most_similar_and_handles_an_empty_list():
    assert textsim.best_alias("Ivan Petrov", ["Maria Sidorova", "Ivan Petroff"])[0] == "Ivan Petroff"
    assert textsim.best_alias("Ivan", []) == (None, 0.0)

"""Step 2 of the external validation: the engine over the human-labelled Pairs, once. Protocol: pairs_external.py.

Aggregate numbers only are written (reports/pairs_external_scores.json): no names, IDs or pair content.
"""

import multiprocessing
import random
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import pairs_external as pe  # noqa: E402
import textsim  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from scoring import make_record  # noqa: E402
from screening import CANDIDATE_CAP, ScreeningConfig, ScreeningEngine, score_components  # noqa: E402

MAX_QUERY_NAMES = 12
TMP_ID = "__pair__"
ENGINE: ScreeningEngine | None = None


def init_worker() -> None:
    global ENGINE
    ENGINE = BUILT


def build_engine() -> tuple[ScreeningEngine, dict]:
    """The shipped engine over the deployment list, checked against screen(), then set to score one given entity."""
    global ENGINE
    snapshot = common.read_snapshot(common.CURRENT_SNAPSHOT)
    entities = list(snapshot.screened.values())
    engine = ScreeningEngine.from_entities(entities, ScreeningConfig.load())
    rng = random.Random(11)
    sample = rng.sample(entities, 300)
    expected = {}
    for e in sample:
        dob, country = (e["birth_dates"][:1] or [None])[0], (e["countries"][:1] or [None])[0]
        name = e["names"][0]
        got = {m.entity_id: m.score for m in engine.screen(name, dob, country)}
        expected[e["id"]] = (name, dob, country, got.get(e["id"]))
    engine.screener.index.query = lambda _name: [(TMP_ID, 0.0)]  # type: ignore[method-assign]
    ENGINE = engine
    mismatches, compared = 0, 0
    for e in sample:
        name, dob, country, want = expected[e["id"]]
        have = score_against([name], dob, country, (e["names"], e["birth_dates"], e["countries"]))
        if want is not None:  # screen() returns only matches at or over the threshold
            compared += 1
            mismatches += abs(have - want) > 1e-9
    if mismatches or compared < 250:
        sys.exit(f"the pair scoring differs from screen() on {mismatches} of {compared} checked entities; not scoring")
    return engine, {"entities_checked_against_screen": compared, "mismatches": mismatches}


def score_against(query_names: list[str], dob: str | None, country: str | None, record_args: tuple) -> float:
    """The engine's highest final score for these query names against one entity, or -1.0 if it scored none."""
    assert ENGINE is not None
    screener = ENGINE.screener
    screener.records[TMP_ID] = make_record(*record_args)
    best = -1.0
    for name in query_names[:MAX_QUERY_NAMES]:
        comps = screener.components(name, dob, country, keep=CANDIDATE_CAP, blends=(ENGINE.config.blend,))
        if comps:
            best = max(best, score_components(comps, ENGINE.config)[0][1])
    return best


def work(task: tuple) -> tuple:
    label, kind, script, cross, ln, rn, l_dob, l_ctry, r_dob, r_ctry, l_rec, r_rec, bd_both, c_both, shared, schemas = (
        task
    )
    forward = score_against(ln, l_dob, l_ctry, r_rec)
    reverse = score_against(rn, r_dob, r_ctry, l_rec)
    return (
        label,
        kind,
        script,
        cross,
        bd_both,
        c_both,
        forward,
        reverse,
        len(ln) > MAX_QUERY_NAMES or len(rn) > MAX_QUERY_NAMES,
        shared,
        schemas,
    )


def tasks() -> Iterator[tuple]:
    """One task per unique entity pair with a consistent label and a name on each side."""
    labels: dict[tuple[str, str], set[str]] = {}
    for _, rec in pe.iter_pairs():
        labels.setdefault(pe.pair_key(rec), set()).add(rec["judgement"])
    done: set = set()
    stats: Counter = Counter()
    for _, rec in pe.iter_pairs():
        a, b = rec["left"].get("schema"), rec["right"].get("schema")
        if a not in pe.ENTITY_SCHEMAS or b not in pe.ENTITY_SCHEMAS:
            stats["not_an_entity_pair"] += 1
            continue
        key = pe.pair_key(rec)
        if key in done:
            stats["duplicate_line"] += 1
            continue
        done.add(key)
        if len(labels[key]) > 1:
            stats["conflicting_labels_left_out"] += 1
            continue
        ln, rn = pe.names_of(rec["left"]), pe.names_of(rec["right"])
        if not ln or not rn:
            stats["a_side_has_no_name_property"] += 1
            continue
        L, R = rec["left"], rec["right"]
        l_dobs, r_dobs = pe.values(L, ("birthDate",)), pe.values(R, ("birthDate",))
        l_cs, r_cs = pe.values(L, pe.COUNTRY_PROPS), pe.values(R, pe.COUNTRY_PROPS)
        yield (
            1 if rec["judgement"] == "positive" else 0,
            pe.entity_type(L, R),
            pe.pair_script_bucket(ln, rn),
            pe.cross_script(ln, rn),
            ln, rn,
            l_dobs[0] if l_dobs else None, l_cs[0] if l_cs else None,
            r_dobs[0] if r_dobs else None, r_cs[0] if r_cs else None,
            (ln, l_dobs, l_cs), (rn, r_dobs, r_cs),
            bool(l_dobs and r_dobs), bool(l_cs and r_cs), shares_a_name(ln, rn),
            f"{L.get('schema')}/{R.get('schema')}",
        )  # fmt: skip
    yield ("stats", stats)


def name_key(name: str) -> str:
    return " ".join(sorted(textsim.tokens(name)))


def shares_a_name(left: list[str], right: list[str]) -> bool:
    """True if some name of one side equals some name of the other as the same tokens in any order, ignoring case and
    accents (analysis/textsim.py, which does not use the matcher). Such a pair is trivial for any name matcher."""
    return bool({name_key(n) for n in left} & {name_key(n) for n in right})


def confusion(label: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((label == 1) & pred).sum())
    fp = int(((label == 0) & pred).sum())
    fn = int(((label == 1) & ~pred).sum())
    tn = int(((label == 0) & ~pred).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def rates(c: dict) -> dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    pos, neg = tp + fn, fp + tn
    recall = tp / pos if pos else None
    precision = tp / (tp + fp) if tp + fp else None
    fpr = fp / neg if neg else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0 if pos else None
    # the same recall and false-alert rate at the hold-outs' balance (equal numbers of positives and negatives)
    p50 = recall / (recall + fpr) if recall is not None and fpr is not None and recall + fpr else None
    f1_50 = 2 * p50 * recall / (p50 + recall) if p50 and recall else None
    return {
        **c, "pairs": tp + fp + fn + tn, "positives": pos, "negatives": neg,
        "positive_share": pos / (pos + neg) if pos + neg else None,
        "precision": precision, "recall": recall, "f1": f1, "false_positive_rate": fpr,
        "accuracy": (tp + tn) / (pos + neg) if pos + neg else None,
        "precision_if_half_the_pairs_were_positive": p50, "f1_if_half_the_pairs_were_positive": f1_50,
    }  # fmt: skip


def auc(label: np.ndarray, score: np.ndarray) -> float | None:
    pos, neg = int((label == 1).sum()), int((label == 0).sum())
    if not pos or not neg:
        return None
    ranks = rankdata(score)
    return float((ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def f1_interval(c: dict, draws: int = 1000, seed: int = 11) -> list[float]:
    """95% percentile interval for F1 from resampling the four confusion cells (multinomial, fixed seed)."""
    n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    if not n:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    cells = rng.multinomial(n, [c["tp"] / n, c["fp"] / n, c["fn"] / n, c["tn"] / n], size=draws)
    tp, fp, fn = cells[:, 0], cells[:, 1], cells[:, 2]
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))]


def summarise(table: np.ndarray, mask: np.ndarray, direction: str, threshold: float) -> dict:
    label = table["label"][mask]
    score = table[direction][mask]
    pred = score >= threshold
    out = rates(confusion(label, pred))
    out["auc"] = auc(label, np.where(score < 0, 0.0, score))
    out["f1_95_interval"] = f1_interval(confusion(label, pred))
    out["no_candidate_pairs"] = int((score < 0).sum())
    return out


def main() -> None:
    global BUILT
    out_path = common.OUT / "pairs_external_scores.json"
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists; the pairs are scored once")
    if not (common.OUT / "pairs_external_alignment.json").exists():
        sys.exit("the alignment report comes first; run the profile step")
    started = time.time()
    BUILT, self_check = build_engine()
    print(f"engine built in {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    rows, stats = [], Counter()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(max(multiprocessing.cpu_count() - 2, 1), initializer=init_worker) as pool:

        def stream() -> Iterator[tuple]:
            for t in tasks():
                if t[0] == "stats":
                    stats.update(t[1])
                else:
                    yield t

        for i, r in enumerate(pool.imap(work, stream(), chunksize=400)):
            rows.append(r)
            if i % 50000 == 0:
                print(f"{i} pairs scored, {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    dtype = [("label", "i1"), ("kind", "U30"), ("script", "U20"), ("cross", "?"), ("bd", "?"), ("ct", "?"),
             ("forward", "f8"), ("reverse", "f8"), ("capped", "?"), ("shared", "?"), ("schemas", "U40")]  # fmt: skip
    table = np.array(rows, dtype=dtype)
    threshold = pe.THRESHOLD
    everything = np.ones(len(table), dtype=bool)
    result: dict = {
        "file_sha256": pe.SHA256,
        "snapshot": common.read_snapshot(common.CURRENT_SNAPSHOT).version,
        "engine": "the shipped engine, deletion_keys off, threshold 0.86, nothing tuned",
        "protocol": "see analysis/pairs_external.py",
        "self_check_against_screen": self_check,
        "pairs_scored": len(table),
        "pairs_not_scored": dict(stats),
        "pairs_where_either_side_had_more_than_12_names": int(table["capped"].sum()),
    }
    for direction, name in (
        ("forward", "left_names_as_queries_against_right"),
        ("reverse", "right_names_as_queries_against_left"),
    ):
        block = {"overall": summarise(table, everything, direction, threshold)}
        block["by_entity_type"] = {
            k: summarise(table, table["kind"] == k, direction, threshold) for k in sorted(set(table["kind"]))
        }
        block["by_script"] = {
            k: summarise(table, table["script"] == k, direction, threshold) for k in sorted(set(table["script"]))
        }
        block["cross_script_pairs"] = summarise(table, table["cross"], direction, threshold)
        block["same_script_pairs"] = summarise(table, ~table["cross"], direction, threshold)
        block["by_type_and_script"] = {
            f"{k}/{s}": summarise(table, (table["kind"] == k) & (table["script"] == s), direction, threshold)
            for k in sorted(set(table["kind"])) for s in sorted(set(table["script"]))
            if ((table["kind"] == k) & (table["script"] == s)).sum() >= 200
        }  # fmt: skip
        block["shares_an_identical_name"] = summarise(table, table["shared"], direction, threshold)
        block["shares_no_identical_name"] = summarise(table, ~table["shared"], direction, threshold)
        block["shares_no_identical_name_by_type"] = {
            k: summarise(table, ~table["shared"] & (table["kind"] == k), direction, threshold)
            for k in sorted(set(table["kind"]))
        }
        block["shares_no_identical_name_by_script"] = {
            k: summarise(table, ~table["shared"] & (table["script"] == k), direction, threshold)
            for k in sorted(set(table["script"]))
        }
        block["shares_an_identical_name_and_birth_date_on_both_sides"] = summarise(
            table, table["shared"] & table["bd"], direction, threshold
        )
        block["shares_an_identical_name_and_no_birth_date_on_both_sides"] = summarise(
            table, table["shared"] & ~table["bd"], direction, threshold
        )
        block["shares_no_identical_name_and_birth_date_on_both_sides"] = summarise(
            table, ~table["shared"] & table["bd"], direction, threshold
        )
        block["shares_no_identical_name_and_no_birth_date_on_both_sides"] = summarise(
            table, ~table["shared"] & ~table["bd"], direction, threshold
        )
        block["shares_no_identical_name_by_schema_pair"] = {
            k: summarise(table, ~table["shared"] & (table["schemas"] == k), direction, threshold)
            for k in sorted(set(table["schemas"]))
            if (~table["shared"] & (table["schemas"] == k)).sum() >= 100
        }
        block["by_schema_pair"] = {
            k: summarise(table, table["schemas"] == k, direction, threshold)
            for k in sorted(set(table["schemas"]))
            if (table["schemas"] == k).sum() >= 100
        }
        block["birth_date_on_both_sides"] = summarise(table, table["bd"], direction, threshold)
        block["birth_date_not_on_both_sides"] = summarise(table, ~table["bd"], direction, threshold)
        block["country_on_both_sides"] = summarise(table, table["ct"], direction, threshold)
        block["country_not_on_both_sides"] = summarise(table, ~table["ct"], direction, threshold)
        block["neither_birth_date_nor_country_on_both_sides"] = summarise(
            table, ~table["bd"] & ~table["ct"], direction, threshold
        )
        result[name] = block
    common.write_json(out_path, result)
    ov = result["left_names_as_queries_against_right"]["overall"]
    print({k: ov[k] for k in ("pairs", "precision", "recall", "f1", "false_positive_rate", "auc")})
    print(f"wrote {out_path}, {time.time() - started:.0f}s")


BUILT: ScreeningEngine | None = None

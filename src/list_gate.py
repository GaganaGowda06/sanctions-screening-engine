"""List-update regression gate: does moving to a newer OpenSanctions snapshot hurt the matcher on the frozen hold-out?

    python src/screen_cli.py gate --new-snapshot data/snapshots/<version>/entities.ftm.json

The spent hold-out (holdout/screening_holdout_v3.json) is a fixed regression benchmark here; nothing is tuned on it.
Each positive query names the listed entity it should find. A new snapshot can merge, split or delist entities, so:

  1. Resolve every positive's expected entity against the new snapshot:
       present in the new screening list      -> kept
       else a referent link, in either direction -> remapped (a new entity whose referents contain the old ID, or the
                                                  old entity's referents naming an ID that is in the new list)
       else                                   -> dropped: the entity is delisted (or no longer a named target)
     Negatives are never dropped. Dropped queries are counted by script and reason, never listed.
  2. Build the engine on each snapshot with the same configuration. Positives are screened against a list in which
     the entity's tested alias is held out, by the same rule that built the hold-out (the names kept for the group's
     script), applied to the entity's names in that snapshot. Negatives are screened against the full list. This is
     the research path, screen() unchanged; no audit records are written.
  3. On the resolved set, compute F1 (the headline definition of the Stage 2 evaluation) and the Arabic false-alert
     burden (the alert rate on synthetic clean Arabic-script names, the Stage 3 metric), for both snapshots.
  4. A paired bootstrap over the queries kept in both runs (the same resampled queries in both, positives and
     negatives resampled separately, bootstrap_draws draws, fixed seed) gives the F1 delta and the Arabic burden delta
     with 95% percentile intervals.
  5. The gate FAILS when the F1 interval lies entirely below minus the F1 tolerance, or the Arabic burden interval lies
     entirely above plus the burden tolerance. The tolerances (monitoring_config.json) are judgement: how much loss in
     F1, or gain in false alerts on Arabic names, is acceptable for a list update. "Entirely" is strict: an interval
     that touches a tolerance passes.
  6. Also reported: new alerts on hold-out negatives that appear only with the new list and are caused by entities
     that exist only in the new snapshot (the false-alert cost of list growth), in total and for Arabic-script names.

The report is aggregate: counts, rates and intervals, no query text and no entity names. If the "new" snapshot is
older than the current one the run is a REVERSE-TIME CHURN TEST and every result says so.

Not production code: it calls screen() and must not import the audit modules (a test guards that).
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import evaluate_scoring as ev
from evaluate_candidates import script_class
from normalise import normalise
from profile_data import COUNTRY_PROPS, NAME_PROPS
from screening import ScreeningConfig, ScreeningEngine, query_script

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "holdout" / "screening_holdout_v3.json"
CURRENT_SNAPSHOT = ROOT / "data" / "sanctions" / "entities.ftm.json"
CONFIG_PATH = Path(__file__).with_name("monitoring_config.json")
REPORT_DIR = ROOT / "reports"
REVERSE_LABEL = "REVERSE-TIME CHURN TEST"


class SnapshotRefused(ValueError):
    """The snapshot cannot be used (missing, unverifiable, or the same as the current one)."""


@dataclass(frozen=True)
class GateConfig:
    f1_tolerance: float = 0.01  # judgement: an F1 loss of up to one point from a list update is acceptable
    arabic_burden_tolerance: float = 0.02  # judgement: up to two points more false alerts on Arabic names
    bootstrap_draws: int = 1000
    seed: int = 11

    def __post_init__(self) -> None:
        if not 0 <= self.f1_tolerance < 1 or not 0 <= self.arabic_burden_tolerance < 1:
            raise ValueError("tolerances must be fractions from 0 up to (not including) 1")
        if self.bootstrap_draws < 100:
            raise ValueError("bootstrap_draws must be at least 100")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "GateConfig":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            data.get("gate_f1_tolerance", cls.f1_tolerance),
            data.get("gate_arabic_burden_tolerance", cls.arabic_burden_tolerance),
            data.get("gate_bootstrap_draws", cls.bootstrap_draws),
            data.get("gate_seed", cls.seed),
        )


# ---------- snapshots ----------


@dataclass(frozen=True)
class Snapshot:
    version: str
    entities: dict[str, dict]  # every named entity: id, schema, target, names, birth_dates, countries, referents
    total_lines: int

    @property
    def screened(self) -> dict[str, dict]:
        """The screening list: named target entities, what the engine actually screens against."""
        return {i: e for i, e in self.entities.items() if e["target"]}


def verify_snapshot(path: Path) -> tuple[str, str]:
    """(version, sha1) of an entities file, checked against the VERSION file beside it (written by the downloaders
    from the index). Raises SnapshotRefused if there is no VERSION file or the file does not match it."""
    path = Path(path)
    if not path.is_file():
        raise SnapshotRefused(f"{path} is not a file")
    version_file = path.with_name("VERSION")
    if not version_file.is_file():
        raise SnapshotRefused(f"{version_file} is missing: the snapshot cannot be verified against the index")
    recorded = dict(line.split("=", 1) for line in version_file.read_text().splitlines() if "=" in line)
    version, sha1 = recorded.get("version"), recorded.get("sha1")
    if not version or not sha1:
        raise SnapshotRefused(f"{version_file} has no version or sha1")
    digest = hashlib.sha1()  # noqa: S324  (the index publishes SHA-1)
    with open(path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != sha1:
        raise SnapshotRefused(f"{path} does not match the SHA-1 recorded in {version_file}")
    return version, sha1


def read_snapshot(path: Path) -> Snapshot:
    """Read a snapshot's named entities, with their referents. Checked against its VERSION file first."""
    version, _ = verify_snapshot(path)
    entities, lines = {}, 0
    with open(path, "rb") as fh:
        for line in fh:
            lines += 1
            try:
                ent = json.loads(line)
                props = ent.get("properties") or {}
                names = list(
                    dict.fromkeys(
                        v.strip() for p in NAME_PROPS for v in props.get(p) or () if isinstance(v, str) and v.strip()
                    )
                )
            except (ValueError, AttributeError):
                continue
            if names:
                entities[ent["id"]] = {
                    "id": ent["id"],
                    "schema": ent.get("schema"),
                    "target": ent.get("target") is True,
                    "names": names,
                    "birth_dates": [v for v in props.get("birthDate") or () if isinstance(v, str) and v],
                    "countries": [v for p in COUNTRY_PROPS for v in props.get(p) or () if isinstance(v, str) and v],
                    "referents": [r for r in ent.get("referents") or () if isinstance(r, str)],
                }
    return Snapshot(version, entities, lines)


def fingerprint(entity: dict) -> tuple:
    return (
        entity["schema"],
        tuple(sorted(entity["names"])),
        tuple(sorted(entity["birth_dates"])),
        tuple(sorted(entity["countries"])),
    )


def diff_lists(old: Snapshot, new: Snapshot) -> dict:
    """Counts of entities added, removed and changed on the screening list (named targets), plus file sizes."""
    old_list, new_list = old.screened, new.screened
    added, removed = new_list.keys() - old_list.keys(), old_list.keys() - new_list.keys()
    changed = [i for i in old_list.keys() & new_list.keys() if fingerprint(old_list[i]) != fingerprint(new_list[i])]
    return {
        "screening_list_old": len(old_list),
        "screening_list_new": len(new_list),
        "added": len(added),
        "removed": len(removed),
        "changed": len(changed),
        "unchanged": len(old_list.keys() & new_list.keys()) - len(changed),
        "entity_lines_old": old.total_lines,
        "entity_lines_new": new.total_lines,
    }


# ---------- resolving the hold-out against the new snapshot ----------


def referent_index(snapshot: Snapshot) -> dict[str, list[str]]:
    """referent ID -> IDs of the screening-list entities that list it as a referent."""
    index: dict[str, list[str]] = defaultdict(list)
    for entity_id, entity in snapshot.screened.items():
        for referent in entity["referents"]:
            index[referent].append(entity_id)
    return index


def resolve_entity(
    old_id: str, old: Snapshot, new: Snapshot, new_referents: dict[str, list[str]]
) -> tuple[str | None, str]:
    """(new ID or None, outcome) for the entity a positive query expects.

    Outcomes: kept, remapped, dropped_absent (not in the new snapshot at all), dropped_not_a_target (in the new file but
    no longer a named target). Remapping looks for a referent link in both directions and takes the smallest ID if there
    are several (outcome remapped_ambiguous)."""
    screened = new.screened
    if old_id in screened:
        return old_id, "kept"
    candidates = set(new_referents.get(old_id, ()))  # a new entity whose referents contain the old ID
    old_entity = old.entities.get(old_id)
    if old_entity:  # the old entity's referents naming an ID that is in the new list
        candidates |= {r for r in old_entity["referents"] if r in screened}
    if candidates:
        return min(candidates), "remapped" if len(candidates) == 1 else "remapped_ambiguous"
    return None, "dropped_not_a_target" if old_id in new.entities else "dropped_absent"


def kept_names_for(group: str, query: str, names: list[str]) -> list[str]:
    """The names an entity keeps in the list while a positive is tested: the hold-out's own rule (see
    evaluate_scoring.build_positives), applied to the names in a given snapshot."""
    if group == "variant":
        same = normalise(query)
        return [n for n in names if normalise(n) != same]
    if group == "other_same_script":
        same = normalise(query)
        return [n for n in names if script_class(n) == "other" and normalise(n) != same]
    kept_class = ev.GROUP_SPEC[group][1]
    return [n for n in names if script_class(n) == kept_class]


def load_holdout(path: Path = HOLDOUT) -> tuple[list[dict], dict[str, list[str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = payload["queries"]
    for q in queries:
        q["script"] = query_script(q["name"])
    return queries, payload["kept_names"]


def resolve_holdout(queries: list[dict], old: Snapshot, new: Snapshot) -> tuple[list[dict], dict]:
    """Attach `new_id` to each positive that survives, and count what happened. Negatives are never dropped."""
    new_referents = referent_index(new)
    outcomes: Counter = Counter()
    by_script: dict[str, Counter] = {"kept": Counter(), "remapped": Counter(), "dropped": Counter()}
    dropped_by_script: Counter = Counter()
    dropped_by_reason: Counter = Counter()
    resolved = []
    for q in queries:
        if q["kind"] != "positive":
            resolved.append({**q, "new_id": None})
            continue
        new_id, outcome = resolve_entity(q["true_id"], old, new, new_referents)
        if new_id is None:
            outcomes["dropped"] += 1
            dropped_by_script[q["script"]] += 1
            by_script["dropped"][q["script"]] += 1
            dropped_by_reason[outcome] += 1
            continue
        kind = "kept" if outcome == "kept" else "remapped"
        outcomes[kind] += 1
        by_script[kind][q["script"]] += 1
        if outcome == "remapped_ambiguous":
            outcomes["remapped_ambiguous"] += 1
        resolved.append({**q, "new_id": new_id})
    positives = sum(q["kind"] == "positive" for q in queries)
    negatives = sum(q["kind"] != "positive" for q in queries)
    counts = {
        "positives": positives,
        "kept": outcomes["kept"],
        "remapped": outcomes["remapped"],
        "remapped_ambiguous": outcomes["remapped_ambiguous"],
        "dropped": outcomes["dropped"],
        "dropped_by_script": dict(sorted(dropped_by_script.items())),
        "by_script": {
            script: {kind: by_script[kind][script] for kind in ("kept", "remapped", "dropped")}
            for script in sorted(set().union(*by_script.values()))
        },
        "dropped_by_reason": dict(sorted(dropped_by_reason.items())),
        "negatives": negatives,
        "negatives_dropped": negatives - sum(q["kind"] != "positive" for q in resolved),  # always 0
    }
    return resolved, counts


# ---------- engines and queries ----------


def engine_for(
    snapshot: Snapshot, config: ScreeningConfig, held: dict[str, list[str]] | None = None
) -> ScreeningEngine:
    """An engine over the snapshot's screening list; entities in `held` keep only the names given there."""
    held = held or {}
    entities = [
        {
            "id": e["id"],
            "names": held.get(e["id"], e["names"]),
            "birth_dates": e["birth_dates"],
            "countries": e["countries"],
        }
        for e in snapshot.screened.values()
    ]
    return ScreeningEngine.from_entities(entities, config)


def screen_set(engine: ScreeningEngine, queries: list[dict], expected: list[str | None]) -> list[dict]:
    """Research path: screen() unchanged, nothing logged."""
    out = []
    for q, want in zip(queries, expected, strict=True):
        ids = {m.entity_id for m in engine.screen(q["name"], q["dob"], q["country"])}
        out.append(
            {
                "alert": bool(ids),
                "true_alert": want in ids if want else False,
                "wrong_only": bool(ids) and (want not in ids if want else True),
                "ids": ids,
            }
        )
    return out


def run_snapshot(
    snapshot: Snapshot,
    queries: list[dict],
    expected: list[str | None],
    held_names: dict[str, list[str]],
    config: ScreeningConfig,
) -> list[dict]:
    """Outcomes for every query in order: positives against the held-out list, negatives against the full list."""
    held_engine = engine_for(snapshot, config, held_names)
    full_engine = engine_for(snapshot, config)
    out: list[dict | None] = [None] * len(queries)
    for kind_is_positive, engine in ((True, held_engine), (False, full_engine)):
        idx = [i for i, q in enumerate(queries) if (q["kind"] == "positive") == kind_is_positive]
        results = screen_set(engine, [queries[i] for i in idx], [expected[i] for i in idx])
        for i, r in zip(idx, results, strict=True):
            out[i] = r
    return out  # type: ignore[return-value]


# ---------- metrics and the paired bootstrap ----------


def f1_from(tp: float, wrong: float, false_alerts: float, n_positive: float) -> float:
    fp = false_alerts + wrong
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_positive if n_positive else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def headline_arrays(queries: list[dict], old: list[dict], new: list[dict]) -> dict:
    """Arrays over the headline positives and headline negatives, plus the Arabic synthetic-negative mask."""
    pos = [i for i, q in enumerate(queries) if q["kind"] == "positive" and q["group"] in ev.HEADLINE_GROUPS]
    neg = [i for i, q in enumerate(queries) if q["kind"] != "positive" and q["group"] in ev.HEADLINE_NEG_SCRIPTS]
    arabic = np.array([queries[i]["kind"] == "synthetic" and queries[i]["script"] == "Arabic" for i in neg], dtype=bool)

    def col(results: list[dict], idx: list[int], key: str) -> np.ndarray:
        return np.array([results[i][key] for i in idx], dtype=float)

    return {
        "pos_tp_old": col(old, pos, "true_alert"),
        "pos_tp_new": col(new, pos, "true_alert"),
        "pos_wrong_old": col(old, pos, "wrong_only"),
        "pos_wrong_new": col(new, pos, "wrong_only"),
        "neg_old": col(old, neg, "alert"),
        "neg_new": col(new, neg, "alert"),
        "arabic": arabic,
    }


def paired_bootstrap(a: dict, draws: int, seed: int) -> dict:
    """Paired bootstrap of the F1 delta (new minus old) and the Arabic burden delta (new minus old).

    Each draw resamples the positives and the negatives with replacement, the same resampled queries for the old and
    the new run. Intervals are the 2.5th and 97.5th percentiles of the draws. The burden is the alert rate on the
    Arabic synthetic negatives in the resample."""
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(a["pos_tp_old"]), len(a["neg_old"])
    pos_idx = rng.integers(0, n_pos, size=(draws, n_pos)) if n_pos else np.zeros((draws, 0), dtype=int)
    neg_idx = rng.integers(0, n_neg, size=(draws, n_neg)) if n_neg else np.zeros((draws, 0), dtype=int)

    def f1(tp: np.ndarray, wrong: np.ndarray, neg: np.ndarray) -> np.ndarray:
        t, w = tp[pos_idx].sum(axis=1), wrong[pos_idx].sum(axis=1)
        fp = neg[neg_idx].sum(axis=1) + w
        precision = np.divide(t, t + fp, out=np.zeros(draws), where=(t + fp) > 0)
        recall = t / n_pos if n_pos else np.zeros(draws)
        denom = precision + recall
        return np.divide(2 * precision * recall, denom, out=np.zeros(draws), where=denom > 0)

    def burden(neg: np.ndarray) -> np.ndarray:
        mask = a["arabic"][neg_idx]
        count = mask.sum(axis=1)
        alerts = (neg[neg_idx] * mask).sum(axis=1)
        return np.divide(alerts, count, out=np.full(draws, np.nan), where=count > 0)

    f1_delta = f1(a["pos_tp_new"], a["pos_wrong_new"], a["neg_new"]) - f1(
        a["pos_tp_old"], a["pos_wrong_old"], a["neg_old"]
    )
    out = {
        "f1": {
            "old": f1_from(a["pos_tp_old"].sum(), a["pos_wrong_old"].sum(), a["neg_old"].sum(), n_pos),
            "new": f1_from(a["pos_tp_new"].sum(), a["pos_wrong_new"].sum(), a["neg_new"].sum(), n_pos),
            "delta_ci95": [float(np.percentile(f1_delta, 2.5)), float(np.percentile(f1_delta, 97.5))],
        }
    }
    out["f1"]["delta"] = out["f1"]["new"] - out["f1"]["old"]
    n_ar = int(a["arabic"].sum())
    if n_ar:
        b_delta = burden(a["neg_new"]) - burden(a["neg_old"])
        b_delta = b_delta[~np.isnan(b_delta)]
        old_rate = float((a["neg_old"] * a["arabic"]).sum() / n_ar)
        new_rate = float((a["neg_new"] * a["arabic"]).sum() / n_ar)
        out["arabic_burden"] = {
            "n": n_ar,
            "old": old_rate,
            "new": new_rate,
            "delta": new_rate - old_rate,
            "delta_ci95": [float(np.percentile(b_delta, 2.5)), float(np.percentile(b_delta, 97.5))],
        }
    else:
        out["arabic_burden"] = None
    out["draws"], out["seed"] = draws, seed
    return out


def decide(boot: dict, config: GateConfig) -> dict:
    """The gate. Fails when an interval lies ENTIRELY beyond its tolerance (touching the tolerance passes)."""
    reasons = []
    f1_low, f1_high = boot["f1"]["delta_ci95"]
    if f1_high < -config.f1_tolerance:
        reasons.append(f"the F1 interval [{f1_low:+.4f}, {f1_high:+.4f}] lies entirely below -{config.f1_tolerance}")
    burden = boot["arabic_burden"]
    if burden is not None:
        b_low, b_high = burden["delta_ci95"]
        if b_low > config.arabic_burden_tolerance:
            reasons.append(
                f"the Arabic burden interval [{b_low:+.4f}, {b_high:+.4f}] lies entirely above "
                f"+{config.arabic_burden_tolerance}"
            )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "f1_tolerance": config.f1_tolerance,
        "arabic_burden_tolerance": config.arabic_burden_tolerance,
        "tolerances_are": "judgement: how much F1 loss, or extra false alerts on Arabic names, a list update may cost",
    }


# ---------- new entities and their alerts ----------


def new_only_ids(old: Snapshot, new: Snapshot) -> set[str]:
    """IDs on the new screening list that are genuinely new: not on the old screening list, and not a successor of one
    (an entity whose referents contain an old listed ID, or that an old listed entity names as a referent)."""
    old_list = old.screened
    named_by_old = {r for e in old_list.values() for r in e["referents"]}
    out = set()
    for entity_id, entity in new.screened.items():
        if entity_id in old_list or entity_id in named_by_old:
            continue
        if any(r in old_list for r in entity["referents"]):
            continue
        out.add(entity_id)
    return out


def list_growth_alerts(queries: list[dict], old: list[dict], new: list[dict], new_only: set[str]) -> dict:
    """Negatives that alert with the new list and not with the old, and how many are caused by new-only entities."""
    rows = [(q, o, n) for q, o, n in zip(queries, old, new, strict=True) if q["kind"] != "positive"]

    def tally(selector: Callable[[dict], bool]) -> dict:
        subset = [(q, o, n) for q, o, n in rows if selector(q)]
        appear = [(q, o, n) for q, o, n in subset if n["alert"] and not o["alert"]]
        caused = [(q, o, n) for q, o, n in appear if n["ids"] & new_only]
        disappear = [(q, o, n) for q, o, n in subset if o["alert"] and not n["alert"]]
        return {
            "negatives": len(subset),
            "alert_old": sum(o["alert"] for _, o, _ in subset),
            "alert_new": sum(n["alert"] for _, _, n in subset),
            "new_alerts": len(appear),
            "new_alerts_from_new_only_entities": len(caused),
            "alerts_that_went_away": len(disappear),
            "new_alert_rate_from_new_only_entities": len(caused) / len(subset) if subset else 0.0,
        }

    return {
        "new_only_entities_on_the_list": len(new_only),
        "all_negatives": tally(lambda q: True),
        "arabic_script_negatives": tally(lambda q: q["script"] == "Arabic"),
        "arabic_script_synthetic_negatives": tally(lambda q: q["script"] == "Arabic" and q["kind"] == "synthetic"),
    }


# ---------- the gate ----------


def version_stamp(version: str) -> str:
    return version.split("-")[0]


def run_gate(
    old_path: Path,
    new_path: Path,
    holdout: Path = HOLDOUT,
    config: GateConfig | None = None,
    screening_config: ScreeningConfig | None = None,
) -> dict:
    """Run the gate and return the aggregate report. Raises SnapshotRefused if a snapshot cannot be used."""
    config = config or GateConfig.load()
    screening_config = screening_config or ScreeningConfig.load()
    old, new = read_snapshot(old_path), read_snapshot(new_path)
    if old.version == new.version:
        raise SnapshotRefused("the new snapshot is the current one; there is nothing to compare")
    reverse = version_stamp(new.version) < version_stamp(old.version)
    label = REVERSE_LABEL if reverse else "forward list update"

    queries, kept_old = load_holdout(holdout)
    resolved, counts = resolve_holdout(queries, old, new)
    expected_old = [q["true_id"] for q in resolved]
    expected_new = [q["new_id"] for q in resolved]
    held_old = {q["true_id"]: kept_old[q["true_id"]] for q in resolved if q["kind"] == "positive"}
    held_new = {}
    for q in resolved:
        if q["kind"] == "positive":
            held_new[q["new_id"]] = kept_names_for(q["group"], q["name"], new.entities[q["new_id"]]["names"])

    old_results = run_snapshot(old, resolved, expected_old, held_old, screening_config)
    new_results = run_snapshot(new, resolved, expected_new, held_new, screening_config)
    arrays = headline_arrays(resolved, old_results, new_results)
    boot = paired_bootstrap(arrays, config.bootstrap_draws, config.seed)
    verdict = decide(boot, config)
    growth = list_growth_alerts(resolved, old_results, new_results, new_only_ids(old, new))
    sections = {
        "list_diff": diff_lists(old, new),
        "resolution": counts,
        "metrics_on_resolved_set": boot,
        "list_growth_alerts_on_negatives": growth,
        "gate": verdict,
    }
    queries_in_resolved_set = {
        "positives": counts["kept"] + counts["remapped"],
        "headline_positives": int(len(arrays["pos_tp_old"])),
        "negatives": counts["negatives"],
        "headline_negatives": int(len(arrays["neg_old"])),
    }
    if reverse:  # every part of a reverse-time report carries the label
        counts["note"] = (
            "reverse-time: the 'new' list is OLDER, so a dropped positive is one whose entity is not on the older list "
            "(listed after it), not one that was delisted"
        )
        for section in (*sections.values(), queries_in_resolved_set):
            section["label"] = label
    return {
        "label": label,
        "provenance": provenance(holdout),
        "direction": "the new snapshot is OLDER than the current one" if reverse else "the new snapshot is newer",
        "old_snapshot": old.version,
        "new_snapshot": new.version,
        "holdout": Path(holdout).name,
        "list_diff": sections["list_diff"],
        "resolution": sections["resolution"],
        "queries_in_resolved_set": queries_in_resolved_set,
        "metrics_on_resolved_set": sections["metrics_on_resolved_set"],
        "list_growth_alerts_on_negatives": sections["list_growth_alerts_on_negatives"],
        "gate": sections["gate"],
    }


def provenance(holdout: Path) -> dict:
    """Which hold-out and engine are behind each headline number in the gate report, so they are not mistaken for the
    project's other F1 figures."""
    holdout = Path(holdout)
    return {
        "holdout": holdout.name,
        "holdout_sha256": hashlib.sha256(holdout.read_bytes()).hexdigest(),
        "holdout_is": "the third screening hold-out; spent, and used here only as a fixed regression benchmark",
        "engine": "the current engine: the Stage 2 hand-tuned matcher plus the Stage 3 rarity-weighted skeleton, "
        "configuration from src/screening_config.json",
        "f1": "F1 on this hold-out's 1,999 to 2,000 headline positives and 2,000 headline negatives, current engine",
        "arabic_burden": "alert rate on this hold-out's 350 synthetic clean Arabic-script names, current engine",
        "not_the_same_as": {
            "F1 0.777": "the Stage 2 engine (no rarity-weighted skeleton) on the SECOND hold-out, "
            "reports/screening_holdout_v2_evaluation.json",
            "F1 0.776": "the Stage 2 engine on the FIRST hold-out, reports/final_evaluation.json",
            "Arabic burden 22.6%": "the Stage 3 result on this third hold-out, reports/fairness_current.json",
        },
        "decomposition": "reports/stage4_f1_provenance.json (src/f1_provenance.py): the Stage 2 and the current engine "
        "on this same hold-out",
    }


def default_report_path(report: dict) -> Path:
    """reports/stage4_list_update_gate_<new version>.json, with REVERSE_TIME in the name for a reverse-time test."""
    tag = "REVERSE_TIME_" if report["label"] == REVERSE_LABEL else ""
    return REPORT_DIR / f"stage4_list_update_gate_{tag}{report['new_snapshot']}.json"


def write_report(report: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:  # exclusive: a report is never overwritten
        fh.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return out


def summary_lines(report: dict) -> list[str]:
    prefix = f"[{report['label']}] " if report["label"] == REVERSE_LABEL else ""
    d, r, m = report["list_diff"], report["resolution"], report["metrics_on_resolved_set"]
    lines = [
        f"{prefix}{report['old_snapshot']} -> {report['new_snapshot']} ({report['direction']})",
        f"{prefix}screening list {d['screening_list_old']} -> {d['screening_list_new']}: "
        f"{d['added']} added, {d['removed']} removed, {d['changed']} changed",
        f"{prefix}positives {r['positives']}: {r['kept']} kept, {r['remapped']} remapped, {r['dropped']} dropped "
        f"({r['dropped_by_script']}); negatives dropped: {r['negatives_dropped']}",
        f"{prefix}F1 {m['f1']['old']:.4f} -> {m['f1']['new']:.4f} (delta {m['f1']['delta']:+.4f}, "
        f"95% [{m['f1']['delta_ci95'][0]:+.4f}, {m['f1']['delta_ci95'][1]:+.4f}])",
    ]
    if m["arabic_burden"]:
        b = m["arabic_burden"]
        lines.append(
            f"{prefix}Arabic false-alert burden {b['old']:.4f} -> {b['new']:.4f} (delta {b['delta']:+.4f}, "
            f"95% [{b['delta_ci95'][0]:+.4f}, {b['delta_ci95'][1]:+.4f}])"
        )
    g = report["list_growth_alerts_on_negatives"]["all_negatives"]
    lines.append(
        f"{prefix}new alerts on negatives: {g['new_alerts']}, of which {g['new_alerts_from_new_only_entities']} "
        f"from entities only in the new snapshot"
    )
    gate = report["gate"]
    lines.append(f"{prefix}gate: {'PASS' if gate['passed'] else 'FAIL'}" + "".join(f"; {x}" for x in gate["reasons"]))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new-snapshot", required=True, type=Path)
    parser.add_argument("--old-snapshot", type=Path, default=CURRENT_SNAPSHOT)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_gate(args.old_snapshot, args.new_snapshot, args.holdout)
    except (SnapshotRefused, OSError, ValueError) as exc:
        print(f"refusing to run the gate: {exc}", file=sys.stderr)
        return 2
    out = args.out or default_report_path(report)
    try:
        path = write_report(report, out)
    except FileExistsError:
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    print("\n".join(summary_lines(report)))
    print(f"wrote {path}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

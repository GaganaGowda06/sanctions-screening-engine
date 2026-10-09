"""Shared plumbing for the analysis scripts: load a frozen hold-out, build engines, score every query with detail.

Read-only. Nothing here changes scoring, matching or audit code, and nothing writes to the hold-out files. The analysis
scripts call the engine through the research path (Screener.components, the same code screen() uses), never through
the audited path, so no audit records are written.

The scoring detail per query is kept for the true entity (when it is retrieved at all) and for every entity at or
over the threshold, because the error analysis needs the true entity's score even when it is under the threshold.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import evaluate_scoring as ev  # noqa: E402
from f1_provenance import stage2_engine  # noqa: E402
from list_gate import CURRENT_SNAPSHOT, Snapshot, engine_for, read_snapshot  # noqa: E402, F401  (re-exported)
from scoring import Components  # noqa: E402
from screening import CANDIDATE_CAP, ScreeningConfig, ScreeningEngine, query_script, score_components  # noqa: E402

HOLDOUT_V2 = ROOT / "holdout" / "screening_holdout_v2.json"
HOLDOUT_V3 = ROOT / "holdout" / "screening_holdout_v3.json"
OUT = ROOT / "reports"
THRESHOLD = 0.86


def load_holdout(path: Path) -> tuple[list[dict], dict[str, list[str]], str]:
    """The queries (with their query script), the names each positive's entity keeps, and the file's SHA-256, which is
    checked against the checksum recorded beside the file when it was frozen."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    recorded = Path(path).with_suffix(".sha256").read_text().strip()
    if digest != recorded:
        raise SystemExit(f"{path.name} does not match its recorded checksum; refusing to read it")
    payload = json.loads(raw)
    for i, q in enumerate(payload["queries"]):
        q["script"] = query_script(q["name"])
        q["index"] = i
    return payload["queries"], payload["kept_names"], digest


def build_engines(kind: str, snapshot: Snapshot, kept: dict[str, list[str]]) -> tuple[ScreeningEngine, ScreeningEngine]:
    """(engine for positives, engine for negatives). Positives are screened against a list where each sampled entity
    keeps only the names in `kept` (its tested alias removed); negatives against the full list. `kind` is "stage2" (the
    engine as frozen at the end of Stage 2: no rarity-weighted skeleton) or "current" (the shipped engine)."""
    config = ScreeningConfig.load()
    if kind == "stage2":
        return stage2_engine(snapshot, config, kept), stage2_engine(snapshot, config)
    if kind == "current":
        return engine_for(snapshot, config, kept), engine_for(snapshot, config)
    raise ValueError(kind)


def entry(c: Components, score: float, used: bool) -> dict:
    return {
        "id": c.entity_id,
        "score": round(score, 6),
        "name": round(c.names[0], 6),
        "skeleton": round(c.skeleton, 6),
        "skeleton_used": bool(used),
        "dob_adj": round(c.dob, 6),
        "country_adj": round(c.country, 6),
        "arabic": bool(c.arabic),
    }


def score_query(
    engine: ScreeningEngine, q: dict, name: str | None = None, dob: str | None = "__q__", country: str | None = "__q__"
) -> dict:
    """One query through the engine's components path. Returns the true entity's entry (None if it was not retrieved or
    scored under the minimum name score), every entry at or over the threshold, and the best other score."""
    name = q["name"] if name is None else name
    dob = q["dob"] if dob == "__q__" else dob
    country = q["country"] if country == "__q__" else country
    config = engine.config
    threshold = config.threshold_for(query_script(name))
    started = time.perf_counter()
    comps = engine.screener.components(
        name, dob, country, keep=CANDIDATE_CAP, blends=(config.blend,), always=q.get("true_id")
    )
    seconds = time.perf_counter() - started
    scored = score_components(comps, config)
    true = next((entry(c, s, u) for c, s, u in scored if c.entity_id == q.get("true_id")), None)
    alerts = sorted((entry(c, s, u) for c, s, u in scored if s >= threshold), key=lambda e: (-e["score"], e["id"]))
    others = [s for c, s, _ in scored if c.entity_id != q.get("true_id")]
    return {
        "threshold": threshold,
        "true": true,
        "alerts": alerts[:25],
        "n_alerts": len(alerts),
        "best_other": round(max(others, default=0.0), 6),
        "components_seconds": seconds,
    }


def score_holdout(queries: list[dict], pos_engine: ScreeningEngine, neg_engine: ScreeningEngine) -> list[dict]:
    rows = []
    for q in queries:
        engine = pos_engine if q["kind"] == "positive" else neg_engine
        rows.append({**{k: q[k] for k in ("index", "kind", "group", "script", "name", "dob", "country", "true_id")},
                     **score_query(engine, q)})  # fmt: skip
    return rows


def headline_metrics(rows: list[dict]) -> dict:
    """The Stage 2 headline definition (evaluate_scoring.metrics_at): true-entity score at or over the threshold is a
    hit, a miss that alerted on some other entity is a wrong alert (a false positive), a negative that alerts is a false
    positive. Positives are the headline groups; negatives are the Latin, Cyrillic and Arabic query scripts."""
    pos = [r for r in rows if r["kind"] == "positive" and r["group"] in ev.HEADLINE_GROUPS]
    neg = [r for r in rows if r["kind"] != "positive" and r["group"] in ev.HEADLINE_NEG_SCRIPTS]
    hit = [r["true"] is not None and r["true"]["score"] >= r["threshold"] for r in pos]
    wrong = [(not h) and r["best_other"] >= r["threshold"] for h, r in zip(hit, pos, strict=True)]
    false_alert = [r["best_other"] >= r["threshold"] for r in neg]
    tp, wr, fa = sum(hit), sum(wrong), sum(false_alert)
    precision = tp / (tp + wr + fa) if tp + wr + fa else 0.0
    recall = tp / len(pos)
    return {
        "headline_positives": len(pos),
        "headline_negatives": len(neg),
        "true_alerts": tp,
        "wrong_entity_alerts_on_positives": wr,
        "false_alerts_on_negatives": fa,
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fa / len(neg),
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

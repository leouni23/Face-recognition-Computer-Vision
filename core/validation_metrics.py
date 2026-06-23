"""Open-set 1:N identification metrics from a labelled validation session.

The evaluated task is **open-set 1:N identification** (a watchlist / access-control
scenario: a gallery of enrolled subjects, probes that include non-enrolled people the
system must reject). Following NIST FRVT/FRTE 1:N and ISO/IEC 19795, the primary metrics
are **FPIR** and **FNIR**, reported with a DET curve and EER — NOT 1:1 verification
metrics. FAR/FRR are provided only as colloquial aliases.

Inputs (under {data_dir}/validation/<session_id>/):
  - detections.jsonl : one record per detected face per frame, carrying the raw cosine
                       distance, the best-match person id, and the full candidate ranking.
  - labels.jsonl     : ground truth per detection — `true_person_id` (an enrolled subject)
                       OR `non_mate: true`. Threshold-independent, so every metric is
                       recomputable at any threshold offline without re-running the cameras.

Unit of analysis = **event/track** (primary): consecutive detections of the same subject
on a camera are aggregated by majority vote into one "search". The per-frame view is
reported as secondary (video frames are correlated → the effective sample size is smaller).
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Verdict taxonomy (stable keys + Italian display labels) — open-set 1:N.
VERDICTS = ("mate_correct", "mate_miss", "swap", "false_positive", "non_mate_correct")
VERDICT_LABELS = {
    "mate_correct": "Mate corretto",        # enrolled identified as themselves → TP
    "mate_miss": "Mate mancato",            # enrolled reported as Unknown        → FNIR
    "swap": "Scambio",                      # enrolled A reported as enrolled B   → FNIR (separate)
    "false_positive": "Falso positivo",     # non-mate reported as an identity    → FPIR (critical)
    "non_mate_correct": "Non-mate corretto",# unknown correctly rejected          → TN
}

_Z95 = 1.959963985  # standard normal quantile for a 95% two-sided interval
_EVENT_GAP_FRAMES = 15  # frames missing before the same subject is counted as a new event (~1s @15fps)
_DET_STEPS = 201
_FPIR_TARGETS = (0.01, 0.003)  # security-oriented operating points (NIST-style)


# ── IO helpers ──────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _key(rec: dict) -> Tuple[str, int, int]:
    return (str(rec["camera_id"]), int(rec["frame_index"]), int(rec["face_id"]))


def _truth(label: dict) -> Optional[Tuple[bool, Optional[int]]]:
    """(is_mate, true_person_id) from a label record, or None if it carries no ground truth."""
    if label.get("non_mate") is True:
        return (False, None)
    tid = label.get("true_person_id")
    if tid is not None:
        return (True, int(tid))
    return None


# ── statistics ────────────────────────────────────────────────────────────────


def wilson_ci(k: int, n: int, z: float = _Z95) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval for a binomial proportion k/n (no SciPy needed)."""
    if n <= 0:
        return (None, None)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6))


def _proportion_block(errors: int, n: int) -> dict:
    """A reported rate (FPIR/FNIR) with Wilson CI, rule-of-3 and the Doddington rule-of-30 flag."""
    lo, hi = wilson_ci(errors, n)
    block = {
        "value": round(errors / n, 6) if n else None,
        "errors": errors,
        "n": n,
        "ci95": [lo, hi],
        "low_error_warning": (0 < n and errors < 30),  # Doddington rule-of-30
    }
    if n and errors == 0:
        block["rule_of_3_upper"] = round(3.0 / n, 6)  # 95% upper bound with zero observed errors
    return block


# ── event construction + per-threshold decisions ────────────────────────────────


def _build_events(labelled: List[dict]) -> List[dict]:
    """Group labelled detections into events (one 'search' each).

    An event = a contiguous run (small frame gaps tolerated) of detections on one camera
    sharing the same ground-truth identity. Grouping by (camera, true identity) first
    handles multiple simultaneous subjects without a real tracker.
    Each detection in `labelled` must carry: camera_id, frame_index, _is_mate, _true_id,
    raw_cosine_distance, best_match_person_id, candidates.
    """
    groups: Dict[Tuple[str, object], List[dict]] = defaultdict(list)
    for d in labelled:
        tkey = d["_true_id"] if d["_is_mate"] else "non_mate"
        groups[(str(d["camera_id"]), tkey)].append(d)

    events: List[dict] = []
    for (cam, tkey), dets in groups.items():
        dets.sort(key=lambda r: (int(r["frame_index"]), int(r["face_id"])))
        run: List[dict] = [dets[0]]
        for d in dets[1:]:
            if int(d["frame_index"]) - int(run[-1]["frame_index"]) <= _EVENT_GAP_FRAMES:
                run.append(d)
            else:
                events.append(_make_event(cam, tkey, run))
                run = [d]
        events.append(_make_event(cam, tkey, run))
    return events


def _make_event(cam: str, tkey, dets: List[dict]) -> dict:
    is_mate = tkey != "non_mate"
    frames = [(d.get("raw_cosine_distance"), d.get("best_match_person_id")) for d in dets]
    return {
        "camera_id": cam,
        "is_mate": is_mate,
        "true_id": tkey if is_mate else None,
        "frames": frames,               # [(raw_distance, best_match_person_id), ...]
        "dets": dets,                   # kept for CMC (candidate rankings) + timeline
        "first_frame": int(dets[0]["frame_index"]),
        "last_frame": int(dets[-1]["frame_index"]),
    }


def _mate_decision(event: dict, t: float) -> str:
    """Plurality outcome of a mate event at threshold t: mate_correct / swap / mate_miss."""
    g = event["true_id"]
    correct = swap = miss = 0
    for raw, best in event["frames"]:
        if raw is not None and raw < t:        # accepted as best-match identity
            if best == g:
                correct += 1
            else:
                swap += 1
        else:                                  # rejected (distance ≥ t or no match)
            miss += 1
    if correct >= swap and correct >= miss:
        return "mate_correct"
    return "swap" if swap >= miss else "mate_miss"


def _nonmate_decision(event: dict, t: float) -> str:
    """Plurality outcome of a non-mate event: false_positive if mostly accepted, else correct."""
    accept = sum(1 for raw, _ in event["frames"] if raw is not None and raw < t)
    return "false_positive" if accept > (len(event["frames"]) - accept) else "non_mate_correct"


def _decide_all(events: List[dict], t: float) -> dict:
    """Verdict counts + FPIR/FNIR error/total tallies for all events at threshold t."""
    counts = {v: 0 for v in VERDICTS}
    mate_n = non_n = fnir_err = fpir_err = 0
    for ev in events:
        if ev["is_mate"]:
            mate_n += 1
            dec = _mate_decision(ev, t)
            counts[dec] += 1
            if dec != "mate_correct":
                fnir_err += 1
        else:
            non_n += 1
            dec = _nonmate_decision(ev, t)
            counts[dec] += 1
            if dec == "false_positive":
                fpir_err += 1
    return {"counts": counts, "mate_n": mate_n, "non_n": non_n,
            "fnir_err": fnir_err, "fpir_err": fpir_err}


# ── DET / EER / operating points / CMC ───────────────────────────────────────────


def _thresholds(events: List[dict]) -> List[float]:
    dists = [raw for ev in events for raw, _ in ev["frames"] if raw is not None]
    if not dists:
        return []
    lo, hi = min(dists), max(dists)
    if hi <= lo:
        hi = lo + 1e-6
    step = (hi - lo) / (_DET_STEPS - 1)
    return [round(lo + i * step, 6) for i in range(_DET_STEPS)]


def _det_curve(events: List[dict]) -> List[dict]:
    curve = []
    for t in _thresholds(events):
        d = _decide_all(events, t)
        fpir = d["fpir_err"] / d["non_n"] if d["non_n"] else None
        fnir = d["fnir_err"] / d["mate_n"] if d["mate_n"] else None
        curve.append({"threshold": t,
                      "fpir": round(fpir, 6) if fpir is not None else None,
                      "fnir": round(fnir, 6) if fnir is not None else None})
    return curve


def _eer(curve: List[dict]) -> dict:
    best = None
    for row in curve:
        if row["fpir"] is None or row["fnir"] is None:
            continue
        gap = abs(row["fpir"] - row["fnir"])
        if best is None or gap < best[0]:
            best = (gap, round((row["fpir"] + row["fnir"]) / 2, 6), row["threshold"])
    return {"value": best[1], "threshold": best[2]} if best else {"value": None, "threshold": None}


def _fnir_at_fpir(curve: List[dict], target: float) -> dict:
    """Best (lowest FNIR) operating point whose FPIR stays within `target`."""
    feasible = [r for r in curve if r["fpir"] is not None and r["fnir"] is not None and r["fpir"] <= target]
    if not feasible:
        return {"fnir": None, "threshold": None, "actual_fpir": None}
    best = min(feasible, key=lambda r: r["fnir"])
    return {"fnir": best["fnir"], "threshold": best["threshold"], "actual_fpir": best["fpir"]}


def _cmc(events: List[dict]) -> Tuple[List[dict], Optional[float]]:
    """Cumulative Match Characteristic on mate events (threshold-independent ranking).

    Per frame, the rank of the true id within `candidates` (ascending distance); the event
    rank is the median per-frame rank. CMC[k] = fraction of mate events with rank ≤ k.
    Returns ([] , None) if no candidate rankings were logged (e.g. legacy sessions).
    """
    mate_ranks: List[int] = []
    max_rank = 0
    for ev in events:
        if not ev["is_mate"]:
            continue
        g = ev["true_id"]
        per_frame = []
        for d in ev["dets"]:
            cand = d.get("candidates") or []
            order = [int(c[0]) for c in cand]
            max_rank = max(max_rank, len(order))
            if g in order:
                per_frame.append(order.index(g) + 1)  # 1-based rank
        if per_frame:
            per_frame.sort()
            mate_ranks.append(per_frame[len(per_frame) // 2])  # median rank
    if not mate_ranks or max_rank == 0:
        return [], None
    n = len(mate_ranks)
    cmc = [{"rank": k, "rate": round(sum(1 for r in mate_ranks if r <= k) / n, 6)}
           for k in range(1, max_rank + 1)]
    return cmc, cmc[0]["rate"]


# ── per-frame (secondary) ─────────────────────────────────────────────────────────


def _per_frame_rates(labelled: List[dict], t: float) -> dict:
    mate = [d for d in labelled if d["_is_mate"]]
    non = [d for d in labelled if not d["_is_mate"]]
    fnir_err = sum(
        1 for d in mate
        if not (d.get("raw_cosine_distance") is not None
                and d["raw_cosine_distance"] < t
                and d.get("best_match_person_id") == d["_true_id"])
    )
    fpir_err = sum(
        1 for d in non
        if d.get("raw_cosine_distance") is not None and d["raw_cosine_distance"] < t
    )
    return {
        "fpir": _proportion_block(fpir_err, len(non)),
        "fnir": _proportion_block(fnir_err, len(mate)),
        "mate_detections": len(mate),
        "non_mate_detections": len(non),
        "note": "Secondario: i frame consecutivi sono correlati → N effettivo minore, CI ottimistici.",
    }


# ── top-level computation ─────────────────────────────────────────────────────────


def _metrics_from(detections: List[dict], threshold: float) -> dict:
    """Full open-set 1:N metrics from detections that already carry `_is_mate`/`_true_id`
    (unlabelled detections are dropped). `threshold` is the operating point."""
    labelled = [d for d in detections if d.get("_truth") is not None]
    events = _build_events(labelled)
    op = _decide_all(events, threshold)

    fpir = _proportion_block(op["fpir_err"], op["non_n"])
    fnir = _proportion_block(op["fnir_err"], op["mate_n"])
    curve = _det_curve(events)
    cmc, rank1 = _cmc(events)

    caveats: List[str] = []
    if op["mate_n"] < 30 or op["non_n"] < 30:
        caveats.append(
            "Campione piccolo: <30 eventi mate o non-mate. Le stime e gli intervalli "
            "vanno letti con cautela (regola del 30 di Doddington)."
        )
    if fpir.get("rule_of_3_upper") is not None:
        caveats.append("Zero falsi positivi osservati: FPIR riportato con limite superiore ~3/N (regola del 3).")
    if fnir.get("rule_of_3_upper") is not None:
        caveats.append("Zero falsi rifiuti osservati: FNIR riportato con limite superiore ~3/N (regola del 3).")
    if not cmc:
        caveats.append("CMC non disponibile: le detection non contengono il ranking dei candidati (sessione legacy).")

    return {
        "schema": "open-set-1N",
        "unit": "event",
        "operating_threshold": threshold,
        "counts": {
            "total_detections": len(detections),
            "labelled_detections": len(labelled),
            "total_events": len(events),
            "mate_events": op["mate_n"],
            "non_mate_events": op["non_n"],
            "verdicts": op["counts"],
            "verdict_labels": VERDICT_LABELS,
        },
        "fpir": fpir,
        "fnir": fnir,
        "aliases": {"far": fpir["value"], "frr": fnir["value"],
                    "note": "FAR/FRR = alias colloquiali di FPIR/FNIR (valutazione open-set 1:N, non verifica 1:1)."},
        "eer": _eer(curve),
        "fnir_at_fpir": {f"{tgt}": _fnir_at_fpir(curve, tgt) for tgt in _FPIR_TARGETS},
        "rank1": rank1,
        "cmc": cmc,
        "det_curve": curve,
        "per_frame": _per_frame_rates(labelled, threshold),
        "caveats": caveats,
    }


def _declared_truth(det: dict) -> Optional[Tuple[bool, Optional[int]]]:
    """Ground truth tagged onto the detection itself by the run context (crossing sessions:
    the operator declared who was crossing). Used when there is no manual label override."""
    if det.get("non_mate") is True:
        return (False, None)
    tid = det.get("true_person_id")
    if tid is not None:
        return (True, int(tid))
    return None


def _attach_truth(detections: List[dict], labels: Dict[Tuple, Tuple[bool, Optional[int]]]) -> None:
    for det in detections:
        truth = labels.get(_key(det))          # manual label wins (override / common sessions)
        if truth is None:
            truth = _declared_truth(det)        # else automatic GT from the declared subject
        det["_truth"] = truth
        if truth is not None:
            det["_is_mate"], det["_true_id"] = truth


def compute_session_metrics(session_dir: Path) -> dict:
    detections = _read_jsonl(session_dir / "detections.jsonl")
    labels: Dict[Tuple, Tuple[bool, Optional[int]]] = {}
    for r in _read_jsonl(session_dir / "labels.jsonl"):
        truth = _truth(r)
        if truth is not None:
            labels[_key(r)] = truth  # last write wins (re-labelling)
    _attach_truth(detections, labels)

    threshold = _session_threshold(session_dir)
    result = _metrics_from(detections, threshold)

    cameras = sorted({str(d["camera_id"]) for d in detections})
    if len(cameras) > 1:
        result["per_camera"] = {
            cam: _metrics_from([d for d in detections if str(d["camera_id"]) == cam], threshold)
            for cam in cameras
        }
    return result


def _session_threshold(session_dir: Path) -> float:
    try:
        meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        return float(meta.get("threshold"))
    except Exception:
        from config.settings import get_settings
        return float(get_settings().match_threshold)


def compute_and_export(session_dir: Path) -> dict:
    """Compute metrics and write metrics.json + det_curve.csv + cmc.csv into the session dir."""
    result = compute_session_metrics(session_dir)
    (session_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (session_dir / "det_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "fpir", "fnir"])
        w.writeheader()
        w.writerows(result["det_curve"])
    with (session_dir / "cmc.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "rate"])
        w.writeheader()
        w.writerows(result["cmc"])
    return result

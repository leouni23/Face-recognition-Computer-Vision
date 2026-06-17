"""Offline accuracy metrics from a labelled validation session.

Inputs (under {data_dir}/validation/<session_id>/):
  - detections.jsonl : one record per detected face per frame (with raw_cosine_distance
                       to the nearest template and the best-match person id),
  - labels.jsonl     : ground-truth verdict per detection, supplied via the review UI.

Verdict keys (security-critical errors made explicit):
  corretta              enrolled person identified as themselves      → TP
  falso_rifiuto         enrolled person reported as Unknown           → FN
  falsa_accettazione    unknown reported as an enrolled identity      → FP  (critical)
  scambio               enrolled A reported as enrolled B (mis-ID)    → FP-type
  sconosciuto_corretto  unknown correctly reported as Unknown         → TN

Operating-threshold rates follow the brief: FAR = FP/(FP+TN), FRR = FN/(FN+TP).
The DET curve / EER come from sweeping the threshold over the raw cosine distances,
split into genuine (nearest template IS the same person: corretta + falso_rifiuto)
and impostor (everyone else) score sets — reproducible without re-running the cameras.
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

VERDICTS = ("corretta", "falso_rifiuto", "falsa_accettazione", "scambio", "sconosciuto_corretto")
_GENUINE = {"corretta", "falso_rifiuto"}
_IMPOSTOR = {"falsa_accettazione", "scambio", "sconosciuto_corretto"}


def _key(rec: dict) -> Tuple:
    return (str(rec["camera_id"]), int(rec["frame_index"]), int(rec["face_id"]))


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


def _metrics_from(detections: List[dict]) -> dict:
    """Compute counts, rates and the DET/EER from a list of detections that already
    carry a 'verdict' field (unlabelled ones are ignored)."""
    counts = {v: 0 for v in VERDICTS}
    genuine: List[float] = []
    impostor: List[float] = []
    labelled = 0

    for det in detections:
        verdict = det.get("verdict")
        if verdict not in VERDICTS:
            continue
        labelled += 1
        counts[verdict] += 1
        d = det.get("raw_cosine_distance")
        if d is None:
            continue
        if verdict in _GENUINE:
            genuine.append(float(d))
        elif verdict in _IMPOSTOR:
            impostor.append(float(d))

    tp, fn = counts["corretta"], counts["falso_rifiuto"]
    fp, tn = counts["falsa_accettazione"], counts["sconosciuto_corretto"]
    swap = counts["scambio"]

    def ratio(num, den):
        return round(num / den, 4) if den else None

    summary = {
        "labelled_detections": labelled,
        "total_detections": len(detections),
        "counts": counts,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "scambio": swap,
        "tpir": ratio(tp, tp + fn + swap),       # correct identification rate (enrolled)
        "far": ratio(fp, fp + tn),               # brief: FP/(FP+TN)
        "frr": ratio(fn, fn + tp),               # brief: FN/(FN+TP)
        "correct_rejection_rate": ratio(tn, fp + tn),
    }
    det_curve, eer, eer_threshold = _det_curve(genuine, impostor)
    return {
        "summary": summary,
        "genuine_scores": len(genuine),
        "impostor_scores": len(impostor),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "det_curve": det_curve,
    }


def compute_session_metrics(session_dir: Path) -> dict:
    detections = _read_jsonl(session_dir / "detections.jsonl")
    labels = {_key(r): r["verdict"] for r in _read_jsonl(session_dir / "labels.jsonl")}
    for det in detections:
        det["verdict"] = labels.get(_key(det))

    result = _metrics_from(detections)

    # Per-camera breakdown (only meaningful with more than one camera)
    cameras = sorted({str(d["camera_id"]) for d in detections})
    if len(cameras) > 1:
        result["per_camera"] = {
            cam: _metrics_from([d for d in detections if str(d["camera_id"]) == cam])
            for cam in cameras
        }
    return result


def _det_curve(genuine: List[float], impostor: List[float], steps: int = 201):
    """Sweep the acceptance threshold over cosine distance.

    A detection "accepts" (claims a match) when distance < threshold:
      FAR(t) = fraction of impostor scores < t   (impostor wrongly accepted)
      FRR(t) = fraction of genuine scores  >= t   (genuine wrongly rejected)
    EER is where FAR and FRR cross.
    """
    if not genuine or not impostor:
        return [], None, None
    g = np.asarray(genuine)
    im = np.asarray(impostor)
    lo = float(min(g.min(), im.min()))
    hi = float(max(g.max(), im.max()))
    if hi <= lo:
        hi = lo + 1e-6
    thresholds = np.linspace(lo, hi, steps)

    curve = []
    best_gap = float("inf")
    eer = eer_t = None
    for t in thresholds:
        far = float((im < t).mean())
        frr = float((g >= t).mean())
        curve.append({"threshold": round(float(t), 6), "far": round(far, 6), "frr": round(frr, 6)})
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            eer = round((far + frr) / 2.0, 6)
            eer_t = round(float(t), 6)
    return curve, eer, eer_t


def compute_and_export(session_dir: Path) -> dict:
    """Compute metrics and write metrics.json + threshold_sweep.csv into the session dir."""
    result = compute_session_metrics(session_dir)
    (session_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (session_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "far", "frr"])
        w.writeheader()
        w.writerows(result["det_curve"])
    return result

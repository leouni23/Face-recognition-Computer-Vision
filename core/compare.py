"""Offline cross-session comparison — Phase 1 (Standard) vs Phase 2 (Optimized-TX2).

Scans the validation root on the external disk, groups sessions by profile (optionally also by
condition preset), and recomputes the open-set 1:N metrics **per group** purely from the saved
`detections.jsonl` + `labels.jsonl` — no camera re-run. Reuses the unchanged
`core.validation_metrics.compute_session_metrics`; to combine several sessions into one group we
concatenate their JSONL into a temp dir, namespacing `camera_id` by session so per-session
frame/face keys never collide.

Used by both `scripts/compare_sessions.py` (CLI) and `GET /api/validation/compare` (UI).
"""
import csv
import io
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from core.validation import validation_root
from core.validation_metrics import compute_session_metrics


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def scan_sessions(root: Optional[Path] = None) -> List[dict]:
    """List every session under `root` with its profile/type/conditions (re-scanned from disk
    each call, so an unmounted→remounted external disk just works). Sessions predating the
    profile field default to `standard`."""
    root = root or validation_root()
    out: List[dict] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        manifest = d / "session.json"
        if not manifest.is_file():
            continue
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        perf = meta.get("performance") or {}
        out.append({
            "session_id": meta.get("session_id", d.name),
            "dir": str(d),
            "profile": meta.get("profile") or perf.get("profile") or "standard",
            "session_type": meta.get("session_type"),
            "record_video": meta.get("record_video"),
            "performance": perf,
            "presets": list((meta.get("presets") or {}).keys()),
            "start": meta.get("start"),
            "threshold": meta.get("threshold"),
        })
    return out


def _combine_metrics(sessions: List[dict], preset: Optional[str] = None) -> Optional[dict]:
    """Concatenate the JSONL of `sessions` (optionally only detections of `preset`) into a temp
    session dir and run the standard metric computation over the union."""
    tmp = Path(tempfile.mkdtemp(prefix="faceid_cmp_"))
    try:
        # Nested with (not parenthesized multi-with — that syntax is 3.9+).
        with (tmp / "detections.jsonl").open("w", encoding="utf-8") as det:
            with (tmp / "labels.jsonl").open("w", encoding="utf-8") as lab:
                any_det = False
                for s in sessions:
                    sid = s["session_id"]
                    sd = Path(s["dir"])
                    for rec in _read_jsonl(sd / "detections.jsonl"):
                        if preset is not None and (rec.get("preset_id") or "none") != preset:
                            continue
                        rec["camera_id"] = f"{sid}::{rec.get('camera_id')}"  # avoid cross-session key clashes
                        det.write(json.dumps(rec) + "\n")
                        any_det = True
                    for rec in _read_jsonl(sd / "labels.jsonl"):
                        rec["camera_id"] = f"{sid}::{rec.get('camera_id')}"
                        lab.write(json.dumps(rec) + "\n")
        if not any_det:
            return None
        return compute_session_metrics(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _slim(m: Optional[dict]) -> dict:
    if not m:
        return {"fpir": None, "fnir": None, "eer": None, "rank1": None, "counts": {}}
    val = lambda x: x.get("value") if isinstance(x, dict) else x
    return {
        "fpir": val(m.get("fpir")), "fnir": val(m.get("fnir")),
        "eer": (m.get("eer") or {}).get("value") if isinstance(m.get("eer"), dict) else m.get("eer"),
        "rank1": val(m.get("rank1")),
        "counts": m.get("counts", {}),
    }


def compare(root: Optional[Path] = None, by_preset: bool = False) -> dict:
    """Group sessions by profile (and optionally preset), recompute combined metrics per group,
    and emit Standard-vs-Optimized deltas. Pure offline; no camera."""
    root = root or validation_root()
    sessions = scan_sessions(root)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for s in sessions:
        groups[s["profile"]].append(s)

    by_profile = {}
    for prof, sess in groups.items():
        entry = {"sessions": len(sess), "overall": _slim(_combine_metrics(sess))}
        if by_preset:
            presets = sorted({(r.get("preset_id") or "none")
                              for s in sess for r in _read_jsonl(Path(s["dir"]) / "detections.jsonl")})
            entry["by_preset"] = {p: _slim(_combine_metrics(sess, preset=p)) for p in presets}
        by_profile[prof] = entry

    deltas = {}
    std, opt = by_profile.get("standard"), by_profile.get("optimized-tx2")
    if std and opt:
        for k in ("fpir", "fnir", "eer", "rank1"):
            a, b = std["overall"].get(k), opt["overall"].get(k)
            deltas[k] = (b - a) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None

    return {
        "root": str(root),
        "n_sessions": len(sessions),
        "profiles": by_profile,
        "deltas_optimized_minus_standard": deltas,
        "sessions": sessions,
    }


def to_csv(comparison: dict) -> str:
    """Flatten the per-profile overall metrics to CSV (one row per profile) for the paper."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["profile", "sessions", "fpir", "fnir", "eer", "rank1",
                "mate_events", "non_mate_events"])
    for prof, e in comparison.get("profiles", {}).items():
        o = e["overall"]
        c = o.get("counts", {})
        w.writerow([prof, e["sessions"], o.get("fpir"), o.get("fnir"), o.get("eer"),
                    o.get("rank1"), c.get("mate_events"), c.get("non_mate_events")])
    return buf.getvalue()

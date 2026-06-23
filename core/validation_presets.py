"""Condition presets + subject registry for the validation experiment (persisted JSON).

Two reusable, operator-configured datasets, stored under the data dir so they survive
restarts and are shared across sessions:

- **Condition presets** (`validation_presets.json`): the controlled environmental conditions
  of a crossing — camera position relative to the subject and lighting. Reused per crossing;
  the full parameters are snapshotted into each session's manifest so the record is
  self-contained even if a preset is later edited.
- **Subject registry** (`validation_subjects.json`): `S1…Sn` enrolled subjects mapped to a
  gallery person_id (mate), and `U1…Un` anonymous unknowns (non-mate; a number only, never a
  name — privacy-preserving). Drives the per-run ground truth: declaring who is crossing makes
  the ground truth automatic.

Pure data layer (no Flask, no pipeline) so it can be reused by the web routes and the manager.
"""
import json
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import get_settings

_settings = get_settings()
_lock = threading.Lock()

_CAMERA_ORIENT = ("frontale", "laterale", "angolata")
_LIGHT_ARRANGE = ("stesso_lato", "opposto", "entrambi", "altro")


def _data_dir() -> Path:
    d = Path(_settings.data_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _presets_path() -> Path:
    return _data_dir() / "validation_presets.json"


def _subjects_path() -> Path:
    return _data_dir() / "validation_subjects.json"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic on the same filesystem


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "preset"


def _num_or_none(v) -> Optional[float]:
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


# ── Condition presets ───────────────────────────────────────────────────────────


def normalize_preset(raw: dict) -> dict:
    """Coerce an incoming preset to the canonical schema (lenient: bad enums fall back)."""
    cam = raw.get("camera") or {}
    light = raw.get("lighting") or {}
    orient = cam.get("orientation")
    arrange = light.get("arrangement")
    try:
        count = int(light.get("count")) if light.get("count") not in (None, "") else 1
    except (TypeError, ValueError):
        count = 1
    return {
        "preset_id": raw.get("preset_id") or _slug(raw.get("name", "")),
        "name": (raw.get("name") or "Preset").strip()[:60],
        "camera": {
            "orientation": orient if orient in _CAMERA_ORIENT else _CAMERA_ORIENT[0],
            "angle_deg": _num_or_none(cam.get("angle_deg")),
        },
        "lighting": {
            "count": max(0, count),
            "arrangement": arrange if arrange in _LIGHT_ARRANGE else _LIGHT_ARRANGE[0],
            "angle_deg": _num_or_none(light.get("angle_deg")),
        },
        "notes": (raw.get("notes") or "").strip()[:500],
    }


def load_presets() -> List[dict]:
    return _read_json(_presets_path(), [])


def get_preset(preset_id: str) -> Optional[dict]:
    return next((p for p in load_presets() if p.get("preset_id") == preset_id), None)


def upsert_preset(raw: dict) -> dict:
    """Create or update a preset (by preset_id). New ids are made unique from the name."""
    preset = normalize_preset(raw)
    with _lock:
        presets = load_presets()
        idx = next((i for i, p in enumerate(presets) if p.get("preset_id") == preset["preset_id"]), None)
        if idx is None:
            # ensure a unique id for a brand-new preset
            base, ids = preset["preset_id"], {p.get("preset_id") for p in presets}
            n = 2
            while preset["preset_id"] in ids:
                preset["preset_id"] = f"{base}-{n}"; n += 1
            presets.append(preset)
        else:
            presets[idx] = preset
        _write_json(_presets_path(), presets)
    return preset


def duplicate_preset(preset_id: str) -> Optional[dict]:
    src = get_preset(preset_id)
    if src is None:
        return None
    copy = dict(src)
    copy["preset_id"] = ""  # force a new unique id
    copy["name"] = f"{src.get('name', 'Preset')} (copia)"
    return upsert_preset(copy)


def delete_preset(preset_id: str) -> bool:
    with _lock:
        presets = load_presets()
        kept = [p for p in presets if p.get("preset_id") != preset_id]
        if len(kept) == len(presets):
            return False
        _write_json(_presets_path(), kept)
    return True


# ── Subject registry ─────────────────────────────────────────────────────────────


def _empty_registry() -> dict:
    return {"enrolled": [], "unknown": []}


def load_subjects() -> dict:
    reg = _read_json(_subjects_path(), _empty_registry())
    reg.setdefault("enrolled", [])
    reg.setdefault("unknown", [])
    return reg


def save_subjects(raw: dict) -> dict:
    """Persist the subject registry. Enrolled labels map to a gallery person_id (mate);
    unknown labels are anonymous (a number only — never a name)."""
    enrolled = []
    for e in raw.get("enrolled", []) or []:
        label = str(e.get("label", "")).strip()
        pid = e.get("person_id")
        if not label:
            continue
        try:
            enrolled.append({"label": label, "person_id": int(pid)})
        except (TypeError, ValueError):
            continue  # an enrolled subject without a valid person_id is dropped
    unknown = []
    for u in raw.get("unknown", []) or []:
        label = str((u.get("label") if isinstance(u, dict) else u) or "").strip()
        if label:
            unknown.append({"label": label})  # label only, by design
    reg = {"enrolled": enrolled, "unknown": unknown}
    with _lock:
        _write_json(_subjects_path(), reg)
    return reg


def subject_truth(label: str) -> Optional[dict]:
    """Ground truth for a declared subject label (threshold-independent):
    an enrolled subject → {'true_person_id': pid} (mate); an unknown → {'non_mate': True};
    unknown label → None."""
    if not label:
        return None
    reg = load_subjects()
    for e in reg["enrolled"]:
        if e["label"] == label:
            return {"true_person_id": e["person_id"]}
    for u in reg["unknown"]:
        if u["label"] == label:
            return {"non_mate": True}
    return None


def all_subject_labels() -> List[str]:
    reg = load_subjects()
    return [e["label"] for e in reg["enrolled"]] + [u["label"] for u in reg["unknown"]]

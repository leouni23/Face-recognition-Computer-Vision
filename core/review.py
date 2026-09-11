"""Offline review state: data-source root, archive (read-only) mode, per-campaign review state.

After a lab experiment the data tree is copied 1:1 from the device to a workstation and reviewed
there. That review must never mutate the copy: exclusions, verdicts and notes go into a NEW file
per campaign (`review_state.json`, next to `campaign.json`) — `session.json`, `detections.jsonl`
and the footage stay byte-identical.

Two roots coexist on purpose:
  * `core.validation.validation_root()` — where NEW sessions are recorded (device / working disk);
  * `review_root()` here            — what the review UI reads (usually the copied archive).
Pointing the review at an archive flips ARCHIVE MODE on: writes that would produce new data
(sessions, enrolment, profile switches) are refused, while review/metrics/export stay available.
"""
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from core.validation import validation_root

REVIEW_STATE_FILE = "review_state.json"
_STATE_VERSION = 1

# UI-chosen root + archive override live outside the data tree and outside anything pydantic reads
# at boot (the dotenv incident: the boot must never depend on a runtime-written file).
_CONFIG_PATH = Path(os.environ.get("FACEID_REVIEW_CONFIG",
                                   str(Path.home() / ".config" / "faceid-review.json")))
_cfg_lock = threading.Lock()
_state_lock = threading.Lock()


# ── UI configuration (review root + archive override) ─────────────────────────────

def _load_cfg() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_cfg(cfg: dict) -> None:
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(_CONFIG_PATH, json.dumps(cfg, indent=2, ensure_ascii=False))
    except OSError as exc:
        logger.warning("Review: impossibile salvare la configurazione ({})".format(exc))


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + os.replace: a crash mid-write can never truncate the state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def review_root() -> Path:
    """Root the review reads. Defaults to the recording root when the operator hasn't chosen one."""
    with _cfg_lock:
        chosen = (_load_cfg().get("review_root") or "").strip()
    return Path(chosen) if chosen else validation_root()


def set_review_root(path: Optional[str]) -> dict:
    """Point the review at `path` (empty/None → back to the recording root). Returns validate_root."""
    with _cfg_lock:
        cfg = _load_cfg()
        cfg["review_root"] = (path or "").strip()
        _save_cfg(cfg)
    info = validate_root(str(review_root()))
    logger.info("Review: root dati = {} ({} campagne)".format(review_root(), info.get("n_campaigns")))
    return info


def validate_root(path: str) -> dict:
    """Is `path` a usable validation tree? Feedback for the UI: N campaigns / N sessions found."""
    out = {"ok": False, "path": path or "", "n_campaigns": 0, "n_sessions": 0,
           "is_recording_root": False, "errors": []}
    if not path:
        out["errors"].append("Percorso vuoto")
        return out
    p = Path(path)
    if not p.exists():
        out["errors"].append("Percorso inesistente")
        return out
    if not p.is_dir():
        out["errors"].append("Il percorso non è una cartella")
        return out
    try:
        from core.compare import scan_sessions
        sessions = scan_sessions(p)
        campaigns = [d for d in p.iterdir() if d.is_dir() and (d / "campaign.json").is_file()]
    except OSError as exc:
        out["errors"].append("Cartella illeggibile: {}".format(exc))
        return out
    out["n_campaigns"] = len(campaigns)
    out["n_sessions"] = len(sessions)
    out["is_recording_root"] = (p.resolve() == validation_root().resolve())
    if not sessions and not campaigns:
        out["errors"].append("Nessuna campagna o sessione trovata in questa cartella")
        return out
    out["ok"] = True
    return out


# ── Archive mode (read-only) ──────────────────────────────────────────────────────

def _cameras_connected() -> bool:
    try:
        from core.camera_manager import camera_manager
        return any(c.get("status") == "connected" for c in camera_manager.list_cameras())
    except Exception:
        return False


def archive_mode() -> dict:
    """Read-only archive mode: AUTOMATIC when the review root isn't the recording root, or when no
    camera is connected (nothing could be recorded anyway). The operator can force it on/off."""
    with _cfg_lock:
        override = _load_cfg().get("archive_override")  # True / False / None
    try:
        different_root = review_root().resolve() != validation_root().resolve()
    except OSError:
        different_root = True
    cams = _cameras_connected()
    auto = bool(different_root or not cams)
    if different_root:
        reason = "la root di revisione non è quella di registrazione (copia dell'esperimento)"
    elif not cams:
        reason = "nessuna camera connessa"
    else:
        reason = "root di lavoro con camere attive"
    active = auto if override is None else bool(override)
    return {"archive": active, "auto": auto, "override": override, "reason": reason,
            "review_root": str(review_root()), "recording_root": str(validation_root())}


def set_archive_override(value: Optional[bool]) -> dict:
    """Force archive mode on (True) / off (False), or None to go back to automatic."""
    with _cfg_lock:
        cfg = _load_cfg()
        cfg["archive_override"] = None if value is None else bool(value)
        _save_cfg(cfg)
    return archive_mode()


# ── Per-campaign review state (exclusions / verdicts / notes) ─────────────────────

def _empty_state() -> dict:
    return {"version": _STATE_VERSION, "updated_at": None,
            "campaign": {"excluded": False, "reason": "", "is_test": False},
            "sessions": {}}


def campaign_dir(folder: str) -> Optional[Path]:
    """Resolve a campaign folder under the review root (rejects traversal)."""
    if not folder or "/" in folder or "\\" in folder or folder in (".", ".."):
        return None
    root = review_root().resolve()
    d = (root / folder).resolve()
    if d.is_dir() and root in d.parents:
        return d
    return None


def load_state(folder: str) -> dict:
    d = campaign_dir(folder)
    if d is None:
        return _empty_state()
    try:
        raw = json.loads((d / REVIEW_STATE_FILE).read_text(encoding="utf-8"))
    except Exception:
        return _empty_state()
    state = _empty_state()
    state.update({k: v for k, v in raw.items() if k in ("version", "updated_at")})
    if isinstance(raw.get("campaign"), dict):
        state["campaign"].update(raw["campaign"])
    if isinstance(raw.get("sessions"), dict):
        state["sessions"] = raw["sessions"]
    return state


def _save_state(folder: str, state: dict) -> dict:
    d = campaign_dir(folder)
    if d is None:
        raise ValueError("Campagna non trovata: {}".format(folder))
    state["version"] = _STATE_VERSION
    state["updated_at"] = datetime.now().isoformat()
    _atomic_write(d / REVIEW_STATE_FILE, json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True))
    return state


def update_sessions(folder: str, session_ids: List[str], **fields) -> dict:
    """Merge `fields` into the review state of each session id (batch-friendly). Only the given
    keys change: exclusions and verdicts recorded earlier are preserved."""
    allowed = ("excluded", "reason", "reviewed", "verdict", "notes")
    patch = {k: v for k, v in fields.items() if k in allowed and v is not None}
    with _state_lock:
        state = load_state(folder)
        now = datetime.now().isoformat()
        for sid in session_ids or []:
            entry = state["sessions"].get(sid) or {}
            entry.update(patch)
            entry["updated_at"] = now
            state["sessions"][sid] = entry
        return _save_state(folder, state)


def flag_campaign(folder: str, excluded: Optional[bool] = None, reason: Optional[str] = None,
                  is_test: Optional[bool] = None) -> dict:
    """Mark a whole campaign as excluded / test — persisted, never a deletion."""
    with _state_lock:
        state = load_state(folder)
        if excluded is not None:
            state["campaign"]["excluded"] = bool(excluded)
        if reason is not None:
            state["campaign"]["reason"] = reason
        if is_test is not None:
            state["campaign"]["is_test"] = bool(is_test)
        return _save_state(folder, state)


def excluded_sessions(folder: str) -> set:
    """Session ids excluded from the analysis in this campaign."""
    return {sid for sid, e in (load_state(folder).get("sessions") or {}).items()
            if isinstance(e, dict) and e.get("excluded")}


def excluded_everywhere(root: Optional[Path] = None) -> set:
    """Session ids excluded anywhere under `root` — sessions of an excluded campaign included, so
    metrics/compare/export can filter with a single set."""
    root = root or review_root()
    out = set()
    if not root.is_dir():
        return out
    try:
        campaigns = [d for d in root.iterdir() if d.is_dir() and (d / "campaign.json").is_file()]
    except OSError:
        return out
    for d in campaigns:
        state = load_state(d.name)
        if state["campaign"].get("excluded"):
            try:
                out.update(s.name for s in d.iterdir() if (s / "session.json").is_file())
            except OSError:
                pass
        else:
            out.update(excluded_sessions(d.name))
    return out


def list_review_campaigns() -> List[dict]:
    """Campaigns under the review root, newest first, each with included/excluded counts and the
    profiles present — the most recent is the one the UI preselects (it's the valid experiment)."""
    from core.compare import scan_sessions
    root = review_root()
    if not root.is_dir():
        return []
    try:
        dirs = [d for d in root.iterdir() if d.is_dir() and (d / "campaign.json").is_file()]
    except OSError:
        return []
    by_group: Dict[str, List[dict]] = {}
    for s in scan_sessions(root):
        by_group.setdefault(s.get("group") or "", []).append(s)
    out = []
    for d in dirs:
        try:
            meta = json.loads((d / "campaign.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        state = load_state(d.name)
        sessions = by_group.get(d.name, [])
        excl = excluded_sessions(d.name)
        out.append({
            "folder": d.name, "dir": str(d),
            "name": meta.get("name", d.name),
            "created_at": meta.get("created_at"), "closed_at": meta.get("closed_at"),
            "is_test": bool(meta.get("is_test")) or bool(state["campaign"].get("is_test")),
            "excluded": bool(state["campaign"].get("excluded")),
            "exclude_reason": state["campaign"].get("reason", ""),
            "n_sessions": len(sessions),
            "n_excluded": sum(1 for s in sessions if s["session_id"] in excl),
            "n_reviewed": sum(1 for s in sessions
                              if (state["sessions"].get(s["session_id"]) or {}).get("reviewed")),
            "profiles": sorted({s.get("profile") for s in sessions if s.get("profile")}),
        })
    out.sort(key=lambda c: c["folder"], reverse=True)
    return out


def review_sessions(folder: str) -> List[dict]:
    """Sessions of a campaign enriched with their review state + the counters the list shows."""
    from core.compare import scan_sessions
    d = campaign_dir(folder)
    if d is None:
        return []
    state = load_state(folder)
    out = []
    for s in scan_sessions(review_root()):
        if s.get("group") != folder:
            continue
        st = state["sessions"].get(s["session_id"]) or {}
        sd = Path(s["dir"])
        out.append(dict(s, **{
            "excluded": bool(st.get("excluded")),
            "reason": st.get("reason", ""),
            "reviewed": bool(st.get("reviewed")),
            "verdict": st.get("verdict"),
            "notes": st.get("notes", ""),
            "n_detections": _count_lines(sd / "detections.jsonl"),
            "n_labels": _count_lines(sd / "labels.jsonl"),
            "has_video": (sd / "video").is_dir(),
        }))
    out.sort(key=lambda s: s["session_id"])
    return out


def _count_lines(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0

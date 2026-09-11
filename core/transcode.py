"""On-demand H.264 transcoding for the offline review player.

Recorded footage uses the `mp4v` FOURCC (MPEG-4 Part 2): ffmpeg and OpenCV read it fine, but
browsers CANNOT decode it in a <video> element — the player just shows a grey rectangle. The
review needs real playback (seek, speed, frame stepping), so each segment is transcoded once to
H.264/yuv420p with `+faststart` and cached.

The cache lives OUTSIDE the data tree (~/.cache/faceid-review by default): the 1:1 copy of the
experiment must stay byte-identical, and transcoded files are always regenerable. A per-file lock
keeps two concurrent requests from launching two ffmpeg runs on the same segment.
"""
import hashlib
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

CACHE_DIR = Path(os.environ.get("FACEID_REVIEW_CACHE",
                                str(Path.home() / ".cache" / "faceid-review")))

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_jobs: Dict[str, dict] = {}          # cache key → {"state": pending|running|done|error, ...}
_ffmpeg_checked: Optional[str] = None


def ffmpeg_path() -> Optional[str]:
    """Path to ffmpeg, or None. Cached after the first lookup."""
    global _ffmpeg_checked
    if _ffmpeg_checked is None:
        _ffmpeg_checked = shutil.which("ffmpeg") or ""
    return _ffmpeg_checked or None


def _key(src: Path) -> str:
    """Cache key from path + size + mtime: a re-copied or edited source re-transcodes on its own."""
    try:
        st = src.stat()
        raw = "{}|{}|{}".format(src.resolve(), st.st_size, int(st.st_mtime))
    except OSError:
        raw = str(src)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def cached_path(src: Path) -> Path:
    return CACHE_DIR / (_key(src) + ".mp4")


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        lk = _locks.get(key)
        if lk is None:
            lk = _locks[key] = threading.Lock()
        return lk


def status(src: Path) -> dict:
    """State of the transcode for `src` (no work started here)."""
    key = _key(src)
    out = cached_path(src)
    if out.is_file() and out.stat().st_size > 0:
        return {"state": "done", "ready": True, "path": str(out)}
    job = _jobs.get(key) or {}
    state = job.get("state", "absent")
    return {"state": state, "ready": False, "error": job.get("error"),
            "ffmpeg": bool(ffmpeg_path())}


def ensure_h264(src: Path, timeout: int = 1800) -> dict:
    """Transcode `src` to H.264 if not cached yet (blocking) → {ready, path} or {error}."""
    src = Path(src)
    if not src.is_file():
        return {"ready": False, "error": "File video non trovato: {}".format(src)}
    out = cached_path(src)
    if out.is_file() and out.stat().st_size > 0:
        return {"ready": True, "path": str(out), "cached": True}
    exe = ffmpeg_path()
    if not exe:
        # Explicit, never a silent muted player: the UI shows this and falls back to JPEG frames.
        return {"ready": False,
                "error": "ffmpeg non disponibile: installalo per riprodurre i video "
                         "(i file registrati usano mp4v, che i browser non decodificano)."}
    key = _key(src)
    with _lock_for(key):
        if out.is_file() and out.stat().st_size > 0:      # produced while we waited on the lock
            return {"ready": True, "path": str(out), "cached": True}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.mp4")
        _jobs[key] = {"state": "running", "src": str(src)}
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            _jobs[key] = {"state": "error", "error": "timeout di transcodifica"}
            _unlink(tmp)
            return {"ready": False, "error": "Transcodifica troppo lenta (timeout)"}
        except OSError as exc:
            _jobs[key] = {"state": "error", "error": str(exc)}
            _unlink(tmp)
            return {"ready": False, "error": "Transcodifica fallita: {}".format(exc)}
        if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            err = (proc.stderr or b"").decode("utf-8", "ignore").strip()[-300:]
            _jobs[key] = {"state": "error", "error": err or "ffmpeg fallito"}
            _unlink(tmp)
            return {"ready": False, "error": "Transcodifica fallita: {}".format(err or "ffmpeg")}
        os.replace(str(tmp), str(out))
        _jobs[key] = {"state": "done"}
        logger.info("Review: transcodifica H.264 completata ({} → {})".format(src.name, out.name))
        return {"ready": True, "path": str(out), "cached": False}


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def transcode_many(sources, progress: Optional[dict] = None) -> dict:
    """Pre-transcode a whole campaign (background thread): `progress` is updated in place so the
    UI can poll it."""
    srcs = [Path(s) for s in sources]
    prog = progress if progress is not None else {}
    prog.update({"total": len(srcs), "done": 0, "errors": [], "running": True})
    for s in srcs:
        res = ensure_h264(s)
        if not res.get("ready"):
            prog["errors"].append({"file": s.name, "error": res.get("error")})
        prog["done"] += 1
    prog["running"] = False
    return prog


def cache_size_bytes() -> int:
    try:
        return sum(f.stat().st_size for f in CACHE_DIR.glob("*.mp4"))
    except OSError:
        return 0

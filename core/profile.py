"""Runtime performance profile — Standard vs Optimized-TX2.

`standard` reproduces today's exact pipeline (buffalo_l, FP32/CUDA, full resolution, no
frame-skip / tracker / batch). `optimized-tx2` layers on the Jetson optimizations, each
parameter overridable via the OPT_* settings. A single image flips between profiles at
runtime; the active profile + its parameters are snapshotted into every validation session
so Phase-1 (Standard) vs Phase-2 (Optimized) stay directly comparable offline.
"""
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

from config.settings import get_settings

STANDARD = "standard"
OPTIMIZED = "optimized-tx2"


# NamedTuple (immutable, Python 3.6-compatible — `dataclasses` is 3.7+).
class Profile(NamedTuple):
    name: str
    model_pack: str                      # InsightFace pack (e.g. buffalo_l / buffalo_s)
    precision: str                       # fp32 | fp16 | int8 (TensorRT, optimized only)
    det_size: Optional[Tuple[int, int]]  # (w, h) detection downsample, or None = full res
    frame_skip: int                      # process 1 frame every N (>= 1)
    tracker: bool                        # IoU tracker carries identity (skip re-embedding)
    batch_embed: bool                    # embed all faces of a frame in one forward pass
    engine_cache_dir: str                # where TensorRT engines are cached (external disk)

    @property
    def is_optimized(self) -> bool:
        return self.name == OPTIMIZED


def _engine_cache_dir(s) -> str:
    return s.trt_engine_cache_dir.strip() or str(Path(s.data_dir) / "engines")


def get_profile() -> Profile:
    """Resolve the active profile from settings (read live each call, so a runtime profile
    switch via /api/settings/profile takes effect immediately)."""
    s = get_settings()
    if (s.performance_profile or "").strip().lower() == OPTIMIZED:
        w, h = int(s.opt_det_width), int(s.opt_det_height)
        det = (w, h) if w > 0 and h > 0 else None
        return Profile(
            name=OPTIMIZED, model_pack=s.opt_model_pack, precision=(s.opt_precision or "fp16").lower(),
            det_size=det, frame_skip=max(1, int(s.opt_frame_skip)),
            tracker=bool(s.opt_tracker), batch_embed=bool(s.opt_batch_embed),
            engine_cache_dir=_engine_cache_dir(s),
        )
    # Standard — today's exact behaviour; all OPT_* knobs ignored.
    return Profile(
        name=STANDARD, model_pack="buffalo_l", precision="fp32", det_size=None,
        frame_skip=1, tracker=False, batch_embed=False, engine_cache_dir=_engine_cache_dir(s),
    )


def profile_summary() -> dict:
    """Self-describing snapshot for session.json / PROTOCOL.md (paper comparability)."""
    p = get_profile()
    return {
        "profile": p.name, "model_pack": p.model_pack, "precision": p.precision,
        "det_size": list(p.det_size) if p.det_size else None,
        "frame_skip": p.frame_skip, "tracker": p.tracker, "batch_embed": p.batch_embed,
    }


def profile_slug() -> str:
    """Short tag for the session folder name: `standard` → `standard`, else `optTX2`."""
    return "standard" if get_profile().name == STANDARD else "optTX2"

"""Lightweight IoU face tracker for the Optimized-TX2 profile.

Carries a face across frames so the **expensive embedding** (the ArcFace forward pass) is run
only when a box can't be matched to an existing track; the cheap matching/ranking step still
runs every frame off the cached embedding, so per-frame results/details stay complete. This is
the big throughput/power win on the TX2. Used only when the profile enables the tracker —
Standard never instantiates it.

Boxes are `FaceLocation = (top, right, bottom, left)`.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

FaceLocation = Tuple[int, int, int, int]


def _iou(a: FaceLocation, b: FaceLocation) -> float:
    at, ar, ab, al = a
    bt, br, bb, bl = b
    ix1, iy1 = max(al, bl), max(at, bt)
    ix2, iy2 = min(ar, br), min(ab, bb)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ar - al) * max(0, ab - at)
    area_b = max(0, br - bl) * max(0, bb - bt)
    return inter / float(area_a + area_b - inter + 1e-9)


class FaceTracker:
    """Greedy IoU association with embedding caching and stale-track aging."""

    def __init__(self, iou_thr: float = 0.3, max_age: int = 30):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self._tracks: Dict[int, dict] = {}  # tid → {bbox, emb, age}
        self._next = 0

    def step(self, locations: List[FaceLocation]) -> List[Tuple[int, Optional[np.ndarray]]]:
        """Associate each detection to a track (greedy IoU); create new tracks for unmatched
        boxes; age out stale ones. Returns, per detection, `(track_id, cached_embedding|None)`
        — `None` means this face must be embedded this frame."""
        for tr in self._tracks.values():
            tr["age"] += 1
        used = set()
        out: List[Tuple[int, Optional[np.ndarray]]] = []
        for loc in locations:
            best_tid, best_iou = None, self.iou_thr
            for tid, tr in self._tracks.items():
                if tid in used:
                    continue
                i = _iou(loc, tr["bbox"])
                if i >= best_iou:
                    best_iou, best_tid = i, tid
            if best_tid is None:
                self._next += 1
                best_tid = self._next
                self._tracks[best_tid] = {"bbox": loc, "emb": None, "age": 0}
            else:
                self._tracks[best_tid]["bbox"] = loc
                self._tracks[best_tid]["age"] = 0
            used.add(best_tid)
            out.append((best_tid, self._tracks[best_tid]["emb"]))
        for tid in [t for t, tr in self._tracks.items() if tr["age"] > self.max_age]:
            del self._tracks[tid]
        return out

    def set_embedding(self, tid: int, emb: np.ndarray) -> None:
        if tid in self._tracks:
            self._tracks[tid]["emb"] = emb

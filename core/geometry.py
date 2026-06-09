"""Planar homography helpers for projecting camera-pixel points onto a floor map.

A homography is a 3x3 matrix H mapping points from the camera image plane to the
floor-plan plane. It is estimated from >=4 point correspondences
(pixel (px,py) in the frame  <->  map (mx,my) on the floor plan) via cv2.findHomography.
Projecting a frame point: (mx,my) = H @ (px,py,1), then divide by the third coord.

Note: H models the *ground plane*. Face detections sit ~1.5 m above the floor, so the
projected position is an approximation suitable for zone-level tracking, not centimetres.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

PointPair = dict  # {"px": float, "py": float, "mx": float, "my": float}


def compute_homography(points: List[PointPair]) -> np.ndarray:
    """Estimate the 3x3 homography from >=4 pixel<->map correspondences."""
    if len(points) < 4:
        raise ValueError("Servono almeno 4 coppie di punti per l'omografia")
    src = np.array([[float(p["px"]), float(p["py"])] for p in points], dtype=np.float64)
    dst = np.array([[float(p["mx"]), float(p["my"])] for p in points], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise ValueError("Omografia non calcolabile: punti degeneri o allineati")
    return H


def project_point(H, px: float, py: float) -> Optional[Tuple[float, float]]:
    """Project a single camera-pixel point through H onto the map plane."""
    H = np.asarray(H, dtype=np.float64)
    v = H @ np.array([px, py, 1.0], dtype=np.float64)
    if v[2] == 0:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])

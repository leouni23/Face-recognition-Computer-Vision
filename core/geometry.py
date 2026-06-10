"""Camera→floor-map projection helpers. Two calibration modes:

HOMOGRAPHY — for cameras that see the floor. A 3x3 matrix H maps frame pixels on the
ground plane to the floor plan, estimated from >=4 point correspondences via
cv2.findHomography. Fails geometrically when the floor is not in view.

POLAR — for cameras that do NOT see the floor (e.g. a desk webcam looking across the
room). Uses the pinhole model with the near-constant physical size of the human face:
    distance  d = k / h_px          (h_px = face bbox height in pixels)
    bearing   β = heading + s · (x_norm − 0.5)
    position  P = C + d · (cos β, sin β)        (C = camera position on the map)
The constants (heading, angular scale s ≈ horizontal FOV, distance constant k) are
solved by least squares from >=2 reference samples: the subject stands at a known map
point while the system records the face's x_norm and h_px.

Both modes give zone-level accuracy (not centimetres): homography because faces sit
~1.5 m above the modelled plane, polar because face size varies between people (±10%).
"""
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

PointPair = dict  # {"px": float, "py": float, "mx": float, "my": float}
PolarSample = dict  # {"mx": float, "my": float, "x_norm": float, "h_px": float}


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


def solve_polar_calibration(cam: Tuple[float, float], samples: List[PolarSample]) -> dict:
    """Solve {heading, scale, k} from >=2 reference samples taken at known map points.

    For each sample i (subject standing at map point P_i, face seen at x_norm_i with
    height h_px_i):  β_i = atan2(P_i − C)  and  d_i = |P_i − C|.
    Linear least-squares fit of  β_i = heading + scale · u_i  (u_i = x_norm_i − 0.5),
    and  k = mean(d_i · h_px_i).
    """
    if len(samples) < 2:
        raise ValueError("Servono almeno 2 campioni di riferimento")
    cx, cy = float(cam[0]), float(cam[1])

    u, beta, k_vals = [], [], []
    for s in samples:
        dx, dy = float(s["mx"]) - cx, float(s["my"]) - cy
        d = math.hypot(dx, dy)
        h_px = float(s["h_px"])
        if d < 5.0:
            raise ValueError("Un campione coincide con la posizione della camera")
        if h_px <= 0:
            raise ValueError("Campione con altezza volto non valida")
        u.append(float(s["x_norm"]) - 0.5)
        beta.append(math.atan2(dy, dx))
        k_vals.append(d * h_px)

    if max(u) - min(u) < 0.05:
        raise ValueError(
            "Campioni troppo vicini orizzontalmente nell'inquadratura: "
            "posizionati in punti più distanti tra loro (sinistra/destra del campo visivo)"
        )

    # Unwrap bearings around the first sample so the linear fit is not broken by ±π wrap
    beta = [beta[0] + math.remainder(b - beta[0], math.tau) for b in beta]

    u_arr, b_arr = np.array(u), np.array(beta)
    u_mean, b_mean = u_arr.mean(), b_arr.mean()
    scale = float(((u_arr - u_mean) * (b_arr - b_mean)).sum() / ((u_arr - u_mean) ** 2).sum())
    heading = float(b_mean - scale * u_mean)

    return {
        "cam": [cx, cy],
        "heading": heading,
        "scale": scale,
        "k": float(np.mean(k_vals)),
    }


def project_polar(params: dict, x_norm: float, h_px: float) -> Optional[Tuple[float, float]]:
    """Project a face (horizontal position + pixel height) onto the map via polar params."""
    if h_px <= 0:
        return None
    cx, cy = params["cam"]
    bearing = params["heading"] + params["scale"] * (x_norm - 0.5)
    d = params["k"] / h_px
    return float(cx + d * math.cos(bearing)), float(cy + d * math.sin(bearing))

from typing import List, Tuple

import cv2
import face_recognition
import numpy as np

FaceLocation = Tuple[int, int, int, int]  # top, right, bottom, left


def detect_faces(
    frame: np.ndarray, scale: float = 0.5, model: str = "hog"
) -> List[FaceLocation]:
    """Detect face bounding boxes on a downscaled copy for real-time performance."""
    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small = small[:, :, ::-1]  # BGR → RGB (face_recognition expects RGB)
    locations = face_recognition.face_locations(rgb_small, model=model)
    inv = 1.0 / scale
    return [
        (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
        for t, r, b, l in locations
    ]


def compute_encodings(
    frame: np.ndarray, locations: List[FaceLocation]
) -> List[np.ndarray]:
    """Compute 128-d face embeddings for each detected location."""
    rgb = frame[:, :, ::-1]
    return face_recognition.face_encodings(rgb, known_face_locations=locations)

"""Rule-based gesture recognition from MediaPipe hand landmarks.

No ML model to train: we infer which fingers are extended from the 21 hand
landmarks and map the finger pattern to a named gesture. This is fast and
reliable for a fixed vocabulary of gestures.

Landmark indices (MediaPipe Hands):
    0 wrist
    1-4  thumb  (CMC, MCP, IP, TIP)
    5-8  index  (MCP, PIP, DIP, TIP)
    9-12 middle
    13-16 ring
    17-20 pinky
"""

from __future__ import annotations

import numpy as np

WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
INDEX_TIP = 8


def _fingers_up(landmarks: np.ndarray, handedness: str) -> dict[str, bool]:
    """Return which fingers are extended.

    For index..pinky a finger is "up" when its tip is above (smaller y than)
    its PIP joint. For the thumb we compare x against the IP joint, accounting
    for left/right hand orientation.
    """
    up: dict[str, bool] = {}
    for name in ("index", "middle", "ring", "pinky"):
        tip_y = landmarks[FINGER_TIPS[name], 1]
        pip_y = landmarks[FINGER_PIPS[name], 1]
        up[name] = tip_y < pip_y

    # Thumb: extended when the tip is to the outer side of the IP joint.
    # In the mirrored frame a "Right" hand's thumb points left when extended.
    tip_x = landmarks[THUMB_TIP, 0]
    ip_x = landmarks[THUMB_IP, 0]
    if handedness == "Right":
        up["thumb"] = tip_x < ip_x
    else:
        up["thumb"] = tip_x > ip_x
    return up


def _pinch_distance(landmarks: np.ndarray) -> float:
    """Normalized distance between thumb tip and index tip."""
    return float(np.linalg.norm(landmarks[THUMB_TIP, :2] - landmarks[INDEX_TIP, :2]))


def classify(
    landmarks: np.ndarray,
    handedness: str = "Right",
    pinch_threshold: float = 0.06,
) -> str:
    """Classify a hand pose into a named gesture.

    Returns one of: pinch, fist, open_palm, pointing, victory, thumbs_up,
    unknown.
    """
    # Pinch takes priority: thumb and index tips touching is unambiguous and
    # overlaps with several finger patterns.
    if _pinch_distance(landmarks) < pinch_threshold:
        return "pinch"

    up = _fingers_up(landmarks, handedness)
    thumb = up["thumb"]
    index = up["index"]
    middle = up["middle"]
    ring = up["ring"]
    pinky = up["pinky"]
    count = sum((thumb, index, middle, ring, pinky))

    if count == 0:
        return "fist"
    if thumb and not any((index, middle, ring, pinky)):
        return "thumbs_up"
    if index and not any((thumb, middle, ring, pinky)):
        return "pointing"
    if index and middle and not any((ring, pinky)):
        return "victory"
    if count == 5:
        return "open_palm"
    return "unknown"


def index_tip_position(landmarks: np.ndarray) -> tuple[float, float]:
    """Normalized (x, y) of the index fingertip, used for cursor control."""
    return float(landmarks[INDEX_TIP, 0]), float(landmarks[INDEX_TIP, 1])

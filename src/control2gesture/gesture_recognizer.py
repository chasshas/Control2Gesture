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


def is_pinch(landmarks: np.ndarray, pinch_threshold: float = 0.06) -> bool:
    """True when the thumb and index tips are close enough to count as a pinch."""
    return _pinch_distance(landmarks) < pinch_threshold


def classify(
    landmarks: np.ndarray,
    handedness: str = "Right",
    pinch_threshold: float = 0.06,
) -> str:
    """Classify a hand pose into a named gesture.

    Returns one of: pinch, fist, open_palm, pointing, victory, three,
    thumbs_up, unknown.
    """
    # Pinch takes priority: thumb and index tips touching is unambiguous and
    # overlaps with several finger patterns.
    if is_pinch(landmarks, pinch_threshold):
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
    if index and middle and ring and not any((thumb, pinky)):
        return "three"
    if count == 5:
        return "open_palm"
    return "unknown"


def index_tip_position(landmarks: np.ndarray) -> tuple[float, float]:
    """Normalized (x, y) of the index fingertip, used for cursor control."""
    return float(landmarks[INDEX_TIP, 0]), float(landmarks[INDEX_TIP, 1])


def hands_distance(landmarks_a: np.ndarray, landmarks_b: np.ndarray) -> float:
    """Normalized distance between two hands, index tip to index tip.

    Used by two-hand gestures (e.g. zoom) to measure how far apart the hands
    are. Both inputs are (21, 3) landmark arrays; only x/y are used.
    """
    a = landmarks_a[INDEX_TIP, :2]
    b = landmarks_b[INDEX_TIP, :2]
    return float(np.linalg.norm(a - b))


# Two-hand gestures, keyed by (left_hand_pose, right_hand_pose) using
# MediaPipe's on-screen handedness labels. Symmetric entries (same pose on both
# hands) also match when handedness is ambiguous; asymmetric entries — a
# different pose per hand — need a clear Left/Right split to orient them.
_TWO_HAND_COMBOS = {
    # Symmetric: both hands show the same pose.
    ("pinch", "pinch"): "two_hand_pinch",
    ("open_palm", "open_palm"): "two_hand_open",
    ("fist", "fist"): "two_hand_fist",
    ("victory", "victory"): "two_hand_victory",
    # Asymmetric: a different pose per hand.
    ("fist", "pointing"): "left_fist_right_point",
    ("pointing", "fist"): "left_point_right_fist",
    ("open_palm", "fist"): "left_open_right_fist",
}


def classify_hands(
    hands: list[tuple[np.ndarray, str]],
    pinch_threshold: float = 0.06,
) -> list[str | None]:
    """Classify each detected hand and return ``[left_gesture, right_gesture]``.

    ``hands`` is a list of ``(landmarks, handedness)`` pairs (0, 1, or 2 of
    them), as produced from :class:`HandTracker` results. The output is always a
    two-slot list ordered by side using MediaPipe's handedness labels, e.g.
    ``["fist", "pointing"]``; a side with no hand present is ``None`` (e.g.
    ``["fist", None]`` when only the left hand is up).
    """
    sides: dict[str, str | None] = {"Left": None, "Right": None}
    for landmarks, handedness in hands:
        if handedness in sides:
            sides[handedness] = classify(landmarks, handedness, pinch_threshold)
    return [sides["Left"], sides["Right"]]


def classify_two_hands(
    landmarks_a: np.ndarray,
    handedness_a: str,
    landmarks_b: np.ndarray,
    handedness_b: str,
    pinch_threshold: float = 0.06,
) -> str:
    """Classify a pose that uses both hands together.

    Each hand is classified independently (see :func:`classify_hands`), then the
    ``[left, right]`` pair is looked up in :data:`_TWO_HAND_COMBOS`. Asymmetric
    combos are oriented by MediaPipe handedness, so hand order in the arguments
    does not matter.

    Returns one of: two_hand_pinch, two_hand_open, two_hand_fist,
    two_hand_victory, left_fist_right_point, left_point_right_fist,
    left_open_right_fist, unknown.

    The distance-driven ones (``two_hand_pinch`` -> zoom, ``two_hand_open`` ->
    volume) use :func:`hands_distance` in the :class:`ActionMapper`: moving the
    hands apart or together drives the action up/down.
    """
    left, right = classify_hands(
        [(landmarks_a, handedness_a), (landmarks_b, handedness_b)],
        pinch_threshold,
    )
    if left is not None and right is not None:
        return _TWO_HAND_COMBOS.get((left, right), "unknown")

    # Handedness ambiguous (same label or "Unknown"): without a reliable
    # Left/Right split we can only trust symmetric (same-pose) combos.
    gesture_a = classify(landmarks_a, handedness_a, pinch_threshold)
    gesture_b = classify(landmarks_b, handedness_b, pinch_threshold)
    if gesture_a == gesture_b:
        return _TWO_HAND_COMBOS.get((gesture_a, gesture_b), "unknown")
    return "unknown"

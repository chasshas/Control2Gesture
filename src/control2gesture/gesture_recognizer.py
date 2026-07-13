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
MIDDLE_MCP = 9
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
INDEX_TIP = 8


def _hand_scale(landmarks: np.ndarray) -> float:
    """A stable measure of hand size (wrist to middle-knuckle distance).

    Knuckle (MCP) positions barely move as fingers curl, so this stays steady
    across every pose while scaling with how large the hand appears in
    frame -- i.e. how close it is to the camera. Distance-based thresholds
    (like thumb clearance) are expressed as a multiple of this instead of a
    fixed image-space distance, so one setting works at any distance instead
    of needing to be re-tuned every time the hand moves closer or farther.
    """
    return float(np.linalg.norm(landmarks[WRIST, :2] - landmarks[MIDDLE_MCP, :2]))


def _fingers_up(landmarks: np.ndarray, handedness: str) -> dict[str, bool]:
    """Return which fingers are extended.

    For index..pinky a finger is "up" when its tip is above (smaller y than)
    its PIP joint. For the thumb we compare x against the IP joint, accounting
    for left/right hand orientation. This is the literal, unpadded test used
    by every pose *other* than the closed-hand check; the fist gate has its
    own independently-tunable margin, see ``_all_fingers_folded``.
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


def _all_fingers_folded(landmarks: np.ndarray, fold_margin: float = 0.0) -> bool:
    """True when index/middle/ring/pinky are all folded (the fist/thumbs_up gate).

    A finger only counts as folded once its tip is clearly below (larger y
    than) its PIP joint by more than ``fold_margin``, so a finger merely
    curling toward the thumb during a pinch isn't misread as a fist. This
    margin is independent of the plain extension test ``_fingers_up`` uses for
    every other pose, so tuning fist sensitivity no longer affects how easily
    pointing/victory/three/four/open_palm trigger.
    """
    for name in ("index", "middle", "ring", "pinky"):
        tip_y = landmarks[FINGER_TIPS[name], 1]
        pip_y = landmarks[FINGER_PIPS[name], 1]
        if tip_y < pip_y + fold_margin:
            return False
    return True


def _pinch_distance(landmarks: np.ndarray) -> float:
    """Normalized distance between thumb tip and index tip."""
    return float(np.linalg.norm(landmarks[THUMB_TIP, :2] - landmarks[INDEX_TIP, :2]))


def _thumb_points_out(landmarks: np.ndarray, handedness: str) -> bool:
    """True when the thumb is extended away from the palm (out to the side or up).

    Combines the horizontal test used elsewhere with a vertical one so a thumb
    raised straight up (a thumbs-up) still counts even when its tip is roughly
    level with the IP joint horizontally.
    """
    tip = landmarks[THUMB_TIP]
    ip = landmarks[THUMB_IP]
    if handedness == "Right":
        horizontal = tip[0] < ip[0]
    else:
        horizontal = tip[0] > ip[0]
    vertical = tip[1] < ip[1]  # tip above the IP joint (smaller y)
    return bool(horizontal or vertical)


def _thumb_clear_of_folded_fingers(
    landmarks: np.ndarray, thumb_clear_margin: float = 0.5
) -> bool:
    """True when the thumb tip is held well clear of the folded fingers.

    This is what separates a thumbs-up from a fist: in a fist the thumb wraps
    across the folded fingers, so its tip lands close to the index/middle
    fingertips (which is also why a fist can trip the pinch check). A thumbs-up
    holds the thumb out, away from them. ``thumb_clear_margin`` is a multiple
    of :func:`_hand_scale`, not a fixed distance, so the same threshold holds
    up whether the hand fills the frame or is held farther from the camera.
    """
    tip = landmarks[THUMB_TIP, :2]
    index_tip = landmarks[FINGER_TIPS["index"], :2]
    middle_tip = landmarks[FINGER_TIPS["middle"], :2]
    nearest = min(
        float(np.linalg.norm(tip - index_tip)),
        float(np.linalg.norm(tip - middle_tip)),
    )
    return nearest > thumb_clear_margin * _hand_scale(landmarks)


def is_pinch(landmarks: np.ndarray, pinch_threshold: float = 0.06) -> bool:
    """True when the thumb and index tips are close enough to count as a pinch."""
    return _pinch_distance(landmarks) < pinch_threshold


def classify(
    landmarks: np.ndarray,
    handedness: str = "Right",
    pinch_threshold: float = 0.06,
    fist_fold_margin: float = 0.03,
    thumb_clear_margin: float = 0.5,
) -> str:
    """Classify a hand pose into a named gesture.

    Returns one of: pinch, fist, open_palm, pointing, victory, three, four,
    thumbs_up, unknown.

    Each closed-hand disambiguation has its own threshold, tunable
    independently: ``fist_fold_margin`` decides how clearly folded a hand must
    be to count as closed at all, ``thumb_clear_margin`` (a multiple of hand
    size, see :func:`_hand_scale`) then decides whether the thumb held out from
    that closed hand reads as a thumbs-up, and ``pinch_threshold`` is a
    separate, unrelated check for an open hand with the thumb and index
    touching.
    """
    up = _fingers_up(landmarks, handedness)
    thumb = up["thumb"]
    index = up["index"]
    middle = up["middle"]
    ring = up["ring"]
    pinky = up["pinky"]
    fingers_folded = _all_fingers_folded(landmarks, fist_fold_margin)

    # Closed hand (all four fingers folded). A fist is easily confused with a
    # thumbs-up (the wrapped thumb reads as "extended") and with a pinch (that
    # same thumb sits right next to the index tip). Resolve the closed hand
    # here, before the pinch check, so a fist can no longer trip either: it is
    # a thumbs-up only when the thumb is both pointing out and held clear of the
    # folded fingers; otherwise it is a fist.
    if fingers_folded:
        if _thumb_points_out(landmarks, handedness) and _thumb_clear_of_folded_fingers(
            landmarks, thumb_clear_margin
        ):
            return "thumbs_up"
        return "fist"

    # Pinch: thumb and index tips touching while the hand is otherwise open.
    # Only reachable when not a closed fist, so a wrapped thumb no longer counts.
    if is_pinch(landmarks, pinch_threshold):
        return "pinch"

    count = sum((thumb, index, middle, ring, pinky))
    if index and not any((thumb, middle, ring, pinky)):
        return "pointing"
    if index and middle and not any((ring, pinky)):
        return "victory"
    if index and middle and ring and not any((thumb, pinky)):
        return "three"
    if index and middle and ring and pinky and not thumb:
        return "four"
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


def _resolve_threshold(value: float | dict[str, float], handedness: str) -> float:
    """Pick the per-hand value out of a ``{"Left": ..., "Right": ...}`` override,
    or pass a plain float through unchanged for the common shared-value case."""
    return value[handedness] if isinstance(value, dict) else value


def classify_hands(
    hands: list[tuple[np.ndarray, str]],
    pinch_threshold: float | dict[str, float] = 0.06,
    fist_fold_margin: float | dict[str, float] = 0.03,
    thumb_clear_margin: float | dict[str, float] = 0.5,
) -> list[str | None]:
    """Classify each detected hand and return ``[left_gesture, right_gesture]``.

    This is the single recognition entry point: the whole system speaks in this
    ``[left, right]`` pair, and :meth:`Config.action_for` looks the pair up
    directly to decide what to do.

    ``hands`` is a list of ``(landmarks, handedness)`` pairs (0, 1, or 2 of
    them), as produced from :class:`HandTracker` results. The output is always a
    two-slot list ordered by side using MediaPipe's handedness labels, e.g.
    ``["fist", "pointing"]``; a side with no hand present is ``None`` (e.g.
    ``[None, "pointing"]`` when only the right hand is up, ``[None, None]`` when
    no hand is up).

    Each threshold accepts either a single float shared by both hands, or a
    ``{"Left": ..., "Right": ...}`` dict to tune a side independently (see
    :meth:`Settings.gesture_thresholds`).
    """
    sides: dict[str, str | None] = {"Left": None, "Right": None}
    for landmarks, handedness in hands:
        if handedness in sides:
            sides[handedness] = classify(
                landmarks,
                handedness,
                _resolve_threshold(pinch_threshold, handedness),
                _resolve_threshold(fist_fold_margin, handedness),
                _resolve_threshold(thumb_clear_margin, handedness),
            )
    return [sides["Left"], sides["Right"]]


"""Tests for the rule-based gesture recognizer.

Landmarks are synthetic (21, 3) arrays. Image y grows downward, so a finger is
"up" when its tip y is smaller than its PIP y.
"""

import numpy as np
import pytest

from control2gesture import gesture_recognizer as gr


def base_pose() -> np.ndarray:
    """A right hand with all fingers folded and thumb tucked (a fist)."""
    lm = np.zeros((21, 3), dtype=np.float32)

    # Wrist and a middle-knuckle reference, giving a hand_scale of 0.15 --
    # used to make the thumb-clearance check scale with hand size.
    lm[gr.WRIST] = [0.5, 0.70, 0.0]
    lm[gr.MIDDLE_MCP] = [0.5, 0.55, 0.0]

    # Four fingers folded: tip below (larger y than) pip.
    for tip, pip in zip(
        gr.FINGER_TIPS.values(), gr.FINGER_PIPS.values()
    ):
        lm[pip] = [0.5, 0.50, 0.0]
        lm[tip] = [0.5, 0.60, 0.0]

    # Thumb tucked: for a "Right" hand, tucked means tip_x >= ip_x.
    lm[gr.THUMB_IP] = [0.50, 0.55, 0.0]
    lm[gr.THUMB_TIP] = [0.56, 0.55, 0.0]

    # Keep thumb tip and index tip far apart so it isn't read as a pinch.
    lm[gr.INDEX_TIP] = [0.30, 0.60, 0.0]
    return lm


def raise_finger(lm: np.ndarray, name: str) -> None:
    tip = gr.FINGER_TIPS[name]
    lm[tip, 1] = 0.30  # move tip well above its pip


def raise_thumb(lm: np.ndarray) -> None:
    lm[gr.THUMB_TIP, 0] = 0.40  # tip_x < ip_x => extended (Right hand)


def test_fist():
    assert gr.classify(base_pose(), "Right") == "fist"


def test_pointing():
    lm = base_pose()
    raise_finger(lm, "index")
    assert gr.classify(lm, "Right") == "pointing"


def test_victory():
    lm = base_pose()
    raise_finger(lm, "index")
    raise_finger(lm, "middle")
    assert gr.classify(lm, "Right") == "victory"


def test_three():
    lm = base_pose()
    raise_finger(lm, "index")
    raise_finger(lm, "middle")
    raise_finger(lm, "ring")
    assert gr.classify(lm, "Right") == "three"


def test_four():
    lm = base_pose()
    for name in ("index", "middle", "ring", "pinky"):
        raise_finger(lm, name)
    assert gr.classify(lm, "Right") == "four"


def test_thumbs_up():
    lm = base_pose()
    raise_thumb(lm)
    assert gr.classify(lm, "Right") == "thumbs_up"


def test_open_palm():
    lm = base_pose()
    for name in ("index", "middle", "ring", "pinky"):
        raise_finger(lm, name)
    raise_thumb(lm)
    assert gr.classify(lm, "Right") == "open_palm"


def test_pinch():
    assert gr.classify(pinch_pose(), "Right", pinch_threshold=0.06) == "pinch"


def test_fist_with_wrapped_thumb_is_not_thumbs_up_or_pinch():
    """Regression: a real fist wraps the thumb across the folded fingers, so its
    tip reads as "extended" and lands next to the index tip. It must still be a
    fist, not thumbs_up or pinch."""
    lm = base_pose()
    # Thumb wrapped inward (reads as extended) and close to the index tip.
    lm[gr.THUMB_TIP, :2] = [0.34, 0.58]  # index tip is [0.30, 0.60]
    assert gr.is_pinch(lm, 0.06)  # tips are close enough to look like a pinch...
    assert gr.classify(lm, "Right") == "fist"  # ...but it's still a fist


def test_thumbs_up_becomes_fist_with_a_stricter_thumb_clear_margin():
    """A thumb held just barely clear of the folded fingers reads as thumbs_up
    with a lenient margin, but requiring more clearance (relative to hand
    size) should read it as fist instead -- independent of fist_fold_margin,
    which only gates the closed-hand check."""
    lm = base_pose()
    raise_thumb(lm)
    lm[gr.THUMB_TIP, :2] = [0.36, 0.58]  # index tip is [0.30, 0.60]: ~0.42x hand_scale away
    assert gr.classify(lm, "Right", thumb_clear_margin=0.3) == "thumbs_up"
    assert gr.classify(lm, "Right", thumb_clear_margin=0.6) == "fist"


def test_fist_fold_margin_does_not_affect_other_poses():
    """fist_fold_margin only gates the closed-hand (fist/thumbs_up) check; it
    must not change whether victory is detected."""
    lm = base_pose()
    raise_finger(lm, "index")
    raise_finger(lm, "middle")
    assert gr.classify(lm, "Right", fist_fold_margin=0.03) == "victory"
    assert gr.classify(lm, "Right", fist_fold_margin=0.20) == "victory"


def test_index_tip_position():
    lm = base_pose()
    x, y = gr.index_tip_position(lm)
    assert x == pytest.approx(0.30)
    assert y == pytest.approx(0.60)


def pinch_pose() -> np.ndarray:
    """A hand pinching: thumb tip resting on the index tip while the other
    fingers stay extended (an open hand, not a closed fist)."""
    lm = base_pose()
    for name in ("middle", "ring", "pinky"):
        raise_finger(lm, name)
    lm[gr.THUMB_TIP, :2] = lm[gr.INDEX_TIP, :2]
    return lm


def test_pinch_with_mildly_curled_fingers_is_not_fist():
    """Regression: pinching naturally curls the other fingers a little, putting
    their tips just barely past the pip line rather than fully folded. That
    shouldn't be read as a closed fist."""
    lm = pinch_pose()
    for name in ("middle", "ring", "pinky"):
        pip_y = lm[gr.FINGER_PIPS[name], 1]
        lm[gr.FINGER_TIPS[name], 1] = pip_y + 0.01  # just past the pip, barely folded
    assert gr.classify(lm, "Right", fist_fold_margin=0.03) == "pinch"


def test_is_pinch():
    assert gr.is_pinch(pinch_pose(), pinch_threshold=0.06)
    assert not gr.is_pinch(base_pose(), pinch_threshold=0.06)


def test_hands_distance():
    a = base_pose()
    b = base_pose()
    a[gr.INDEX_TIP, :2] = [0.20, 0.50]
    b[gr.INDEX_TIP, :2] = [0.50, 0.50]
    assert gr.hands_distance(a, b) == pytest.approx(0.30)


# --- [left, right] per-hand form --------------------------------------------
# base_pose()'s thumb is Right-hand specific, so build poses whose thumb matches
# the hand we label them with.


def handed_fist(handedness: str) -> np.ndarray:
    """A fist whose tucked thumb reads correctly for the given hand."""
    lm = base_pose()
    ip_x = lm[gr.THUMB_IP, 0]
    lm[gr.THUMB_TIP, 0] = ip_x + 0.06 if handedness == "Right" else ip_x - 0.06
    return lm


def handed_pointing(handedness: str) -> np.ndarray:
    lm = handed_fist(handedness)
    raise_finger(lm, "index")
    return lm


def test_classify_hands_pair():
    left, right = handed_fist("Left"), handed_pointing("Right")
    assert gr.classify_hands(
        [(left, "Left"), (right, "Right")]
    ) == ["fist", "pointing"]


def test_classify_hands_order_independent():
    # Result is oriented by handedness, not argument order.
    left, right = handed_fist("Left"), handed_pointing("Right")
    assert gr.classify_hands(
        [(right, "Right"), (left, "Left")]
    ) == ["fist", "pointing"]


def test_classify_hands_single_hand_leaves_other_none():
    assert gr.classify_hands([(handed_pointing("Right"), "Right")]) == [
        None,
        "pointing",
    ]


def test_classify_hands_empty():
    assert gr.classify_hands([]) == [None, None]


def test_classify_hands_per_hand_thresholds():
    """A dict threshold lets one hand use a different value than the other."""
    left, right = handed_fist("Left"), handed_fist("Right")
    # Widen the thumb-clearance requirement only for the right hand so its
    # (identically wrapped) thumb now reads as fist while the left is untouched
    # by the default margin.
    left[gr.THUMB_TIP, :2] = [left[gr.INDEX_TIP, 0] + 0.20, left[gr.INDEX_TIP, 1]]
    right[gr.THUMB_TIP, :2] = [right[gr.INDEX_TIP, 0] + 0.20, right[gr.INDEX_TIP, 1]]
    for lm, handedness in ((left, "Left"), (right, "Right")):
        lm[gr.THUMB_TIP, 1] = lm[gr.THUMB_IP, 1] - 0.02  # thumb points out
    result = gr.classify_hands(
        [(left, "Left"), (right, "Right")],
        thumb_clear_margin={"Left": 0.2, "Right": 0.6},
    )
    assert result == ["thumbs_up", "fist"]

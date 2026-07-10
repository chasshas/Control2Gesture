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


def test_pinch_takes_priority():
    lm = base_pose()
    # Bring thumb tip onto the index tip.
    lm[gr.THUMB_TIP, :2] = lm[gr.INDEX_TIP, :2]
    assert gr.classify(lm, "Right", pinch_threshold=0.06) == "pinch"


def test_index_tip_position():
    lm = base_pose()
    x, y = gr.index_tip_position(lm)
    assert x == pytest.approx(0.30)
    assert y == pytest.approx(0.60)

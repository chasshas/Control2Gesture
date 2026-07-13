"""Tests for loading the [left, right] pair-keyed gesture config."""

import textwrap

import pytest

from control2gesture.config import Settings, load_config


def write_config(tmp_path, body: str):
    path = tmp_path / "gestures.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_action_for_pair(tmp_path):
    cfg = load_config(
        write_config(
            tmp_path,
            """
            gestures:
              - gesture: [null, pinch]
                action: left_click
              - gesture: [pinch, pinch]
                action: zoom
            """,
        )
    )
    # Single-hand pair: right hand pinch, left side empty.
    assert cfg.action_for([None, "pinch"]) == {"action": "left_click"}
    # Two-hand pair.
    assert cfg.action_for(["pinch", "pinch"]) == {"action": "zoom"}


def test_action_for_unmapped_pair_is_none(tmp_path):
    cfg = load_config(write_config(tmp_path, "gestures: []\n"))
    assert cfg.action_for([None, "fist"]) == {"action": "none"}


def test_extra_action_fields_are_kept(tmp_path):
    cfg = load_config(
        write_config(
            tmp_path,
            """
            gestures:
              - gesture: [null, thumbs_up]
                action: scroll_up
                amount: 3
              - gesture: [fist, fist]
                action: hotkey
                keys: ["ctrl", "up"]
            """,
        )
    )
    assert cfg.action_for([None, "thumbs_up"]) == {"action": "scroll_up", "amount": 3}
    assert cfg.action_for(["fist", "fist"]) == {
        "action": "hotkey",
        "keys": ["ctrl", "up"],
    }


def test_missing_gesture_key_raises(tmp_path):
    with pytest.raises(ValueError, match="gesture"):
        load_config(
            write_config(
                tmp_path,
                """
                gestures:
                  - action: left_click
                """,
            )
        )


def test_gesture_must_be_two_element_list(tmp_path):
    with pytest.raises(ValueError, match="two-element"):
        load_config(
            write_config(
                tmp_path,
                """
                gestures:
                  - gesture: [pinch]
                    action: left_click
                """,
            )
        )


def test_gesture_thresholds_falls_back_to_shared_value_by_default():
    s = Settings(pinch_threshold=0.08, fist_fold_margin=0.04, thumb_clear_margin=0.12)
    pinch, fist, thumb = s.gesture_thresholds()
    assert pinch == {"Left": 0.08, "Right": 0.08}
    assert fist == {"Left": 0.04, "Right": 0.04}
    assert thumb == {"Left": 0.12, "Right": 0.12}


def test_gesture_thresholds_per_hand_override_wins():
    s = Settings(
        pinch_threshold=0.06,
        pinch_threshold_left=0.09,
        fist_fold_margin=0.03,
        fist_fold_margin_right=0.05,
    )
    pinch, fist, _ = s.gesture_thresholds()
    assert pinch == {"Left": 0.09, "Right": 0.06}
    assert fist == {"Left": 0.03, "Right": 0.05}


def test_duplicate_pair_raises(tmp_path):
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_config(
            write_config(
                tmp_path,
                """
                gestures:
                  - gesture: [null, pinch]
                    action: left_click
                  - gesture: [null, pinch]
                    action: right_click
                """,
            )
        )

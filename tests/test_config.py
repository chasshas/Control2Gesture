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
    s = Settings(pinch_threshold=0.08, fist_fold_margin=0.04, thumb_straight_threshold=0.6)
    pinch, fist, thumb = s.gesture_thresholds()
    assert pinch == {"Left": 0.08, "Right": 0.08}
    assert fist == {"Left": 0.04, "Right": 0.04}
    assert thumb == {"Left": 0.6, "Right": 0.6}


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


def test_any_wildcard_matches_regardless_of_other_side(tmp_path):
    cfg = load_config(
        write_config(
            tmp_path,
            """
            gestures:
              - gesture: [any, pinch]
                action: left_click
            """,
        )
    )
    assert cfg.action_for([None, "pinch"]) == {"action": "left_click"}
    assert cfg.action_for(["fist", "pinch"]) == {"action": "left_click"}
    assert cfg.action_for(["pinch", "pinch"]) == {"action": "left_click"}
    # Right side doesn't match -> still unmapped.
    assert cfg.action_for(["fist", "fist"]) == {"action": "none"}


def test_exact_pair_wins_over_any_wildcard(tmp_path):
    cfg = load_config(
        write_config(
            tmp_path,
            """
            gestures:
              - gesture: [any, pinch]
                action: left_click
              - gesture: [fist, pinch]
                action: right_click
            """,
        )
    )
    assert cfg.action_for(["fist", "pinch"]) == {"action": "right_click"}
    assert cfg.action_for(["open_palm", "pinch"]) == {"action": "left_click"}


def test_more_specific_wildcard_wins_over_double_any(tmp_path):
    cfg = load_config(
        write_config(
            tmp_path,
            """
            gestures:
              - gesture: [any, any]
                action: left_click
              - gesture: [any, pinch]
                action: right_click
            """,
        )
    )
    assert cfg.action_for(["fist", "pinch"]) == {"action": "right_click"}
    assert cfg.action_for(["fist", "open_palm"]) == {"action": "left_click"}


def test_orthogonal_wildcards_both_match(tmp_path):
    """[any, pinch] and [pinch, any] constrain different sides, so neither
    dominates the other -- an actual [pinch, pinch] should fire both."""
    cfg = load_config(
        write_config(
            tmp_path,
            """
            gestures:
              - gesture: [any, pinch]
                action: left_click
              - gesture: [pinch, any]
                action: right_click
            """,
        )
    )
    specs = cfg.actions_for(["pinch", "pinch"])
    assert {s["action"] for s in specs} == {"left_click", "right_click"}

    # Off the overlap, only the relevant one matches.
    assert cfg.actions_for(["fist", "pinch"]) == [{"action": "left_click"}]
    assert cfg.actions_for(["pinch", "fist"]) == [{"action": "right_click"}]


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

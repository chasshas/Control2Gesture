"""Tests for ActionMapper dispatch, debounce, and the enable/disable toggle."""

import textwrap

from control2gesture.action_mapper import ActionMapper
from control2gesture.config import load_config


def write_config(tmp_path, body: str):
    path = tmp_path / "gestures.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_config(path)


class FakeController:
    """Records calls instead of touching the real mouse/keyboard."""

    def __init__(self):
        self.calls: list[tuple] = []

    def reset_cursor_origin(self):
        self.calls.append(("reset_cursor_origin",))

    def left_click(self):
        self.calls.append(("left_click",))

    def right_click(self):
        self.calls.append(("right_click",))

    def double_click(self):
        self.calls.append(("double_click",))

    def move_cursor(self, nx, ny):
        self.calls.append(("move_cursor", nx, ny))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))

    def zoom(self, steps):
        self.calls.append(("zoom", steps))

    def change_volume(self, steps):
        self.calls.append(("change_volume", steps))

    def press_keys(self, keys):
        self.calls.append(("press_keys", keys))

    def hotkey(self, keys):
        self.calls.append(("hotkey", keys))


def build_mapper(tmp_path):
    config = write_config(
        tmp_path,
        """
        settings:
          stable_frames: 1
        gestures:
          - gesture: [four, four]
            action: toggle_control
          - gesture: [null, pinch]
            action: left_click
        """,
    )
    controller = FakeController()
    return ActionMapper(config, controller), controller


def clicks(controller):
    return [c for c in controller.calls if c[0] == "left_click"]


def test_enabled_by_default(tmp_path):
    mapper, _ = build_mapper(tmp_path)
    assert mapper.enabled is True


def test_toggle_gesture_disables_and_reenables(tmp_path):
    mapper, controller = build_mapper(tmp_path)

    mapper.handle([None, "pinch"])
    assert len(clicks(controller)) == 1

    mapper.handle(["four", "four"])
    assert mapper.enabled is False

    # A gesture that would normally click is suppressed while disabled.
    mapper.reset()
    mapper.handle([None, "pinch"])
    assert len(clicks(controller)) == 1

    mapper.reset()
    mapper.handle(["four", "four"])
    assert mapper.enabled is True

    mapper.reset()
    mapper.handle([None, "pinch"])
    assert len(clicks(controller)) == 2


def test_toggle_enabled_matches_keyboard_shortcut_path(tmp_path):
    mapper, controller = build_mapper(tmp_path)

    assert mapper.toggle_enabled() is False
    assert mapper.enabled is False

    mapper.handle([None, "pinch"])
    assert clicks(controller) == []

    assert mapper.toggle_enabled() is True
    mapper.handle([None, "pinch"])
    assert len(clicks(controller)) == 1


def test_toggle_gesture_is_one_shot(tmp_path):
    mapper, _ = build_mapper(tmp_path)

    mapper.handle(["four", "four"])
    assert mapper.enabled is False
    # Holding the same pair across more frames must not flip it back.
    mapper.handle(["four", "four"])
    mapper.handle(["four", "four"])
    assert mapper.enabled is False


def test_orthogonal_any_wildcards_both_fire_on_overlap(tmp_path):
    """[any, pinch] and [pinch, any] each govern one hand; on an actual
    [pinch, pinch] neither dominates the other, so both actions should fire,
    each with its own one-shot latch."""
    config = write_config(
        tmp_path,
        """
        settings:
          stable_frames: 1
        gestures:
          - gesture: [any, pinch]
            action: left_click
          - gesture: [pinch, any]
            action: right_click
        """,
    )
    controller = FakeController()
    mapper = ActionMapper(config, controller)

    mapper.handle(["pinch", "pinch"])
    assert len(clicks(controller)) == 1
    assert len([c for c in controller.calls if c[0] == "right_click"]) == 1

    # Holding the same pair must not re-fire either one-shot.
    mapper.handle(["pinch", "pinch"])
    assert len(clicks(controller)) == 1
    assert len([c for c in controller.calls if c[0] == "right_click"]) == 1

    # Dropping to only the right-hand pinch keeps that one's action live and
    # lets the left-hand one re-arm for next time.
    mapper.reset()
    mapper.handle([None, "pinch"])
    assert len(clicks(controller)) == 2
    assert len([c for c in controller.calls if c[0] == "right_click"]) == 1


def test_per_gesture_stable_frames_overrides_the_global_default(tmp_path):
    """A gesture's own `stable_frames` wins over settings.stable_frames, so a
    pinch-click can fire on the very first frame even with a slower global
    default for everything else."""
    config = write_config(
        tmp_path,
        """
        settings:
          stable_frames: 5
        gestures:
          - gesture: [null, pinch]
            action: left_click
            stable_frames: 1
          - gesture: [null, victory]
            action: right_click
        """,
    )
    controller = FakeController()
    mapper = ActionMapper(config, controller)

    mapper.handle([None, "pinch"])
    assert clicks(controller) == [("left_click",)]

    mapper.reset()
    mapper.handle([None, "victory"])
    assert [c for c in controller.calls if c[0] == "right_click"] == []
    for _ in range(4):
        mapper.handle([None, "victory"])
    assert [c for c in controller.calls if c[0] == "right_click"] == [("right_click",)]

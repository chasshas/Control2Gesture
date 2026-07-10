"""Map recognized gestures to controller actions, with debouncing.

Actions come in two flavours:

* **Continuous** (``move_cursor``, ``scroll_up``, ``scroll_down``) run on every
  frame the gesture is held.
* **One-shot** (clicks, key presses) fire exactly once, when a gesture becomes
  stable, and will not fire again until a different gesture is seen in between.
"""

from __future__ import annotations

import logging

from .config import Config
from .controller import Controller

log = logging.getLogger(__name__)

CONTINUOUS_ACTIONS = {"move_cursor", "scroll_up", "scroll_down"}


class ActionMapper:
    def __init__(self, config: Config, controller: Controller) -> None:
        self.config = config
        self.controller = controller
        self.stable_frames = config.settings.stable_frames

        self._candidate: str | None = None   # gesture currently stabilizing
        self._stable_count = 0
        self._active: str | None = None       # gesture confirmed stable
        self._fired_oneshot = False           # one-shot already fired for _active

    def reset(self) -> None:
        """Call when no hand is present, so the next gesture fires cleanly."""
        self._candidate = None
        self._stable_count = 0
        self._active = None
        self._fired_oneshot = False

    def handle(self, gesture: str, cursor_xy: tuple[float, float] | None) -> None:
        """Process one frame's gesture and dispatch the mapped action."""
        # Track stability: a gesture must persist `stable_frames` frames.
        if gesture == self._candidate:
            self._stable_count += 1
        else:
            self._candidate = gesture
            self._stable_count = 1

        if self._stable_count < self.stable_frames:
            return

        # Gesture is stable. Detect transition to reset one-shot latch.
        if gesture != self._active:
            self._active = gesture
            self._fired_oneshot = False

        spec = self.config.action_for(gesture)
        action = spec.get("action", "none")
        if action == "none":
            return

        if action in CONTINUOUS_ACTIONS:
            self._run_continuous(action, spec, cursor_xy)
        elif not self._fired_oneshot:
            self._run_oneshot(action, spec)
            self._fired_oneshot = True

    def _run_continuous(self, action: str, spec: dict, cursor_xy) -> None:
        if action == "move_cursor":
            if cursor_xy is not None:
                self.controller.move_cursor(*cursor_xy)
        elif action == "scroll_up":
            self.controller.scroll(int(spec.get("amount", 3)))
        elif action == "scroll_down":
            self.controller.scroll(-int(spec.get("amount", 3)))

    def _run_oneshot(self, action: str, spec: dict) -> None:
        if action == "left_click":
            self.controller.left_click()
        elif action == "right_click":
            self.controller.right_click()
        elif action == "double_click":
            self.controller.double_click()
        elif action == "key":
            self.controller.press_keys(list(spec.get("keys", [])))
        elif action == "hotkey":
            self.controller.hotkey(list(spec.get("keys", [])))
        else:
            log.warning("Unknown action: %s", action)

"""Map recognized gestures to controller actions, with debouncing.

Actions come in two flavours:

* **Continuous** (``move_cursor``, ``scroll_up``, ``scroll_down``, ``zoom``) run
  on every frame the gesture is held.
* **One-shot** (clicks, key presses) fire exactly once, when a gesture becomes
  stable, and will not fire again until a different gesture is seen in between.

Single-hand gestures arrive via :meth:`ActionMapper.handle`; two-hand gestures
(e.g. ``two_hand_pinch`` -> ``zoom``) arrive via
:meth:`ActionMapper.handle_two_hands`. Both share one debounce state machine, so
switching between one- and two-hand gestures resets the one-shot latch cleanly.
"""

from __future__ import annotations

import logging

from .config import Config
from .controller import Controller

log = logging.getLogger(__name__)

CONTINUOUS_ACTIONS = {"move_cursor", "scroll_up", "scroll_down"}

# Two-hand actions driven by the change in inter-hand distance (apart/together).
TWO_HAND_DISTANCE_ACTIONS = {"zoom", "volume"}


class ActionMapper:
    def __init__(self, config: Config, controller: Controller) -> None:
        self.config = config
        self.controller = controller
        self.stable_frames = config.settings.stable_frames

        self._candidate: str | None = None   # gesture currently stabilizing
        self._stable_count = 0
        self._active: str | None = None       # gesture confirmed stable
        self._fired_oneshot = False           # one-shot already fired for _active
        self._prev_distance: float | None = None  # last inter-hand zoom distance

    def reset(self) -> None:
        """Call when no hand is present, so the next gesture fires cleanly."""
        self._candidate = None
        self._stable_count = 0
        self._active = None
        self._fired_oneshot = False
        self._prev_distance = None

    def _stabilize(self, gesture: str) -> bool:
        """Advance the debounce machine; return True once ``gesture`` is stable.

        A gesture must persist ``stable_frames`` consecutive frames. On the
        transition to a new stable gesture the one-shot latch and zoom-distance
        tracking are reset so the next gesture starts clean.
        """
        if gesture == self._candidate:
            self._stable_count += 1
        else:
            self._candidate = gesture
            self._stable_count = 1

        if self._stable_count < self.stable_frames:
            return False

        if gesture != self._active:
            self._active = gesture
            self._fired_oneshot = False
            self._prev_distance = None
        return True

    def handle(self, gesture: str, cursor_xy: tuple[float, float] | None) -> None:
        """Process one frame's single-hand gesture and dispatch its action."""
        if not self._stabilize(gesture):
            return

        spec = self.config.action_for(gesture)
        action = spec.get("action", "none")
        if action == "none":
            return

        if action in CONTINUOUS_ACTIONS:
            self._run_continuous(action, spec, cursor_xy)
        elif not self._fired_oneshot:
            self._run_oneshot(action, spec)
            self._fired_oneshot = True

    def handle_two_hands(self, gesture: str, distance: float) -> None:
        """Process one frame's two-hand gesture.

        Distance-driven actions (``zoom``, ``volume``) translate the change in
        ``distance`` into signed steps (hands apart -> up, together -> down).
        Any other mapped action (e.g. a ``hotkey``) fires once as a one-shot,
        the same as single-hand gestures.
        """
        if not self._stabilize(gesture):
            return

        spec = self.config.action_for(gesture)
        action = spec.get("action", "none")
        if action == "none":
            return

        if action in TWO_HAND_DISTANCE_ACTIONS:
            self._run_distance(action, distance)
        elif not self._fired_oneshot:
            self._run_oneshot(action, spec)
            self._fired_oneshot = True

    def _run_distance(self, action: str, distance: float) -> None:
        """Emit signed steps from the change in inter-hand distance."""
        if self._prev_distance is None:
            self._prev_distance = distance
            return

        deadzone = self.config.settings.two_hand_deadzone
        delta = distance - self._prev_distance
        if abs(delta) < deadzone:
            return

        # Ratchet: only advance the reference once we've moved a full deadzone,
        # so slow drift doesn't accumulate but real motion tracks smoothly.
        steps = int(delta / deadzone) * self.config.settings.two_hand_step
        if action == "zoom":
            self.controller.zoom(steps)
        elif action == "volume":
            self.controller.change_volume(steps)
        self._prev_distance = distance

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

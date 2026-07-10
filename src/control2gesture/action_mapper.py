"""Map recognized gestures to controller actions, with debouncing.

Actions come in two flavours:

* **Continuous** (``move_cursor``, ``scroll_up``, ``scroll_down``, ``zoom``) run
  on every frame the gesture is held.
* **One-shot** (clicks, key presses) fire exactly once, when a gesture becomes
  stable, and will not fire again until a different gesture is seen in between.

Every frame the recognizer produces one ``[left, right]`` gesture pair, and
:meth:`ActionMapper.handle` maps that pair to an action. Whether an action is
continuous, distance-driven, or one-shot is decided by the action name, not by
how many hands are up, so single- and two-hand gestures share one debounce
state machine and switching between them resets the one-shot latch cleanly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .config import Config, GesturePair
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

        self._candidate: GesturePair | None = None  # pair currently stabilizing
        self._stable_count = 0
        self._active: GesturePair | None = None      # pair confirmed stable
        self._fired_oneshot = False           # one-shot already fired for _active
        self._prev_distance: float | None = None  # last inter-hand zoom distance

    def reset(self) -> None:
        """Call when no hand is present, so the next gesture fires cleanly."""
        self._candidate = None
        self._stable_count = 0
        self._active = None
        self._fired_oneshot = False
        self._prev_distance = None

    def _stabilize(self, pair: GesturePair) -> bool:
        """Advance the debounce machine; return True once ``pair`` is stable.

        A gesture pair must persist ``stable_frames`` consecutive frames. On the
        transition to a new stable pair the one-shot latch and zoom-distance
        tracking are reset so the next gesture starts clean.
        """
        if pair == self._candidate:
            self._stable_count += 1
        else:
            self._candidate = pair
            self._stable_count = 1

        if self._stable_count < self.stable_frames:
            return False

        if pair != self._active:
            self._active = pair
            self._fired_oneshot = False
            self._prev_distance = None
        return True

    def handle(
        self,
        pair: Sequence[str | None],
        cursor_xy: tuple[float, float] | None = None,
        distance: float | None = None,
    ) -> None:
        """Process one frame's ``[left, right]`` gesture pair and dispatch it.

        The action mapped to ``pair`` decides how it runs:

        * **Continuous** (``move_cursor``, scroll) run every frame, using
          ``cursor_xy`` (the driving hand's index tip) where needed.
        * **Distance-driven** (``zoom``, ``volume``) translate the change in
          ``distance`` (inter-hand spread) into signed steps; they no-op until a
          second hand supplies a ``distance``.
        * **One-shot** (clicks, keys, hotkeys) fire exactly once per stable pair.
        """
        key: GesturePair = (pair[0], pair[1])
        if not self._stabilize(key):
            return

        spec = self.config.action_for(pair)
        action = spec.get("action", "none")
        if action == "none":
            return

        if action in CONTINUOUS_ACTIONS:
            self._run_continuous(action, spec, cursor_xy)
        elif action in TWO_HAND_DISTANCE_ACTIONS:
            if distance is not None:
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

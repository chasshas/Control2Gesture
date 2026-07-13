"""Map recognized gestures to controller actions, with debouncing.

Actions come in two flavours:

* **Continuous** (``move_cursor``, ``scroll_up``, ``scroll_down``, ``zoom``) run
  on every frame the gesture is held.
* **One-shot** (clicks, key presses) fire exactly once, when a gesture becomes
  stable, and will not fire again until a different gesture is seen in between.

`toggle_control` is a special one-shot: it flips :attr:`ActionMapper.enabled`
instead of calling the controller, and it is the one action that still fires
while control is disabled (so the same gesture/keyboard shortcut turns it back
on). Every other action is suppressed while disabled.

Every frame the recognizer produces one ``[left, right]`` gesture pair, and
:meth:`ActionMapper.handle` looks up every config *pattern* that pair matches
(see :meth:`Config.matching_patterns`) and dispatches each one. Usually
there's exactly one match, so this reads like one debounce state machine --
but ``any``-wildcard patterns constraining different sides (``[any, pinch]``
and ``[pinch, any]``) can both match at once (e.g. on an actual ``[pinch,
pinch]``), and then both run, each with its own independent debounce/one-shot
state so one doesn't reset the other's latch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .config import Config, GesturePair
from .controller import Controller

log = logging.getLogger(__name__)

CONTINUOUS_ACTIONS = {"move_cursor", "scroll_up", "scroll_down"}

# Two-hand actions driven by the change in inter-hand distance (apart/together).
TWO_HAND_DISTANCE_ACTIONS = {"zoom", "volume"}

# One-shot action that flips `ActionMapper.enabled` instead of touching the
# controller; it always dispatches, even while control is disabled, so it can
# turn itself back on.
TOGGLE_ACTION = "toggle_control"


class ActionMapper:
    def __init__(self, config: Config, controller: Controller) -> None:
        self.config = config
        self.controller = controller
        self.stable_frames = config.settings.stable_frames
        # While False, every action except `toggle_control` is suppressed;
        # gestures/keys are still recognized so control can be turned back on.
        self.enabled = True

        # One debounce/one-shot/distance state per matched config pattern,
        # keyed by that pattern (which may contain the "any" wildcard) --
        # not by the raw [left, right] pair, so two wildcard patterns that
        # both match the same pair track independently.
        self._pattern_state: dict[GesturePair, dict[str, Any]] = {}

    def reset(self) -> None:
        """Call when no hand is present, so the next gesture fires cleanly."""
        self._pattern_state.clear()
        self.controller.reset_cursor_origin()

    def toggle_enabled(self) -> bool:
        """Flip whether gesture-driven control is active; return the new state.

        Called both from the `toggle_control` gesture and the keyboard
        shortcut, so the two stay in sync on one flag.
        """
        self.enabled = not self.enabled
        log.info("Gesture control %s", "enabled" if self.enabled else "disabled")
        self.controller.reset_cursor_origin()
        return self.enabled

    def handle(
        self,
        pair: Sequence[str | None],
        cursor_xy: tuple[float, float] | None = None,
        distance: float | None = None,
    ) -> None:
        """Process one frame's ``[left, right]`` gesture pair and dispatch it.

        Every config pattern the pair matches (usually one) is stabilized and
        dispatched independently -- see the module docstring for how multiple
        wildcard patterns can match, and both fire, on the same pair.
        """
        matches = self.config.matching_patterns(pair)
        matched_patterns = {pattern for pattern, _ in matches}
        for pattern in list(self._pattern_state):
            if pattern not in matched_patterns:
                del self._pattern_state[pattern]

        for pattern, spec in matches:
            self._handle_pattern(pattern, spec, cursor_xy, distance)

    def _handle_pattern(
        self,
        pattern: GesturePair,
        spec: dict[str, Any],
        cursor_xy: tuple[float, float] | None,
        distance: float | None,
    ) -> None:
        """Advance one pattern's debounce state and dispatch its action.

        The action mapped to ``pattern`` decides how it runs:

        * **Continuous** (``move_cursor``, scroll) run every frame, using
          ``cursor_xy`` (the driving hand's index tip) where needed.
        * **Distance-driven** (``zoom``, ``volume``) translate the change in
          ``distance`` (inter-hand spread) into signed steps; they no-op until a
          second hand supplies a ``distance``.
        * **One-shot** (clicks, keys, hotkeys) fire exactly once per stable
          pattern.

        `toggle_control` always dispatches, even while control is disabled, so
        the same gesture can turn it back on; every other action is
        suppressed while :attr:`enabled` is False.

        The number of consecutive frames required before the pattern is
        considered stable is :attr:`stable_frames` by default, but a gesture
        entry can set its own ``stable_frames`` in ``gestures.yaml`` to react
        faster (e.g. ``1`` for an instant click) or slower than the rest.
        """
        required_frames = int(spec.get("stable_frames", self.stable_frames))
        state = self._pattern_state.setdefault(
            pattern, {"count": 0, "active": False, "fired": False, "prev_distance": None}
        )
        state["count"] += 1
        if state["count"] < required_frames:
            return

        if not state["active"]:
            state["active"] = True
            state["fired"] = False
            state["prev_distance"] = None
            # Newly-stable pattern: drop any move_cursor baseline so resuming
            # pointing elsewhere doesn't read as one big relative jump.
            self.controller.reset_cursor_origin()

        action = spec.get("action", "none")
        if action == "none":
            return

        if action == TOGGLE_ACTION:
            if not state["fired"]:
                self.toggle_enabled()
                state["fired"] = True
            return

        if not self.enabled:
            return

        if action in CONTINUOUS_ACTIONS:
            self._run_continuous(action, spec, cursor_xy)
        elif action in TWO_HAND_DISTANCE_ACTIONS:
            if distance is not None:
                self._run_distance(state, action, distance)
        elif not state["fired"]:
            self._run_oneshot(action, spec)
            state["fired"] = True

    def _run_distance(self, state: dict[str, Any], action: str, distance: float) -> None:
        """Emit signed steps from the change in inter-hand distance."""
        if state["prev_distance"] is None:
            state["prev_distance"] = distance
            return

        deadzone = self.config.settings.two_hand_deadzone
        delta = distance - state["prev_distance"]
        if abs(delta) < deadzone:
            return

        # Ratchet: only advance the reference once we've moved a full deadzone,
        # so slow drift doesn't accumulate but real motion tracks smoothly.
        steps = int(delta / deadzone) * self.config.settings.two_hand_step
        if action == "zoom":
            self.controller.zoom(steps)
        elif action == "volume":
            self.controller.change_volume(steps)
        state["prev_distance"] = distance

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

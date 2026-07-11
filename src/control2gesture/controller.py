"""Keyboard/mouse control via pyautogui, with cursor smoothing.

On macOS the app running this must be granted Accessibility permission
(System Settings -> Privacy & Security -> Accessibility) or the OS will
silently ignore synthetic mouse/keyboard events.
"""

from __future__ import annotations

import sys

import pyautogui

# App-level zoom is Cmd +/- on macOS, Ctrl +/- elsewhere.
_ZOOM_MODIFIER = "command" if sys.platform == "darwin" else "ctrl"
_MAX_STEPS_PER_FRAME = 5  # cap presses per frame so a big jump can't runaway


class Controller:
    def __init__(
        self,
        smoothing: float = 0.5,
        sensitivity: float = 3.0,
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        # Disable the fail-safe: moving into a screen corner should not raise.
        pyautogui.FAILSAFE = False
        # Zero delay between calls keeps cursor tracking responsive.
        pyautogui.PAUSE = 0

        self.screen_w, self.screen_h = screen_size or pyautogui.size()
        self.smoothing = max(0.0, min(0.95, smoothing))
        self.sensitivity = max(0.0, sensitivity)
        self._prev_x, self._prev_y = pyautogui.position()
        # Smoothed fingertip position (normalized) from the last frame; None
        # means "no baseline yet" so the next call seeds it instead of moving.
        self._smoothed_finger: tuple[float, float] | None = None

    def reset_cursor_origin(self) -> None:
        """Drop the fingertip baseline so the next move_cursor starts fresh.

        Without this, resuming the pointing gesture after doing something
        else (or after the hand briefly left frame) would read as one huge
        jump, since move_cursor only tracks frame-to-frame deltas.
        """
        self._smoothed_finger = None

    def move_cursor(self, nx: float, ny: float) -> None:
        """Move the cursor by the fingertip's frame-to-frame delta.

        ``nx``/``ny`` are normalized (0..1) fingertip coords. Movement is
        relative (joystick-style): the cursor moves by ``sensitivity`` times
        the change in (smoothed) fingertip position, rather than mapping the
        fingertip to an absolute screen position.
        """
        alpha = 1.0 - self.smoothing
        if self._smoothed_finger is None:
            self._smoothed_finger = (nx, ny)
            return

        prev_snx, prev_sny = self._smoothed_finger
        snx = prev_snx + (nx - prev_snx) * alpha
        sny = prev_sny + (ny - prev_sny) * alpha
        self._smoothed_finger = (snx, sny)

        dx = (snx - prev_snx) * self.sensitivity * self.screen_w
        dy = (sny - prev_sny) * self.sensitivity * self.screen_h

        x = min(max(self._prev_x + dx, 0), self.screen_w - 1)
        y = min(max(self._prev_y + dy, 0), self.screen_h - 1)

        pyautogui.moveTo(x, y)
        self._prev_x, self._prev_y = x, y

    def left_click(self) -> None:
        pyautogui.click(button="left")

    def right_click(self) -> None:
        pyautogui.click(button="right")

    def double_click(self) -> None:
        pyautogui.doubleClick()

    def scroll(self, amount: int) -> None:
        """Positive scrolls up, negative scrolls down."""
        pyautogui.scroll(amount)

    def zoom(self, steps: int) -> None:
        """App-level zoom via the platform modifier + '+'/'-'.

        Positive ``steps`` zooms in, negative zooms out. The count is capped so
        a large jump in one frame cannot fire an unbounded burst of presses.
        """
        if steps == 0:
            return
        key = "+" if steps > 0 else "-"
        for _ in range(min(abs(steps), _MAX_STEPS_PER_FRAME)):
            pyautogui.hotkey(_ZOOM_MODIFIER, key)

    def change_volume(self, steps: int) -> None:
        """Raise (positive) or lower (negative) the system volume.

        Uses the media volume keys. These are recognized on Windows/Linux; on
        macOS they may be ignored depending on the keyboard/driver, so treat
        this as best-effort. Count is capped per frame like :meth:`zoom`.
        """
        if steps == 0:
            return
        key = "volumeup" if steps > 0 else "volumedown"
        for _ in range(min(abs(steps), _MAX_STEPS_PER_FRAME)):
            pyautogui.press(key)

    def press_keys(self, keys: list[str]) -> None:
        """Press keys one after another."""
        for key in keys:
            pyautogui.press(key)

    def hotkey(self, keys: list[str]) -> None:
        """Press keys together as a chord (e.g. ['ctrl', 'c'])."""
        if keys:
            pyautogui.hotkey(*keys)

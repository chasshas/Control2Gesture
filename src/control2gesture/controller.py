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
        margin: float = 0.15,
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        # Disable the fail-safe: moving into a screen corner should not raise.
        pyautogui.FAILSAFE = False
        # Zero delay between calls keeps cursor tracking responsive.
        pyautogui.PAUSE = 0

        self.screen_w, self.screen_h = screen_size or pyautogui.size()
        self.smoothing = max(0.0, min(0.95, smoothing))
        self.margin = max(0.0, min(0.4, margin))
        self._prev_x, self._prev_y = pyautogui.position()

    def _remap(self, value: float) -> float:
        """Map a normalized coord through the active region [margin, 1-margin]."""
        span = 1.0 - 2 * self.margin
        if span <= 0:
            return value
        clamped = min(max(value, self.margin), 1.0 - self.margin)
        return (clamped - self.margin) / span

    def move_cursor(self, nx: float, ny: float) -> None:
        """Move the cursor from normalized fingertip coords (0..1)."""
        target_x = self._remap(nx) * self.screen_w
        target_y = self._remap(ny) * self.screen_h

        alpha = 1.0 - self.smoothing
        x = self._prev_x + (target_x - self._prev_x) * alpha
        y = self._prev_y + (target_y - self._prev_y) * alpha

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

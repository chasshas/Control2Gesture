"""Keyboard/mouse control via pyautogui, with cursor smoothing.

On macOS the app running this must be granted Accessibility permission
(System Settings -> Privacy & Security -> Accessibility) or the OS will
silently ignore synthetic mouse/keyboard events.
"""

from __future__ import annotations

import pyautogui


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

    def press_keys(self, keys: list[str]) -> None:
        """Press keys one after another."""
        for key in keys:
            pyautogui.press(key)

    def hotkey(self, keys: list[str]) -> None:
        """Press keys together as a chord (e.g. ['ctrl', 'c'])."""
        if keys:
            pyautogui.hotkey(*keys)

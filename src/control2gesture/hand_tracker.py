"""Thin wrapper around MediaPipe Hands for real-time landmark detection."""

from __future__ import annotations

from dataclasses import dataclass

import mediapipe as mp
import numpy as np


@dataclass
class HandResult:
    """Detected hand landmarks for a single hand.

    ``landmarks`` is a (21, 3) array of normalized (x, y, z) coordinates in the
    range [0, 1] for x/y (z is relative depth). ``handedness`` is "Left" or
    "Right" as labelled by MediaPipe (in the mirrored image frame).
    """

    landmarks: np.ndarray
    handedness: str


class HandTracker:
    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles

    def process(self, frame_rgb: np.ndarray) -> list[HandResult]:
        """Run detection on an RGB frame and return per-hand results."""
        results = self._hands.process(frame_rgb)
        hands: list[HandResult] = []
        if not results.multi_hand_landmarks:
            return hands

        handedness_list = results.multi_handedness or []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
            label = "Unknown"
            if idx < len(handedness_list):
                label = handedness_list[idx].classification[0].label
            hands.append(HandResult(landmarks=coords, handedness=label))
        return hands

    def draw(self, frame_bgr: np.ndarray, raw_results=None) -> None:
        """(Optional) draw landmarks. Kept simple; main.py draws its own overlay."""
        # Intentionally lightweight — overlay drawing lives in main.py so this
        # module stays free of presentation concerns.

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

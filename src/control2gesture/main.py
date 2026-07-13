"""Entry point: capture webcam, recognize gestures, drive the mouse/keyboard."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from . import gesture_recognizer as gr
from .action_mapper import ActionMapper
from .config import Config, load_config
from .controller import Controller
from .hand_tracker import HandResult, HandTracker

log = logging.getLogger("control2gesture")

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "gestures.yaml"
_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


def _draw_hand(frame: np.ndarray, hand: HandResult) -> None:
    """Draw one hand's landmarks and connections on the BGR frame."""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]

    for a, b in _CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for px, py in pts:
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)


def _draw_banner(
    frame: np.ndarray,
    hands_pair: list[str | None],
    action: str,
    enabled: bool,
) -> None:
    """Draw the ``[left, right]`` gesture pair, its action, and on/off state."""
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
    left, right = hands_pair
    status = "ON" if enabled else "OFF (space or [four, four] to resume)"
    text = f"L:{left or '-'}  R:{right or '-'}   action: {action}   control: {status}"
    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run(config: Config) -> None:
    s = config.settings

    cap = cv2.VideoCapture(s.camera_index)
    if s.camera_fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*s.camera_fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.frame_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {s.camera_index}")

    controller = Controller(smoothing=s.cursor_smoothing, sensitivity=s.cursor_sensitivity)
    mapper = ActionMapper(config, controller)
    pinch_t, fist_m, thumb_m = s.gesture_thresholds()

    log.info(
        "Starting. Press 'q' (or Ctrl+C) to quit, space or [four, four] to "
        "toggle gesture control on/off."
    )
    try:
        with HandTracker(
            max_hands=s.max_hands,
            detection_confidence=s.detection_confidence,
            tracking_confidence=s.tracking_confidence,
            model_complexity=s.model_complexity,
        ) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.warning("Failed to read frame; retrying.")
                    continue

                if s.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = tracker.process(rgb)

                # The whole system speaks one [left, right] gesture pair.
                hands_pair = gr.classify_hands(
                    [(h.landmarks, h.handedness) for h in hands],
                    pinch_t,
                    fist_m,
                    thumb_m,
                )
                log.debug("hands: %s", hands_pair)

                if hands:
                    # cursor_xy drives move_cursor; distance drives zoom/volume
                    # (only meaningful with two hands).
                    cursor_xy = gr.index_tip_position(hands[0].landmarks)
                    distance = (
                        gr.hands_distance(hands[0].landmarks, hands[1].landmarks)
                        if len(hands) >= 2
                        else None
                    )
                    mapper.handle(hands_pair, cursor_xy, distance)
                    action = config.action_for(hands_pair).get("action", "none")
                    if s.show_window:
                        for hand in hands:
                            _draw_hand(frame, hand)
                        _draw_banner(frame, hands_pair, action, mapper.enabled)
                else:
                    mapper.reset()

                if s.show_window:
                    cv2.imshow("Control2Gesture", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord(" "):
                        mapper.toggle_enabled()
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MediaPipe hand-gesture keyboard/mouse controller."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to gesture config YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()

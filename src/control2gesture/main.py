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
    gesture: str,
    action: str,
    hands_pair: list[str | None] | None = None,
) -> None:
    """Draw the gesture/action status banner across the top of the frame.

    ``hands_pair`` is the ``[left, right]`` per-hand detection; when given it is
    shown alongside the combined gesture/action.
    """
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
    text = f"gesture: {gesture}   action: {action}"
    if hands_pair is not None:
        left, right = hands_pair
        text = f"L:{left or '-'}  R:{right or '-'}   {text}"
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {s.camera_index}")

    controller = Controller(smoothing=s.cursor_smoothing, margin=s.cursor_margin)
    mapper = ActionMapper(config, controller)

    log.info("Starting. Press 'q' in the window (or Ctrl+C) to quit.")
    try:
        with HandTracker(
            max_hands=s.max_hands,
            detection_confidence=s.detection_confidence,
            tracking_confidence=s.tracking_confidence,
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

                # Per-hand detection as [left_gesture, right_gesture].
                hands_pair = gr.classify_hands(
                    [(h.landmarks, h.handedness) for h in hands], s.pinch_threshold
                )
                log.debug("hands: %s", hands_pair)

                if len(hands) >= 2:
                    # Two hands: recognize a combined gesture (e.g. zoom) from
                    # the pair and drive it with the inter-hand distance.
                    a, b = hands[0], hands[1]
                    gesture = gr.classify_two_hands(
                        a.landmarks, a.handedness,
                        b.landmarks, b.handedness,
                        s.pinch_threshold,
                    )
                    distance = gr.hands_distance(a.landmarks, b.landmarks)
                    mapper.handle_two_hands(gesture, distance)
                    action = config.action_for(gesture).get("action", "none")
                    if s.show_window:
                        _draw_hand(frame, a)
                        _draw_hand(frame, b)
                        _draw_banner(frame, gesture, action, hands_pair)
                elif hands:
                    hand = hands[0]
                    gesture = gr.classify(
                        hand.landmarks, hand.handedness, s.pinch_threshold
                    )
                    cursor_xy = gr.index_tip_position(hand.landmarks)
                    mapper.handle(gesture, cursor_xy)
                    action = config.action_for(gesture).get("action", "none")
                    if s.show_window:
                        _draw_hand(frame, hand)
                        _draw_banner(frame, gesture, action, hands_pair)
                else:
                    mapper.reset()

                if s.show_window:
                    cv2.imshow("Control2Gesture", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
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

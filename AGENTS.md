# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. This is
the canonical agent guide; `CLAUDE.md` points here.

## What this project is

Control2Gesture turns webcam hand gestures into OS mouse/keyboard events. The
pipeline is deliberately linear and each stage lives in its own module:

```
main.py ─▶ hand_tracker.py ─▶ gesture_recognizer.py ─▶ action_mapper.py ─▶ controller.py
 loop        MediaPipe            landmarks→name          debounce+map        pyautogui
```

Keep this separation. Recognition logic must not import `pyautogui`; OS control
must not import `mediapipe`. This keeps `gesture_recognizer.py` unit-testable
without a camera or a display.

## Environment

- Use **conda**: `conda env create -f environment.yml && conda activate control2gesture`.
- Then install the package: `pip install -e .` (required once per environment —
  the `src/` layout means `python -m control2gesture` fails with
  `No module named control2gesture` until it's installed).
- Python is pinned to **3.11** (MediaPipe supports 3.9–3.12).
- If you change a dependency, update **both** `environment.yml` and
  `requirements.txt`.

## Running & testing

```bash
python -m control2gesture       # run the app (needs a webcam + display)
pytest                          # run tests (no camera/display needed)
```

- Tests target the pure logic in `gesture_recognizer.py` using synthetic
  landmark arrays. **Any new recognition rule must come with a test.** Do not
  write tests that require a camera, a screen, or that move the real cursor.
- Runtime verification (the camera loop, `pyautogui`) needs hardware and, on
  macOS, Accessibility + Camera permissions — you generally cannot verify these
  from a headless agent context. Say so rather than claiming it was tested.

## Conventions

- Landmarks are numpy `(21, 3)` arrays of normalized coords; **image y grows
  downward**, so "finger up" means `tip.y < pip.y`. Thumb uses x and depends on
  handedness.
- Config is data, not code. New behaviour that a user might want to tune belongs
  in `config/gestures.yaml` and `config.py`'s `Settings`, not hardcoded.
- Actions are **continuous** (run every frame) or **one-shot** (fire once per
  gesture, via the debounce in `ActionMapper`). When adding an action, decide
  which it is; continuous ones must be added to `CONTINUOUS_ACTIONS`.
- Type hints and short docstrings on public functions. Match the existing style.

## How to extend

- **New gesture:** add a branch in `gesture_recognizer.classify()` returning a
  new name + a test, then map it in `config/gestures.yaml`.
- **New action:** add a handler in `ActionMapper._run_oneshot` /
  `_run_continuous`, and a `Controller` method for the actual OS call.

## Safety

The app moves the real mouse and presses real keys, and `pyautogui`'s fail-safe
is disabled for smooth tracking. Never add code that could enter an
uninterruptible input loop — always leave the `q`/`Ctrl+C` exit intact. Do not
add telemetry, network calls, or anything that captures/transmits the webcam
feed.

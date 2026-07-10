# CLAUDE.md

This project's agent guidance lives in **[AGENTS.md](./AGENTS.md)** — read it
before making changes.

Quick reference:

- **Setup:** `conda env create -f environment.yml && conda activate control2gesture && pip install -e .`
- **Run:** `python -m control2gesture`
- **Test:** `pytest` (pure logic only; no camera/display required)
- **Architecture:** `main → hand_tracker → gesture_recognizer → action_mapper → controller`.
  Keep recognition free of `pyautogui` and OS control free of `mediapipe`.
- **Gestures are `[left, right]` pairs:** recognition returns a
  `[left_gesture, right_gesture]` pair (`None` per empty side); `gestures.yaml`
  keys actions by that pair (`gesture: [left, right]`, `null` for an empty side).
- **Extend a gesture:** add a pose rule + test in `gesture_recognizer.py`, then
  map the pair in `config/gestures.yaml`.

See [AGENTS.md](./AGENTS.md) for conventions, safety notes, and details.

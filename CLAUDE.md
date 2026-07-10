# CLAUDE.md

This project's agent guidance lives in **[AGENTS.md](./AGENTS.md)** — read it
before making changes.

Quick reference:

- **Setup:** `conda env create -f environment.yml && conda activate control2gesture`
- **Run:** `python -m control2gesture`
- **Test:** `pytest` (pure logic only; no camera/display required)
- **Architecture:** `main → hand_tracker → gesture_recognizer → action_mapper → controller`.
  Keep recognition free of `pyautogui` and OS control free of `mediapipe`.
- **Extend a gesture:** add a rule + test in `gesture_recognizer.py`, then map it
  in `config/gestures.yaml`.

See [AGENTS.md](./AGENTS.md) for conventions, safety notes, and details.

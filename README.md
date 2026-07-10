# Control2Gesture

Control your **mouse and keyboard with hand gestures**, using your webcam.
Control2Gesture uses [MediaPipe](https://developers.google.com/mediapipe) to
track your hand in real time, classifies the pose into a named gesture, and
runs a pre-mapped action (move cursor, click, scroll, press a key, …).

No model training required — gestures are recognized from hand-landmark
geometry, so it works out of the box and is easy to extend.

## How it works

```
webcam ─▶ HandTracker ─▶ gesture_recognizer ─▶ ActionMapper ─▶ Controller
 (cv2)     (MediaPipe)     (landmarks→name)      (debounce)     (pyautogui)
```

1. **HandTracker** (`hand_tracker.py`) runs MediaPipe Hands and returns 21
   normalized landmarks per hand.
2. **gesture_recognizer** (`gesture_recognizer.py`) decides which fingers are
   extended and maps the pattern to a gesture name.
3. **ActionMapper** (`action_mapper.py`) looks the gesture up in the config and
   dispatches the action, debouncing one-shot actions so they fire once per
   gesture.
4. **Controller** (`controller.py`) performs the actual OS mouse/keyboard event
   via `pyautogui`, with cursor smoothing.

## Gestures (default mapping)

| Gesture      | Pose                        | Action        |
|--------------|-----------------------------|---------------|
| `pointing`   | index finger only           | move cursor   |
| `pinch`      | thumb + index tips touching | left click    |
| `victory`    | index + middle (peace sign) | right click   |
| `thumbs_up`  | thumb only                  | scroll up     |
| `three`      | index + middle + ring       | scroll down   |
| `fist`       | closed hand                 | none          |
| `open_palm`  | all fingers spread          | none          |

**Two-hand gestures** — both hands the same pose (symmetric):

| Gesture            | Pose                    | Action                                   |
|--------------------|-------------------------|------------------------------------------|
| `two_hand_pinch`   | pinch with both hands   | zoom in/out as hands move apart/together |
| `two_hand_open`    | both open palms         | volume up/down as hands move apart/together |
| `two_hand_fist`    | both fists              | one-shot hotkey (default: show desktop)  |
| `two_hand_victory` | both peace signs        | one-shot hotkey (default: screenshot)    |

**Two-hand gestures** — a different pose per hand (asymmetric):

| Gesture                  | Pose (left + right)       | Action (default)         |
|--------------------------|---------------------------|--------------------------|
| `left_fist_right_point`  | left fist + right index   | one-shot hotkey (next tab)|
| `left_point_right_fist`  | left index + right fist   | one-shot hotkey (prev tab)|
| `left_open_right_fist`   | left open palm + right fist | one-shot hotkey (undo) |

With `max_hands: 2` (the default) the app tracks both hands at once. The
distance-driven pair (`two_hand_pinch` → zoom, `two_hand_open` → volume) reacts
to how far apart the hands are: pull apart to increase, bring together to
decrease. Every other two-hand gesture fires a configurable hotkey once per
gesture.

Asymmetric gestures are oriented by which hand is **Left** vs **Right** on
screen (MediaPipe's handedness). If left/right feel swapped, flip the two
mappings in `config/gestures.yaml` or toggle `flip_horizontal`. All default
hotkey `keys` are macOS-oriented — each mapping lists the Windows equivalent in
a comment.

Edit `config/gestures.yaml` to remap any gesture to any action.

## Setup (conda)

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate control2gesture

# Install this package into the env (required, once per environment).
# Editable (-e) means code edits take effect without reinstalling.
pip install -e .
```

The `pip install -e .` step is required because of the `src/` layout — without
it `python -m control2gesture` reports `No module named control2gesture`. Rerun
it only if you recreate the environment.

> Prefer pip-only? `pip install -r requirements.txt && pip install -e .` also works.

### macOS permissions

macOS requires **Accessibility** and **Camera** permissions for synthetic
input and webcam access:

- **System Settings → Privacy & Security → Camera** → enable your terminal / IDE.
- **System Settings → Privacy & Security → Accessibility** → enable your
  terminal / IDE. Without this, the mouse/keyboard will *not* move even though
  the app runs.

## Run

```bash
conda activate control2gesture
python -m control2gesture              # uses config/gestures.yaml
python -m control2gesture -c my.yaml   # custom config
python -m control2gesture -v           # verbose logging
```

A preview window opens with your tracked hand and the current gesture/action.
Press **`q`** in the window (or `Ctrl+C` in the terminal) to quit.

> **Safety:** the app moves your real cursor and presses real keys. Test with a
> throwaway window focused. `pyautogui`'s fail-safe is disabled for smooth
> tracking, so `Ctrl+C`/`q` is your stop switch.

## Configuration

`config/gestures.yaml` has two sections:

- **`settings`** — camera index/resolution, MediaPipe confidences, cursor
  smoothing/margin, pinch threshold, and `stable_frames` (how many frames a
  gesture must persist before a one-shot action fires).
- **`gestures`** — maps each gesture name to an action. Action types:
  `none`, `move_cursor`, `left_click`, `right_click`, `double_click`,
  `scroll_up`, `scroll_down`, `zoom`/`volume` (two-hand), `key` (sequential),
  `hotkey` (chord). The distance-driven two-hand actions are tuned by
  `max_hands`, `two_hand_deadzone` (how far the hands must move per step) and
  `two_hand_step`.

Example — make the peace sign copy:

```yaml
gestures:
  victory:
    action: hotkey
    keys: ["ctrl", "c"]   # use "command" on macOS
```

## Development

```bash
pytest                 # run the recognizer tests (no camera needed)
```

Project layout:

```
Control2Gesture/
├── config/gestures.yaml            # gesture → action mapping
├── src/control2gesture/
│   ├── main.py                     # camera loop + overlay (entry point)
│   ├── hand_tracker.py             # MediaPipe wrapper
│   ├── gesture_recognizer.py       # landmarks → gesture name
│   ├── action_mapper.py            # gesture → action, with debounce
│   ├── controller.py               # pyautogui mouse/keyboard
│   └── config.py                   # YAML config loader
├── tests/test_gesture_recognizer.py
├── environment.yml                 # conda environment
├── requirements.txt                # pip fallback
├── CLAUDE.md / AGENTS.md           # guidance for AI coding agents
└── README.md
```

## Extending

To add a new single-hand gesture, add a branch in
`gesture_recognizer.classify()` returning a new name, then map that name in
`config/gestures.yaml`. To add a new action type, add a handler in
`ActionMapper` and (if continuous) list it in `CONTINUOUS_ACTIONS`.

Per-hand detection is available as `gesture_recognizer.classify_hands(hands)`,
which returns the pair `[left_gesture, right_gesture]` (a side with no hand is
`None`, e.g. `["fist", "pointing"]` or `["fist", None]`). The preview banner
shows it live as `L:… R:…`, and `-v` logs it each frame.

Two-hand *combined* gestures are entries in the `_TWO_HAND_COMBOS` table in
`gesture_recognizer.py`, keyed by `(left_hand_pose, right_hand_pose)` — add a
row (symmetric or asymmetric), map the new name in `config/gestures.yaml`, and
add a test with synthetic landmarks. They are dispatched by
`ActionMapper.handle_two_hands()`; `zoom`/`volume` are distance-driven, anything
else fires as a one-shot.

## License

MIT (see project owner).

# Control2Gesture

Control your **mouse and keyboard with hand gestures**, using your webcam.
Control2Gesture uses [MediaPipe](https://developers.google.com/mediapipe) to
track your hands in real time, classifies each hand's pose into a
`[left, right]` gesture pair, and runs a pre-mapped action (move cursor, click,
scroll, press a key, …).

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
   extended on each hand and produces one `[left, right]` pose pair.
3. **ActionMapper** (`action_mapper.py`) looks that pair up in the config and
   dispatches the action, debouncing one-shot actions so they fire once per
   gesture.
4. **Controller** (`controller.py`) performs the actual OS mouse/keyboard event
   via `pyautogui`, with cursor smoothing.

## Gestures (default mapping)

Every gesture is a `[left, right]` pair of hand poses. `null` means "no hand on
that side", so a single-hand gesture uses `null` for the empty side. Left/Right
follow the on-screen (mirrored) handedness. Recognized poses: `fist`,
`open_palm`, `pointing`, `victory`, `three`, `thumbs_up`, `pinch`.

**Single hand** — right hand up (`[null, <pose>]`):

| Pair                  | Pose                        | Action      |
|-----------------------|-----------------------------|-------------|
| `[null, pointing]`    | index finger only           | move cursor |
| `[null, pinch]`       | thumb + index tips touching | left click  |
| `[null, victory]`     | index + middle (peace sign) | right click |
| `[null, thumbs_up]`   | thumb only                  | scroll up   |
| `[null, three]`       | index + middle + ring       | scroll down |

**Two hands, the same pose:**

| Pair                     | Pose                  | Action                                   |
|--------------------------|-----------------------|------------------------------------------|
| `[pinch, pinch]`         | pinch with both hands | zoom in/out as hands move apart/together |
| `[open_palm, open_palm]` | both open palms       | volume up/down as hands move apart/together |
| `[fist, fist]`           | both fists            | one-shot hotkey (default: show desktop)  |
| `[victory, victory]`     | both peace signs      | one-shot hotkey (default: screenshot)    |

**Two hands, a different pose per hand:**

| Pair                  | Pose (left + right)         | Action (default)          |
|-----------------------|-----------------------------|---------------------------|
| `[fist, pointing]`    | left fist + right index     | one-shot hotkey (next tab)|
| `[pointing, fist]`    | left index + right fist     | one-shot hotkey (prev tab)|
| `[open_palm, fist]`   | left open palm + right fist | one-shot hotkey (undo)    |

With `max_hands: 2` (the default) the app tracks both hands at once. The
distance-driven pairs (`[pinch, pinch]` → zoom, `[open_palm, open_palm]` →
volume) react to how far apart the hands are: pull apart to increase, bring
together to decrease. Every other two-hand pair fires a configurable hotkey once
per gesture.

Pairs are oriented by which hand is **Left** vs **Right** on screen (MediaPipe's
handedness). If left/right feel swapped, flip the pair in
`config/gestures.yaml` or toggle `flip_horizontal`. All default hotkey `keys`
are macOS-oriented — each mapping lists the Windows equivalent in a comment.

Edit `config/gestures.yaml` to remap any pair to any action.

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
- **`gestures`** — a list of entries, each keyed by a `[left, right]` pose pair
  (`null` for an empty side) and mapped to an action. Action types:
  `none`, `move_cursor`, `left_click`, `right_click`, `double_click`,
  `scroll_up`, `scroll_down`, `zoom`/`volume` (two-hand), `key` (sequential),
  `hotkey` (chord). The distance-driven two-hand actions are tuned by
  `max_hands`, `two_hand_deadzone` (how far the hands must move per step) and
  `two_hand_step`.

Example — make a right-hand peace sign copy:

```yaml
gestures:
  - gesture: [null, victory]
    action: hotkey
    keys: ["ctrl", "c"]   # use "command" on macOS
```

### GUI action mapper

Prefer not to hand-edit YAML? A small desktop editor lets you build the
`[left, right]` pose → action map visually and import/export it as YAML:

```bash
python -m control2gesture.gui     # or: control2gesture-gui
```

It opens `config/gestures.yaml` by default. Add/Edit/Duplicate/Remove rows,
pick each hand's pose and the action from dropdowns (with the extra `amount` /
`keys` fields shown only when the chosen action uses them), then **Save** back to
the same file or **Save As / Export** to a new one. **Open / Import** loads any
existing map. It validates as you go (unknown poses, duplicate pairs, missing
keys), and what it writes is the exact schema the app reads — no camera or
MediaPipe needed, so you can edit maps anywhere. Built on Tkinter (bundled with
Python; no extra dependency).

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
│   ├── gesture_recognizer.py       # landmarks → [left, right] pose pair
│   ├── action_mapper.py            # pair → action, with debounce
│   ├── controller.py               # pyautogui mouse/keyboard
│   └── config.py                   # YAML config loader
├── tests/                          # test_gesture_recognizer.py, test_config.py
├── environment.yml                 # conda environment
├── requirements.txt                # pip fallback
├── CLAUDE.md / AGENTS.md           # guidance for AI coding agents
└── README.md
```

## Extending

Recognition speaks one `[left, right]` pair.
`gesture_recognizer.classify_hands(hands)` returns
`[left_gesture, right_gesture]`, with `None` on a side that has no hand (e.g.
`["fist", "pointing"]`, `[None, "pointing"]`, or `[None, None]`). The preview
banner shows it live as `L:… R:…`, and `-v` logs it each frame. `ActionMapper`
looks that pair up in the config and dispatches; `zoom`/`volume` are
distance-driven, `move_cursor`/scroll are continuous, everything else fires once
per stable pair.

To add a new **pose**, add a branch in `gesture_recognizer.classify()` returning
a new pose name plus a test with synthetic landmarks, then map the `[left,
right]` pair(s) that use it in `config/gestures.yaml` — single-hand as
`[null, <pose>]`, two-hand as `[<left>, <right>]`. To add a new **action type**,
add a handler in `ActionMapper` and (if continuous) list it in
`CONTINUOUS_ACTIONS` (or `TWO_HAND_DISTANCE_ACTIONS` for a distance-driven one).

## License

MIT (see project owner).

"""Load and validate the gesture configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    flip_horizontal: bool = True
    max_hands: int = 2
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.6
    cursor_smoothing: float = 0.5
    cursor_margin: float = 0.15
    pinch_threshold: float = 0.06
    stable_frames: int = 3
    # Distance-driven two-hand gestures (zoom, volume): how much the inter-hand
    # distance must change (in normalized units) to emit one step, and how many
    # key presses each step sends.
    two_hand_deadzone: float = 0.03
    two_hand_step: int = 1
    show_window: bool = True


# A gesture is identified by the (left, right) pair the recognizer produces;
# either side may be None when no hand is on that side.
GesturePair = tuple[str | None, str | None]


@dataclass
class Config:
    settings: Settings = field(default_factory=Settings)
    # Maps a (left, right) gesture pair -> action spec, e.g. {"action": "zoom"}.
    gestures: dict[GesturePair, dict[str, Any]] = field(default_factory=dict)

    def action_for(self, pair: Sequence[str | None]) -> dict[str, Any]:
        """Look up the action for a ``[left, right]`` gesture pair."""
        return self.gestures.get((pair[0], pair[1]), {"action": "none"})


def load_config(path: str | Path) -> Config:
    """Read a YAML config file into a :class:`Config`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    settings_raw = raw.get("settings", {}) or {}
    # Only pass keys the dataclass knows about, so unknown keys fail loudly.
    known = Settings.__dataclass_fields__.keys()
    unknown = set(settings_raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown settings keys in {path}: {sorted(unknown)}")
    settings = Settings(**settings_raw)

    gestures = _parse_gestures(raw.get("gestures", []) or [], path)
    return Config(settings=settings, gestures=gestures)


def _parse_gestures(
    raw: Any, path: Path
) -> dict[GesturePair, dict[str, Any]]:
    """Turn the YAML ``gestures`` list into a ``(left, right) -> spec`` map.

    Each entry is a mapping with a ``gesture: [left, right]`` two-element list
    (either side may be ``null`` for "no hand there") plus its action fields.
    """
    if not isinstance(raw, list):
        raise ValueError(
            f"'gestures' in {path} must be a list of "
            "'{{gesture: [left, right], action: ...}}' entries"
        )

    gestures: dict[GesturePair, dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, dict) or "gesture" not in entry:
            raise ValueError(
                f"Each gesture entry in {path} needs a 'gesture: [left, right]' "
                f"key; got: {entry!r}"
            )
        pair = entry["gesture"]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"'gesture' in {path} must be a two-element [left, right] list; "
                f"got: {pair!r}"
            )
        key: GesturePair = (pair[0], pair[1])
        if key in gestures:
            raise ValueError(f"Duplicate gesture pair {list(key)} in {path}")
        gestures[key] = {k: v for k, v in entry.items() if k != "gesture"}
    return gestures

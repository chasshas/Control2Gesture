"""Load and validate the gesture configuration."""

from __future__ import annotations

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
    max_hands: int = 1
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.6
    cursor_smoothing: float = 0.5
    cursor_margin: float = 0.15
    pinch_threshold: float = 0.06
    stable_frames: int = 3
    show_window: bool = True


@dataclass
class Config:
    settings: Settings = field(default_factory=Settings)
    # Maps gesture name -> action spec, e.g. {"action": "left_click"}.
    gestures: dict[str, dict[str, Any]] = field(default_factory=dict)

    def action_for(self, gesture: str) -> dict[str, Any]:
        return self.gestures.get(gesture, {"action": "none"})


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

    gestures = raw.get("gestures", {}) or {}
    return Config(settings=settings, gestures=gestures)

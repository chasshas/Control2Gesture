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
    camera_fourcc: str = "MJPG"    # most webcams need this to hit 30fps at 720p
    flip_horizontal: bool = True
    max_hands: int = 2
    model_complexity: int = 0      # 0=lite (fast), 1=full (more accurate, slower)
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.6
    cursor_smoothing: float = 0.5
    cursor_sensitivity: float = 3.0
    # Shared defaults for the three closed-hand/pinch thresholds. Each has an
    # optional per-hand override below (``*_left`` / ``*_right``) so either
    # side can be finetuned independently instead of moving both hands at once.
    pinch_threshold: float = 0.06
    fist_fold_margin: float = 0.03
    # How straight the thumb must be (see gesture_recognizer.thumb_straightness,
    # -1 = bent back on itself, 1 = fully extended) to read a closed hand as
    # thumbs_up rather than fist. A joint angle, not a distance, so it needs no
    # calibration against hand size or camera distance.
    thumb_straight_threshold: float = 0.5
    pinch_threshold_left: float | None = None
    pinch_threshold_right: float | None = None
    fist_fold_margin_left: float | None = None
    fist_fold_margin_right: float | None = None
    thumb_straight_threshold_left: float | None = None
    thumb_straight_threshold_right: float | None = None
    stable_frames: int = 3
    # Distance-driven two-hand gestures (zoom, volume): how much the inter-hand
    # distance must change (in normalized units) to emit one step, and how many
    # key presses each step sends.
    two_hand_deadzone: float = 0.03
    two_hand_step: int = 1
    show_window: bool = True

    def gesture_thresholds(
        self,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Resolve pinch/fist/thumb thresholds per hand for
        :func:`gesture_recognizer.classify_hands`.

        A ``*_left``/``*_right`` override wins for that hand; otherwise both
        hands fall back to the shared base value.
        """

        def resolve(base: float, left: float | None, right: float | None) -> dict[str, float]:
            return {
                "Left": base if left is None else left,
                "Right": base if right is None else right,
            }

        return (
            resolve(self.pinch_threshold, self.pinch_threshold_left, self.pinch_threshold_right),
            resolve(
                self.fist_fold_margin, self.fist_fold_margin_left, self.fist_fold_margin_right
            ),
            resolve(
                self.thumb_straight_threshold,
                self.thumb_straight_threshold_left,
                self.thumb_straight_threshold_right,
            ),
        )


# A gesture is identified by the (left, right) pair the recognizer produces;
# either side may be None when no hand is on that side.
GesturePair = tuple[str | None, str | None]

# Wildcard usable in a config entry's ``gesture`` side to mean "match whatever
# this hand is doing, including no hand at all" -- e.g. ``[any, pinch]`` fires
# on a right-hand pinch no matter what the left hand is up to. Never appears
# in a pair the recognizer produces, only in config-authored patterns.
ANY = "any"


def _pattern_matches(pattern: GesturePair, pair: GesturePair) -> bool:
    return (pattern[0] == ANY or pattern[0] == pair[0]) and (
        pattern[1] == ANY or pattern[1] == pair[1]
    )


def _dominates(a: GesturePair, b: GesturePair) -> bool:
    """True when pattern ``a`` is at least as specific as ``b`` on every side
    and strictly more specific on at least one -- so ``b`` is redundant
    wherever both match.

    ``[any, pinch]`` dominates ``[any, any]`` (same left, more specific
    right), so only the former fires. But ``[any, pinch]`` and ``[pinch,
    any]`` constrain *different* sides, so neither dominates the other --
    both stay live and both fire, e.g. on an actual ``[pinch, pinch]`` pair.
    """
    rank_a = (a[0] != ANY, a[1] != ANY)
    rank_b = (b[0] != ANY, b[1] != ANY)
    return rank_a[0] >= rank_b[0] and rank_a[1] >= rank_b[1] and rank_a != rank_b


@dataclass
class Config:
    settings: Settings = field(default_factory=Settings)
    # Maps a (left, right) gesture pair -> action spec, e.g. {"action": "zoom"}.
    gestures: dict[GesturePair, dict[str, Any]] = field(default_factory=dict)

    def matching_patterns(
        self, pair: Sequence[str | None]
    ) -> list[tuple[GesturePair, dict[str, Any]]]:
        """Return every configured ``(pattern, spec)`` that matches ``pair``,
        minus any pattern dominated by a more specific matching one (see
        :func:`_dominates`).

        This is almost always a single entry, but wildcard patterns that
        constrain different sides can both survive and both fire at once --
        e.g. ``[any, pinch]`` and ``[pinch, any]`` both match ``[pinch,
        pinch]``, since neither is more specific than the other.
        """
        key = (pair[0], pair[1])
        candidates = [
            (pattern, spec)
            for pattern, spec in self.gestures.items()
            if _pattern_matches(pattern, key)
        ]
        return [
            (pattern, spec)
            for pattern, spec in candidates
            if not any(_dominates(other, pattern) for other, _ in candidates if other != pattern)
        ]

    def actions_for(self, pair: Sequence[str | None]) -> list[dict[str, Any]]:
        """Like :meth:`matching_patterns`, but just the action specs."""
        return [spec for _, spec in self.matching_patterns(pair)]

    def action_for(self, pair: Sequence[str | None]) -> dict[str, Any]:
        """Look up a single action for a ``[left, right]`` gesture pair.

        A convenience for callers that only want one label (e.g. the preview
        banner); when more than one pattern matches (see
        :meth:`matching_patterns`), this returns the first. Real dispatch
        should use :meth:`matching_patterns` so every matching action fires.
        """
        specs = self.actions_for(pair)
        return specs[0] if specs else {"action": "none"}


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

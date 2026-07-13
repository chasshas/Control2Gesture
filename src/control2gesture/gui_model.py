"""Pure data model for the gesture action-mapping editor GUI.

This module holds everything the editor needs *except* the Tk widgets, so it can
be unit-tested without a display: the vocabulary of poses/actions, an editable
``Mapping`` row, a ``GestureDocument`` (settings + mappings), and the YAML
import/export + validation helpers.

Keep the vocabularies below in sync with the recognizer and the action mapper:

* ``POSES`` mirrors the pose names ``gesture_recognizer.classify`` can return.
* ``ACTIONS`` mirrors the action names ``action_mapper.ActionMapper`` handles.

The YAML this reads and writes is exactly the schema ``config.load_config``
consumes, so a document exported here loads straight back into the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Pose names the recognizer can produce (see gesture_recognizer.classify).
POSES: list[str] = [
    "fist",
    "open_palm",
    "pointing",
    "victory",
    "three",
    "four",
    "thumbs_up",
    "pinch",
]

# Action names the mapper understands (see action_mapper / controller).
ACTIONS: list[str] = [
    "none",
    "move_cursor",
    "left_click",
    "right_click",
    "double_click",
    "scroll_up",
    "scroll_down",
    "zoom",
    "volume",
    "key",
    "hotkey",
    "toggle_control",
]

# Actions that carry an integer ``amount`` (scroll speed).
ACTIONS_WITH_AMOUNT: set[str] = {"scroll_up", "scroll_down"}
# Actions that carry a ``keys`` list (a key sequence or a chord).
ACTIONS_WITH_KEYS: set[str] = {"key", "hotkey"}

DEFAULT_AMOUNT = 3

# How an empty hand-side (``None``) is shown in the UI.
EMPTY_SIDE = "—"


class _FlowList(list):
    """A list that YAML dumps inline, e.g. ``[null, pinch]`` instead of a block."""


def _flow_list_representer(dumper: yaml.SafeDumper, data: _FlowList):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    )


yaml.SafeDumper.add_representer(_FlowList, _flow_list_representer)


def side_to_display(side: str | None) -> str:
    """Render a hand-side value for the UI (``None`` -> the empty marker)."""
    return EMPTY_SIDE if side is None else side


def display_to_side(text: str | None) -> str | None:
    """Parse a UI hand-side value back to a pose name or ``None``."""
    if text is None:
        return None
    text = text.strip()
    return None if text in ("", EMPTY_SIDE) else text


@dataclass
class Mapping:
    """One editable row: a ``[left, right]`` pose pair and the action it fires."""

    left: str | None = None
    right: str | None = None
    action: str = "none"
    amount: int = DEFAULT_AMOUNT
    keys: list[str] = field(default_factory=list)
    # Any other entry fields the GUI doesn't have a dedicated widget for (e.g.
    # a per-gesture ``stable_frames`` override), kept so editing a mapping in
    # the GUI doesn't silently drop settings only the YAML file knows about.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def pair(self) -> tuple[str | None, str | None]:
        return (self.left, self.right)

    def details(self) -> str:
        """A short human summary of the action's extra fields, for the table."""
        if self.action in ACTIONS_WITH_AMOUNT:
            return f"amount={self.amount}"
        if self.action in ACTIONS_WITH_KEYS:
            return "+".join(self.keys) if self.keys else "(no keys)"
        return ""

    def to_entry(self) -> dict[str, Any]:
        """Serialize to a ``config.load_config`` gesture entry.

        Only the fields relevant to the chosen action are emitted, so the YAML
        stays as tidy as the hand-authored config.
        """
        entry: dict[str, Any] = {
            "gesture": _FlowList([self.left, self.right]),
            "action": self.action,
        }
        if self.action in ACTIONS_WITH_AMOUNT:
            entry["amount"] = int(self.amount)
        elif self.action in ACTIONS_WITH_KEYS:
            entry["keys"] = _FlowList(list(self.keys))
        entry.update(self.extra)
        return entry

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> Mapping:
        """Build a ``Mapping`` from a raw YAML gesture entry."""
        pair = entry.get("gesture")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                "each gesture entry needs a two-element [left, right] list; "
                f"got: {pair!r}"
            )
        action = entry.get("action", "none")
        amount = entry.get("amount", DEFAULT_AMOUNT)
        keys = entry.get("keys", []) or []
        known = {"gesture", "action", "amount", "keys"}
        extra = {k: v for k, v in entry.items() if k not in known}
        return cls(
            left=pair[0],
            right=pair[1],
            action=action,
            amount=int(amount) if amount is not None else DEFAULT_AMOUNT,
            keys=[str(k) for k in keys],
            extra=extra,
        )


@dataclass
class GestureDocument:
    """A whole editable config: the ``settings`` block plus the mapping rows.

    ``settings`` is kept as a raw dict so the editor round-trips it untouched
    (the GUI edits mappings, not tuning knobs).
    """

    settings: dict[str, Any] = field(default_factory=dict)
    mappings: list[Mapping] = field(default_factory=list)


def document_from_raw(raw: dict[str, Any] | None) -> GestureDocument:
    """Build a document from a parsed-YAML mapping."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping with 'settings'/'gestures'")

    settings = raw.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError("'settings' must be a mapping")

    raw_gestures = raw.get("gestures") or []
    if not isinstance(raw_gestures, list):
        raise ValueError("'gestures' must be a list of entries")

    mappings = [Mapping.from_entry(e) for e in raw_gestures]
    return GestureDocument(settings=dict(settings), mappings=mappings)


def document_from_yaml(text: str) -> GestureDocument:
    """Parse a YAML string into a :class:`GestureDocument`."""
    return document_from_raw(yaml.safe_load(text) or {})


def load_document(path: str | Path) -> GestureDocument:
    """Read a YAML config file into a :class:`GestureDocument`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return document_from_yaml(f.read())


_HEADER = (
    "# Control2Gesture action map — edited with the mapping GUI.\n"
    "# 'settings' tunes the camera/cursor; 'gestures' maps each [left, right]\n"
    "# hand-pose pair to an action (null = no hand on that side).\n"
)


def dump_document(doc: GestureDocument) -> str:
    """Serialize a document to YAML text that ``config.load_config`` accepts."""
    body = {
        "settings": doc.settings,
        "gestures": [m.to_entry() for m in doc.mappings],
    }
    dumped = yaml.safe_dump(
        body,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return f"{_HEADER}{dumped}"


def save_document(path: str | Path, doc: GestureDocument) -> None:
    """Write a document to a YAML file."""
    Path(path).write_text(dump_document(doc), encoding="utf-8")


def validate_mappings(mappings: list[Mapping]) -> list[str]:
    """Return a list of human-readable problems; empty means the map is valid.

    Mirrors the rules ``config.load_config`` enforces (valid poses, no duplicate
    pairs) plus the editor's own (at least one hand, required action fields), so
    the user hears about problems here rather than at app launch.
    """
    errors: list[str] = []
    seen: dict[tuple[str | None, str | None], int] = {}

    for i, m in enumerate(mappings, start=1):
        label = f"Row {i} ({side_to_display(m.left)}, {side_to_display(m.right)})"

        for side_name, side in (("left", m.left), ("right", m.right)):
            if side is not None and side not in POSES:
                errors.append(f"{label}: unknown {side_name} pose '{side}'.")

        if m.left is None and m.right is None:
            errors.append(f"{label}: a mapping must use at least one hand.")

        if m.action not in ACTIONS:
            errors.append(f"{label}: unknown action '{m.action}'.")

        if m.action in ACTIONS_WITH_KEYS and not m.keys:
            errors.append(f"{label}: action '{m.action}' needs at least one key.")

        if m.action in ACTIONS_WITH_AMOUNT and m.amount <= 0:
            errors.append(f"{label}: action '{m.action}' needs a positive amount.")

        if m.pair in seen:
            errors.append(
                f"{label}: duplicate pair, already mapped in row {seen[m.pair]}."
            )
        else:
            seen[m.pair] = i

    return errors


def parse_keys(text: str) -> list[str]:
    """Split a user-typed key string into a list.

    Accepts commas or '+' as separators, e.g. ``"command, shift, 4"`` or
    ``"command+shift+4"``.
    """
    parts = text.replace("+", ",").split(",")
    return [p.strip() for p in parts if p.strip()]

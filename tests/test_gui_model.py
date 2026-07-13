"""Tests for the GUI's pure data model (no Tk / display required)."""

import textwrap

from control2gesture import gui_model as gm
from control2gesture.config import load_config
from control2gesture.gui_model import GestureDocument, Mapping


def test_roundtrip_yaml_to_document_to_yaml(tmp_path):
    text = textwrap.dedent(
        """
        settings:
          camera_index: 0
          max_hands: 2
        gestures:
          - gesture: [null, pointing]
            action: move_cursor
          - gesture: [null, thumbs_up]
            action: scroll_up
            amount: 5
          - gesture: [fist, fist]
            action: hotkey
            keys: ["ctrl", "up"]
        """
    )
    doc = gm.document_from_yaml(text)

    assert doc.settings["max_hands"] == 2
    assert [m.pair for m in doc.mappings] == [
        (None, "pointing"),
        (None, "thumbs_up"),
        ("fist", "fist"),
    ]
    assert doc.mappings[1].amount == 5
    assert doc.mappings[2].keys == ["ctrl", "up"]

    # Re-dumping and reloading preserves the data.
    reloaded = gm.document_from_yaml(gm.dump_document(doc))
    assert [m.pair for m in reloaded.mappings] == [m.pair for m in doc.mappings]
    assert reloaded.mappings[1].amount == 5
    assert reloaded.mappings[2].keys == ["ctrl", "up"]


def test_unrecognized_entry_fields_survive_a_roundtrip():
    """A field the GUI has no dedicated widget for (e.g. a per-gesture
    `stable_frames` override) must not be silently dropped when a mapping is
    loaded and re-saved."""
    text = textwrap.dedent(
        """
        gestures:
          - gesture: [null, pinch]
            action: left_click
            stable_frames: 1
        """
    )
    doc = gm.document_from_yaml(text)
    assert doc.mappings[0].extra == {"stable_frames": 1}

    reloaded = gm.document_from_yaml(gm.dump_document(doc))
    assert reloaded.mappings[0].extra == {"stable_frames": 1}
    assert reloaded.mappings[0].to_entry()["stable_frames"] == 1


def test_exported_yaml_loads_in_app_config(tmp_path):
    """A document exported here must load straight back into the real app."""
    doc = GestureDocument(
        settings={"max_hands": 2},
        mappings=[
            Mapping(right="pinch", action="left_click"),
            Mapping(left="fist", right="fist", action="hotkey", keys=["ctrl", "up"]),
            Mapping(right="thumbs_up", action="scroll_up", amount=4),
        ],
    )
    out = tmp_path / "exported.yaml"
    gm.save_document(out, doc)

    cfg = load_config(out)
    assert cfg.action_for([None, "pinch"]) == {"action": "left_click"}
    assert cfg.action_for(["fist", "fist"]) == {"action": "hotkey", "keys": ["ctrl", "up"]}
    assert cfg.action_for([None, "thumbs_up"]) == {"action": "scroll_up", "amount": 4}


def test_to_entry_only_emits_relevant_fields():
    assert Mapping(right="pinch", action="left_click").to_entry() == {
        "gesture": [None, "pinch"],
        "action": "left_click",
    }
    scroll = Mapping(right="thumbs_up", action="scroll_down", amount=2).to_entry()
    assert scroll["amount"] == 2 and "keys" not in scroll
    chord = Mapping(left="fist", right="fist", action="hotkey", keys=["a"]).to_entry()
    assert chord["keys"] == ["a"] and "amount" not in chord


def test_validate_flags_both_hands_empty():
    errors = gm.validate_mappings([Mapping(action="left_click")])
    assert any("at least one hand" in e for e in errors)


def test_validate_flags_duplicate_pair():
    errors = gm.validate_mappings(
        [
            Mapping(right="pinch", action="left_click"),
            Mapping(right="pinch", action="right_click"),
        ]
    )
    assert any("duplicate" in e.lower() for e in errors)


def test_validate_flags_missing_keys_and_bad_amount():
    errors = gm.validate_mappings(
        [
            Mapping(right="fist", action="hotkey", keys=[]),
            Mapping(right="thumbs_up", action="scroll_up", amount=0),
        ]
    )
    assert any("needs at least one key" in e for e in errors)
    assert any("positive amount" in e for e in errors)


def test_validate_flags_unknown_pose():
    errors = gm.validate_mappings([Mapping(right="banana", action="left_click")])
    assert any("unknown right pose" in e for e in errors)


def test_valid_map_has_no_errors():
    assert gm.validate_mappings([Mapping(right="pointing", action="move_cursor")]) == []


def test_any_wildcard_is_a_valid_side():
    errors = gm.validate_mappings([Mapping(left=gm.ANY, right="pinch", action="left_click")])
    assert errors == []


def test_parse_keys_accepts_commas_and_plus():
    assert gm.parse_keys("command, shift, 4") == ["command", "shift", "4"]
    assert gm.parse_keys("command+shift+4") == ["command", "shift", "4"]
    assert gm.parse_keys("  ") == []


def test_display_side_conversion():
    assert gm.side_to_display(None) == gm.EMPTY_SIDE
    assert gm.side_to_display("fist") == "fist"
    assert gm.display_to_side(gm.EMPTY_SIDE) is None
    assert gm.display_to_side("  ") is None
    assert gm.display_to_side("fist") == "fist"

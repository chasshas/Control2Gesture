"""Tkinter editor for the gesture action map, with YAML import/export.

Run it with ``python -m control2gesture.gui`` (or the ``control2gesture-gui``
console script). It edits the same ``config/gestures.yaml`` the app reads: add,
edit, and remove ``[left, right]`` pose -> action rows, then Save/Export to YAML
or Open/Import an existing map.

All the data logic lives in :mod:`control2gesture.gui_model`; this file is only
the Tk view, so it stays free of recognition and OS-control imports.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import gui_model as gm
from .gui_model import GestureDocument, Mapping

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "gestures.yaml"

_SIDE_CHOICES = [gm.EMPTY_SIDE, gm.ANY, *gm.POSES]


class MappingDialog(tk.Toplevel):
    """Modal add/edit dialog for a single :class:`Mapping`.

    On OK it writes the edited mapping to :attr:`result`; on Cancel it stays
    ``None``.
    """

    def __init__(self, parent: tk.Misc, mapping: Mapping, title: str) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.result: Mapping | None = None

        self._left = tk.StringVar(value=gm.side_to_display(mapping.left))
        self._right = tk.StringVar(value=gm.side_to_display(mapping.right))
        self._action = tk.StringVar(value=mapping.action)
        self._amount = tk.StringVar(value=str(mapping.amount))
        self._keys = tk.StringVar(value="+".join(mapping.keys))

        self._build()
        self._sync_fields()

        self.bind("<Return>", lambda _e: self._on_ok())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.grab_set()

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=12)
        frm.grid(sticky="nsew")

        ttk.Label(frm, text="Left hand").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frm, textvariable=self._left, values=_SIDE_CHOICES,
            state="readonly", width=16,
        ).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="Right hand").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frm, textvariable=self._right, values=_SIDE_CHOICES,
            state="readonly", width=16,
        ).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="Action").grid(row=2, column=0, sticky="w", pady=4)
        action_box = ttk.Combobox(
            frm, textvariable=self._action, values=gm.ACTIONS,
            state="readonly", width=16,
        )
        action_box.grid(row=2, column=1, sticky="ew", pady=4)
        action_box.bind("<<ComboboxSelected>>", lambda _e: self._sync_fields())

        self._amount_label = ttk.Label(frm, text="Amount")
        self._amount_label.grid(row=3, column=0, sticky="w", pady=4)
        self._amount_entry = ttk.Entry(frm, textvariable=self._amount, width=18)
        self._amount_entry.grid(row=3, column=1, sticky="ew", pady=4)

        self._keys_label = ttk.Label(frm, text="Keys (a+b+c)")
        self._keys_label.grid(row=4, column=0, sticky="w", pady=4)
        self._keys_entry = ttk.Entry(frm, textvariable=self._keys, width=18)
        self._keys_entry.grid(row=4, column=1, sticky="ew", pady=4)

        self._hint = ttk.Label(frm, text="", foreground="#666", wraplength=240)
        self._hint.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 8))

        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=2, sticky="e")
        ttk.Button(btns, text="Cancel", command=self.destroy).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="OK", command=self._on_ok).grid(row=0, column=1)

    def _sync_fields(self) -> None:
        """Enable only the extra fields the chosen action uses."""
        action = self._action.get()
        wants_amount = action in gm.ACTIONS_WITH_AMOUNT
        wants_keys = action in gm.ACTIONS_WITH_KEYS

        self._amount_entry.configure(state="normal" if wants_amount else "disabled")
        self._keys_entry.configure(state="normal" if wants_keys else "disabled")

        if wants_amount:
            self._hint.configure(text="Amount: wheel clicks per frame (e.g. 3).")
        elif wants_keys:
            self._hint.configure(
                text="Keys: '+'-separated, e.g. command+shift+4 "
                "(sequence for 'key', chord for 'hotkey')."
            )
        else:
            self._hint.configure(text="")

    def _on_ok(self) -> None:
        action = self._action.get()
        mapping = Mapping(
            left=gm.display_to_side(self._left.get()),
            right=gm.display_to_side(self._right.get()),
            action=action,
        )
        if action in gm.ACTIONS_WITH_AMOUNT:
            try:
                mapping.amount = int(self._amount.get())
            except ValueError:
                messagebox.showerror("Invalid amount", "Amount must be an integer.", parent=self)
                return
        if action in gm.ACTIONS_WITH_KEYS:
            mapping.keys = gm.parse_keys(self._keys.get())

        errors = gm.validate_mappings([mapping])
        if errors:
            messagebox.showerror("Invalid mapping", "\n".join(errors), parent=self)
            return

        self.result = mapping
        self.destroy()


class GestureMapperApp:
    """The main editor window."""

    def __init__(self, root: tk.Tk, initial_path: Path | None = None) -> None:
        self.root = root
        self.doc = GestureDocument()
        self.path: Path | None = None
        self.dirty = False

        root.title("Control2Gesture — Action Mapper")
        root.geometry("720x460")
        self._build()

        if initial_path and Path(initial_path).exists():
            self._load_path(Path(initial_path))
        else:
            self._refresh()

        root.protocol("WM_DELETE_WINDOW", self._on_quit)

    # -- UI construction ---------------------------------------------------

    def _build(self) -> None:
        self._build_menu()

        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(fill="x")
        for text, cmd in (
            ("New", self._new),
            ("Open…", self._open),
            ("Save", self._save),
            ("Save As…", self._save_as),
        ):
            ttk.Button(toolbar, text=text, command=cmd).pack(side="left", padx=2)

        table = ttk.Frame(self.root, padding=(8, 0))
        table.pack(fill="both", expand=True)

        columns = ("left", "right", "action", "details")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse")
        for col, heading, width in (
            ("left", "Left hand", 130),
            ("right", "Right hand", 130),
            ("action", "Action", 150),
            ("details", "Details", 200),
        ):
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda _e: self._edit())

        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scroll.set)

        actions = ttk.Frame(self.root, padding=(8, 6))
        actions.pack(fill="x")
        for text, cmd in (
            ("Add", self._add),
            ("Edit", self._edit),
            ("Duplicate", self._duplicate),
            ("Remove", self._remove),
        ):
            ttk.Button(actions, text=text, command=cmd).pack(side="left", padx=2)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w", padding=4).pack(fill="x")

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self._new, accelerator="Ctrl+N")
        filemenu.add_command(label="Open / Import…", command=self._open, accelerator="Ctrl+O")
        filemenu.add_command(label="Save", command=self._save, accelerator="Ctrl+S")
        filemenu.add_command(label="Save As / Export…", command=self._save_as)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self._on_quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        self.root.bind("<Control-n>", lambda _e: self._new())
        self.root.bind("<Control-o>", lambda _e: self._open())
        self.root.bind("<Control-s>", lambda _e: self._save())

    # -- table / state sync ------------------------------------------------

    def _refresh(self) -> None:
        """Redraw the table and the status bar from the current document."""
        self.tree.delete(*self.tree.get_children())
        for i, m in enumerate(self.doc.mappings):
            self.tree.insert(
                "", "end", iid=str(i),
                values=(
                    gm.side_to_display(m.left),
                    gm.side_to_display(m.right),
                    m.action,
                    m.details(),
                ),
            )
        where = str(self.path) if self.path else "untitled"
        flag = " *" if self.dirty else ""
        n = len(self.doc.mappings)
        self.status.set(f"{where}{flag}  —  {n} mapping(s)")

    def _selected_index(self) -> int | None:
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._refresh()

    # -- file actions ------------------------------------------------------

    def _confirm_discard(self) -> bool:
        """Return True if it's OK to throw away unsaved changes."""
        if not self.dirty:
            return True
        return messagebox.askyesno(
            "Discard changes?",
            "You have unsaved changes. Discard them?",
            parent=self.root,
        )

    def _new(self) -> None:
        if not self._confirm_discard():
            return
        self.doc = GestureDocument()
        self.path = None
        self.dirty = False
        self._refresh()

    def _open(self) -> None:
        if not self._confirm_discard():
            return
        initial = str(self.path or DEFAULT_CONFIG)
        chosen = filedialog.askopenfilename(
            title="Open / Import gesture map",
            initialfile=Path(initial).name,
            initialdir=str(Path(initial).parent),
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if chosen:
            self._load_path(Path(chosen))

    def _load_path(self, path: Path) -> None:
        try:
            self.doc = gm.load_document(path)
        except Exception as exc:  # surface any parse/schema error to the user
            messagebox.showerror("Could not open file", str(exc), parent=self.root)
            return
        self.path = path
        self.dirty = False
        self._refresh()

    def _save(self) -> None:
        if self.path is None:
            self._save_as()
            return
        self._write_to(self.path)

    def _save_as(self) -> None:
        chosen = filedialog.asksaveasfilename(
            title="Save / Export gesture map",
            defaultextension=".yaml",
            initialfile=self.path.name if self.path else "gestures.yaml",
            initialdir=str((self.path or DEFAULT_CONFIG).parent),
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if chosen:
            self._write_to(Path(chosen))

    def _write_to(self, path: Path) -> None:
        errors = gm.validate_mappings(self.doc.mappings)
        if errors:
            proceed = messagebox.askyesno(
                "Validation warnings",
                "This map has problems:\n\n"
                + "\n".join(errors)
                + "\n\nSave anyway?",
                parent=self.root,
            )
            if not proceed:
                return
        try:
            gm.save_document(path, self.doc)
        except Exception as exc:
            messagebox.showerror("Could not save file", str(exc), parent=self.root)
            return
        self.path = path
        self.dirty = False
        self._refresh()

    # -- mapping actions ---------------------------------------------------

    def _add(self) -> None:
        dialog = MappingDialog(self.root, Mapping(right="pointing", action="move_cursor"), "Add mapping")
        self.root.wait_window(dialog)
        if dialog.result is not None:
            self.doc.mappings.append(dialog.result)
            self._mark_dirty()

    def _edit(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        dialog = MappingDialog(self.root, self.doc.mappings[idx], "Edit mapping")
        self.root.wait_window(dialog)
        if dialog.result is not None:
            self.doc.mappings[idx] = dialog.result
            self._mark_dirty()

    def _duplicate(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        src = self.doc.mappings[idx]
        clone = Mapping(src.left, src.right, src.action, src.amount, list(src.keys))
        self.doc.mappings.insert(idx + 1, clone)
        self._mark_dirty()

    def _remove(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        del self.doc.mappings[idx]
        self._mark_dirty()

    # -- lifecycle ---------------------------------------------------------

    def _on_quit(self) -> None:
        if self._confirm_discard():
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    GestureMapperApp(root, initial_path=DEFAULT_CONFIG)
    root.mainloop()


if __name__ == "__main__":
    main()

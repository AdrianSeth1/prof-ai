"""
module_store.py — Canvas-style module layer for Prof-AI.

A *module* is a named, ORDERED list of document filenames. The real document
bytes live once in the canonical document store (see DOCUMENTS_DIR); a module
only holds references. The same file may appear in many modules — that is the
whole point, mirroring how Canvas modules reference files in the course Files
area.

This file is the single source of truth for everything about modules:
  - persistence (load/save modules.json)
  - CRUD (create / rename / delete)
  - membership (add / remove / reorder docs)   <- canonical, no inline mutation
  - resolution (expand_selection)              <- order-preserving
  - display formatting (module_dd_choices / unified_source_choices)

app.py should import from here and never reimplement any of it.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Paths. Keep these next to the rest of the flat project files.
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
MODULES_PATH = ROOT / "modules.json"
CLASSES_PATH = ROOT / "classes.json"
MANIFEST_PATH = ROOT / "manifest.json"
# Canonical store for the real document bytes. One copy per file, shared by
# every module that references it. This is the SAME directory ingest.py uses
# (DOCS_DIR = Path("docs")), so modules reference the already-ingested files
# in place — no migration, no manifest rewrite, no orphaned embeddings.
DOCUMENTS_DIR = ROOT / "docs"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load_modules() -> list[dict]:
    """Return the list of module dicts, or [] if the store doesn't exist yet."""
    if not MODULES_PATH.exists():
        return []
    try:
        with MODULES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_modules(modules: list[dict]) -> None:
    """Write the module list back to disk (atomic-ish via temp + replace)."""
    tmp = MODULES_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(modules, f, indent=2, ensure_ascii=False)
    tmp.replace(MODULES_PATH)


# --------------------------------------------------------------------------- #
# Class persistence
# --------------------------------------------------------------------------- #
def load_classes() -> list[dict]:
    """Return the list of class dicts, or [] if the store doesn't exist yet."""
    if not CLASSES_PATH.exists():
        return []
    try:
        with CLASSES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_classes(classes: list[dict]) -> None:
    tmp = CLASSES_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)
    tmp.replace(CLASSES_PATH)


def _new_class_id() -> str:
    return f"cls_{uuid.uuid4().hex[:12]}"


def get_class(classes: list[dict], class_id: str) -> dict | None:
    return next((c for c in classes if c.get("id") == class_id), None)


# --------------------------------------------------------------------------- #
# Validation — cached against manifest.json's mtime so the hot path
# (expand_selection) doesn't re-read & re-parse the manifest on every call.
# --------------------------------------------------------------------------- #
_valid_cache: set[str] | None = None
_valid_cache_mtime: float | None = None


def _valid_doc_filenames() -> set[str]:
    """
    Set of filenames known to the ingestion manifest. Cached; invalidated
    automatically when manifest.json changes on disk.
    """
    global _valid_cache, _valid_cache_mtime
    if not MANIFEST_PATH.exists():
        return set()
    mtime = MANIFEST_PATH.stat().st_mtime
    if _valid_cache is not None and mtime == _valid_cache_mtime:
        return _valid_cache
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()
    # manifest is {relative_path: mtime}; keys look like "docs\\file.pdf"
    # (full relative path, backslash-separated on Windows). Modules store BARE
    # filenames, so normalize keys to basenames for comparison. PureWindowsPath
    # handles the backslash regardless of the OS this runs on.
    from pathlib import PureWindowsPath
    if isinstance(manifest, dict):
        _valid_cache = {PureWindowsPath(k).name for k in manifest.keys()}
    else:
        _valid_cache = set()
    _valid_cache_mtime = mtime
    return _valid_cache


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #
def get_module(modules: list[dict], module_id: str) -> dict | None:
    return next((m for m in modules if m.get("id") == module_id), None)


def _new_id() -> str:
    return f"mod_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
def create_module(name: str, class_id: str | None = None) -> dict:
    """Create an empty module with the given display name. Returns the new dict."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Module name cannot be empty.")
    modules = load_modules()
    module = {
        "id": _new_id(),
        "name": name,
        "documents": [],  # ORDERED list of filenames
        "created_at": datetime.now().isoformat(),
        "class_id": class_id,
    }
    modules.append(module)
    save_modules(modules)
    return module


def rename_module(module_id: str, new_name: str) -> None:
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("Module name cannot be empty.")
    modules = load_modules()
    module = get_module(modules, module_id)
    if module is None:
        raise KeyError(f"No module with id {module_id!r}")
    module["name"] = new_name
    save_modules(modules)


def delete_module(module_id: str) -> None:
    """Delete the module reference only. Never touches the shared document bytes."""
    modules = load_modules()
    modules = [m for m in modules if m.get("id") != module_id]
    save_modules(modules)


# --------------------------------------------------------------------------- #
# Membership — the canonical path. app.py must call THESE, not mutate inline.
# All three preserve / control order explicitly (Canvas-style).
# --------------------------------------------------------------------------- #
def add_docs_to_module(module_id: str, filenames: Iterable[str]) -> None:
    """
    Append the given filenames to a module, in the order given, skipping any
    already present (so order of existing items is never disturbed) and any
    not known to the manifest.
    """
    valid = _valid_doc_filenames()
    modules = load_modules()
    module = get_module(modules, module_id)
    if module is None:
        raise KeyError(f"No module with id {module_id!r}")
    existing = module.setdefault("documents", [])
    seen = set(existing)
    for fn in filenames:
        if fn in valid and fn not in seen:
            existing.append(fn)
            seen.add(fn)
    save_modules(modules)


def remove_docs_from_module(module_id: str, filenames: Iterable[str]) -> None:
    """Remove the given filenames from a module. Order of the rest is preserved."""
    drop = set(filenames)
    modules = load_modules()
    module = get_module(modules, module_id)
    if module is None:
        raise KeyError(f"No module with id {module_id!r}")
    module["documents"] = [fn for fn in module.get("documents", []) if fn not in drop]
    save_modules(modules)


def set_module_docs(module_id: str, filenames: list[str]) -> None:
    """
    Replace a module's document list wholesale with the given ordered list.
    This is what a 'Save Documents' multi-select picker maps to: the picker
    already encodes the desired final set + order. Invalid/duplicate entries
    are filtered while preserving first-seen order.
    """
    valid = _valid_doc_filenames()
    modules = load_modules()
    module = get_module(modules, module_id)
    if module is None:
        raise KeyError(f"No module with id {module_id!r}")
    cleaned: list[str] = []
    seen: set[str] = set()
    for fn in filenames:
        if fn in valid and fn not in seen:
            cleaned.append(fn)
            seen.add(fn)
    module["documents"] = cleaned
    save_modules(modules)


def reorder_module_docs(module_id: str, ordered_filenames: list[str]) -> None:
    """
    Set an explicit display order. Any current docs missing from the supplied
    order are appended at the end (so a partial reorder can't drop files).
    """
    modules = load_modules()
    module = get_module(modules, module_id)
    if module is None:
        raise KeyError(f"No module with id {module_id!r}")
    current = module.get("documents", [])
    current_set = set(current)
    new_order = [fn for fn in ordered_filenames if fn in current_set]
    new_order += [fn for fn in current if fn not in set(new_order)]
    module["documents"] = new_order
    save_modules(modules)


# --------------------------------------------------------------------------- #
# Resolution — order-preserving dedup (NOT set-based, so Canvas order survives)
# --------------------------------------------------------------------------- #
def expand_selection(selection: Iterable[str]) -> list[str]:
    """
    Take a mixed list of module IDs and bare filenames and return a flat,
    de-duplicated list of filenames in first-seen order.

    Order rule: items are resolved left-to-right; a module expands to its
    documents in stored order; the first occurrence of any filename wins and
    later duplicates are dropped. This keeps a deterministic, Canvas-like
    sequence instead of the arbitrary order a set would give.
    """
    modules = load_modules()
    by_id = {m["id"]: m for m in modules}
    out: list[str] = []
    seen: set[str] = set()

    def _emit(fn: str) -> None:
        if fn not in seen:
            out.append(fn)
            seen.add(fn)

    for item in selection:
        if item in by_id:
            for fn in by_id[item].get("documents", []):
                _emit(fn)
        else:
            _emit(item)  # bare filename
    return out


# --------------------------------------------------------------------------- #
# Display formatting — moved here from app.py so the data model owns its own
# presentation. Returns Gradio-style (label, value) choice tuples.
# --------------------------------------------------------------------------- #
def module_dd_choices() -> list[tuple[str, str]]:
    """Choices for a pure module selector: (display name, module id)."""
    return [(m["name"], m["id"]) for m in load_modules()]


def unified_source_choices() -> list[tuple[str, str]]:
    """
    Choices for the Chat / Live Lecture source picker: every module as
    '[Module] name' -> module id, followed by every individual document
    '[Doc] filename' -> filename. Modules first, then loose docs, each in a
    stable order.
    """
    choices: list[tuple[str, str]] = [
        (f"[Module] {m['name']}", m["id"]) for m in load_modules()
    ]
    for fn in sorted(_valid_doc_filenames()):
        choices.append((f"[Doc] {fn}", fn))
    return choices


# --------------------------------------------------------------------------- #
# Class CRUD
# --------------------------------------------------------------------------- #
def create_class(name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("Class name cannot be empty.")
    classes = load_classes()
    cls = {
        "id": _new_class_id(),
        "name": name,
        "created_at": datetime.now().isoformat(),
    }
    classes.append(cls)
    save_classes(classes)
    return cls


def rename_class(class_id: str, new_name: str) -> None:
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("Class name cannot be empty.")
    classes = load_classes()
    cls = get_class(classes, class_id)
    if cls is None:
        raise KeyError(f"No class with id {class_id!r}")
    cls["name"] = new_name
    save_classes(classes)


def delete_class(class_id: str) -> None:
    """Remove the class record. Modules that belonged to it are set to class_id=None (Unassigned)."""
    classes = load_classes()
    classes = [c for c in classes if c.get("id") != class_id]
    save_classes(classes)
    modules = load_modules()
    changed = False
    for m in modules:
        if m.get("class_id") == class_id:
            m["class_id"] = None
            changed = True
    if changed:
        save_modules(modules)


def assign_module_to_class(module_id: str, class_id: str | None) -> None:
    modules = load_modules()
    m = get_module(modules, module_id)
    if m is None:
        return
    m["class_id"] = class_id
    save_modules(modules)


# --------------------------------------------------------------------------- #
# Class display helpers
# --------------------------------------------------------------------------- #
def class_tree() -> list[dict]:
    """Return [{class_id, name, modules}] with synthetic 'Unassigned' group first.

    Modules missing class_id or pointing at a deleted class fall into Unassigned.
    """
    modules = load_modules()
    classes = load_classes()
    class_ids = {c["id"] for c in classes}

    groups: dict[str | None, list[dict]] = {None: []}
    for c in classes:
        groups[c["id"]] = []

    for m in modules:
        cid = m.get("class_id")
        if cid not in class_ids:
            cid = None
        groups[cid].append(m)

    result: list[dict] = [{"class_id": None, "name": "Unassigned", "modules": groups[None]}]
    for c in classes:
        result.append({"class_id": c["id"], "name": c["name"], "modules": groups[c["id"]]})
    return result


def class_dd_choices() -> list[tuple[str, str]]:
    """Choices for a pure class selector: (display name, class id)."""
    return [(c["name"], c["id"]) for c in load_classes()]

"""
modules.py — Module helpers for grouping ingested documents.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

from manifest import load_manifest

MODULES_PATH = Path("modules.json")
_DOC_MANIFEST = Path("manifest.json")
_DOC_EXTS = {".pdf", ".docx", ".pptx"}


def load_modules() -> list[dict]:
    if MODULES_PATH.exists():
        data = json.loads(MODULES_PATH.read_text())
        return data.get("modules", [])
    return []


def save_modules(modules: list[dict]) -> None:
    MODULES_PATH.write_text(json.dumps({"modules": modules}, indent=2))


def create_module(name: str) -> dict:
    modules = load_modules()
    mod = {
        "id": f"mod_{uuid.uuid4().hex[:12]}",
        "name": name.strip(),
        "documents": [],
        "created_at": datetime.now().isoformat(),
    }
    modules.append(mod)
    save_modules(modules)
    return mod


def rename_module(mod_id: str, new_name: str) -> bool:
    modules = load_modules()
    for mod in modules:
        if mod["id"] == mod_id:
            mod["name"] = new_name.strip()
            save_modules(modules)
            return True
    return False


def delete_module(mod_id: str) -> bool:
    modules = load_modules()
    filtered = [m for m in modules if m["id"] != mod_id]
    if len(filtered) == len(modules):
        return False
    save_modules(filtered)
    return True


def add_docs_to_module(mod_id: str, filenames: list[str]) -> bool:
    modules = load_modules()
    for mod in modules:
        if mod["id"] == mod_id:
            existing = set(mod["documents"])
            mod["documents"] = sorted(existing | set(filenames))
            save_modules(modules)
            return True
    return False


def remove_docs_from_module(mod_id: str, filenames: list[str]) -> bool:
    modules = load_modules()
    to_remove = set(filenames)
    for mod in modules:
        if mod["id"] == mod_id:
            mod["documents"] = [d for d in mod["documents"] if d not in to_remove]
            save_modules(modules)
            return True
    return False


def _valid_doc_filenames() -> set[str]:
    manifest = load_manifest(_DOC_MANIFEST)
    return {Path(k).name for k in manifest if Path(k).suffix.lower() in _DOC_EXTS}


def expand_selection(selection: list[str]) -> list[str]:
    """Expand a mixed list of module IDs and filenames to a deduplicated filename list.

    Module IDs start with 'mod_'. Silently skips documents not in the manifest.
    """
    valid = _valid_doc_filenames()
    mods_by_id = {m["id"]: m for m in load_modules()}

    result: list[str] = []
    seen: set[str] = set()

    for item in selection:
        if item in mods_by_id:
            for doc in mods_by_id[item]["documents"]:
                if doc in valid and doc not in seen:
                    result.append(doc)
                    seen.add(doc)
        else:
            if item in valid and item not in seen:
                result.append(item)
                seen.add(item)

    return result

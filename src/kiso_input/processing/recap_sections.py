"""Shared helpers for recap prompt sections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from ..config import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "config"
GLOBAL_PROMPT_PATH = CONFIG_DIR / "system_prompt_global.txt"
EXERCISE_SPECIFIC_PATH = CONFIG_DIR / "exercise_specific_prompts.json"

# Keys for exercise-specific data
EXERCISE_SECTION_KEYS = ["prompt", "example1", "example2"]


def _read_json(path: Path, fallback):
    if not path.exists():
        return json.loads(json.dumps(fallback)) if fallback is not None else None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return json.loads(json.dumps(fallback)) if fallback is not None else None


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_global_prompt() -> str:
    """Load the global system prompt from file."""
    if not GLOBAL_PROMPT_PATH.exists():
        return ""
    return GLOBAL_PROMPT_PATH.read_text(encoding="utf-8").strip()


def save_global_prompt(content: str) -> None:
    """Save the global system prompt to file."""
    GLOBAL_PROMPT_PATH.write_text(content.strip(), encoding="utf-8")


def _empty_exercise_entry() -> Dict[str, str]:
    return {key: "" for key in EXERCISE_SECTION_KEYS}


def _load_exercise_store() -> Dict[str, Dict[str, str]]:
    data = _read_json(EXERCISE_SPECIFIC_PATH, {})
    return data if isinstance(data, dict) else {}


def _save_exercise_store(store: Dict[str, Dict[str, str]]) -> None:
    _write_json(EXERCISE_SPECIFIC_PATH, store)


def get_exercise_sections(
    exercise_name: Optional[str],
) -> Dict[str, str]:
    """Get exercise-specific sections (prompt, example1, example2)."""
    if not exercise_name:
        return _empty_exercise_entry()
    store = _load_exercise_store()
    entry = store.get(exercise_name, {}).copy()
    # Ensure all keys exist
    for key in EXERCISE_SECTION_KEYS:
        if key not in entry or not isinstance(entry[key], str):
            entry[key] = ""
    return entry


def save_exercise_sections(exercise_name: str, updates: Dict[str, str]) -> None:
    """Save updates to exercise-specific sections."""
    store = _load_exercise_store()
    entry = store.get(exercise_name, _empty_exercise_entry())
    for key, value in updates.items():
        if key in EXERCISE_SECTION_KEYS and isinstance(value, str):
            entry[key] = value.strip()
    store[exercise_name] = entry
    _save_exercise_store(store)


def assemble_system_prompt(
    global_prompt: str,
    exercise_prompt: str,
) -> str:
    """Assemble the complete system prompt from global and exercise-specific parts."""
    parts = []
    if global_prompt.strip():
        parts.append(global_prompt.strip())
    if exercise_prompt.strip():
        parts.append(exercise_prompt.strip())
    return "\n\n".join(parts).strip()


def assembled_prompt_for_exercise(
    exercise_name: Optional[str],
    global_prompt_override: Optional[str] = None,
    exercise_sections_override: Optional[Dict[str, str]] = None,
) -> str:
    """Build the complete system prompt for an exercise."""
    global_prompt = global_prompt_override if global_prompt_override is not None else load_global_prompt()
    exercise_sections = exercise_sections_override or get_exercise_sections(exercise_name)
    exercise_prompt = exercise_sections.get("prompt", "")
    return assemble_system_prompt(global_prompt, exercise_prompt)

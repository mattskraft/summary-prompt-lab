"""Shared helpers for recap prompt sections and word-limit management."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..config import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_DIR = CONFIG_DIR / "prompts"
GLOBAL_SECTION_FILES: Dict[str, Path] = {
    "rolle": PROMPTS_DIR / "system_prompt_rolle.txt",
    "eingabeformat": PROMPTS_DIR / "system_prompt_eingabeformat.txt",
    "stil": PROMPTS_DIR / "system_prompt_stil.txt",
}
EXERCISE_SECTION_TEMPLATES: Dict[str, Path] = {
    "anweisungen": PROMPTS_DIR / "system_prompt_anweisungen.txt",
    "ausgabeformat": PROMPTS_DIR / "system_prompt_ausgabeformat.txt",
}
EXERCISE_SPECIFIC_PATH = PROMPTS_DIR / "exercise_specific_prompts.json"
LEGACY_PROMPTS_PATH = CONFIG_DIR / "exercise_system_prompts.json"
WORD_LIMITS_PATH = PROMPTS_DIR / "recap_word_limits.json"

DEFAULT_WORD_LIMITS = {
    "answer_counts": [1, 2, 3],
    "max_words": [15, 30, 50],
}
WORD_LIMIT_COLUMNS = 3
SECTION_ORDER = ["rolle", "eingabeformat", "anweisungen", "stil", "ausgabeformat"]
EXERCISE_SECTION_KEYS = ["anweisungen", "ausgabeformat", "example1", "example2"]
SECTION_HEADER_MAP = {
    "rolle und aufgabe": "rolle",
    "eingabeformat": "eingabeformat",
    "anweisungen": "anweisungen",
    "stil und ton": "stil",
    "ausgabeformat": "ausgabeformat",
}


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


def load_word_limit_config() -> Dict[str, List[int]]:
    data = _read_json(WORD_LIMITS_PATH, DEFAULT_WORD_LIMITS)
    if not isinstance(data, dict):
        data = DEFAULT_WORD_LIMITS.copy()
    counts = data.get("answer_counts", DEFAULT_WORD_LIMITS["answer_counts"])
    words = data.get("max_words", DEFAULT_WORD_LIMITS["max_words"])
    if len(counts) != WORD_LIMIT_COLUMNS or len(words) != WORD_LIMIT_COLUMNS:
        counts = DEFAULT_WORD_LIMITS["answer_counts"]
        words = DEFAULT_WORD_LIMITS["max_words"]
    pairs = sorted(zip(counts, words), key=lambda item: int(item[0]))
    return {
        "answer_counts": [int(pair[0]) for pair in pairs],
        "max_words": [int(pair[1]) for pair in pairs],
    }


def save_word_limit_config(answer_counts: List[int], max_words: List[int]) -> None:
    if len(answer_counts) != WORD_LIMIT_COLUMNS or len(max_words) != WORD_LIMIT_COLUMNS:
        raise ValueError(f"Es werden genau {WORD_LIMIT_COLUMNS} Werte benötigt.")
    pairs = sorted(zip(answer_counts, max_words), key=lambda item: int(item[0]))
    payload = {
        "answer_counts": [int(pair[0]) for pair in pairs],
        "max_words": [int(pair[1]) for pair in pairs],
    }
    _write_json(WORD_LIMITS_PATH, payload)


def choose_max_words(num_answers: int, config: Optional[Dict[str, List[int]]] = None) -> int:
    cfg = config or load_word_limit_config()
    for threshold, words in zip(cfg["answer_counts"], cfg["max_words"]):
        if num_answers <= threshold:
            return words
    return cfg["max_words"][-1]


def max_words_default(config: Optional[Dict[str, List[int]]] = None) -> int:
    cfg = config or load_word_limit_config()
    return cfg["max_words"][-1]


def load_global_section(section_key: str) -> str:
    path = GLOBAL_SECTION_FILES.get(section_key)
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def save_global_section(section_key: str, content: str) -> None:
    path = GLOBAL_SECTION_FILES.get(section_key)
    if not path:
        raise KeyError(f"Unbekannter Abschnitt: {section_key}")
    path.write_text(content.strip(), encoding="utf-8")


def load_global_sections() -> Dict[str, str]:
    return {key: load_global_section(key) for key in GLOBAL_SECTION_FILES}


def _empty_exercise_entry() -> Dict[str, str]:
    return {key: "" for key in EXERCISE_SECTION_KEYS}


def _load_exercise_store() -> Dict[str, Dict[str, str]]:
    data = _read_json(EXERCISE_SPECIFIC_PATH, {})
    return data if isinstance(data, dict) else {}


def _save_exercise_store(store: Dict[str, Dict[str, str]]) -> None:
    _write_json(EXERCISE_SPECIFIC_PATH, store)


def _extract_sections_from_text(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    matches = list(re.finditer(r"(?mi)^#{1,2}\s+([^\n]+)\s*$", text))
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        header = match.group(1).strip().lower()
        key = SECTION_HEADER_MAP.get(header)
        if not key:
            continue
        body = text[start:end].strip()
        title = match.group(1).strip()
        sections[key] = f"## {title}\n{body}"
    return sections


def _migrate_from_legacy(exercise_name: str) -> Dict[str, str]:
    if not LEGACY_PROMPTS_PATH.exists():
        return {}
    legacy = _read_json(LEGACY_PROMPTS_PATH, {})
    if not isinstance(legacy, dict) or exercise_name not in legacy:
        return {}
    entry = legacy[exercise_name]
    result = _empty_exercise_entry()
    if isinstance(entry, dict):
        system_prompt = entry.get("system_prompt", "")
        if isinstance(system_prompt, str):
            sections = _extract_sections_from_text(system_prompt)
            for key in ("anweisungen", "ausgabeformat"):
                if key in sections:
                    result[key] = sections[key]
        for example_key in ("example1", "example2"):
            value = entry.get(example_key)
            if isinstance(value, str):
                result[example_key] = value
    elif isinstance(entry, str):
        sections = _extract_sections_from_text(entry)
        for key in ("anweisungen", "ausgabeformat"):
            if key in sections:
                result[key] = sections[key]
    return result


def _default_exercise_section(section_key: str, config: Optional[Dict[str, List[int]]] = None) -> str:
    template_path = EXERCISE_SECTION_TEMPLATES.get(section_key)
    if not template_path or not template_path.exists():
        return ""
    return template_path.read_text(encoding="utf-8").strip()


def get_exercise_sections(
    exercise_name: Optional[str],
    word_limit_config: Optional[Dict[str, List[int]]] = None,
    auto_create: bool = True,
) -> Dict[str, str]:
    if not exercise_name:
        entry = _empty_exercise_entry()
        entry["anweisungen"] = _default_exercise_section("anweisungen", word_limit_config)
        entry["ausgabeformat"] = _default_exercise_section("ausgabeformat", word_limit_config)
        return entry
    store = _load_exercise_store()
    entry = store.get(exercise_name, {}).copy()
    changed = False
    if not entry and auto_create:
        entry = _migrate_from_legacy(exercise_name)
        if entry:
            changed = True
    for key in EXERCISE_SECTION_KEYS:
        if key not in entry or not isinstance(entry[key], str):
            entry[key] = ""
    for section_key in ("anweisungen", "ausgabeformat"):
        if not entry[section_key] and auto_create:
            entry[section_key] = _default_exercise_section(section_key, word_limit_config)
            changed = True
    if changed:
        store[exercise_name] = entry
        _save_exercise_store(store)
    return entry


def save_exercise_sections(exercise_name: str, updates: Dict[str, str]) -> None:
    store = _load_exercise_store()
    entry = store.get(exercise_name, _empty_exercise_entry())
    for key, value in updates.items():
        if key in EXERCISE_SECTION_KEYS and isinstance(value, str):
            entry[key] = value.strip()
    store[exercise_name] = entry
    _save_exercise_store(store)


def apply_max_words_to_ausgabeformat(text: str, max_words: int) -> str:
    if not text:
        return text
    if "xxx" not in text.lower():
        return text
    return re.sub(r"xxx", str(max_words), text, flags=re.IGNORECASE)


def assemble_system_prompt(
    global_sections: Dict[str, str],
    exercise_sections: Dict[str, str],
    max_words: int,
) -> str:
    parts: List[str] = []
    for key in SECTION_ORDER:
        if key in GLOBAL_SECTION_FILES:
            text = global_sections.get(key, "")
        elif key == "ausgabeformat":
            text = apply_max_words_to_ausgabeformat(exercise_sections.get("ausgabeformat", ""), max_words)
        else:
            text = exercise_sections.get(key, "")
        text = text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def assembled_prompt_for_exercise(
    exercise_name: Optional[str],
    num_answers: int,
    word_limit_config: Optional[Dict[str, List[int]]] = None,
    exercise_sections_override: Optional[Dict[str, str]] = None,
    global_sections_override: Optional[Dict[str, str]] = None,
) -> str:
    config = word_limit_config or load_word_limit_config()
    max_words = choose_max_words(num_answers, config)
    global_sections = global_sections_override or load_global_sections()
    exercise_sections = exercise_sections_override or get_exercise_sections(exercise_name, config)
    return assemble_system_prompt(global_sections, exercise_sections, max_words)


def iter_section_keys(scope: str) -> Iterable[str]:
    if scope == "global":
        return GLOBAL_SECTION_FILES.keys()
    if scope == "exercise":
        return EXERCISE_SECTION_KEYS
    raise ValueError(f"Unbekannter Scope: {scope}")


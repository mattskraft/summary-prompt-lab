"""Prompt construction helpers for Gemini interactions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import PROMPTS_CONFIG_PATH, PROJECT_ROOT


def _load_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("PyYAML wird benötigt. Installiere es mit: pip install pyyaml") from exc
    return yaml


@lru_cache(maxsize=1)
def _get_prompt_templates() -> Dict[str, str]:
    if not PROMPTS_CONFIG_PATH:
        raise ValueError("PROMPTS_CONFIG_PATH ist nicht konfiguriert.")
    path = Path(PROMPTS_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Prompt-Konfigurationsdatei nicht gefunden: {path}")
    yaml = _load_yaml()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt-Konfiguration muss ein Mapping enthalten: {path}")
    templates: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, str):
            templates[str(key)] = value
    return templates


def build_gemini_prompt_up_to_question(
    segments: List[Dict[str, Any]],
    question_index: int,
    mc_answers: Dict[str, List[str]],
    is_target_question: bool = False,
    free_text_answers: Optional[Dict[str, str]] = None,
) -> str:
    """Create a prompt containing all segments up to a specific question."""
    if free_text_answers is None:
        free_text_answers = {}

    templates = _get_prompt_templates()
    header = templates.get("gemini_answer_header")
    if header is None:
        raise KeyError("Schlüssel 'gemini_answer_header' fehlt in der Prompt-Konfiguration.")
    lines: List[str] = []

    for i in range(question_index + 1):
        segment = segments[i]
        if "Text" in segment:
            clean_text = " ".join(segment["Text"].split())
            lines.append(f"TEXT: {clean_text}")
        elif "Question" in segment:
            question_text = segment["Question"]
            lines.append(f"FRAGE: {question_text}")

            if is_target_question and i == question_index:
                continue

            answer_text: Optional[str] = None
            for j in range(i + 1, len(segments)):
                candidate = segments[j]
                if "Answer" in candidate and "Question" not in candidate:
                    answer_val = candidate.get("Answer")
                    if question_text in mc_answers:
                        selected = mc_answers[question_text]
                        answer_text = ", ".join(selected)
                        break
                    if (
                        isinstance(answer_val, str)
                        and answer_val.strip().lower() in {"free_text", "freetext", "free text"}
                    ):
                        if is_target_question and i == question_index:
                            break
                        if free_text_answers and question_text in free_text_answers:
                            answer_text = free_text_answers[question_text]
                        else:
                            answer_text = "free_text"
                        break
                    if isinstance(answer_val, str):
                        answer_text = answer_val
                        break
                    if isinstance(answer_val, list):
                        answer_text = ", ".join(map(str, answer_val))
                        break

            if answer_text:
                lines.append(f"ANTWORT: {answer_text}")

    return header + "\n" + "\n".join(lines)


@lru_cache(maxsize=1)
def _load_recap_system_prompt() -> str:
    """Load the recap system prompt from the text file."""
    recap_prompt_path = PROJECT_ROOT / "config" / "recap_system_prompt.txt"
    if not recap_prompt_path.exists():
        raise FileNotFoundError(
            f"Recap system prompt file not found: {recap_prompt_path}"
        )
    return recap_prompt_path.read_text(encoding="utf-8").strip()


def build_summary_prompt(segments: List[Dict[str, Any]]) -> str:
    """Create a summary prompt from processed segments."""
    # Use the new recap_system_prompt.txt file instead of YAML
    system_prompt = _load_recap_system_prompt()

    content_lines: List[str] = []

    for segment in segments:
        if "Text" in segment:
            clean_text = " ".join(segment["Text"].split())
            content_lines.append(f"TEXT: {clean_text}")
        elif "Question" in segment:
            question_text = segment["Question"]
            content_lines.append(f"FRAGE: {question_text}")
        elif "Answer" in segment:
            answer_val = segment.get("Answer")
            answer_text: Optional[str] = None

            if isinstance(answer_val, list):
                answer_text = ", ".join(str(item) for item in answer_val)
            elif isinstance(answer_val, (int, float)):
                answer_text = str(int(answer_val))
            elif isinstance(answer_val, str):
                answer_text = answer_val.strip()
            elif isinstance(answer_val, dict):
                if "min" in answer_val and "max" in answer_val:
                    min_val = answer_val.get("min")
                    max_val = answer_val.get("max")
                    if min_val == max_val:
                        answer_text = str(int(min_val))
                    else:
                        answer_text = f"{int(min_val)} - {int(max_val)}"
                else:
                    sub_parts = []
                    for key, value in answer_val.items():
                        sub_parts.append(f"{key}: {value}")
                    answer_text = "; ".join(sub_parts)

            if answer_text:
                content_lines.append(f"ANTWORT: {answer_text}")

    return (
        ("-" * 80)
        + "\nSYSTEM:\n\n"
        + system_prompt
        + "\n\n"
        + ("-" * 80)
        + "\nINHALT:\n\n"
        + "\n\n".join(content_lines)
    )


__all__ = [
    "build_gemini_prompt_up_to_question",
    "build_summary_prompt",
]



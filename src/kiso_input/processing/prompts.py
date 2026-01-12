"""Prompt construction helpers for LLM interactions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .recap_sections import assembled_prompt_for_exercise


# Default header when none is provided
DEFAULT_ANSWER_HEADER = """Du bist ein Nutzer einer therapeutischen App für Menschen mit psychischen Belastungen.
Du beantwortest die folgende Frage basierend auf dem Kontext.
Antworte kurz und authentisch, wie ein echter Nutzer es tun würde.
Halte deine Antwort unter 30 Wörtern.

KONTEXT:"""


def build_gemini_prompt_up_to_question(
    segments: List[Dict[str, Any]],
    question_index: int,
    mc_answers: Dict[str, List[str]],
    is_target_question: bool = False,
    free_text_answers: Optional[Dict[str, str]] = None,
    header: Optional[str] = None,
) -> str:
    """Create a prompt containing all segments up to a specific question.
    
    Args:
        segments: List of segment dictionaries
        question_index: Index of the question to build prompt up to
        mc_answers: Dictionary of MC answers by question text
        is_target_question: Whether this is the target question being answered
        free_text_answers: Dictionary of already-generated free text answers
        header: Optional header text. If not provided, uses default.
    """
    if free_text_answers is None:
        free_text_answers = {}

    # Use provided header or fall back to default
    if header is None:
        header = DEFAULT_ANSWER_HEADER
    
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


def build_summary_prompt(segments: List[Dict[str, Any]], exercise_name: Optional[str] = None) -> str:
    """Create a summary prompt from processed segments.
    
    Args:
        segments: List of segment dictionaries
        exercise_name: Optional exercise name to load exercise-specific system prompt
    """
    system_prompt = assembled_prompt_for_exercise(exercise_name)

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

"""Formatting helpers for prompt segments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def format_answer_for_text(answer: Any) -> Optional[str]:
    """Convert answer structures into plain-text strings."""
    if answer is None:
        return None
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        return "; ".join(str(item) for item in answer)
    if isinstance(answer, dict):
        min_value = answer.get("min") if "min" in answer else answer.get("minText")
        max_value = answer.get("max") if "max" in answer else answer.get("maxText")
        if min_value is not None or max_value is not None:
            min_str = str(min_value).strip() if min_value is not None else ""
            max_str = str(max_value).strip() if max_value is not None else ""
            if min_str and max_str:
                return f"{min_str} - {max_str}"
            return min_str or max_str
        return "; ".join(f"{key}: {value}" for key, value in answer.items())
    return str(answer)


def prompt_segments_to_text(segments: List[Dict[str, Any]]) -> str:
    """Format segments as a human-readable text representation."""
    lines: List[str] = []
    for segment in segments:
        if "Text" in segment:
            lines.append(f"TEXT: {segment['Text']}")
        elif "Question" in segment:
            lines.append(f"QUESTION: {segment['Question']}")
        elif "Answer" in segment:
            answer_text = format_answer_for_text(segment["Answer"])
            if answer_text:
                lines.append(f"ANSWER: {answer_text}")
    return "\n".join(line.replace("\n", " ").strip() for line in lines if line.strip())


__all__ = [
    "format_answer_for_text",
    "prompt_segments_to_text",
]



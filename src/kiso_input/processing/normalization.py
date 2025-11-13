"""Utility functions for normalizing user input structures."""

from __future__ import annotations

from typing import Any, Dict, List


def sanitize_text(raw_text: str) -> str:
    """Remove underscores, asterisks, and collapse whitespace."""
    cleaned = raw_text.replace("\n\n", " ")
    cleaned = cleaned.replace("_", "").replace("*", "")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def normalize_entries(container: Any) -> List[Dict[str, Any]]:
    """Extract a list of entry dictionaries from a heterogeneous container."""
    if isinstance(container, list):
        return container
    if isinstance(container, dict):
        for key in ("blocks", "items", "content"):
            value = container.get(key)
            if isinstance(value, list):
                return value
        if any(key in container for key in ("Text", "Question", "Answer", "Branch")):
            return [container]
    return []


def normalize_answer_value(answer: Any) -> Any:
    """Normalise answer shapes into stable structures while sanitising strings."""
    if answer is None:
        return None
    if isinstance(answer, str):
        sanitized = sanitize_text(answer)
        return sanitized if sanitized else None
    if isinstance(answer, list):
        sanitized_items: List[str] = []
        for element in answer:
            sanitized = sanitize_text(str(element))
            if sanitized:
                sanitized_items.append(sanitized)
        return sanitized_items if sanitized_items else None
    if isinstance(answer, dict):
        normalized: Dict[str, Any] = {}
        for key, value in answer.items():
            if isinstance(value, str):
                sanitized = sanitize_text(value)
                if sanitized:
                    normalized[key] = sanitized
            else:
                normalized[key] = value
        return normalized if normalized else None
    return answer


__all__ = [
    "sanitize_text",
    "normalize_entries",
    "normalize_answer_value",
]



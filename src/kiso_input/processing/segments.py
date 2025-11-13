"""Segment construction utilities for Kiso input processing."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple

from .normalization import normalize_answer_value, normalize_entries, sanitize_text


FREE_TEXT_TOKENS = {"free_text", "freetext", "free text"}


def emit_segments_from_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a single entry into a sequence of segment dictionaries."""
    segments: List[Dict[str, Any]] = []

    text_value = entry.get("Text")
    if isinstance(text_value, str):
        sanitized_text = sanitize_text(text_value)
        if sanitized_text:
            segments.append({"Text": sanitized_text})

    question_value = entry.get("Question", entry.get("question"))
    if isinstance(question_value, str):
        sanitized_question = sanitize_text(question_value)
        if sanitized_question:
            segments.append({"Question": sanitized_question})

    answer_handled = False

    answer_options = entry.get("AnswerOptions")
    if isinstance(answer_options, list):
        sanitized_options: List[str] = []
        for opt in answer_options:
            sanitized_opt = sanitize_text(str(opt))
            if sanitized_opt:
                sanitized_options.append(sanitized_opt)

        allow_multiple = bool(entry.get("AllowMultiple", True))
        selected_raw = entry.get("Answer")
        selected_normalized = (
            normalize_answer_value(selected_raw) if selected_raw is not None else None
        )
        selected_list: List[str] = []
        if isinstance(selected_normalized, list):
            selected_list = selected_normalized
        elif isinstance(selected_normalized, str) and selected_normalized:
            selected_list = [selected_normalized]

        segments.append(
            {
                "Answer": selected_list,
                "AnswerOptions": sanitized_options,
                "AllowMultiple": allow_multiple,
            }
        )
        answer_handled = True

    elif isinstance(answer_options, dict):
        normalized_options = normalize_answer_value(answer_options)
        if normalized_options is not None:
            normalized_answer = normalize_answer_value(entry.get("Answer"))
            segments.append(
                {
                    "Answer": normalized_answer,
                    "AnswerOptions": normalized_options,
                }
            )
            answer_handled = True

    elif isinstance(answer_options, str):
        if answer_options.strip().lower() in FREE_TEXT_TOKENS:
            segments.append({"Answer": "free_text"})
            answer_handled = True

    if answer_handled:
        return segments

    raw_answer = None
    if "Answer" in entry:
        raw_answer = entry.get("Answer")
    elif "answer" in entry:
        raw_answer = entry.get("answer")
    elif isinstance(entry.get("options"), list):
        raw_answer = entry.get("options")
    elif isinstance(entry.get("type"), str):
        entry_type = entry["type"].lower()
        if entry_type in ("free_text", "free-text", "text"):
            raw_answer = "free_text"

    normalized_answer = normalize_answer_value(raw_answer)
    if normalized_answer is not None:
        segments.append({"Answer": normalized_answer})

    return segments


def process_branch(
    branch: Dict[str, Any],
    story_nodes: Dict[str, Any],
    rng: random.Random,
    max_depth: int = 5,
) -> List[Dict[str, Any]]:
    """Select a random branch option, append the choice, and resolve its story node."""
    segments: List[Dict[str, Any]] = []
    if max_depth <= 0:
        return segments

    branch_items: List[Tuple[str, Any]] = []
    if isinstance(branch, dict):
        branch_items = list(branch.items())
    elif isinstance(branch, list):
        for option in branch:
            if isinstance(option, dict):
                branch_items.extend(option.items())
    if not branch_items:
        return segments

    selected_key, selected_node_name = rng.choice(branch_items)
    sanitized_choice = sanitize_text(str(selected_key))
    if sanitized_choice:
        segments.append({"Answer": sanitized_choice})

    node_data = story_nodes.get(selected_node_name)
    if node_data is None:
        return segments

    entries = normalize_entries(node_data)
    for nd in entries:
        segments.extend(emit_segments_from_entry(nd))
        branch_data = nd.get("Branch")
        if isinstance(branch_data, (dict, list)):
            segments.extend(
                process_branch(branch_data, story_nodes, rng, max_depth=max_depth - 1)
            )

    return segments


def get_prompt_segments_from_exercise(
    exercise_name: str,
    json_struct_path: str,
    json_sn_struct_path: str,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """Gather normalized segments for an exercise based on the provided JSON structures."""
    if not os.path.exists(json_struct_path):
        raise FileNotFoundError(f"Exercises JSON not found: {json_struct_path}")
    if not os.path.exists(json_sn_struct_path):
        raise FileNotFoundError(f"Story-nodes JSON not found: {json_sn_struct_path}")

    with open(json_struct_path, "r", encoding="utf-8") as exercise_file:
        ex_data = json.load(exercise_file)
    with open(json_sn_struct_path, "r", encoding="utf-8") as story_nodes_file:
        sn_data = json.load(story_nodes_file)

    def find_exercise(data: Any, name: str) -> Any:
        if isinstance(data, dict):
            if name in data:
                return data[name]
            for value in data.values():
                result = find_exercise(value, name)
                if result is not None:
                    return result
        return None

    exercise_container = find_exercise(ex_data, exercise_name)
    if exercise_container is None:
        raise KeyError(f"Exercise '{exercise_name}' not found in {json_struct_path}")

    rng = random.Random(seed)
    entries = normalize_entries(exercise_container)
    segments: List[Dict[str, Any]] = []

    for entry in entries:
        segments.extend(emit_segments_from_entry(entry))
        branch_data = entry.get("Branch")
        if isinstance(branch_data, (dict, list)):
            segments.extend(process_branch(branch_data, sn_data, rng))

    return segments


__all__ = [
    "FREE_TEXT_TOKENS",
    "emit_segments_from_entry",
    "process_branch",
    "get_prompt_segments_from_exercise",
]



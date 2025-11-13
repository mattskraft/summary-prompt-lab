"""High-level exports for the Kiso input processing package."""

from __future__ import annotations

from .processing.classification import (
    assess_free_text_answers,
    classify_self_harm,
    extract_free_text_answers,
    load_self_harm_lexicon,
)
from .processing.formatting import (
    format_answer_for_text,
    prompt_segments_to_text,
)
from .processing.gemini import (
    extract_json_array_from_gemini_output,
    generate_answers_with_gemini,
    generate_summary_with_gemini,
)
from .processing.prompts import (
    build_gemini_prompt_up_to_question,
    build_summary_prompt,
)
from .processing.segments import (
    FREE_TEXT_TOKENS,
    emit_segments_from_entry,
    get_prompt_segments_from_exercise,
    process_branch,
)

__all__ = [
    "assess_free_text_answers",
    "classify_self_harm",
    "extract_free_text_answers",
    "load_self_harm_lexicon",
    "format_answer_for_text",
    "prompt_segments_to_text",
    "extract_json_array_from_gemini_output",
    "generate_answers_with_gemini",
    "generate_summary_with_gemini",
    "build_gemini_prompt_up_to_question",
    "build_summary_prompt",
    "FREE_TEXT_TOKENS",
    "emit_segments_from_entry",
    "get_prompt_segments_from_exercise",
    "process_branch",
]



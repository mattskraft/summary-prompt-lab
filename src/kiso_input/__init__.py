"""Kiso input processing utilities."""

from .processing.classification import (
    assess_free_text_answers,
    classify_self_harm,
    extract_free_text_answers,
    load_self_harm_lexicon,
)
from .processing.cloud_apis import (
    extract_json_array_from_gemini_output,
    generate_answers_with_gemini,
    generate_summary_with_gemini,
)
from .processing.formatting import (
    format_answer_for_text,
    prompt_segments_to_text,
)
from .processing.prompts import (
    build_gemini_prompt_up_to_question,
    build_summary_prompt,
)
from .processing.segments import (
    get_prompt_segments_from_exercise,
)

__all__ = [
    "assess_free_text_answers",
    "build_gemini_prompt_up_to_question",
    "build_summary_prompt",
    "classify_self_harm",
    "extract_free_text_answers",
    "extract_json_array_from_gemini_output",
    "format_answer_for_text",
    "generate_answers_with_gemini",
    "generate_summary_with_gemini",
    "get_prompt_segments_from_exercise",
    "load_self_harm_lexicon",
    "prompt_segments_to_text",
]



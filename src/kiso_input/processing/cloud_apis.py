"""Cloud API integrations for answer generation and summarisation (Mistral)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

from .prompts import build_gemini_prompt_up_to_question
from ..config import MISTRAL_MODEL_SUMMARY, MISTRAL_MODEL_ANSWERS


def generate_summary_with_mistral(
    prompt: str,
    api_key: str,
    model: str = None,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a summary using the Mistral API from a prompt string.
    
    Args:
        prompt: The complete prompt string to send to Mistral
        api_key: Mistral API key
        model: Model name to use (default: mistral-small-latest)
        max_tokens: Maximum tokens to generate (default: 200)
        temperature: Sampling temperature (default: 0.7)
        top_p: Top-p sampling parameter (default: 0.9)
        
    Returns:
        Generated summary text
    """
    if model is None:
        model = MISTRAL_MODEL_SUMMARY
        
    try:
        from mistralai import Mistral  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mistralai package not installed. Install it with: pip install mistralai"
        ) from exc

    print(
        f"[Mistral Recap] model={model}, temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}"
    )

    with Mistral(api_key=api_key) as mistral:
        res = mistral.chat.complete(
            model=model,
            messages=[
                {
                    "content": prompt,
                    "role": "user",
                },
            ],
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        summary = res.choices[0].message.content.strip()
    
    return summary


def stream_summary_with_mistral(
    prompt: str,
    api_key: str,
    model: str = None,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Stream a summary using the Mistral API from a prompt string.
    
    This is a generator that yields tokens as they arrive from the API.
    Use with Streamlit's st.write_stream() for real-time display.
    
    Args:
        prompt: The complete prompt string to send to Mistral
        api_key: Mistral API key
        model: Model name to use (default: mistral-small-latest)
        max_tokens: Maximum tokens to generate (default: 200)
        temperature: Sampling temperature (default: 0.7)
        top_p: Top-p sampling parameter (default: 0.9)
        
    Yields:
        String chunks as they arrive from the API
    """
    if model is None:
        model = MISTRAL_MODEL_SUMMARY
        
    try:
        from mistralai import Mistral  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mistralai package not installed. Install it with: pip install mistralai"
        ) from exc

    print(
        f"[Mistral Recap Stream] model={model}, temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}"
    )

    with Mistral(api_key=api_key) as mistral:
        stream = mistral.chat.stream(
            model=model,
            messages=[
                {
                    "content": prompt,
                    "role": "user",
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        for chunk in stream:
            content = chunk.data.choices[0].delta.content
            if content:
                yield content


def generate_answers_with_mistral(
    segments: List[Dict[str, Any]],
    api_key: str,
    model: str = None,
    temperature: float = 0.9,
    top_p: float = 0.8,
    debug: bool = False,
    return_debug_info: bool = False,
    seed: Optional[int] = None,
    system_prompt: Optional[str] = None,
    max_words: Optional[int] = None,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Generate user answers using Mistral and merge them into the segment structure.

    Args:
        segments: List of segment dictionaries
        api_key: Mistral API key
        model: Mistral model to use
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        debug: Enable debug output
        return_debug_info: Return debug information along with segments
        seed: Random seed for reproducible MC/slider answer generation
        system_prompt: Optional custom system prompt to use instead of default
        max_words: Optional max words setting to make MC selection proportional

    Returns:
        The merged segments and optionally debug information.
    """
    if model is None:
        model = MISTRAL_MODEL_ANSWERS
    
    # Track errors during generation
    generation_errors: List[Dict[str, str]] = []
        
    try:
        from mistralai import Mistral  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "mistralai package not installed. Install it with: pip install mistralai"
        ) from exc

    rng = random.Random(seed)
    mc_answers: Dict[str, List[str]] = {}
    slider_answers: Dict[str, int] = {}

    # Generate random MC and slider answers
    for index, segment in enumerate(segments):
        if "Answer" in segment and "Question" not in segment:
            options = segment.get("AnswerOptions")
            if isinstance(options, list) and options:
                allow_multiple = bool(segment.get("AllowMultiple", True))
                for back_index in range(index - 1, -1, -1):
                    question_segment = segments[back_index]
                    if "Question" in question_segment and "Answer" not in question_segment:
                        question_text = question_segment.get("Question")
                        if question_text:
                            option_count = len(options)
                            if allow_multiple:
                                if max_words is not None:
                                    # Calculate proportional selection based on max_words
                                    max_words_clamped = max(1, min(40, max_words))
                                    selection_ratio = 0.1 + (max_words_clamped - 1) * (0.9 / 39)
                                    num_selections = max(1, min(option_count, int(option_count * selection_ratio)))
                                else:
                                    num_selections = rng.randint(1, option_count)
                                selected = rng.sample(options, num_selections)
                            else:
                                selected = [rng.choice(options)]
                            mc_answers[question_text] = selected
                            if debug:
                                mode = "Mehrfach" if allow_multiple else "Einfach"
                                proportional_info = ""
                                if allow_multiple and max_words is not None:
                                    proportional_info = f" (proportional to max_words={max_words})"
                                print(
                                    f"🎲 MC ({mode}) Question at index {back_index}: "
                                    f"Selected {len(selected)}/{option_count} options{proportional_info}"
                                )
                                print(f"   Question: {question_text[:60]}...")
                                print(f"   Selected: {selected}")
                        break
            else:
                # Handle slider questions
                slider_config = None
                if isinstance(options, dict):
                    slider_config = options
                elif isinstance(segment.get("Answer"), dict):
                    slider_config = segment.get("Answer")

                if isinstance(slider_config, dict):
                    min_numeric = slider_config.get("min")
                    max_numeric = slider_config.get("max")
                    try:
                        min_int = int(float(min_numeric)) if min_numeric is not None else None
                        max_int = int(float(max_numeric)) if max_numeric is not None else None
                        if min_int is not None and max_int is not None and min_int <= max_int:
                            for back_index in range(index - 1, -1, -1):
                                question_segment = segments[back_index]
                                if "Question" in question_segment and "Answer" not in question_segment:
                                    question_text = question_segment.get("Question")
                                    if question_text:
                                        selected_value = rng.randint(min_int, max_int)
                                        slider_answers[question_text] = selected_value
                                        if debug:
                                            print(
                                                f"🎲 Slider Question at index {back_index}: "
                                                f"Selected value {selected_value} "
                                                f"(from {min_int}-{max_int})"
                                            )
                                            print(f"   Question: {question_text[:60]}...")
                                    break
                    except (TypeError, ValueError):
                        if debug:
                            print(f"⚠️ Slider config invalid at index {index}: {slider_config}")

    # For free text questions, we'll use Mistral to generate answers
    free_text_question_indices = []

    # Find free text questions and their indices
    for index, segment in enumerate(segments):
        if "Answer" in segment and "Question" not in segment:
            answer_val = segment.get("Answer")
            if (isinstance(answer_val, str) and 
                answer_val.strip().lower() in {"free_text", "freetext", "free text"}):
                # Find the corresponding question
                for back_index in range(index - 1, -1, -1):
                    question_segment = segments[back_index]
                    if "Question" in question_segment and "Answer" not in question_segment:
                        question_text = question_segment.get("Question")
                        if question_text:
                            free_text_question_indices.append((back_index, question_text))
                        break

    # Generate free text answers using Mistral with full context
    free_text_answers_by_text: Dict[str, str] = {}
    free_text_answers_by_index: Dict[int, str] = {}  # Track by question index to avoid duplicates
    
    if free_text_question_indices:
        for idx, (question_idx, question_text) in enumerate(free_text_question_indices, start=1):
            try:
                # Build context prompt with custom header if provided
                context_prompt = build_gemini_prompt_up_to_question(
                    segments,
                    question_idx,
                    mc_answers,
                    is_target_question=True,
                    free_text_answers=free_text_answers_by_text,
                    header=system_prompt,
                )
                
                if debug:
                    print(f"\n{'='*80}")
                    print(f"🤖 MISTRAL API CALL - Free Text Question {idx}/{len(free_text_question_indices)}")
                    print(f"{'='*80}")
                    print(f"Question: {question_text}")
                    print(f"Question Index: {question_idx}")
                    print(f"Model: {model}")
                    print(f"Temperature: {temperature}, Top-p: {top_p}")
                    print(f"\nComplete Context Prompt:")
                    print(f"{'─'*40}")
                    print(context_prompt)
                    print(f"{'─'*40}")
                
                with Mistral(api_key=api_key) as mistral:
                    res = mistral.chat.complete(
                        model=model,
                        messages=[{"role": "user", "content": context_prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=100,  # Keep answers short
                    )
                    
                    if res.choices and res.choices[0].message:
                        answer = res.choices[0].message.content.strip()
                        free_text_answers_by_index[question_idx] = answer
                        # Only store in by_text if we haven't seen this question before
                        if question_text not in free_text_answers_by_text:
                            free_text_answers_by_text[question_text] = answer
                        
                        if debug:
                            print(f"\nMistral Response:")
                            print(f"{'─'*40}")
                            print(answer)
                            print(f"{'─'*40}")
                            print(f"✅ Answer stored for question")
                            print(f"{'='*80}\n")
                            print(f"📝 Free text answer for: {question_text[:60]}...")
                            print(f"   Answer: {answer}")
                    
            except Exception as e:
                # Capture error for reporting
                generation_errors.append({
                    "question": question_text[:60] if question_text else "unknown",
                    "error": str(e),
                })
                
                # Use a fallback answer with error info
                error_msg = f"Fehler: {str(e)[:100]}"
                free_text_answers_by_index[question_idx] = error_msg
                if question_text not in free_text_answers_by_text:
                    free_text_answers_by_text[question_text] = error_msg

    # Merge answers into segments
    merged_segments = []
    for segment_idx, segment in enumerate(segments):
        new_segment = segment.copy()
        
        if "Question" in segment and "Answer" not in segment:
            merged_segments.append(new_segment)
        elif "Answer" in segment and "Question" not in segment:
            # Find the corresponding question by looking backwards in original segments
            question_text = None
            question_idx = None
            for back_index in range(segment_idx - 1, -1, -1):
                back_segment = segments[back_index]
                if "Question" in back_segment and "Answer" not in back_segment:
                    question_text = back_segment.get("Question")
                    question_idx = back_index
                    break
            
            if question_text:
                # Replace with generated answers - prioritize index-based lookup for free text
                if question_text in mc_answers:
                    new_segment["Answer"] = mc_answers[question_text]
                elif question_text in slider_answers:
                    new_segment["Answer"] = slider_answers[question_text]
                elif question_idx is not None and question_idx in free_text_answers_by_index:
                    new_segment["Answer"] = free_text_answers_by_index[question_idx]
                elif question_text in free_text_answers_by_text:
                    new_segment["Answer"] = free_text_answers_by_text[question_text]
            
            merged_segments.append(new_segment)
        else:
            merged_segments.append(new_segment)

    debug_info = {
        "model_used": model,
        "temperature": temperature,
        "top_p": top_p,
        "total_segments": len(segments),
        "mc_questions_generated": len(mc_answers),
        "slider_questions_generated": len(slider_answers),
        "free_text_questions_found": len(free_text_question_indices),
        "free_text_answers_generated": len(free_text_answers_by_index),
        "errors": generation_errors,
    }

    if return_debug_info:
        return merged_segments, debug_info
    return merged_segments


__all__ = [
    "generate_summary_with_mistral",
    "stream_summary_with_mistral",
    "generate_answers_with_mistral",
]

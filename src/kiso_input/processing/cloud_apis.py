"""Cloud API integrations for answer generation and summarisation (Gemini and Mistral)."""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .prompts import build_gemini_prompt_up_to_question, build_summary_prompt
from ..config import GEMINI_MODEL_SUMMARY, GEMINI_MODEL_ANSWERS, MISTRAL_MODEL_SUMMARY, MISTRAL_MODEL_ANSWERS


def generate_summary_with_gemini(
    segments: List[Dict[str, Any]],
    api_key: str,
    model: str = None,
    debug: bool = False,
) -> str:
    """Generate a therapeutic summary using the Gemini API from segments."""
    if model is None:
        model = GEMINI_MODEL_SUMMARY
        
    try:
        from google import genai  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "google-genai package not installed. Install it with: pip install google-genai"
        ) from exc

    prompt = build_summary_prompt(segments)

    if debug:
        print("=== SUMMARY PROMPT ===")
        print(prompt)
        print("=" * 50)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 200},
    )

    if hasattr(response, "text") and response.text:
        summary = response.text.strip()
    else:
        summary = response.candidates[0].content.parts[0].text.strip()

    if debug:
        print("=== SUMMARY RESPONSE ===")
        print(summary)
        print("=" * 50)

    return summary


def generate_summary_with_gemini_from_prompt(
    prompt: str,
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Generate a summary using the Gemini API from a prompt string.
    
    Args:
        prompt: The complete prompt string to send to Gemini
        api_key: Gemini API key
        model: Model name to use (default: gemini-2.5-flash-lite)
        max_tokens: Maximum tokens to generate (default: 200)
        temperature: Sampling temperature (default: 0.7)
        
    Returns:
        Generated summary text
    """
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-genai package not installed. Install it with: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": temperature, "top_p": 0.9, "max_output_tokens": max_tokens},
    )

    if hasattr(response, "text") and response.text:
        summary = response.text.strip()
    else:
        summary = response.candidates[0].content.parts[0].text.strip()

    return summary


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
        model: Model name to use (default: mistral-medium-latest)
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
        model: Model name to use (default: mistral-medium-latest)
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
    Similar to generate_answers_with_gemini but uses Mistral API.

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
    
    # Log input info
    print(f"\n{'='*60}")
    print(f"🤖 [Mistral Answers] Starting answer generation")
    print(f"   Model: {model}")
    print(f"   Total segments: {len(segments)}")
    answer_segments = [s for s in segments if "Answer" in s]
    question_segments = [s for s in segments if "Question" in s]
    print(f"   Answer segments: {len(answer_segments)}")
    print(f"   Question segments: {len(question_segments)}")
    
    # Show answer types
    for i, seg in enumerate(segments):
        if "Answer" in seg:
            answer_val = seg.get("Answer")
            options = seg.get("AnswerOptions")
            answer_type = "unknown"
            if isinstance(options, list):
                answer_type = "MC"
            elif isinstance(options, dict) or isinstance(answer_val, dict):
                answer_type = "slider"
            elif isinstance(answer_val, str) and answer_val.strip().lower() in {"free_text", "freetext", "free text"}:
                answer_type = "free_text"
            elif isinstance(answer_val, str):
                answer_type = f"text ({answer_val[:30]}...)" if len(str(answer_val)) > 30 else f"text ({answer_val})"
            print(f"   Segment {i}: {answer_type}")
    print(f"{'='*60}")
        
    try:
        from mistralai import Mistral  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "mistralai package not installed. Install it with: pip install mistralai"
        ) from exc

    rng = random.Random(seed)
    mc_answers: Dict[str, List[str]] = {}
    slider_answers: Dict[str, int] = {}

    # Generate random MC and slider answers (same logic as Gemini version)
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
                # Handle slider questions (same logic as Gemini version)
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

    # Always log diagnostic info for debugging
    print(f"\n📊 [Mistral Answers] MC questions found: {len(mc_answers)}")
    print(f"📊 [Mistral Answers] Slider questions found: {len(slider_answers)}")

    # For free text questions, we'll use Mistral to generate answers
    free_text_answers: Dict[str, str] = {}
    free_text_question_indices = []

    # Find free text questions and their indices (same logic as Gemini)
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

    print(f"📊 [Mistral Answers] Free text questions found: {len(free_text_question_indices)}")
    if not free_text_question_indices:
        print("⚠️ [Mistral Answers] No free text questions to generate - Mistral will not be called")

    # Generate free text answers using Mistral with full context
    free_text_answers_by_text: Dict[str, str] = {}
    free_text_answers_by_index: Dict[int, str] = {}  # Track by question index to avoid duplicates
    
    if free_text_question_indices:
        for idx, (question_idx, question_text) in enumerate(free_text_question_indices, start=1):
            try:
                # Build complete context prompt (same as Gemini approach)
                context_prompt = build_gemini_prompt_up_to_question(
                    segments,
                    question_idx,
                    mc_answers,
                    is_target_question=True,
                    free_text_answers=free_text_answers_by_text,
                )
                
                # Replace system prompt if custom one is provided
                if system_prompt:
                    # Find where the actual content starts (after the header)
                    lines = context_prompt.split('\n')
                    content_start = 0
                    for i, line in enumerate(lines):
                        if line.startswith('TEXT:') or line.startswith('FRAGE:'):
                            content_start = i
                            break
                    
                    # Replace everything before content with our custom system prompt
                    if content_start > 0:
                        content_lines = lines[content_start:]
                        context_prompt = system_prompt + '\n\n' + '\n'.join(content_lines)
                    else:
                        # Fallback: just prepend the system prompt
                        context_prompt = system_prompt + '\n\n' + context_prompt
                
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
                # Always log the error to console so it's visible
                print(f"❌ MISTRAL API ERROR: {e}")
                print(f"   Question: {question_text[:60]}...")
                print(f"   Model: {model}")
                if debug:
                    print(f"\n{'─'*40}")
                    print(f"Full error details:")
                    print(f"{'─'*40}")
                    import traceback
                    traceback.print_exc()
                    print(f"{'='*80}\n")
                # Use a fallback answer with error info
                error_msg = f"Fehler: {str(e)[:100]}"
                free_text_answers_by_index[question_idx] = error_msg
                if question_text not in free_text_answers_by_text:
                    free_text_answers_by_text[question_text] = error_msg

    # Merge answers into segments (same logic as Gemini version)
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
        "mc_questions": len(mc_answers),
        "slider_questions": len(slider_answers),
        "free_text_questions": len(free_text_answers_by_index),
        "model_used": model,
        "temperature": temperature,
        "top_p": top_p,
    }

    # Log summary
    print(f"\n{'='*60}")
    print(f"✅ [Mistral Answers] Generation complete")
    print(f"   MC answers generated: {len(mc_answers)}")
    print(f"   Slider answers generated: {len(slider_answers)}")
    print(f"   Free text answers generated: {len(free_text_answers_by_index)}")
    print(f"   Total merged segments: {len(merged_segments)}")
    print(f"{'='*60}\n")

    if return_debug_info:
        return merged_segments, debug_info
    return merged_segments


def extract_json_array_from_gemini_output(output: str) -> List[Dict[str, Any]]:
    """Extract a JSON array from raw Gemini output, tolerant of code fences."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.IGNORECASE)
    try:
        obj = json.loads(stripped)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    match = re.search(r"\[\s*(?:.|\n)*?\]", stripped, re.DOTALL)
    if match:
        fragment = match.group(0)
        try:
            parsed = json.loads(fragment)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    raise ValueError("LLM-Output enthält keine parsebare JSON-Liste.")


def generate_answers_with_gemini(
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
    Generate user answers using Gemini and merge them into the segment structure.

    Args:
        segments: List of segment dictionaries
        api_key: Gemini API key
        model: Gemini model to use
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
        model = GEMINI_MODEL_ANSWERS
        
    try:
        from google import genai  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "google-genai package not installed. Install it with: pip install google-genai"
        ) from exc

    rng = random.Random(seed)
    mc_answers: Dict[str, List[str]] = {}
    slider_answers: Dict[str, int] = {}

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
                                    # Scale max_words (typically 1-40) to selection ratio (0.1-1.0)
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
                                print(f"🎲 MC ({mode}) Question at index {back_index}: "
                                      f"Selected {len(selected)}/{option_count} options{proportional_info}")
                                print(f"   Question: {question_text[:60]}...")
                                print(f"   Selected: {selected}")
                        break
            else:
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

    if debug:
        print(f"\n📊 Total MC questions found: {len(mc_answers)}")
        print(f"📊 Total Slider questions found: {len(slider_answers)}")

    free_text_question_indices: List[Tuple[int, str]] = []
    for index, segment in enumerate(segments):
        if "Answer" in segment and "Question" not in segment:
            answer_val = segment.get("Answer")
            if isinstance(answer_val, str) and answer_val.strip().lower() in {
                "free_text",
                "freetext",
                "free text",
            }:
                for back_index in range(index - 1, -1, -1):
                    question_segment = segments[back_index]
                    if "Question" in question_segment and "Answer" not in question_segment:
                        question_text = question_segment.get("Question")
                        if question_text:
                            free_text_question_indices.append((back_index, question_text))
                        break

    free_text_answers: Dict[Tuple[int, str], str] = {}
    free_text_answers_by_text: Dict[str, str] = {}
    all_raw_outputs: List[str] = []

    if free_text_question_indices:
        client = genai.Client(api_key=api_key)

        for idx, (question_idx, question_text) in enumerate(free_text_question_indices, start=1):
            prompt = build_gemini_prompt_up_to_question(
                segments,
                question_idx,
                mc_answers,
                is_target_question=True,
                free_text_answers=free_text_answers_by_text,
            )
            
            # Replace system prompt if custom one is provided
            if system_prompt:
                # Split prompt into header and content
                lines = prompt.split('\n')
                # Find where the content starts (after the header)
                content_start = 0
                for i, line in enumerate(lines):
                    if line.startswith('TEXT:') or line.startswith('FRAGE:') or line.startswith('ANTWORT:'):
                        content_start = i
                        break
                
                # Replace header with custom system prompt
                content_lines = lines[content_start:]
                prompt = system_prompt + '\n' + '\n'.join(content_lines)

            if debug:
                print(f"\n{'=' * 80}")
                print(f"GEMINI CALL {idx} - QUESTION INDEX {question_idx}")
                print(f"Question Text: {question_text}")
                print(f"{'=' * 80}")
                print("\n--- PROMPT ---")
                print(prompt)
                print("\n--- CALLING GEMINI ---")

            # Try with full prompt first
            max_retries = 2
            retry_count = 0
            response = None
            current_prompt = prompt
            
            while retry_count < max_retries:
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=current_prompt,
                        config={"temperature": temperature, "top_p": top_p, "max_output_tokens": 2000},
                    )
                    
                    # Check if we got a valid response
                    if (response.candidates and 
                        response.candidates[0].content and 
                        hasattr(response.candidates[0].content, 'parts') and 
                        response.candidates[0].content.parts):
                        break  # Success, exit retry loop
                    
                    # If we hit MAX_TOKENS and have no content, try with shorter prompt
                    finish_reason = getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None
                    if str(finish_reason) == 'MAX_TOKENS' and retry_count < max_retries - 1:
                        print(f"Retry {retry_count + 1}: Shortening prompt due to MAX_TOKENS")
                        # Truncate the prompt to roughly half its length
                        lines = current_prompt.split('\n')
                        if len(lines) > 10:
                            # Keep system prompt and first half of content
                            current_prompt = '\n'.join(lines[:len(lines)//2])
                        else:
                            # If already short, just add instruction to be brief
                            current_prompt = current_prompt + "\n\nBitte antworte sehr kurz und prägnant."
                        retry_count += 1
                    else:
                        break  # No more retries or different error
                        
                except Exception as e:
                    print(f"Error calling Gemini API with model '{model}': {e}")
                    if retry_count == max_retries - 1:
                        raise
                    retry_count += 1

            if hasattr(response, "text") and response.text:
                raw_output = response.text.strip()
            else:
                # Add error handling for response structure
                try:
                    if not response.candidates:
                        raise ValueError("No candidates in response")
                    
                    candidate = response.candidates[0]
                    if not candidate.content:
                        raise ValueError("No content in candidate")
                    
                    # Check finish reason first
                    if hasattr(candidate, 'finish_reason'):
                        if str(candidate.finish_reason) == 'MAX_TOKENS':
                            print(f"Warning: Response was truncated due to max tokens limit")
                        elif str(candidate.finish_reason) == 'SAFETY':
                            raise ValueError("Response blocked due to safety filters")
                        elif str(candidate.finish_reason) not in ['STOP', 'MAX_TOKENS']:
                            print(f"Warning: Unexpected finish reason: {candidate.finish_reason}")
                    
                    # Handle missing parts (common when response is truncated)
                    if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                        # Try to get text from alternative locations or provide fallback
                        if hasattr(candidate.content, 'text') and candidate.content.text:
                            raw_output = candidate.content.text.strip()
                        else:
                            # If no content available due to MAX_TOKENS, provide a fallback response
                            finish_reason = getattr(candidate, 'finish_reason', 'unknown')
                            if str(finish_reason) == 'MAX_TOKENS':
                                print(f"Warning: Response completely truncated due to max tokens. Using fallback response.")
                                # Provide a minimal fallback response instead of crashing
                                raw_output = "[Response truncated due to token limit]"
                            else:
                                raise ValueError(f"No content parts available. Finish reason: {finish_reason}")
                    else:
                        if not candidate.content.parts[0].text:
                            raise ValueError("No text in first part")
                        raw_output = candidate.content.parts[0].text.strip()
                    
                except (IndexError, AttributeError, ValueError) as e:
                    print(f"Error accessing response structure: {e}")
                    print(f"Response: {response}")
                    if hasattr(response, 'candidates'):
                        print(f"Candidates: {response.candidates}")
                        if response.candidates:
                            print(f"First candidate: {response.candidates[0]}")
                    raise RuntimeError(f"Failed to extract text from Gemini response: {e}")

            all_raw_outputs.append(raw_output)

            if debug:
                print("\n--- GEMINI RESPONSE ---")
                print(raw_output)
                print(f"{'=' * 80}\n")

            answer_text = re.sub(
                r"^```(?:json|text)?\s*|\s*```$",
                "",
                raw_output,
                flags=re.IGNORECASE,
            ).strip()

            free_text_answers[(question_idx, question_text)] = answer_text
            free_text_answers_by_text[question_text] = answer_text

            if debug:
                preview = answer_text[:50] + ("..." if len(answer_text) > 50 else "")
                print(f"✅ Stored answer for question index {question_idx}: {preview}")

    debug_info: Optional[Dict[str, Any]] = None
    if return_debug_info:
        free_text_answers_debug = {
            f"{idx}: {text}": ans for (idx, text), ans in free_text_answers.items()
        }
        debug_info = {
            "raw_outputs": all_raw_outputs
            if free_text_question_indices
            else ["No free_text questions, skipped Gemini call"],
            "free_text_answers": free_text_answers_debug,
            "free_text_question_indices": [(idx, text) for idx, text in free_text_question_indices],
            "mc_answers": mc_answers,
            "slider_answers": slider_answers,
            "free_text_questions_count": len(free_text_question_indices),
        }

    question_to_answer_idx: Dict[str, int] = {}
    for idx, segment in enumerate(segments):
        if "Answer" in segment and "Question" not in segment:
            for back_index in range(idx - 1, -1, -1):
                question_segment = segments[back_index]
                if "Question" in question_segment and "Answer" not in question_segment:
                    question_text = question_segment.get("Question")
                    if question_text:
                        question_to_answer_idx[question_text] = idx
                    break

    result_segments: List[Dict[str, Any]] = []

    for idx, segment in enumerate(segments):
        if "Text" in segment:
            result_segments.append(segment.copy())
        elif "Question" in segment:
            question_text = segment["Question"]
            question_idx = idx
            new_question_seg = segment.copy()
            result_segments.append(new_question_seg)

            answer_idx = question_to_answer_idx.get(question_text)
            if answer_idx is not None:
                answer_segment = segments[answer_idx]
                original_answer = answer_segment.get("Answer")
                original_options = answer_segment.get("AnswerOptions")
                allow_multiple = answer_segment.get("AllowMultiple")
                new_answer_seg = answer_segment.copy()

                if question_text in mc_answers:
                    if isinstance(original_options, list):
                        new_answer_seg["AnswerOptions"] = original_options
                    new_answer_seg["Answer"] = mc_answers[question_text]
                    if allow_multiple is not None:
                        new_answer_seg["AllowMultiple"] = allow_multiple
                    if debug:
                        print(f"✅ Merged MC answer for question index {question_idx}")
                elif question_text in slider_answers:
                    if isinstance(original_options, dict):
                        new_answer_seg["AnswerOptions"] = original_options.copy()
                    elif isinstance(original_answer, dict):
                        new_answer_seg["AnswerOptions"] = original_answer.copy()
                    new_answer_seg["Answer"] = slider_answers[question_text]
                    if debug:
                        print(f"✅ Merged slider answer for question index {question_idx}")
                elif (
                    isinstance(original_answer, str)
                    and original_answer.strip().lower() in {"free_text", "freetext", "free text"}
                ):
                    generated_answer = free_text_answers.get((question_idx, question_text))
                    if generated_answer is not None:
                        new_answer_seg["Answer"] = generated_answer
                        if debug:
                            preview = generated_answer[:50] + (
                                "..." if len(generated_answer) > 50 else ""
                            )
                            print(f"✅ Matched answer for question index {question_idx}: {preview}")
                elif isinstance(original_options, list) and len(original_options) > 0:
                    if debug:
                        print(f"⚠️ MC question at index {question_idx} not found in mc_answers")
                result_segments.append(new_answer_seg)
        elif "Answer" in segment and "Question" not in segment:
            continue
        else:
            result_segments.append(segment.copy())

    if return_debug_info:
        return result_segments, debug_info  # type: ignore[return-value]
    return result_segments


__all__ = [
    "generate_summary_with_gemini",
    "generate_summary_with_gemini_from_prompt",
    "generate_summary_with_mistral",
    "stream_summary_with_mistral",
    "extract_json_array_from_gemini_output",
    "generate_answers_with_gemini",
    "generate_answers_with_mistral",
]


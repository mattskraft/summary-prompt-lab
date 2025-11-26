"""Cloud API integrations for answer generation and summarisation (Gemini and Mistral)."""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .prompts import build_gemini_prompt_up_to_question, build_summary_prompt


def generate_summary_with_gemini(
    segments: List[Dict[str, Any]],
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
    debug: bool = False,
) -> str:
    """Generate a therapeutic summary using the Gemini API from segments."""
    try:
        from google import genai
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
        from google import genai
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
    model: str = "mistral-small-latest",
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
    try:
        from mistralai import Mistral
    except ImportError as exc:
        raise ImportError(
            "mistralai package not installed. Install it with: pip install mistralai"
        ) from exc

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
    model: str = "gemini-2.5-flash",
    temperature: float = 0.9,
    top_p: float = 0.8,
    debug: bool = False,
    return_debug_info: bool = False,
    seed: Optional[int] = None,
    system_prompt: Optional[str] = None,
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

    Returns:
        The merged segments and optionally debug information.
    """
    try:
        from google import genai
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
                                num_selections = rng.randint(1, option_count)
                                selected = rng.sample(options, num_selections)
                            else:
                                selected = [rng.choice(options)]
                            mc_answers[question_text] = selected
                            if debug:
                                mode = "Mehrfach" if allow_multiple else "Einfach"
                                print(f"🎲 MC ({mode}) Question at index {back_index}: "
                                      f"Selected {len(selected)}/{option_count} options")
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

            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"temperature": temperature, "top_p": top_p, "max_output_tokens": 1000},
                )
            except Exception as e:
                print(f"Error calling Gemini API with model '{model}': {e}")
                raise

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
                            # If no content available, provide a meaningful error
                            raise ValueError(f"No content parts available. Finish reason: {getattr(candidate, 'finish_reason', 'unknown')}")
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
    "extract_json_array_from_gemini_output",
    "generate_answers_with_gemini",
]


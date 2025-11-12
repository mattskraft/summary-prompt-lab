"""Helper utilities for processing user input structures for MVP prototypes."""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union


def _sanitize_text(raw_text: str) -> str:
    """Remove underscores, asterisks, and line breaks while normalizing whitespace."""
    cleaned = raw_text.replace("\n\n", " ")
    cleaned = cleaned.replace("_", "").replace("*", "")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _normalize_entries(container: Any) -> List[Dict[str, Any]]:
    """
    Try to extract a list of entry dictionaries from various container shapes:
    - already a list of dictionaries
    - dictionary with lists under 'blocks', 'items', or 'content'
    - single story node that resembles a question/answer structure
    """
    if isinstance(container, list):
        return container
    if isinstance(container, dict):
        for key in ("blocks", "items", "content"):
            if isinstance(container.get(key), list):
                return container[key]
        if any(
            key in container
            for key in ("Text", "Question", "Answer", "Branch")
        ):
            return [container]
    return []


def _normalize_answer_value(answer: Any) -> Any:
    """
    Normalize "Answer" values into consistent structures while sanitizing strings.
    - string → sanitized string or None
    - list → list of sanitized strings (empty entries removed)
    - dict → preserve keys, sanitize string values
    """
    if answer is None:
        return None
    if isinstance(answer, str):
        sanitized = _sanitize_text(answer)
        return sanitized if sanitized else None
    if isinstance(answer, list):
        sanitized_items: List[str] = []
        for element in answer:
            sanitized = _sanitize_text(str(element))
            if sanitized:
                sanitized_items.append(sanitized)
        return sanitized_items if sanitized_items else None
    if isinstance(answer, dict):
        normalized: Dict[str, Any] = {}
        for key, value in answer.items():
            if isinstance(value, str):
                sanitized = _sanitize_text(value)
                if sanitized:
                    normalized[key] = sanitized
            else:
                normalized[key] = value
        return normalized if normalized else None
    return answer


def _emit_segments_from_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single entry into a sequence of segments (Text, Question, Answer).
    """
    segments: List[Dict[str, Any]] = []

    text_value = entry.get("Text")
    if isinstance(text_value, str):
        sanitized_text = _sanitize_text(text_value)
        if sanitized_text:
            segments.append({"Text": sanitized_text})

    question_value = entry.get("Question", entry.get("question"))
    if isinstance(question_value, str):
        sanitized_question = _sanitize_text(question_value)
        if sanitized_question:
            segments.append({"Question": sanitized_question})

    answer_handled = False

    # New schema: explicit AnswerOptions (+ optional AllowMultiple/Answer)
    if isinstance(entry.get("AnswerOptions"), list):
        sanitized_options: List[str] = []
        for opt in entry.get("AnswerOptions", []):
            sanitized_opt = _sanitize_text(str(opt))
            if sanitized_opt:
                sanitized_options.append(sanitized_opt)

        allow_multiple = bool(entry.get("AllowMultiple", True))

        selected_raw = entry.get("Answer")
        selected_normalized = _normalize_answer_value(selected_raw) if selected_raw is not None else None
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

    elif isinstance(entry.get("AnswerOptions"), dict):
        normalized_options = _normalize_answer_value(entry.get("AnswerOptions"))
        if normalized_options is not None:
            normalized_answer = _normalize_answer_value(entry.get("Answer"))
            segments.append(
                {
                    "Answer": normalized_answer,
                    "AnswerOptions": normalized_options,
                }
            )
            answer_handled = True
    elif isinstance(entry.get("AnswerOptions"), str):
        # Treat string "free_text" (or variants) as a free text placeholder
        ao_str = str(entry.get("AnswerOptions")).strip().lower()
        if ao_str in {"free_text", "freetext", "free text"}:
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

    normalized_answer = _normalize_answer_value(raw_answer)
    if normalized_answer is not None:
        segments.append({"Answer": normalized_answer})

    return segments


def _process_branch(
    branch: Dict[str, Any],
    story_nodes: Dict[str, Any],
    rng: random.Random,
    max_depth: int = 5,
) -> List[Dict[str, Any]]:
    """
    Select a random branch option, append the choice, and process the referenced story node.
    """
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
    sanitized_choice = _sanitize_text(str(selected_key))
    if sanitized_choice:
        segments.append({"Answer": sanitized_choice})

    node_data = story_nodes.get(selected_node_name)
    if node_data is None:
        return segments

    entries = _normalize_entries(node_data)
    for nd in entries:
        segments.extend(_emit_segments_from_entry(nd))
        if "Branch" in nd and isinstance(nd["Branch"], (dict, list)):
            segments.extend(
                _process_branch(nd["Branch"], story_nodes, rng, max_depth=max_depth - 1)
            )

    return segments


def get_prompt_segments_from_exercise(
    exercise_name: str,
    json_struct_path: str,
    json_sn_struct_path: str,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Gather normalized segments for an exercise based on the provided JSON structures."""
    rng = random.Random(seed)

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

    entries = _normalize_entries(exercise_container)
    segments: List[Dict[str, Any]] = []

    for entry in entries:
        segments.extend(_emit_segments_from_entry(entry))
        if "Branch" in entry and isinstance(entry["Branch"], (dict, list)):
            segments.extend(_process_branch(entry["Branch"], sn_data, rng))

    return segments


def _format_answer_for_text(answer: Any) -> Optional[str]:
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
            answer_text = _format_answer_for_text(segment["Answer"])
            if answer_text:
                lines.append(f"ANSWER: {answer_text}")
    return "\n".join(line.replace("\n", " ").strip() for line in lines if line.strip())


def build_gemini_prompt_up_to_question(
    segments: List[Dict[str, Any]], 
    question_index: int,
    mc_answers: Dict[str, List[str]],
    is_target_question: bool = False,
    free_text_answers: Optional[Dict[str, str]] = None
) -> str:
    """
    Build a prompt for Gemini API with all segments up to and including a specific question.
    Includes all texts, questions, and answers (with MC answers filled in) up to the target question.
    If is_target_question is True, the answer for the target question is not included (Gemini should generate it).
    free_text_answers: Dictionary of already-generated free_text answers to include in the prompt.
    """
    if free_text_answers is None:
        free_text_answers = {}
    
    header = (
        "Im folgenden gibt es eine Abfolge von Texten, Fragen und Antworten.\n"
        "Formuliere eine realistische Antwort für die letzte Frage in 1-3 Sätzen.\n"
        "Formuliere NUR die Antwort, keine weiteren Texte oder Struktur.\n"
        "Sprache: Deutsch, natürlich klingend.\n"
    )
    lines = []
    
    # Process segments up to and including the question at question_index
    for i in range(question_index + 1):
        seg = segments[i]
        if "Text" in seg:
            clean_text = " ".join(seg["Text"].split())
            lines.append(f"TEXT: {clean_text}")
        elif "Question" in seg:
            question_text = seg["Question"]
            lines.append(f'FRAGE: {question_text}')
            
            # For the target question, don't include the answer (Gemini should generate it)
            if is_target_question and i == question_index:
                # Don't add answer - Gemini should generate it
                continue
            
            # Find the corresponding Answer segment
            answer_text = None
            for j in range(i + 1, len(segments)):
                if "Answer" in segments[j] and "Question" not in segments[j]:
                    answer_val = segments[j].get("Answer")
                    
                    # Check if it's a MC question with random answer
                    if question_text in mc_answers:
                        selected = mc_answers[question_text]
                        answer_text = ", ".join(selected)
                        break
                    # Check if it's free_text
                    elif isinstance(answer_val, str) and answer_val.strip().lower() in {"free_text", "freetext", "free text"}:
                        # If this is the target question, don't include answer
                        if is_target_question and i == question_index:
                            break
                        # If we have a generated answer for this question, use it
                        if free_text_answers and question_text in free_text_answers:
                            answer_text = free_text_answers[question_text]
                        else:
                            # Otherwise, it's still free_text (not yet generated)
                            answer_text = "free_text"
                        break
                    # Otherwise it's already a concrete answer
                    elif isinstance(answer_val, str):
                        answer_text = answer_val
                        break
                    elif isinstance(answer_val, list):
                        answer_text = ", ".join(map(str, answer_val))
                        break
            
            if answer_text:
                lines.append(f"ANTWORT: {answer_text}")
    
    return header + "\n" + "\n".join(lines)


def build_summary_prompt(segments: List[Dict[str, Any]]) -> str:
    """
    Build a prompt for summarizing an exercise session.
    
    Args:
        segments: List of segment dictionaries (Text/Question/Answer) with filled answers
        
    Returns:
        Formatted prompt string with SYSTEM and INHALT sections
    """
    system_prompt = (
        "Du bist eine therapeutische Assistenz.\n"
        "Fasse den folgenden INHALT zusammen.\n"
        "Beziehe die TEXT-Blöcke und die FRAGE-Blöcke als Kontext für die Zusammenfassung mit ein.\n"
        "Aber fasse hauptsächlich die ANTWORT-Blöcke zusammen.\n"
        "Nutze ausschließlich die vorhandenen Texte, Fragen und Antworten. Erfinde keine neuen Inhalte.\n"
        "Benutze warmen, empathischen Sprache. Sprich die antwortende Person in der Du-Form an.\n"
        "Antworte auf Deutsch. Maximal 120 Wörter."
    )
    
    content_lines = []
    
    for seg in segments:
        if "Text" in seg:
            clean_text = " ".join(seg["Text"].split())
            content_lines.append(f"TEXT: {clean_text}")
        elif "Question" in seg:
            question_text = seg["Question"]
            content_lines.append(f"FRAGE: {question_text}")
        elif "Answer" in seg:
            answer_val = seg.get("Answer")
            answer_text = None
            
            # Handle different answer types
            if isinstance(answer_val, list):
                # MC question - show selected options
                answer_text = ", ".join(str(item) for item in answer_val)
            elif isinstance(answer_val, (int, float)):
                # Slider question - show the value
                answer_text = str(int(answer_val))
            elif isinstance(answer_val, str):
                # Free text or other string answer
                answer_text = answer_val.strip()
            elif isinstance(answer_val, dict):
                # Slider with min/max dict - extract the value
                if "min" in answer_val and "max" in answer_val:
                    min_val = answer_val.get("min")
                    max_val = answer_val.get("max")
                    # If min == max, it's a single value
                    if min_val == max_val:
                        answer_text = str(int(min_val))
                    else:
                        answer_text = f"{int(min_val)} - {int(max_val)}"
                else:
                    # Other dict format
                    answer_text = _format_answer_for_text(answer_val)
            
            if answer_text:
                content_lines.append(f"ANTWORT: {answer_text}")
    
    return ("-" * 80) + f"\nSYSTEM:\n\n{system_prompt}\n\n" + ("-" * 80) + "\nINHALT:\n\n" + "\n\n".join(content_lines)


def generate_summary_with_gemini(
    segments: List[Dict[str, Any]],
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
    debug: bool = False,
) -> str:
    """
    Generate a therapeutic summary using Gemini API.
    
    Args:
        segments: List of segment dictionaries (Text/Question/Answer) with filled answers
        api_key: Gemini API key
        model: Model name to use
        debug: Print debug info to console
        
    Returns:
        Generated summary text
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai package not installed. Install it with: pip install google-genai")
    
    # Build the summary prompt
    prompt = build_summary_prompt(segments)
    
    if debug:
        print("=== SUMMARY PROMPT ===")
        print(prompt)
        print("=" * 50)
    
    # Initialize client and call Gemini
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 200},
    )
    
    # Extract response text
    if hasattr(resp, "text") and resp.text:
        summary = resp.text.strip()
    else:
        summary = resp.candidates[0].content.parts[0].text.strip()
    
    if debug:
        print("=== SUMMARY RESPONSE ===")
        print(summary)
        print("=" * 50)
    
    return summary


def extract_json_array_from_gemini_output(output: str) -> List[Dict[str, Any]]:
    """
    Extract JSON array from Gemini API output.
    Removes code fences and extracts the first valid JSON list.
    Handles both {"Frage": "...", "Antwort": "..."} and {"Question": "...", "GeneratedUserAnswer": "..."} formats.
    """
    s = output.strip()
    # Remove code fences
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    # Try direct parsing
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # Fallback: extract first [...] sequence
    m = re.search(r"\[\s*(?:.|\n)*?\]", s, re.DOTALL)
    if m:
        frag = m.group(0)
        try:
            return json.loads(frag)
        except Exception:
            pass
    raise ValueError("LLM-Output enthält keine parsebare JSON-Liste.")


def generate_answers_with_gemini(
    segments: List[Dict[str, Any]],
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
    debug: bool = False,
    return_debug_info: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Generate user answers using Gemini API and merge them back into segments.
    
    Args:
        segments: List of segment dictionaries (Text/Question/Answer)
        api_key: Gemini API key
        model: Model name to use
        debug: Print debug info to console
        return_debug_info: Return debug info as second return value
        
    Returns:
        List of segments with generated answers merged in, and optionally debug info
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai package not installed. Install it with: pip install google-genai")
    
    # Step 1: Randomly answer multiple choice questions (1 to N-1 choices for N options)
    # and slider questions (random min/max within range)
    # Note: segments are separate - Question and Answer are in different segments
    rng = random.Random()
    mc_answers: Dict[str, List[str]] = {}  # question -> selected answers
    slider_answers: Dict[str, int] = {}  # question -> selected value
    
    for i, seg in enumerate(segments):
        if "Answer" in seg and "Question" not in seg:
            options = seg.get("AnswerOptions")
            # Multiple choice questions with explicit options
            if isinstance(options, list) and options:
                allow_multiple = bool(seg.get("AllowMultiple", True))
                for j in range(i - 1, -1, -1):
                    if "Question" in segments[j] and "Answer" not in segments[j]:
                        question_text = segments[j].get("Question")
                        if question_text:
                            n = len(options)
                            if allow_multiple:
                                num_selections = rng.randint(1, n)
                                selected = rng.sample(options, num_selections)
                            else:
                                selected = [rng.choice(options)]
                            mc_answers[question_text] = selected
                            if debug:
                                mode = "Mehrfach" if allow_multiple else "Einfach"
                                print(f"🎲 MC ({mode}) Question at index {j}: Selected {len(selected)}/{n} options")
                                print(f"   Question: {question_text[:60]}...")
                                print(f"   Selected: {selected}")
                        break
            else:
                # Slider questions stored either in AnswerOptions or Answer as dict
                slider_config = None
                if isinstance(options, dict):
                    slider_config = options
                elif isinstance(seg.get("Answer"), dict):
                    slider_config = seg.get("Answer")

                if isinstance(slider_config, dict):
                    min_numeric = slider_config.get("min")
                    max_numeric = slider_config.get("max")
                    try:
                        min_int = int(float(min_numeric)) if min_numeric is not None else None
                        max_int = int(float(max_numeric)) if max_numeric is not None else None
                        if min_int is not None and max_int is not None and min_int <= max_int:
                            for j in range(i - 1, -1, -1):
                                if "Question" in segments[j] and "Answer" not in segments[j]:
                                    question_text = segments[j].get("Question")
                                    if question_text:
                                        selected_value = rng.randint(min_int, max_int)
                                        slider_answers[question_text] = selected_value
                                        if debug:
                                            print(
                                                f"🎲 Slider Question at index {j}: Selected value {selected_value} "
                                                f"(from {min_int}-{max_int})"
                                            )
                                            print(f"   Question: {question_text[:60]}...")
                                    break
                    except (TypeError, ValueError):
                        pass
    
    if debug:
        print(f"\n📊 Total MC questions found: {len(mc_answers)}")
        print(f"📊 Total Slider questions found: {len(slider_answers)}")
    
    # Step 2: Find free_text questions and their indices
    # Note: segments are separate - Question and Answer are in different segments
    # Use (question_index, question_text) tuple as key to avoid collisions from duplicate question texts
    free_text_question_indices: List[Tuple[int, str]] = []  # List of (question_index, question_text) tuples
    question_index_to_text: Dict[int, str] = {}  # Map question index to question text for exact matching
    
    for i, seg in enumerate(segments):
        # Check if this is an Answer segment with free_text
        if "Answer" in seg and "Question" not in seg:
            answer_val = seg.get("Answer")
            if isinstance(answer_val, str) and answer_val.strip().lower() in {"free_text", "freetext", "free text"}:
                # Find the preceding Question segment
                for j in range(i - 1, -1, -1):
                    if "Question" in segments[j] and "Answer" not in segments[j]:
                        question_text = segments[j].get("Question")
                        if question_text:
                            free_text_question_indices.append((j, question_text))
                            question_index_to_text[j] = question_text
                        break
    
    # Step 3: Call Gemini for each free_text question individually
    # Process them sequentially so each prompt includes answers from previous Gemini calls
    # Use (question_index, question_text) tuple as key to ensure unique matching
    free_text_answers: Dict[Tuple[int, str], str] = {}  # (question_index, question_text) -> generated answer
    free_text_answers_by_text: Dict[str, str] = {}  # question_text -> generated answer (for prompt building)
    all_raw_outputs: List[str] = []
    
    if free_text_question_indices:
        # Initialize client once
        client = genai.Client(api_key=api_key)
        
        # Loop through each free_text question in order
        for question_idx, question_text in free_text_question_indices:
            # Build prompt with all segments up to and including this question
            # Include previous free_text answers that have already been generated
            prompt = build_gemini_prompt_up_to_question(
                segments, question_idx, mc_answers, 
                is_target_question=True,
                free_text_answers=free_text_answers_by_text
            )
            
            if debug:
                print(f"\n{'='*80}")
                print(f"GEMINI CALL {len(free_text_answers) + 1} - QUESTION INDEX {question_idx}")
                print(f"Question Text: {question_text}")
                print(f"{'='*80}")
                print("\n--- PROMPT ---")
                print(prompt)
                print("\n--- CALLING GEMINI ---")
            
            # Call Gemini
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0.9, "top_p": 0.8, "max_output_tokens": 200},
            )
            
            # Extract response text
            if hasattr(resp, "text") and resp.text:
                raw_output = resp.text.strip()
            else:
                raw_output = resp.candidates[0].content.parts[0].text.strip()
            
            all_raw_outputs.append(raw_output)
            
            if debug:
                print("\n--- GEMINI RESPONSE ---")
                print(raw_output)
                print(f"{'='*80}\n")
            
            # Parse response as plain text (not JSON)
            # Remove any code fences or extra formatting
            answer_text = raw_output
            answer_text = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", answer_text, flags=re.IGNORECASE)
            answer_text = answer_text.strip()
            
            # Store the answer using both key formats
            # Use tuple key for exact matching by index
            free_text_answers[(question_idx, question_text)] = answer_text
            # Use text key for prompt building (may have duplicates, but that's OK for context)
            free_text_answers_by_text[question_text] = answer_text
            
            if debug:
                print(f"✅ Stored answer for question index {question_idx}: {answer_text[:50]}...")
                print(f"=== PARSED ANSWER FOR QUESTION {question_idx} ===")
                print(answer_text)
                print("=" * 50)
    
    # Step 4: Merge both random MC answers and Gemini free_text answers back into segments
    result_segments = []
    debug_info = None
    
    if return_debug_info:
        # Convert tuple keys to strings for JSON serialization
        free_text_answers_debug = {f"{idx}: {text}": ans for (idx, text), ans in free_text_answers.items()}
        debug_info = {
            "raw_outputs": all_raw_outputs if free_text_question_indices else ["No free_text questions, skipped Gemini call"],
            "free_text_answers": free_text_answers_debug,
            "free_text_question_indices": [(idx, text) for idx, text in free_text_question_indices],
            "mc_answers": mc_answers,
            "slider_answers": slider_answers,
            "free_text_questions_count": len(free_text_question_indices),
        }
    
    # Build a map of question -> answer segment index for easier lookup
    question_to_answer_idx: Dict[str, int] = {}
    for i, seg in enumerate(segments):
        if "Answer" in seg and "Question" not in seg:
            # Find the preceding Question segment
            for j in range(i - 1, -1, -1):
                if "Question" in segments[j] and "Answer" not in segments[j]:
                    question_text = segments[j].get("Question")
                    if question_text:
                        question_to_answer_idx[question_text] = i
                    break
    
    for i, seg in enumerate(segments):
        if "Text" in seg:
            result_segments.append(seg.copy())
        elif "Question" in seg:
            question_text = seg["Question"]
            question_idx = i  # Current index is the question index
            new_seg = seg.copy()
            result_segments.append(new_seg)
            
            # Check if there's a corresponding Answer segment
            answer_idx = question_to_answer_idx.get(question_text)
            if answer_idx is not None:
                answer_seg = segments[answer_idx]
                original_answer = answer_seg.get("Answer")
                original_options = answer_seg.get("AnswerOptions")
                allow_multiple = answer_seg.get("AllowMultiple")
                new_answer_seg = answer_seg.copy()
                
                # Check if it's a MC question (has random answer)
                if question_text in mc_answers:
                    # MC question: use randomly selected answers
                    if isinstance(original_options, list):
                        new_answer_seg["AnswerOptions"] = original_options
                    new_answer_seg["Answer"] = mc_answers[question_text]  # Randomly selected answers
                    if allow_multiple is not None:
                        new_answer_seg["AllowMultiple"] = allow_multiple
                    if debug:
                        print(f"✅ Merged MC answer for question index {question_idx}")
                        print(f"   AnswerOptions: {len(original_options) if isinstance(original_options, list) else 'n/a'} options")
                        print(f"   Answer (selected): {mc_answers[question_text]}")
                
                # Check if it's a slider question (has random answer)
                elif question_text in slider_answers:
                    # Slider question: use randomly selected value and preserve options
                    if isinstance(original_options, dict):
                        new_answer_seg["AnswerOptions"] = original_options.copy()
                    elif isinstance(original_answer, dict):
                        new_answer_seg["AnswerOptions"] = original_answer.copy()
                    new_answer_seg["Answer"] = slider_answers[question_text]
                    if debug:
                        print(f"✅ Merged slider answer for question index {question_idx}")
                        print(f"   Value: {slider_answers[question_text]}")
                
                # Check if it's a free_text question (has Gemini answer)
                elif isinstance(original_answer, str) and original_answer.strip().lower() in {"free_text", "freetext", "free text"}:
                    # Get the generated answer using (question_index, question_text) tuple for exact matching
                    generated_answer = free_text_answers.get((question_idx, question_text))
                    
                    if generated_answer is not None:
                        new_answer_seg["Answer"] = generated_answer
                        if debug:
                            print(f"✅ Matched answer for question index {question_idx}: {generated_answer[:50]}...")
                    else:
                        if debug:
                            print(f"⚠️ No answer found for question index {question_idx}, text: {question_text[:50]}...")
                            # Debug: show what keys we have
                            print(f"   Available keys: {list(free_text_answers.keys())[:3]}...")
                    # If no answer found, keep as free_text (user can fill manually)
                
                # Debug: check if this should have been an MC question but wasn't found
                elif isinstance(original_options, list) and len(original_options) > 0:
                    if debug:
                        print(f"⚠️ MC question at index {question_idx} not found in mc_answers")
                        print(f"   Question text: {question_text[:60]}...")
                        print(f"   Available MC keys: {list(mc_answers.keys())[:2]}...")
                
                result_segments.append(new_answer_seg)
        elif "Answer" in seg and "Question" not in seg:
            # This Answer segment should have been handled when we processed its Question
            # Skip it here to avoid duplicates
            pass
        else:
            # Keep other segments as-is (e.g., Answer segments from branches)
            result_segments.append(seg.copy())
    
    if return_debug_info:
        return result_segments, debug_info
    return result_segments


__all__ = [
    "get_prompt_segments_from_exercise",
    "prompt_segments_to_text",
    "build_gemini_prompt_up_to_question",
    "build_summary_prompt",
    "generate_summary_with_gemini",
    "extract_json_array_from_gemini_output",
    "generate_answers_with_gemini",
]


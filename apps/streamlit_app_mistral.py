import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import from the kiso_input package
try:
    from kiso_input.config import (
        STRUCT_JSON_PATH,
        SN_JSON_PATH,
        SAFETY_LEXICON_PATH,
    )
    import kiso_input.config as config_module
    
    # Get APP_PASSWORD from Streamlit secrets first, then fallback to config/.env
    APP_PASSWORD = None
    PASSWORD_FROM_SECRETS = False
    
    APP_PASSWORD = getattr(config_module, "APP_PASSWORD", None)
    
    try:
        if hasattr(st, "secrets"):
            try:
                _ = len(st.secrets)
                if "APP_PASSWORD" in st.secrets:
                    APP_PASSWORD = st.secrets["APP_PASSWORD"]
                    PASSWORD_FROM_SECRETS = True
                elif hasattr(st.secrets, "get"):
                    app_pwd = st.secrets.get("APP_PASSWORD")
                    if app_pwd:
                        APP_PASSWORD = app_pwd
                        PASSWORD_FROM_SECRETS = True
            except Exception:
                pass
    except (AttributeError, KeyError, TypeError):
        pass
    
    # Get GEMINI_API_KEY
    GEMINI_API_KEY = None
    try:
        if hasattr(st, "secrets"):
            try:
                _ = len(st.secrets)
                GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or None
            except Exception:
                pass
    except (AttributeError, KeyError, TypeError):
        pass
    
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = getattr(config_module, "GEMINI_API_KEY", None)
    
    # Get MISTRAL_API_KEY
    MISTRAL_API_KEY = None
    try:
        if hasattr(st, "secrets"):
            try:
                _ = len(st.secrets)
                MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY") or None
            except Exception:
                pass
    except (AttributeError, KeyError, TypeError):
        pass
    
    if not MISTRAL_API_KEY:
        MISTRAL_API_KEY = getattr(config_module, "MISTRAL_API_KEY", None)
    
    # Load synth_answers_prompt.txt for Gemini system prompt
    SYNTH_ANSWERS_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "synth_answers_prompt.txt"
    SYNTH_ANSWERS_PROMPT = ""
    if SYNTH_ANSWERS_PROMPT_PATH.exists():
        SYNTH_ANSWERS_PROMPT = SYNTH_ANSWERS_PROMPT_PATH.read_text(encoding="utf-8").strip()
    
    from kiso_input import (
        assess_free_text_answers,
        generate_answers_with_gemini,
        generate_answers_with_mistral,
        get_prompt_segments_from_exercise,
        load_self_harm_lexicon,
    )
    from kiso_input.processing.cloud_apis import (
        generate_summary_with_mistral,
    )
    from kiso_input.processing.recap_sections import (
        assemble_system_prompt,
        choose_max_words,
        get_exercise_sections,
        load_global_section,
        load_global_sections,
        load_word_limit_config,
        save_exercise_sections,
        save_global_section,
        save_word_limit_config,
    )
except ImportError as e:
    st.error(f"""
    ❌ Import error: {e}
    
    Please install the package in development mode:
    ```bash
    cd {PROJECT_ROOT}
    pip install -e .
    ```
    
    Or ensure the src directory is in your Python path.
    """)
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


EXERCISE_PROMPTS_STORE = PROJECT_ROOT / "config" / "prompts" / "exercise_specific_prompts.json"
SUMMARY_SWITCH_PATH = PROJECT_ROOT / "config" / "prompts" / "summary_switch.json"
SECTION_UI_CONFIG = [
    {"key": "rolle", "label": "Rolle & Aufgabe", "scope": "global"},
    {"key": "eingabeformat", "label": "Eingabeformat", "scope": "global"},
    {"key": "anweisungen", "label": "Anweisungen", "scope": "exercise"},
    {"key": "stil", "label": "Stil & Ton", "scope": "global"},
    {"key": "ausgabeformat", "label": "Ausgabeformat", "scope": "exercise"},
]
GLOBAL_SECTION_KEYS = [cfg["key"] for cfg in SECTION_UI_CONFIG if cfg["scope"] == "global"]
EXERCISE_SECTION_EDITOR_KEYS = ["anweisungen", "ausgabeformat"]
WORD_LIMIT_COLUMNS = 3
WORD_LIMIT_SESSION_PREFIX = "recap_word_limits"


def ensure_exercise_prompt_store() -> Path:
    EXERCISE_PROMPTS_STORE.parent.mkdir(parents=True, exist_ok=True)
    if not EXERCISE_PROMPTS_STORE.exists():
        EXERCISE_PROMPTS_STORE.write_text("{}", encoding="utf-8")
    return EXERCISE_PROMPTS_STORE


def ensure_summary_switch_store() -> Path:
    SUMMARY_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SUMMARY_SWITCH_PATH.exists():
        SUMMARY_SWITCH_PATH.write_text("{}", encoding="utf-8")
    return SUMMARY_SWITCH_PATH


def load_summary_switch_config() -> Dict[str, str]:
    ensure_summary_switch_store()
    try:
        raw = SUMMARY_SWITCH_PATH.read_text(encoding="utf-8").strip() or "{}"
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        st.warning(f"Ungültige summary_switch.json: {exc}")
        return {}


def save_summary_switch_config(config: Dict[str, str]) -> None:
    ensure_summary_switch_store()
    SUMMARY_SWITCH_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summary_switch_is_on(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.strip().lower().startswith("summary")


def confirm_action(dialog_key: str, message: str, on_confirm) -> None:
    """Show a confirmation dialog and call on_confirm when accepted."""
    if not st.session_state.get(dialog_key):
        return
    confirmation_box = st.container()
    with confirmation_box:
        st.write(f"**Bestätigung erforderlich**\n\n{message}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ja", key=f"{dialog_key}_yes", use_container_width=True):
                try:
                    on_confirm()
                finally:
                    st.session_state[dialog_key] = False
                st.rerun()
        with col2:
            if st.button("Nein", key=f"{dialog_key}_no", use_container_width=True):
                st.session_state[dialog_key] = False
                st.rerun()


def section_state_key(section_key: str, scope: str, session_key: str) -> str:
    if scope == "global":
        return f"global_section_{section_key}"
    return f"{session_key}_{section_key}_section"


def section_exercise_tracker_key(section_state: str) -> str:
    return f"{section_state}_exercise"


def initialize_word_limit_inputs(config: Dict[str, List[int]]) -> None:
    answers = config["answer_counts"]
    words = config["max_words"]
    for idx in range(WORD_LIMIT_COLUMNS):
        ans_key = f"{WORD_LIMIT_SESSION_PREFIX}_answers_{idx}"
        words_key = f"{WORD_LIMIT_SESSION_PREFIX}_words_{idx}"
        if ans_key not in st.session_state:
            st.session_state[ans_key] = int(answers[idx])
        if words_key not in st.session_state:
            st.session_state[words_key] = int(words[idx])


def current_word_limit_config() -> Dict[str, List[int]]:
    answers = [int(st.session_state[f"{WORD_LIMIT_SESSION_PREFIX}_answers_{idx}"]) for idx in range(WORD_LIMIT_COLUMNS)]
    words = [int(st.session_state[f"{WORD_LIMIT_SESSION_PREFIX}_words_{idx}"]) for idx in range(WORD_LIMIT_COLUMNS)]
    pairs = sorted(zip(answers, words), key=lambda item: item[0])
    return {
        "answer_counts": [pair[0] for pair in pairs],
        "max_words": [pair[1] for pair in pairs],
    }


def save_exercise_payload(exercise_name: str, payload: Dict[str, str]) -> bool:
    try:
        save_exercise_sections(exercise_name, payload)
        return True
    except Exception as exc:
        st.error(f"Fehler beim Speichern: {exc}")
        return False


def collect_global_section_values(session_key: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for cfg in SECTION_UI_CONFIG:
        if cfg["scope"] != "global":
            continue
        key = section_state_key(cfg["key"], "global", session_key)
        values[cfg["key"]] = st.session_state.get(key, "")
    return values


def collect_exercise_section_values(session_key: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for section_key in EXERCISE_SECTION_EDITOR_KEYS:
        key = section_state_key(section_key, "exercise", session_key)
        values[section_key] = st.session_state.get(key, "")
    return values


def print_prompt_debug(context: str, prompt: str, params: Dict[str, Any]) -> None:
    print(f"[Recap:{context}] Prompt:\n{prompt}\n")  # noqa: T201
    print(f"[Recap:{context}] Params: {params}\n")  # noqa: T201




def initialize_section_states(
    sel_uebung: str,
    session_key: str,
    global_defaults: Dict[str, str],
    exercise_sections: Dict[str, str],
) -> None:
    for cfg in SECTION_UI_CONFIG:
        state_key = section_state_key(cfg["key"], cfg["scope"], session_key)
        if cfg["scope"] == "global":
            if state_key not in st.session_state:
                st.session_state[state_key] = global_defaults.get(cfg["key"], "")
            continue
        tracker_key = section_exercise_tracker_key(state_key)
        if st.session_state.get(tracker_key) != sel_uebung:
            st.session_state[state_key] = exercise_sections.get(cfg["key"], "")
            st.session_state[tracker_key] = sel_uebung


def get_all_exercise_names(hierarchy: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """Extract all exercise names from the hierarchy."""
    exercise_names: List[str] = []
    for thema_dict in hierarchy.values():
        for path_list in thema_dict.values():
            exercise_names.extend(path_list)
    return sorted(set(exercise_names))


def get_all_exercises_with_paths(hierarchy: Dict[str, Dict[str, List[str]]]) -> List[Tuple[str, str, str]]:
    """Get all exercises as (thema, pfad, uebung) tuples in sorted order."""
    exercises: List[Tuple[str, str, str]] = []
    for thema in sorted(hierarchy.keys()):
        for pfad in sorted(hierarchy[thema].keys()):
            for uebung in sorted(hierarchy[thema][pfad]):
                exercises.append((thema, pfad, uebung))
    return exercises


def get_next_exercise(hierarchy: Dict[str, Dict[str, List[str]]], current_uebung: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """Get the next exercise after the current one, wrapping around if at the end."""
    all_exercises = get_all_exercises_with_paths(hierarchy)
    if not all_exercises:
        return None
    
    if not current_uebung:
        return all_exercises[0]
    
    # Find current exercise index
    current_idx = None
    for idx, (_, _, uebung) in enumerate(all_exercises):
        if uebung == current_uebung:
            current_idx = idx
            break
    
    if current_idx is None:
        return all_exercises[0]
    
    # Get next exercise (wrap around)
    next_idx = (current_idx + 1) % len(all_exercises)
    return all_exercises[next_idx]


@st.cache_resource(show_spinner=False)
def load_self_harm_lexicon_cached(path: str) -> Dict[str, Any]:
    return load_self_harm_lexicon(path)


def build_hierarchy(ex_struct: Any) -> Dict[str, Dict[str, List[str]]]:
    """Return {Thema: {Path: [Übung, ...]}} for the merged structured JSON."""
    hierarchy: Dict[str, Dict[str, List[str]]] = {}

    def add_entry(thema: str, pfad: str, uebung: str) -> None:
        hierarchy.setdefault(thema, {}).setdefault(pfad, []).append(uebung)

    if isinstance(ex_struct, dict):
        for thema, val in ex_struct.items():
            if not isinstance(val, dict):
                continue
            for pfad, val2 in val.items():
                if not isinstance(val2, dict):
                    continue
                for uebung, content in val2.items():
                    if isinstance(uebung, str):
                        add_entry(thema, pfad, uebung)
    return hierarchy


def render_segment_header(label: str, color: str, with_background: bool = False) -> None:
    """Render a consistently styled segment header."""
    styles = [
        "font-size:1.1rem",
        "font-weight:700",
        "letter-spacing:0.08em",
        "margin-top:0.75rem",
        "margin-bottom:0.25rem",
        f"color:{color}",
    ]
    if with_background:
        styles.extend(
            [
                "display:inline-block",
                "padding:0.25rem 0.5rem",
                "border-radius:0.4rem",
                "background-color:#1F2933",
            ]
        )
    st.markdown(
        f"<div style=\"{';'.join(styles)}\">{label}</div>",
        unsafe_allow_html=True,
    )


def render_segments_ui(segments: List[Dict[str, Any]], key_prefix: str = "") -> List[Dict[str, Any]]:
    """Render the segments and collect user inputs in a parallel list."""
    filled: List[Dict[str, Any]] = []

    for idx, seg in enumerate(segments):
        if "Text" in seg:
            render_segment_header("TEXT", "#93C5FD")
            st.markdown(seg["Text"])
            filled.append({"Text": seg["Text"]})

        elif "Question" in seg:
            render_segment_header("FRAGE", "#22C55E")
            st.markdown(seg["Question"])
            filled.append({"Question": seg["Question"]})

        elif "Answer" in seg or "AnswerOptions" in seg:
            answer_val = seg.get("Answer")
            answer_options = seg.get("AnswerOptions")
            allow_multiple = bool(seg.get("AllowMultiple", True))

            key = f"{key_prefix}_ans_{idx}"
            render_segment_header("ANTWORT", "#F97316")

            if isinstance(answer_options, list):
                options = answer_options

                default_selection: List[str] = []
                if isinstance(answer_val, list):
                    default_selection = [item for item in answer_val if item in options]
                elif isinstance(answer_val, str) and answer_val in options:
                    default_selection = [answer_val]

                if allow_multiple:
                    user_val = st.multiselect(
                        "Answer Options",
                        options=options,
                        default=default_selection,
                        key=key,
                        label_visibility="collapsed",
                    )
                else:
                    default_value = default_selection[0] if default_selection else None
                    if default_value is not None and default_value in options:
                        selected_value = st.selectbox(
                            "Answer Options",
                            options=options,
                            index=options.index(default_value),
                            key=key,
                            label_visibility="collapsed",
                        )
                    else:
                        selected_value = st.selectbox(
                            "Answer Options",
                            options=options,
                            key=key,
                            label_visibility="collapsed",
                            placeholder="Option auswählen",
                            index=None,
                        )
                    user_val = [selected_value] if selected_value else []

                filled.append(
                    {
                        "Answer": user_val,
                        "AnswerOptions": options,
                        "AllowMultiple": allow_multiple,
                    }
                )

            elif isinstance(answer_options, dict) or isinstance(answer_val, dict):
                slider_config = answer_options if isinstance(answer_options, dict) else answer_val

                min_numeric_raw = slider_config.get("min")
                max_numeric_raw = slider_config.get("max")
                min_text = slider_config.get("minText")
                max_text = slider_config.get("maxText")

                min_numeric: Optional[int] = None
                max_numeric: Optional[int] = None

                try:
                    min_numeric = int(float(min_numeric_raw)) if min_numeric_raw is not None else None
                except (TypeError, ValueError):
                    min_numeric = None
                try:
                    max_numeric = int(float(max_numeric_raw)) if max_numeric_raw is not None else None
                except (TypeError, ValueError):
                    max_numeric = None

                if (
                    min_numeric is not None
                    and max_numeric is not None
                    and min_numeric <= max_numeric
                ):
                    if min_numeric > max_numeric:
                        min_numeric, max_numeric = max_numeric, min_numeric

                    current_answer = answer_val
                    default_value = None
                    if isinstance(current_answer, (int, float)):
                        default_value = int(current_answer)
                    elif isinstance(current_answer, dict) and isinstance(current_answer.get("value"), (int, float)):
                        default_value = int(current_answer["value"])

                    if default_value is None or not (min_numeric <= default_value <= max_numeric):
                        default_value = min_numeric

                    label_cols = st.columns([1, 4, 1])
                    with label_cols[0]:
                        st.caption(min_text if min_text else str(min_numeric))
                    with label_cols[2]:
                        st.caption(max_text if max_text else str(max_numeric))

                    slider_key = f"{key}_slider"
                    slider_value = st.slider(
                        "Value",
                        min_value=min_numeric,
                        max_value=max_numeric,
                        value=default_value,
                        format="%d",
                        key=slider_key,
                        label_visibility="collapsed",
                    )
                    filled.append(
                        {
                            "Answer": int(slider_value),
                            "AnswerOptions": slider_config,
                        }
                    )
                else:
                    st.warning("Slider configuration unvollständig – zeige Rohdaten.")
                    st.json(slider_config)
                    filled.append({"Answer": answer_val, "AnswerOptions": slider_config})

            elif isinstance(answer_val, str) and answer_val.strip().lower() in {"free_text", "freetext", "free text"}:
                user_val = st.text_input("Answer", key=key, label_visibility="collapsed")
                filled.append({"Answer": user_val})

            elif isinstance(answer_val, str):
                user_val = st.text_area(
                    "Answer",
                    value=answer_val,
                    key=key,
                    label_visibility="collapsed",
                )
                filled.append({"Answer": user_val})

            else:
                st.markdown(str(answer_val))
                filled.append({"Answer": answer_val})

    return filled


def has_non_empty_answers(inhalt_text: str) -> bool:
    """Check if INHALT text contains any non-empty answers."""
    if not inhalt_text.strip():
        return False
    
    lines = inhalt_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("ANTWORT: "):
            answer_content = line[9:].strip()  # Remove "ANTWORT: " prefix
            if answer_content:  # Non-empty answer found
                return True
    return False


def has_answer_lines(inhalt_text: str) -> bool:
    """Return True if text already contains at least one ANTWORT line."""
    return count_answer_lines(inhalt_text) > 0


def count_answer_lines(inhalt_text: str) -> int:
    """Count the number of ANTWORT lines in the provided text."""
    if not inhalt_text:
        return 0
    return sum(1 for line in inhalt_text.split("\n") if line.strip().startswith("ANTWORT:"))


def copy_answer_metadata(source: Dict[str, Any], target: Dict[str, Any]) -> None:
    """Copy slider/multiple-choice metadata from source to target."""
    answer_options = source.get("AnswerOptions")
    if not answer_options and isinstance(source.get("Answer"), dict):
        answer_options = source.get("Answer")

    if answer_options and not target.get("AnswerOptions"):
        target["AnswerOptions"] = answer_options

    if "AllowMultiple" in source and "AllowMultiple" not in target:
        target["AllowMultiple"] = source["AllowMultiple"]


def count_answer_segments(segments: List[Dict[str, Any]]) -> int:
    """Count how many answer-bearing segments exist."""
    return sum(1 for seg in segments if "Answer" in seg)


def exercise_has_questions(segments: List[Dict[str, Any]]) -> bool:
    """Return True if any question segments exist."""
    return any("Question" in seg for seg in segments)


def enrich_segments_with_answer_metadata(
    source_segments: List[Dict[str, Any]],
    target_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ensure generated segments keep their slider/MC metadata."""
    enriched: List[Dict[str, Any]] = []
    source_answer_iter = (
        seg for seg in source_segments if "Answer" in seg or "AnswerOptions" in seg
    )

    for seg in target_segments:
        new_seg = seg.copy()
        if "Answer" in new_seg or "AnswerOptions" in new_seg:
            base_seg = next(source_answer_iter, None)
            if base_seg:
                copy_answer_metadata(base_seg, new_seg)

        enriched.append(new_seg)

    return enriched


def segments_to_inhalt(segments: List[Dict[str, Any]], empty_answers: bool = False) -> str:
    """Convert segments to INHALT format (one line per segment).
    
    Args:
        segments: List of segment dictionaries
        empty_answers: If True, use empty strings for answers instead of actual values
    """
    lines: List[str] = []

    def answer_uses_percent(options: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(options, dict):
            return False
        for key in ("minText", "maxText"):
            text_val = options.get(key)
            if isinstance(text_val, str) and text_val.strip().endswith("%"):
                return True
        return False
    
    for segment in segments:
        if "Text" in segment:
            clean_text = " ".join(segment["Text"].split())
            lines.append(f"TEXT: {clean_text}")
        elif "Question" in segment:
            question_text = segment["Question"]
            lines.append(f"FRAGE: {question_text}")
        elif "Answer" in segment:
            if empty_answers:
                # Skip empty answers - don't add ANTWORT line at all
                continue
            else:
                answer_val = segment.get("Answer")
                answer_text: Optional[str] = None
                
                if isinstance(answer_val, list):
                    # Only create text if list is not empty
                    if answer_val:
                        answer_text = ", ".join(str(item) for item in answer_val)
                elif isinstance(answer_val, (int, float)):
                    suffix = "%" if answer_uses_percent(segment.get("AnswerOptions")) else ""
                    answer_text = f"{int(answer_val)}{suffix}"
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
                        if sub_parts:
                            answer_text = "; ".join(sub_parts)
                
                # Only add ANTWORT line if there's actual answer text
                if answer_text:
                    lines.append(f"ANTWORT: {answer_text}")
    return "\n".join(lines)


def inhalt_to_segments(inhalt_text: str) -> List[Dict[str, Any]]:
    """Parse INHALT format back to segments."""
    segments: List[Dict[str, Any]] = []
    lines = inhalt_text.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("TEXT: "):
            segments.append({"Text": line[6:].strip()})
        elif line.startswith("FRAGE: "):
            segments.append({"Question": line[7:].strip()})
        elif line.startswith("ANTWORT: "):
            segments.append({"Answer": line[9:].strip()})
    
    return segments


def replace_answers_in_inhalt(inhalt_text: str, updated_segments: List[Dict[str, Any]]) -> str:
    """Replace ANTWORT blocks in INHALT with answers from the current segments."""
    segments = inhalt_to_segments(inhalt_text)
    new_answers: List[Any] = [seg.get("Answer") for seg in updated_segments if "Answer" in seg]
    
    answer_idx = 0
    for seg in segments:
        if "Answer" in seg and answer_idx < len(new_answers):
            seg["Answer"] = new_answers[answer_idx]
            answer_idx += 1
    
    return segments_to_inhalt(segments)


def merge_filled_answers_into_segments(
    original_segments: List[Dict[str, Any]], 
    filled_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge filled answers from UI back into the original segment structure."""
    result: List[Dict[str, Any]] = []
    filled_idx = 0

    def advance_to_next_answer(idx: int) -> int:
        while idx < len(filled_segments) and "Answer" not in filled_segments[idx]:
            idx += 1
        return idx

    filled_idx = advance_to_next_answer(filled_idx)

    for seg in original_segments:
        if "Text" in seg or "Question" in seg:
            result.append(seg.copy())
            continue

        if "Answer" in seg:
            filled_idx = advance_to_next_answer(filled_idx)
            if filled_idx < len(filled_segments):
                filled_entry = filled_segments[filled_idx]
                if "Answer" in filled_entry:
                    new_seg = seg.copy()
                    new_seg["Answer"] = filled_entry["Answer"]
                    if "AnswerOptions" in filled_entry:
                        new_seg["AnswerOptions"] = filled_entry["AnswerOptions"]
                    if "AllowMultiple" in filled_entry:
                        new_seg["AllowMultiple"] = filled_entry["AllowMultiple"]
                    result.append(new_seg)
                    filled_idx += 1
                    continue
            result.append(seg.copy())
            continue

        result.append(seg.copy())

    return result


# -----------------------------
# UI Layout
# -----------------------------
st.set_page_config(page_title="Summary Prompt Lab", layout="centered")

# Password protection
if APP_PASSWORD and not PASSWORD_FROM_SECRETS:
    if "password_verified" not in st.session_state:
        st.session_state.password_verified = True

st.title("🧪 Summary Prompt Lab")

# Load files first (before sidebar)
files_ok = True
ex_struct = None
story_nodes = None
try:
    if not STRUCT_JSON_PATH or not os.path.exists(STRUCT_JSON_PATH):
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {STRUCT_JSON_PATH}")
    if not SN_JSON_PATH or not os.path.exists(SN_JSON_PATH):
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {SN_JSON_PATH}")
    
    ex_struct = load_json(STRUCT_JSON_PATH)
    story_nodes = load_json(SN_JSON_PATH)
except FileNotFoundError as e:
    files_ok = False
    st.error(str(e))
except Exception as e:
    files_ok = False
    st.error(f"Fehler beim Laden der Dateien: {e}")

if not files_ok:
    st.stop()

hier = build_hierarchy(ex_struct)
if not hier:
    st.warning("Konnte keine Themen/Wege/Übungen aus der Struktur extrahieren.")

# Ensure exercise prompts JSON file exists
all_exercise_names = get_all_exercise_names(hier) if hier else []
if all_exercise_names:
    ensure_exercise_prompt_store()
    ensure_summary_switch_store()

with st.sidebar:
    st.header("Navigation")
    
    thema_key = "nav_selected_thema"
    path_key = "nav_selected_path"
    uebung_key = "nav_selected_uebung"
    
    themen = sorted(hier.keys())
    
    # Navigation buttons in two columns
    nav_button_cols = st.columns(2)
    
    with nav_button_cols[0]:
        if st.button("🎲 Zufällige Übung", use_container_width=True, disabled=not hier):
            if themen:
                zufalls_thema = random.choice(themen)
                available_paths = sorted(hier.get(zufalls_thema, {}).keys())
                if available_paths:
                    zufalls_path = random.choice(available_paths)
                    available_uebungen = sorted(
                        hier.get(zufalls_thema, {}).get(zufalls_path, [])
                    )
                    if available_uebungen:
                        zufalls_uebung = random.choice(available_uebungen)
                        st.session_state[thema_key] = zufalls_thema
                        st.session_state[path_key] = zufalls_path
                        st.session_state[uebung_key] = zufalls_uebung
                        st.rerun()
    
    with nav_button_cols[1]:
        current_uebung = st.session_state.get(uebung_key)
        if st.button("➡️ Nächste Übung", use_container_width=True, disabled=not hier):
            next_exercise = get_next_exercise(hier, current_uebung)
            if next_exercise:
                next_thema, next_path, next_uebung = next_exercise
                st.session_state[thema_key] = next_thema
                st.session_state[path_key] = next_path
                st.session_state[uebung_key] = next_uebung
                st.rerun()
    
    if themen and st.session_state.get(thema_key) not in themen:
        st.session_state[thema_key] = themen[0]
    sel_thema = (
        st.selectbox("Thema", options=themen, key=thema_key)
        if themen
        else None
    )
    
    paths = sorted(hier.get(sel_thema, {}).keys()) if sel_thema else []
    if paths and st.session_state.get(path_key) not in paths:
        st.session_state[path_key] = paths[0]
    elif not paths:
        st.session_state.pop(path_key, None)
    sel_path = (
        st.selectbox("Pfad", options=paths, key=path_key)
        if paths
        else None
    )
    
    uebungen = sorted(hier.get(sel_thema, {}).get(sel_path, [])) if sel_path else []
    if uebungen and st.session_state.get(uebung_key) not in uebungen:
        st.session_state[uebung_key] = uebungen[0]
    elif not uebungen:
        st.session_state.pop(uebung_key, None)
    sel_uebung = (
        st.selectbox("Übung", options=uebungen, key=uebung_key)
        if uebungen
        else None
    )
    
if sel_uebung:
    # Add buttons to sidebar
    with st.sidebar:
        st.markdown("---")
        
        # Session key for this exercise (no longer seed-dependent)
        session_key = f"generated_segments_{sel_uebung}"
        
        # Define state keys for all three text areas (tied to exercise only)
        example1_state_key = f"{sel_uebung}_example1"
        example2_state_key = f"{sel_uebung}_example2"
        mainrecap_inhalt_key = f"{sel_uebung}_mainrecap_inhalt"
        
        # Gemini max words control
        gemini_max_words_key = f"{session_key}_gemini_max_words"
        if gemini_max_words_key not in st.session_state:
            st.session_state[gemini_max_words_key] = 20
        
        st.header("Synthetische Antworten")
        gemini_max_words = st.slider(
            "Maximale Wortanzahl",
            min_value=1,
            max_value=40,
            step=1,
            key=gemini_max_words_key,
        )
        
        # Answer generation button
        mistral_gen_clicked = st.button("🤖 Antworten generieren", use_container_width=True)
        
        if mistral_gen_clicked:
            if not MISTRAL_API_KEY:
                st.error("❌ MISTRAL_API_KEY nicht gesetzt.")
            else:
                try:
                    # Preserve other text areas' state before rerun
                    if example1_state_key in st.session_state:
                        preserved_ex1 = st.session_state[example1_state_key]
                        st.session_state[f"{example1_state_key}_preserve"] = preserved_ex1
                    if example2_state_key in st.session_state:
                        preserved_ex2 = st.session_state[example2_state_key]
                        st.session_state[f"{example2_state_key}_preserve"] = preserved_ex2
                    if mainrecap_inhalt_key in st.session_state:
                        preserved_main = st.session_state[mainrecap_inhalt_key]
                        st.session_state[f"{mainrecap_inhalt_key}_preserve"] = preserved_main
                    
                    # Build fresh segments for this exercise
                    try:
                        current_segments = get_prompt_segments_from_exercise(
                            exercise_name=sel_uebung,
                            json_struct_path=STRUCT_JSON_PATH,
                            json_sn_struct_path=SN_JSON_PATH,
                            seed=None,
                        )
                    except Exception as e:
                        st.error(f"Fehler bei der Segment-Generierung: {e}")
                        st.stop()
                    
                    # Create system prompt with max words
                    system_prompt_with_words = SYNTH_ANSWERS_PROMPT.replace("xxx", str(gemini_max_words))
                    
                    with st.spinner("Generiere Antworten..."):
                        result = generate_answers_with_mistral(
                            segments=current_segments,
                            api_key=MISTRAL_API_KEY,
                            system_prompt=system_prompt_with_words,
                            debug=False,
                            return_debug_info=True,
                            seed=None,
                            max_words=gemini_max_words,
                        )
                        generated_segments, debug_info = result
                        generated_segments = enrich_segments_with_answer_metadata(
                            current_segments, generated_segments
                        )
                        # Store in session state
                        st.session_state[session_key] = generated_segments
                        
                        # Set flag to indicate fresh generation occurred
                        st.session_state[f"{session_key}_fresh_generated"] = True
                        
                        # Clear old widget inputs and pre-populate new ones with generated answers
                        keys_to_remove = []
                        for state_key in list(st.session_state.keys()):
                            # Clear old widget keys
                            if (state_key.startswith(f"{session_key}_original_ans_") or 
                                state_key.startswith(f"{session_key}_generated_ans_")):
                                keys_to_remove.append(state_key)
                        
                        for key in keys_to_remove:
                            del st.session_state[key]
                        
                        # Pre-populate new widget keys with generated answers
                        for idx, seg in enumerate(generated_segments):
                            if "Answer" in seg:
                                widget_key = f"{session_key}_generated_ans_{idx}"
                                answer_val = seg["Answer"]
                                
                                # Set the widget value in session state
                                if isinstance(answer_val, list):
                                    # Check if this is a single-choice MC question
                                    if ("AnswerOptions" in seg and 
                                        isinstance(seg["AnswerOptions"], list) and
                                        not seg.get("AllowMultiple", True)):
                                        # Single-choice MC: store first item as string
                                        st.session_state[widget_key] = answer_val[0] if answer_val else ""
                                    else:
                                        # Multi-choice MC: store as list
                                        st.session_state[widget_key] = answer_val
                                elif isinstance(answer_val, (int, float)):
                                    # For sliders, also set the slider-specific key
                                    st.session_state[widget_key] = int(answer_val)
                                    if "AnswerOptions" in seg and isinstance(seg["AnswerOptions"], dict):
                                        # This is a slider - also set the slider key
                                        slider_key = f"{widget_key}_slider"
                                        st.session_state[slider_key] = int(answer_val)
                                elif isinstance(answer_val, str):
                                    st.session_state[widget_key] = answer_val
                                else:
                                    st.session_state[widget_key] = answer_val
                    st.success("✅ Antworten generiert")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Fehler bei der Antwort-Generierung: {e}")
                    import traceback
                    with st.expander("🔍 Fehlerdetails anzeigen"):
                        st.code(traceback.format_exc(), language="python")
        
        # Transfer buttons in sidebar
        st.subheader("Antworten übertragen")
        
        transfer_cols = st.columns(3)
        
        with transfer_cols[0]:
            transfer_ex1_key = f"{sel_uebung}_transfer_ex1"
            if st.button("📋 Beispiel 1", key=transfer_ex1_key, use_container_width=True):
                # Preserve other text areas' state before transfer
                if example2_state_key in st.session_state:
                    preserved_ex2 = st.session_state[example2_state_key]
                    st.session_state[f"{example2_state_key}_preserve"] = preserved_ex2
                if mainrecap_inhalt_key in st.session_state:
                    preserved_main = st.session_state[mainrecap_inhalt_key]
                    st.session_state[f"{mainrecap_inhalt_key}_preserve"] = preserved_main
                
                # Set transfer flag - will use segments_for_prompt computed later
                st.session_state[f"{sel_uebung}_transfer_ex1_clicked"] = True
                st.session_state[f"{sel_uebung}_transfer_ex1_success"] = True
        
        with transfer_cols[1]:
            transfer_ex2_key = f"{sel_uebung}_transfer_ex2"
            if st.button("📋 Beispiel 2", key=transfer_ex2_key, use_container_width=True):
                # Preserve other text areas' state before transfer
                if example1_state_key in st.session_state:
                    preserved_ex1 = st.session_state[example1_state_key]
                    st.session_state[f"{example1_state_key}_preserve"] = preserved_ex1
                if mainrecap_inhalt_key in st.session_state:
                    preserved_main = st.session_state[mainrecap_inhalt_key]
                    st.session_state[f"{mainrecap_inhalt_key}_preserve"] = preserved_main
                
                # Set transfer flag
                st.session_state[f"{sel_uebung}_transfer_ex2_clicked"] = True
                st.session_state[f"{sel_uebung}_transfer_ex2_success"] = True
        
        with transfer_cols[2]:
            transfer_main_key = f"{sel_uebung}_transfer_main"
            if st.button("📋 Summary", key=transfer_main_key, use_container_width=True):
                # Preserve other text areas' state before transfer
                if example1_state_key in st.session_state:
                    preserved_ex1 = st.session_state[example1_state_key]
                    st.session_state[f"{example1_state_key}_preserve"] = preserved_ex1
                if example2_state_key in st.session_state:
                    preserved_ex2 = st.session_state[example2_state_key]
                    st.session_state[f"{example2_state_key}_preserve"] = preserved_ex2
                
                # Set transfer flag - will use segments_for_prompt computed later
                st.session_state[f"{sel_uebung}_transfer_main_clicked"] = True
                st.session_state[f"{sel_uebung}_transfer_main_success"] = True
        
        # Display transfer success messages right below the buttons
        if st.session_state.get(f"{sel_uebung}_transfer_ex1_success", False):
            st.success("✅ Antworten zu Beispiel 1 übertragen")
            st.session_state[f"{sel_uebung}_transfer_ex1_success"] = False
        
        if st.session_state.get(f"{sel_uebung}_transfer_ex2_success", False):
            st.success("✅ Antworten zu Beispiel 2 übertragen")
            st.session_state[f"{sel_uebung}_transfer_ex2_success"] = False
        
        if st.session_state.get(f"{sel_uebung}_transfer_main_success", False):
            st.success("✅ Antworten zu TEST übertragen")
            st.session_state[f"{sel_uebung}_transfer_main_success"] = False
    
    
    # Initialize session_key if it doesn't exist (don't overwrite existing generated segments)
    if session_key not in st.session_state:
        st.session_state[session_key] = None
    
    # Build segments from processing pipeline
    try:
        segments = get_prompt_segments_from_exercise(
            exercise_name=sel_uebung,
            json_struct_path=STRUCT_JSON_PATH,
            json_sn_struct_path=SN_JSON_PATH,
        )
    except Exception as e:
        st.error(f"Fehler bei der Segment-Generierung: {e}")
        st.stop()

    has_questions = exercise_has_questions(segments)
    summary_switch_config = load_summary_switch_config()
    summary_switch_value = summary_switch_config.get(sel_uebung)

    if summary_switch_value is None:
        summary_switch_value = "summary" if has_questions else "no questions"
        summary_switch_config[sel_uebung] = summary_switch_value
        save_summary_switch_config(summary_switch_config)
    
    # Choose which segments to display
    has_generated = st.session_state.get(session_key) is not None
    fresh_generated = st.session_state.get(f"{session_key}_fresh_generated", False)
    
    # If fresh generation occurred, use the generated segments and clear the flag
    if fresh_generated:
        st.session_state[f"{session_key}_fresh_generated"] = False
        segments_to_display = st.session_state[session_key]
        has_generated = True  # Ensure we use the generated key prefix
    else:
        # Use generated segments if available, otherwise use original
        segments_to_display = (
            st.session_state[session_key] if has_generated else segments
        )
    
    # Always use generated key prefix if we have generated segments
    key_prefix = (
        f"{session_key}_generated" if has_generated else f"{session_key}_original"
    )
    
    st.markdown("---")
    st.subheader("Übung")
    filled_segments = render_segments_ui(segments_to_display, key_prefix=key_prefix)
    
    # Merge filled answers back into segments
    segments_for_prompt = merge_filled_answers_into_segments(
        segments_to_display if has_generated else segments,
        filled_segments
    )
    
    
    # Update session state if segments changed
    if has_generated:
        current_segments = st.session_state[session_key]
        if segments_for_prompt != current_segments:
            st.session_state[session_key] = segments_for_prompt
    
    # Store segments_for_prompt in session state for transfer buttons
    st.session_state[f"{session_key}_segments_for_prompt"] = segments_for_prompt
    
    # Handle transfer button clicks (using stored segments_for_prompt)
    transfer_ex1_clicked = st.session_state.get(f"{sel_uebung}_transfer_ex1_clicked", False)
    if transfer_ex1_clicked:
        existing_text = st.session_state.get(example1_state_key, "")
        existing_count = count_answer_lines(existing_text)
        segment_count = count_answer_segments(segments_for_prompt)
        if existing_count and existing_count == segment_count:
            inhalt = replace_answers_in_inhalt(existing_text, segments_for_prompt)
        else:
            inhalt = segments_to_inhalt(segments_for_prompt)
        st.session_state[example1_state_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_ex1_clicked"] = False
    
    transfer_ex2_clicked = st.session_state.get(f"{sel_uebung}_transfer_ex2_clicked", False)
    if transfer_ex2_clicked:
        existing_text = st.session_state.get(example2_state_key, "")
        existing_count = count_answer_lines(existing_text)
        segment_count = count_answer_segments(segments_for_prompt)
        if existing_count and existing_count == segment_count:
            inhalt = replace_answers_in_inhalt(existing_text, segments_for_prompt)
        else:
            inhalt = segments_to_inhalt(segments_for_prompt)
        st.session_state[example2_state_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_ex2_clicked"] = False
    
    transfer_main_clicked = st.session_state.get(f"{sel_uebung}_transfer_main_clicked", False)
    if transfer_main_clicked:
        existing_text = st.session_state.get(mainrecap_inhalt_key, "")
        existing_count = count_answer_lines(existing_text)
        segment_count = count_answer_segments(segments_for_prompt)
        if existing_count and existing_count == segment_count:
            inhalt = replace_answers_in_inhalt(existing_text, segments_for_prompt)
        else:
            inhalt = segments_to_inhalt(segments_for_prompt)
        st.session_state[mainrecap_inhalt_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_main_clicked"] = False

    st.markdown("---")
    st.subheader("Summary Switch")
    summary_switch_cols = st.columns([3, 1])
    summary_switch_input_key = f"{session_key}_summary_switch_input"
    with summary_switch_cols[0]:
        st.text_input(
            "Summary Switch Eintrag",
            key=summary_switch_input_key,
            value=summary_switch_value,
            label_visibility="collapsed",
        )
    with summary_switch_cols[1]:
        summary_switch_save_clicked = st.button(
            "💾 Speichern",
            key=f"{summary_switch_input_key}_save",
            use_container_width=True,
        )

    if summary_switch_save_clicked:
        new_value = st.session_state.get(summary_switch_input_key, "").strip()
        summary_switch_config[sel_uebung] = new_value
        save_summary_switch_config(summary_switch_config)
        summary_switch_value = new_value
        st.session_state[summary_switch_input_key] = new_value
        st.success("✅ Summary Switch aktualisiert")

    summary_mode_active = summary_switch_is_on(summary_switch_value)

    if not summary_mode_active:
        st.info(
            "Diese Übung ist deaktiviert. Setze den Summary Switch auf 'summary', "
            "um die nachfolgenden Bereiche zu verwenden."
        )
        st.stop()
    
    word_limit_file_config = load_word_limit_config()
    initialize_word_limit_inputs(word_limit_file_config)
    active_word_limit_config = current_word_limit_config()
    forced_sections = st.session_state.pop(f"{session_key}_force_reload_sections", None)
    exercise_sections = forced_sections or get_exercise_sections(sel_uebung, word_limit_file_config)
    if forced_sections:
        for exercise_key in EXERCISE_SECTION_EDITOR_KEYS:
            section_state = section_state_key(exercise_key, "exercise", session_key)
            st.session_state[section_state] = forced_sections.get(exercise_key, "")
            st.session_state[section_exercise_tracker_key(section_state)] = sel_uebung
    global_section_defaults = load_global_sections()
    initialize_section_states(sel_uebung, session_key, global_section_defaults, exercise_sections)
    
    default_example1 = exercise_sections.get("example1") or segments_to_inhalt(segments, empty_answers=True)
    default_example2 = exercise_sections.get("example2") or segments_to_inhalt(segments, empty_answers=True)
    
    # System Prompt Section
    st.markdown("---")
    st.subheader("System-Prompt")
    for cfg in SECTION_UI_CONFIG:
        scope = cfg["scope"]
        key = cfg["key"]
        state_key = section_state_key(key, scope, session_key)
        with st.expander(cfg["label"], expanded=False):
            st.text_area(
                cfg["label"],
                key=state_key,
                height=180,
                label_visibility="collapsed",
            )
            button_cols = st.columns(2)
            with button_cols[0]:
                load_key = f"{state_key}_load"
                if st.button("📥 Laden", key=load_key, use_container_width=True):
                    if scope == "global":
                        st.session_state[state_key] = load_global_section(key)
                    else:
                        fresh_sections = get_exercise_sections(sel_uebung, active_word_limit_config)
                        st.session_state[state_key] = fresh_sections.get(key, "")
                        st.session_state[section_exercise_tracker_key(state_key)] = sel_uebung
                    st.rerun()
            with button_cols[1]:
                save_button_key = f"{state_key}_save"
                if st.button("💾 Speichern", key=save_button_key, use_container_width=True):
                    st.session_state[f"{save_button_key}_dialog"] = True
                dialog_message = (
                    f"Möchten Sie den globalen Abschnitt '{cfg['label']}' wirklich überschreiben?"
                    if scope == "global"
                    else f"Möchten Sie den Abschnitt '{cfg['label']}' für '{sel_uebung}' wirklich speichern?"
                )
                def _save_section(scope=scope, section_key=key, value_key=state_key) -> None:
                    if scope == "global":
                        save_global_section(section_key, st.session_state.get(value_key, ""))
                    else:
                        save_exercise_payload(sel_uebung, {section_key: st.session_state.get(value_key, "")})
                        refreshed_sections = get_exercise_sections(sel_uebung, active_word_limit_config)
                        st.session_state[f"{session_key}_force_reload_sections"] = refreshed_sections
                confirm_action(f"{save_button_key}_dialog", dialog_message, _save_section)
    
    st.markdown("---")
    st.subheader("Wortlimits")
    st.caption("Ordne den Anzahl-Antworten-Schwellen entsprechende Wortlimits zu.")
    limit_cols = st.columns(WORD_LIMIT_COLUMNS)
    for idx, col in enumerate(limit_cols):
        with col:
            st.number_input(
                "Anzahl Antworten",
                min_value=0,
                step=1,
                key=f"{WORD_LIMIT_SESSION_PREFIX}_answers_{idx}",
            )
            st.number_input(
                "Max. Wörter",
                min_value=1,
                step=1,
                key=f"{WORD_LIMIT_SESSION_PREFIX}_words_{idx}",
            )
    save_limits_key = f"{session_key}_save_word_limits"
    if st.button("💾 Wortlimits speichern", key=save_limits_key, use_container_width=True):
        st.session_state[f"{save_limits_key}_dialog"] = True
    def _save_word_limits() -> None:
        config = current_word_limit_config()
        save_word_limit_config(config["answer_counts"], config["max_words"])
    confirm_action(
        f"{save_limits_key}_dialog",
        "Aktuelle Wortlimits dauerhaft speichern?",
        _save_word_limits,
    )
    
    # Beispiel 1 Section
    st.markdown("---")
    st.subheader("Beispiel 1")
    
    # Use exercise-specific key for Beispiel 1
    example1_state_key = f"{sel_uebung}_example1"
    
    # Restore preserved value if it exists (from transfer button click)
    # This MUST happen before any other logic that might affect the state
    preserve_key = f"{example1_state_key}_preserve"
    if preserve_key in st.session_state:
        preserved_value = st.session_state[preserve_key]
        st.session_state[example1_state_key] = preserved_value
        del st.session_state[preserve_key]
    
    
    # Initialize only if key doesn't exist (preserve existing content)
    # IMPORTANT: Only initialize if the key truly doesn't exist, don't overwrite existing content
    # This check happens AFTER restore, so preserved values won't be overwritten
    # CRITICAL: Never overwrite existing text area content, even if segments change
    if example1_state_key not in st.session_state:
        example1_value = exercise_sections.get("example1", "") or default_example1
        st.session_state[example1_state_key] = example1_value
    # Note: Widget with key=example1_state_key manages its own session state
    # We only set it explicitly during transfer or load operations
    
    # Check if we need to update Beispiel 1 from a previous recap generation
    update_ex1_key = f"{session_key}_update_ex1"
    if update_ex1_key in st.session_state:
        st.session_state[example1_state_key] = st.session_state[update_ex1_key]
        del st.session_state[update_ex1_key]
    
    # Create widget - Streamlit manages session state via key
    # Always create the widget the same way - Streamlit will use the value from session state
    example1_text = st.text_area(
        "Beispiel 1 INHALT",
        key=example1_state_key,
        height=300,
        label_visibility="collapsed",
    )
    
    example1_button_cols = st.columns(3)
    with example1_button_cols[0]:
        gen_ex1_key = f"{session_key}_gen_ex1"
        if st.button("🧾 Summary generieren", key=gen_ex1_key, use_container_width=True):
            if not MISTRAL_API_KEY:
                st.error("❌ MISTRAL_API_KEY nicht gesetzt.")
            elif not has_non_empty_answers(example1_text):
                st.warning("⚠️ Keine Antworten vorhanden. Bitte erst Antworten generieren oder manuell eingeben.")
            else:
                try:
                    # Preserve other text areas' state before rerun
                    if example2_state_key in st.session_state:
                        preserved_ex2 = st.session_state[example2_state_key]
                        st.session_state[f"{example2_state_key}_preserve"] = preserved_ex2
                    if mainrecap_inhalt_key in st.session_state:
                        preserved_main = st.session_state[mainrecap_inhalt_key]
                        st.session_state[f"{mainrecap_inhalt_key}_preserve"] = preserved_main
                    
                    answer_count = count_answer_lines(example1_text)
                    max_words_value = choose_max_words(answer_count, active_word_limit_config)
                    global_section_values = collect_global_section_values(session_key)
                    exercise_section_values = collect_exercise_section_values(session_key)
                    system_prompt_for_ex1 = assemble_system_prompt(
                        global_section_values,
                        exercise_section_values,
                        max_words_value,
                    )
                    
                    # Build prompt: system prompt + INHALT
                    prompt = f"{system_prompt_for_ex1}\n\n{example1_text}"
                    print_prompt_debug(
                        "beispiel1",
                        prompt,
                        {"max_tokens": 200, "max_words": max_words_value},
                    )
                    
                    with st.spinner("Generiere Recap..."):
                        recap = generate_summary_with_mistral(
                            prompt=prompt,
                            api_key=MISTRAL_API_KEY,
                            max_tokens=200,
                        )
                    
                    # Add or replace ZUSAMMENFASSUNG line
                    lines = example1_text.split("\n")
                    # Remove existing ZUSAMMENFASSUNG if present
                    lines = [line for line in lines if not line.startswith("ZUSAMMENFASSUNG:")]
                    # Add new ZUSAMMENFASSUNG
                    lines.append(f"ZUSAMMENFASSUNG: {recap}")
                    updated_ex1_text = "\n".join(lines)
                    
                    # Store in temporary key to update Beispiel 1 on next rerun
                    st.session_state[update_ex1_key] = updated_ex1_text
                    
                    st.success("✅ Recap für Beispiel 1 generiert")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Fehler: {e}")
                    import traceback
                    with st.expander("🔍 Fehlerdetails"):
                        st.code(traceback.format_exc())
    
    with example1_button_cols[1]:
        save_ex1_key = f"{session_key}_save_ex1"
        save_ex1_clicked = st.button("💾 Beispiel 1 speichern", key=save_ex1_key, use_container_width=True)
        if save_ex1_clicked:
            confirm_key = f"{save_ex1_key}_dialog"
            if confirm_key not in st.session_state:
                st.session_state[confirm_key] = True
            
            if st.session_state[confirm_key]:
                with st.dialog("Bestätigung") as dialog:
                    st.write(f"Möchten Sie Beispiel 1 für '{sel_uebung}' wirklich überschreiben?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Ja", key=f"{save_ex1_key}_yes", use_container_width=True):
                            if save_exercise_payload(sel_uebung, {"example1": example1_text}):
                                st.session_state[confirm_key] = False
                                dialog.close()
                                st.rerun()
                            else:
                                st.error("❌ Fehler beim Speichern")
                    with col2:
                        if st.button("Nein", key=f"{save_ex1_key}_no", use_container_width=True):
                            st.session_state[confirm_key] = False
                            dialog.close()
                            st.rerun()
    
    with example1_button_cols[2]:
        if st.button("📥 Beispiel 1 laden", key=f"{session_key}_load_ex1", use_container_width=True):
            example1_value = exercise_sections.get("example1", "")
            if not example1_value:
                # Initialize with INHALT from segments with empty answers
                example1_value = segments_to_inhalt(segments, empty_answers=True)
            st.session_state[example1_state_key] = example1_value
            st.rerun()
    
    # Beispiel 2 Section
    st.markdown("---")
    st.subheader("Beispiel 2")
    
    # Use exercise-specific key for Beispiel 2
    example2_state_key = f"{sel_uebung}_example2"
    
    # Restore preserved value if it exists (from transfer button click)
    # This MUST happen before any other logic that might affect the state
    preserve_key = f"{example2_state_key}_preserve"
    if preserve_key in st.session_state:
        preserved_value = st.session_state[preserve_key]
        st.session_state[example2_state_key] = preserved_value
        del st.session_state[preserve_key]
    
    
    # Initialize only if key doesn't exist (preserve existing content)
    # This check happens AFTER restore, so preserved values won't be overwritten
    # CRITICAL: Never overwrite existing text area content, even if segments change
    if example2_state_key not in st.session_state:
        example2_value = exercise_sections.get("example2", "") or default_example2
        st.session_state[example2_state_key] = example2_value
    # Note: Widget with key=example2_state_key manages its own session state
    # We only set it explicitly during transfer or load operations
    
    # Check if we need to update Beispiel 2 from a previous recap generation
    update_ex2_key = f"{session_key}_update_ex2"
    if update_ex2_key in st.session_state:
        st.session_state[example2_state_key] = st.session_state[update_ex2_key]
        del st.session_state[update_ex2_key]
    
    # Create widget - Streamlit manages session state via key
    # Always create the widget the same way - Streamlit will use the value from session state
    example2_text = st.text_area(
        "Beispiel 2 Inhalt",
        key=example2_state_key,
        height=300,
        label_visibility="collapsed",
    )
    
    example2_button_cols = st.columns(3)
    with example2_button_cols[0]:
        gen_ex2_key = f"{session_key}_gen_ex2"
        if st.button("🧾 Summary generieren", key=gen_ex2_key, use_container_width=True):
            if not MISTRAL_API_KEY:
                st.error("❌ MISTRAL_API_KEY nicht gesetzt.")
            elif not has_non_empty_answers(example2_text):
                st.warning("⚠️ Keine Antworten vorhanden. Bitte erst Antworten generieren oder manuell eingeben.")
            else:
                try:
                    # Preserve other text areas' state before rerun
                    if example1_state_key in st.session_state:
                        preserved_ex1 = st.session_state[example1_state_key]
                        st.session_state[f"{example1_state_key}_preserve"] = preserved_ex1
                    if mainrecap_inhalt_key in st.session_state:
                        preserved_main = st.session_state[mainrecap_inhalt_key]
                        st.session_state[f"{mainrecap_inhalt_key}_preserve"] = preserved_main
                    
                    answer_count = count_answer_lines(example2_text)
                    max_words_value = choose_max_words(answer_count, active_word_limit_config)
                    global_section_values = collect_global_section_values(session_key)
                    exercise_section_values = collect_exercise_section_values(session_key)
                    system_prompt_for_ex2 = assemble_system_prompt(
                        global_section_values,
                        exercise_section_values,
                        max_words_value,
                    )
                    prompt = f"{system_prompt_for_ex2}\n\n{example2_text}"
                    print_prompt_debug(
                        "beispiel2",
                        prompt,
                        {"max_tokens": 200, "max_words": max_words_value},
                    )
                    
                    with st.spinner("Generiere Recap..."):
                        recap = generate_summary_with_mistral(
                            prompt=prompt,
                            api_key=MISTRAL_API_KEY,
                            max_tokens=200,
                        )
                    
                    lines = example2_text.split("\n")
                    lines = [line for line in lines if not line.startswith("ZUSAMMENFASSUNG:")]
                    lines.append(f"ZUSAMMENFASSUNG: {recap}")
                    updated_ex2_text = "\n".join(lines)
                    
                    # Store in temporary key to update Beispiel 2 on next rerun
                    st.session_state[update_ex2_key] = updated_ex2_text
                    
                    st.success("✅ Recap für Beispiel 2 generiert")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Fehler: {e}")
                    import traceback
                    with st.expander("🔍 Fehlerdetails"):
                        st.code(traceback.format_exc())
    
    with example2_button_cols[1]:
        save_ex2_key = f"{session_key}_save_ex2"
        save_ex2_clicked = st.button("💾 Beispiel 2 speichern", key=save_ex2_key, use_container_width=True)
        if save_ex2_clicked:
            confirm_key = f"{save_ex2_key}_dialog"
            if confirm_key not in st.session_state:
                st.session_state[confirm_key] = True
            
            if st.session_state[confirm_key]:
                with st.dialog("Bestätigung") as dialog:
                    st.write(f"Möchten Sie Beispiel 2 für '{sel_uebung}' wirklich überschreiben?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Ja", key=f"{save_ex2_key}_yes", use_container_width=True):
                            if save_exercise_payload(sel_uebung, {"example2": example2_text}):
                                st.session_state[confirm_key] = False
                                dialog.close()
                                st.rerun()
                            else:
                                st.error("❌ Fehler beim Speichern")
                    with col2:
                        if st.button("Nein", key=f"{save_ex2_key}_no", use_container_width=True):
                            st.session_state[confirm_key] = False
                            dialog.close()
                            st.rerun()
    
    with example2_button_cols[2]:
        if st.button("📥 Beispiel 2 laden", key=f"{session_key}_load_ex2", use_container_width=True):
            example2_value = exercise_sections.get("example2", "")
            if not example2_value:
                # Initialize with INHALT from segments with empty answers
                example2_value = segments_to_inhalt(segments, empty_answers=True)
            st.session_state[example2_state_key] = example2_value
            st.rerun()
    
    # TEST Section
    st.markdown("---")
    st.subheader("TEST")
    
    # TEST INHALT is constructed from segments (not stored in JSON)
    # Use exercise-specific key for TEST
    mainrecap_inhalt_key = f"{sel_uebung}_mainrecap_inhalt"
    
    # Restore preserved value if it exists (from transfer button click)
    # This MUST happen before any other logic that might affect the state
    preserve_key = f"{mainrecap_inhalt_key}_preserve"
    if preserve_key in st.session_state:
        preserved_value = st.session_state[preserve_key]
        st.session_state[mainrecap_inhalt_key] = preserved_value
        del st.session_state[preserve_key]
    
    # Initialize only if key doesn't exist (preserve existing content)
    # This check happens AFTER restore, so preserved values won't be overwritten
    # CRITICAL: Never overwrite existing text area content, even if segments change
    if mainrecap_inhalt_key not in st.session_state:
        # Initialize with INHALT from segments with empty answers
        st.session_state[mainrecap_inhalt_key] = segments_to_inhalt(segments, empty_answers=True)
    # Note: Widget with key=mainrecap_inhalt_key manages its own session state
    
    # Show TEST Inhalt text area - Streamlit manages session state via key
    # Always create the widget the same way - Streamlit will use the value from session state
    mainrecap_inhalt_text = st.text_area(
        "TEST Inhalt",
        key=mainrecap_inhalt_key,
        height=300,
        label_visibility="collapsed",
    )
    
    # Slider for TEST
    # Mistral temperature and top_p controls for TEST
    main_mistral_temp_key = f"{session_key}_main_mistral_temperature"
    main_mistral_top_p_key = f"{session_key}_main_mistral_top_p"
    if main_mistral_temp_key not in st.session_state:
        st.session_state[main_mistral_temp_key] = 0.7
    if main_mistral_top_p_key not in st.session_state:
        st.session_state[main_mistral_top_p_key] = 0.9
    
    main_mistral_cols = st.columns(2)
    with main_mistral_cols[0]:
        main_mistral_temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            step=0.1,
            key=main_mistral_temp_key,
        )
    with main_mistral_cols[1]:
        main_mistral_top_p = st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key=main_mistral_top_p_key,
        )
    
    # Simple prompt display - just show what was last sent to Mistral
    last_mistral_prompt_key = f"{session_key}_last_mistral_prompt"
    
    # Get the last prompt that was sent to Mistral (empty if none yet)
    last_prompt = st.session_state.get(last_mistral_prompt_key, "")
    
    with st.expander("🔍 Mistral Prompt"):
        if last_prompt:
            st.text_area(
                "Letzter an Mistral gesendeter Prompt",
                value=last_prompt,
                height=400,
                label_visibility="collapsed",
                disabled=True
            )
        else:
            st.info("Prompt wird hier angezeigt, nachdem 'Summary generieren' gedrückt wurde.")
    
    r2_state_key = f"{session_key}_r2"
    if r2_state_key not in st.session_state:
        st.session_state[r2_state_key] = ""
    
    if st.button("🧾 Summary generieren", key=f"{session_key}_gen_main", use_container_width=True):
        if not MISTRAL_API_KEY:
            st.error("❌ MISTRAL_API_KEY nicht gesetzt.")
        elif not has_non_empty_answers(mainrecap_inhalt_text):
            st.warning("⚠️ Keine Antworten vorhanden. Bitte erst Antworten generieren oder manuell eingeben.")
        else:
            try:
                # Preserve other text areas' state before rerun
                if example1_state_key in st.session_state:
                    preserved_ex1 = st.session_state[example1_state_key]
                    st.session_state[f"{example1_state_key}_preserve"] = preserved_ex1
                if example2_state_key in st.session_state:
                    preserved_ex2 = st.session_state[example2_state_key]
                    st.session_state[f"{example2_state_key}_preserve"] = preserved_ex2
                
                answer_count = count_answer_lines(mainrecap_inhalt_text)
                max_words_value = choose_max_words(answer_count, active_word_limit_config)
                system_prompt_for_main = assemble_system_prompt(
                    collect_global_section_values(session_key),
                    collect_exercise_section_values(session_key),
                    max_words_value,
                )
                
                # Get the current text area values (including any generated recaps)
                example1_final = st.session_state.get(example1_state_key, exercise_sections.get("example1", ""))
                example2_final = st.session_state.get(example2_state_key, exercise_sections.get("example2", ""))
                
                # Build complete Mistral prompt
                mistral_prompt = f"{system_prompt_for_main}\n\n# BEISPIELE\n\n## Beispiel 1\n{example1_final}\n\n## Beispiel 2\n{example2_final}\n\n# INHALT\n{mainrecap_inhalt_text}"
                print_prompt_debug(
                    "text",
                    mistral_prompt,
                    {
                        "max_tokens": 200,
                        "temperature": main_mistral_temperature,
                        "top_p": main_mistral_top_p,
                        "max_words": max_words_value,
                    },
                )
                
                with st.spinner("Generiere Zusammenfassung..."):
                    # Store the exact prompt that we're about to send
                    st.session_state[last_mistral_prompt_key] = mistral_prompt
                    
                    recap = generate_summary_with_mistral(
                        prompt=mistral_prompt,
                        api_key=MISTRAL_API_KEY,
                        max_tokens=200,
                        temperature=main_mistral_temperature,
                        top_p=main_mistral_top_p,
                    )
                    st.session_state[r2_state_key] = recap
                
                st.success("✅ Zusammenfassung generiert")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Fehler: {e}")
                import traceback
                with st.expander("🔍 Fehlerdetails"):
                    st.code(traceback.format_exc())
    
    st.text_area(
        "Generierte Zusammenfassung",
        value=st.session_state[r2_state_key],
        key=r2_state_key,
        height=200,
        label_visibility="collapsed",
    )
    
    # Display word count for the generated summary
    summary_text = st.session_state.get(r2_state_key, "")
    if summary_text.strip():
        word_count = len(summary_text.split())
        st.caption(f"📊 Anzahl Wörter: {word_count}")
    else:
        st.caption("📊 Anzahl Wörter: 0")

else:
    st.info("Bitte wähle eine Übung aus.")


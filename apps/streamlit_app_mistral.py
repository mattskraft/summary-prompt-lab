import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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
    SYNTH_ANSWERS_PROMPT_PATH = PROJECT_ROOT / "config" / "synth_answers_prompt.txt"
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


def get_all_exercise_names(hierarchy: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """Extract all exercise names from the hierarchy."""
    exercise_names: List[str] = []
    for thema_dict in hierarchy.values():
        for path_list in thema_dict.values():
            exercise_names.extend(path_list)
    return sorted(set(exercise_names))


def ensure_exercise_prompts_json(exercise_names: List[str]) -> Path:
    """Ensure the exercise_system_prompts.json file exists with all exercises.
    Returns the path to the JSON file."""
    json_path = PROJECT_ROOT / "config" / "exercise_system_prompts.json"
    
    # Load existing JSON if it exists
    existing_data: Dict[str, Dict[str, str]] = {}
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
                # Handle both old format (flat) and new format (nested)
                if loaded and isinstance(list(loaded.values())[0], str):
                    # Old format: convert to new format (empty values)
                    existing_data = {
                        name: {"system_prompt": "", "example1": "", "example2": ""}
                        for name in loaded.keys()
                    }
                else:
                    existing_data = loaded
                    # Remove mainrecap if it exists
                    for name in existing_data:
                        if "mainrecap" in existing_data[name]:
                            del existing_data[name]["mainrecap"]
        except Exception:
            existing_data = {}
    
    # Update with any missing exercises (with empty values)
    updated = False
    for exercise_name in exercise_names:
        if exercise_name not in existing_data:
            existing_data[exercise_name] = {
                "system_prompt": "",
                "example1": "",
                "example2": ""
            }
            updated = True
        elif not isinstance(existing_data[exercise_name], dict):
            # Convert old format entry to new format (empty values)
            existing_data[exercise_name] = {
                "system_prompt": "",
                "example1": "",
                "example2": ""
            }
            updated = True
        elif "mainrecap" in existing_data[exercise_name]:
            # Remove mainrecap if it exists
            del existing_data[exercise_name]["mainrecap"]
            updated = True
    
    # Write back if updated
    if updated:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    return json_path


def load_exercise_prompts_json() -> Dict[str, Dict[str, str]]:
    """Load the exercise prompts JSON file with new structure."""
    json_path = PROJECT_ROOT / "config" / "exercise_system_prompts.json"
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
                # Handle both old format (flat) and new format (nested)
                if loaded and isinstance(list(loaded.values())[0], str):
                    # Old format: convert to new format (empty values)
                    return {
                        name: {"system_prompt": "", "example1": "", "example2": ""}
                        for name in loaded.keys()
                    }
                # Remove mainrecap if it exists
                result = {}
                for name, data in loaded.items():
                    result[name] = {k: v for k, v in data.items() if k != "mainrecap"}
                return result
        except Exception:
            return {}
    return {}


def save_exercise_data(exercise_name: str, data: Dict[str, str]) -> bool:
    """Save exercise data to the JSON file."""
    json_path = PROJECT_ROOT / "config" / "exercise_system_prompts.json"
    try:
        # Load existing data
        existing_data = load_exercise_prompts_json()
        
        # Update the entry
        if exercise_name not in existing_data:
            existing_data[exercise_name] = {
                "system_prompt": "",
                "example1": "",
                "example2": ""
            }
        
        # Filter out mainrecap if present
        filtered_data = {k: v for k, v in data.items() if k != "mainrecap"}
        existing_data[exercise_name].update(filtered_data)
        
        # Write back
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Fehler beim Speichern: {e}")
        return False


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


def segments_to_inhalt(segments: List[Dict[str, Any]], empty_answers: bool = False) -> str:
    """Convert segments to INHALT format (one line per segment).
    
    Args:
        segments: List of segment dictionaries
        empty_answers: If True, use empty strings for answers instead of actual values
    """
    lines: List[str] = []
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
                    answer_text = str(int(answer_val))
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


def replace_answers_in_inhalt(inhalt_text: str, new_answers: List[Dict[str, Any]]) -> str:
    """Replace ANTWORT blocks in INHALT with new answers, preserving structure."""
    segments = inhalt_to_segments(inhalt_text)
    answer_idx = 0
    
    for seg in segments:
        if "Answer" in seg:
            if answer_idx < len(new_answers):
                new_seg = new_answers[answer_idx]
                if "Answer" in new_seg:
                    seg["Answer"] = new_seg["Answer"]
                answer_idx += 1
    
    return segments_to_inhalt(segments)


def update_system_prompt_length(system_prompt: str, max_words: int) -> str:
    """Update the last line of system prompt with max word count."""
    lines = system_prompt.split("\n")
    
    # Find and replace the line with "Länge: Maximal"
    for i in range(len(lines) - 1, -1, -1):
        if "Länge:" in lines[i] or "Maximal" in lines[i]:
            # Replace the number in this line
            lines[i] = re.sub(r"Maximal\s+\d+\s+Wörter", f"Maximal {max_words} Wörter", lines[i])
            lines[i] = re.sub(r"Länge:\s*Maximal\s+\d+\s+Wörter", f"Länge: Maximal {max_words} Wörter", lines[i])
            break
    else:
        # If no line found, append it
        lines.append(f"- Länge: Maximal {max_words} Wörter")
    
    return "\n".join(lines)


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
    ensure_exercise_prompts_json(all_exercise_names)

with st.sidebar:
    st.header("Navigation")
    
    thema_key = "nav_selected_thema"
    path_key = "nav_selected_path"
    uebung_key = "nav_selected_uebung"
    
    themen = sorted(hier.keys())
    
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
                                    st.session_state[widget_key] = answer_val
                                elif isinstance(answer_val, (int, float)):
                                    st.session_state[widget_key] = int(answer_val)
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
                st.rerun()
        
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
                st.rerun()
        
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
                st.rerun()
        
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
        inhalt = segments_to_inhalt(segments_for_prompt)
        st.session_state[example1_state_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_ex1_clicked"] = False
    
    transfer_ex2_clicked = st.session_state.get(f"{sel_uebung}_transfer_ex2_clicked", False)
    if transfer_ex2_clicked:
        inhalt = segments_to_inhalt(segments_for_prompt)
        st.session_state[example2_state_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_ex2_clicked"] = False
    
    transfer_main_clicked = st.session_state.get(f"{sel_uebung}_transfer_main_clicked", False)
    if transfer_main_clicked:
        inhalt = segments_to_inhalt(segments_for_prompt)
        st.session_state[mainrecap_inhalt_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_main_clicked"] = False
    
    # Load exercise data
    exercise_data = load_exercise_prompts_json()
    current_exercise_data = exercise_data.get(sel_uebung, {
        "system_prompt": "",
        "example1": "",
        "example2": ""
    })
    
    # Initialize empty values (only for display/fallback, don't modify in place)
    default_prompt_path = PROJECT_ROOT / "config" / "recap_system_prompt.txt"
    default_system_prompt = ""
    if not current_exercise_data.get("system_prompt"):
        if default_prompt_path.exists():
            default_system_prompt = default_prompt_path.read_text(encoding="utf-8").strip()
    
    default_example1 = ""
    if not current_exercise_data.get("example1"):
        # Initialize with INHALT from segments with empty answers
        default_example1 = segments_to_inhalt(segments, empty_answers=True)
    
    default_example2 = ""
    if not current_exercise_data.get("example2"):
        # Initialize with INHALT from segments with empty answers
        default_example2 = segments_to_inhalt(segments, empty_answers=True)
    
    # System Prompt Section
    st.markdown("---")
    st.subheader("System-Prompt")
    
    system_prompt_state_key = f"{session_key}_system_prompt"
    system_prompt_base_key = f"{session_key}_system_prompt_base"
    
    # Initialize system prompt
    if system_prompt_state_key not in st.session_state:
        base_prompt = current_exercise_data.get("system_prompt", "") or default_system_prompt
        st.session_state[system_prompt_base_key] = base_prompt
        st.session_state[system_prompt_state_key] = base_prompt
    
    # Get current prompt (may have been manually edited)
    base_prompt = st.session_state.get(system_prompt_base_key, current_exercise_data.get("system_prompt", ""))
    current_prompt = st.session_state.get(system_prompt_state_key, "")
    if not current_prompt:
        current_prompt = base_prompt
        st.session_state[system_prompt_state_key] = current_prompt
    
    system_prompt_input = st.text_area(
        "System-Prompt",
        value=current_prompt,
        key=system_prompt_state_key,
        height=200,
        label_visibility="collapsed",
    )
    
    # Slider for system prompt length
    system_length_key = f"{session_key}_system_length"
    if system_length_key not in st.session_state:
        st.session_state[system_length_key] = 30
    
    system_length = st.slider(
        "Maximale Wortanzahl",
        min_value=10,
        max_value=60,
        key=system_length_key,
    )
    
    # Update system prompt with slider value immediately
    # Track slider changes
    slider_changed_key = f"{system_length_key}_last_value"
    if slider_changed_key not in st.session_state:
        st.session_state[slider_changed_key] = system_length
    
    # Update base prompt if slider changed
    if st.session_state[slider_changed_key] != system_length:
        st.session_state[slider_changed_key] = system_length
        # Update base prompt with new length
        updated_base = update_system_prompt_length(base_prompt, system_length)
        st.session_state[system_prompt_base_key] = updated_base
        st.session_state[system_prompt_state_key] = updated_base
    
    # Update base prompt when user edits (remove length line for base)
    if system_prompt_input != current_prompt:
        # User edited - extract base (without length line)
        base_lines = system_prompt_input.split("\n")
        base_lines_clean = []
        for line in base_lines:
            if "Länge:" not in line and "Maximal" not in line:
                base_lines_clean.append(line)
        base_clean = "\n".join(base_lines_clean).strip()
        st.session_state[system_prompt_base_key] = base_clean
    
    system_button_cols = st.columns(2)
    with system_button_cols[0]:
        if st.button("📥 System-Prompt laden", key=f"{session_key}_load_system", use_container_width=True):
            base_prompt = current_exercise_data.get("system_prompt", "")
            st.session_state[system_prompt_base_key] = base_prompt
            updated = update_system_prompt_length(base_prompt, system_length)
            st.session_state[system_prompt_state_key] = updated
            st.session_state[f"{system_length_key}_last_value"] = system_length
            st.rerun()
    
    with system_button_cols[1]:
        save_system_key = f"{session_key}_save_system"
        save_system_clicked = st.button("💾 System-Prompt speichern", key=save_system_key, use_container_width=True)
        if save_system_clicked:
            # Show confirmation dialog
            confirm_key = f"{save_system_key}_dialog"
            if confirm_key not in st.session_state:
                st.session_state[confirm_key] = True
            
            if st.session_state[confirm_key]:
                with st.dialog("Bestätigung") as dialog:
                    st.write(f"Möchten Sie den System-Prompt für '{sel_uebung}' wirklich überschreiben?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Ja", key=f"{save_system_key}_yes", use_container_width=True):
                            # Extract base prompt (without length line) for saving
                            base_lines = system_prompt_input.split("\n")
                            base_lines_clean = [line for line in base_lines if "Länge:" not in line and "Maximal" not in line]
                            base_prompt_clean = "\n".join(base_lines_clean).strip()
                            
                            if save_exercise_data(sel_uebung, {"system_prompt": base_prompt_clean}):
                                st.session_state[system_prompt_base_key] = base_prompt_clean
                                st.session_state[confirm_key] = False
                                dialog.close()
                                st.rerun()
                            else:
                                st.error("❌ Fehler beim Speichern")
                    with col2:
                        if st.button("Nein", key=f"{save_system_key}_no", use_container_width=True):
                            st.session_state[confirm_key] = False
                            dialog.close()
                            st.rerun()
    
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
    
    # Check if transfer button was clicked (check button state after render)
    transfer_ex1_clicked = st.session_state.get(f"{sel_uebung}_transfer_ex1_clicked", False)
    if transfer_ex1_clicked:
        inhalt = segments_to_inhalt(segments_for_prompt)
        st.session_state[example1_state_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_ex1_clicked"] = False
    
    # Initialize only if key doesn't exist (preserve existing content)
    # IMPORTANT: Only initialize if the key truly doesn't exist, don't overwrite existing content
    # This check happens AFTER restore, so preserved values won't be overwritten
    # CRITICAL: Never overwrite existing text area content, even if segments change
    if example1_state_key not in st.session_state:
        example1_value = current_exercise_data.get("example1", "") or default_example1
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
    
    # Slider for Beispiel 1
    example1_length_key = f"{session_key}_example1_length"
    if example1_length_key not in st.session_state:
        st.session_state[example1_length_key] = 30
    
    example1_length = st.slider(
        "Maximale Wortanzahl",
        min_value=10,
        max_value=60,
        key=example1_length_key,
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
                    
                    # Get system prompt with Beispiel 1 slider value
                    system_prompt_for_ex1 = update_system_prompt_length(
                        st.session_state[system_prompt_state_key],
                        example1_length
                    )
                    
                    # Build prompt: system prompt + INHALT
                    prompt = f"{system_prompt_for_ex1}\n\n{example1_text}"
                    
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
                            if save_exercise_data(sel_uebung, {"example1": example1_text}):
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
            example1_value = current_exercise_data.get("example1", "")
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
    
    # Check transfer flag
    transfer_ex2_clicked = st.session_state.get(f"{sel_uebung}_transfer_ex2_clicked", False)
    if transfer_ex2_clicked:
        inhalt = segments_to_inhalt(segments_for_prompt)
        st.session_state[example2_state_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_ex2_clicked"] = False
    
    # Initialize only if key doesn't exist (preserve existing content)
    # This check happens AFTER restore, so preserved values won't be overwritten
    # CRITICAL: Never overwrite existing text area content, even if segments change
    if example2_state_key not in st.session_state:
        example2_value = current_exercise_data.get("example2", "") or default_example2
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
    
    example2_length_key = f"{session_key}_example2_length"
    if example2_length_key not in st.session_state:
        st.session_state[example2_length_key] = 30
    
    example2_length = st.slider(
        "Maximale Wortanzahl",
        min_value=10,
        max_value=60,
        key=example2_length_key,
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
                    
                    system_prompt_for_ex2 = update_system_prompt_length(
                        st.session_state[system_prompt_state_key],
                        example2_length
                    )
                    prompt = f"{system_prompt_for_ex2}\n\n{example2_text}"
                    
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
                            if save_exercise_data(sel_uebung, {"example2": example2_text}):
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
            example2_value = current_exercise_data.get("example2", "")
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
    mainrecap_length_key = f"{session_key}_mainrecap_length"
    if mainrecap_length_key not in st.session_state:
        st.session_state[mainrecap_length_key] = 30
    
    mainrecap_length = st.slider(
        "Maximale Wortanzahl",
        min_value=10,
        max_value=60,
        key=mainrecap_length_key,
    )
    
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
                
                # Build the complete Mistral prompt
                system_prompt_for_main = update_system_prompt_length(
                    st.session_state[system_prompt_state_key],
                    mainrecap_length
                )
                
                # Get the current text area values (including any generated recaps)
                example1_final = st.session_state.get(example1_state_key, current_exercise_data.get("example1", ""))
                example2_final = st.session_state.get(example2_state_key, current_exercise_data.get("example2", ""))
                
                # Build complete Mistral prompt
                mistral_prompt = f"{system_prompt_for_main}\n\n# BEISPIELE\n\n## Beispiel 1\n{example1_final}\n\n## Beispiel 2\n{example2_final}\n\n# INHALT\n{mainrecap_inhalt_text}"
                
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


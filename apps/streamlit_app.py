import json
import os
import random
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
        SUICIDE_LEXICON_PATH,
    )
    import kiso_input.config as config_module
    
    # Get APP_PASSWORD from Streamlit secrets first, then fallback to config/.env
    # Streamlit Cloud provides secrets via st.secrets, not environment variables
    # When running locally, password is loaded from config/.env via config module
    APP_PASSWORD = None
    PASSWORD_FROM_SECRETS = False  # Track if password comes from Streamlit secrets
    
    # First try to get from config module (loads from config/.env when running locally)
    APP_PASSWORD = getattr(config_module, "APP_PASSWORD", None)
    
    # Then try Streamlit secrets (for Streamlit Cloud - overrides local .env)
    try:
        # Try to get from Streamlit secrets (Streamlit Cloud)
        if hasattr(st, "secrets"):
            try:
                # Check if secrets is accessible (may raise StreamlitSecretNotFoundError)
                _ = len(st.secrets)  # This will raise if no secrets file exists
                # If we get here, secrets exist
                if "APP_PASSWORD" in st.secrets:
                    APP_PASSWORD = st.secrets["APP_PASSWORD"]
                    PASSWORD_FROM_SECRETS = True
                elif hasattr(st.secrets, "get"):
                    app_pwd = st.secrets.get("APP_PASSWORD")
                    if app_pwd:
                        APP_PASSWORD = app_pwd
                        PASSWORD_FROM_SECRETS = True
            except Exception:
                # No secrets file found, use config/.env value (already loaded above)
                pass
    except (AttributeError, KeyError, TypeError) as e:
        # Use config/.env value (already loaded above)
        pass
    
    # Get GEMINI_API_KEY from Streamlit secrets first, then fallback to config
    GEMINI_API_KEY = None
    try:
        # Try to get from Streamlit secrets (Streamlit Cloud)
        if hasattr(st, "secrets"):
            try:
                # Check if secrets is accessible (may raise StreamlitSecretNotFoundError)
                _ = len(st.secrets)  # This will raise if no secrets file exists
                # If we get here, secrets exist
                GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or None
            except Exception:
                # No secrets file found, silently fall through to config fallback
                pass
    except (AttributeError, KeyError, TypeError):
        pass
    
    # Fallback to config module if not found in secrets
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = getattr(config_module, "GEMINI_API_KEY", None)
    
    from kiso_input import (
        assess_free_text_answers,
        build_summary_prompt,
        generate_answers_with_gemini,
        generate_summary_with_gemini,
        get_prompt_segments_from_exercise,
        load_self_harm_lexicon,
        prompt_segments_to_text,
    )
    from kiso_input.processing.local_models import (
        generate_summary_with_model,
        get_available_models,
        is_local_models_available,
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
    default_prompt_path = PROJECT_ROOT / "config" / "recap_system_prompt.txt"
    
    # Load default prompt text
    default_prompt_text = ""
    if default_prompt_path.exists():
        default_prompt_text = default_prompt_path.read_text(encoding="utf-8").strip()
    
    # Load existing JSON if it exists
    existing_data: Dict[str, str] = {}
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = {}
    
    # Update with any missing exercises
    updated = False
    for exercise_name in exercise_names:
        if exercise_name not in existing_data:
            existing_data[exercise_name] = default_prompt_text
            updated = True
    
    # Write back if updated
    if updated:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    return json_path


def load_exercise_prompts_json() -> Dict[str, str]:
    """Load the exercise system prompts JSON file."""
    json_path = PROJECT_ROOT / "config" / "exercise_system_prompts.json"
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_exercise_prompt(exercise_name: str, prompt_text: str) -> bool:
    """Save the system prompt for a specific exercise to the JSON file."""
    json_path = PROJECT_ROOT / "config" / "exercise_system_prompts.json"
    try:
        # Load existing data
        existing_data: Dict[str, str] = {}
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
        
        # Update the entry
        existing_data[exercise_name] = prompt_text.strip()
        
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
    """Return {Thema: {Path: [Übung, ...]}} for the merged structured JSON.
    Tries to be forgiving if there are extra wrapper keys.
    """
    hierarchy: Dict[str, Dict[str, List[str]]] = {}

    def add_entry(thema: str, pfad: str, uebung: str) -> None:
        hierarchy.setdefault(thema, {}).setdefault(pfad, []).append(uebung)

    # Expected shape: {Thema: {Path: {Übung: [entries...]}}}
    if isinstance(ex_struct, dict):
        for thema, val in ex_struct.items():
            if not isinstance(val, dict):
                continue
            for pfad, val2 in val.items():
                if not isinstance(val2, dict):
                    continue
                for uebung, content in val2.items():
                    # content is typically a list of dicts (exercise entries)
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
    """Render the segments and collect user inputs in a parallel list.
    We preserve structure and only mutate the Answer fields with UI input values.
    """
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

            # Multiple choice questions driven by AnswerOptions list
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

            # Slider-style answers represented by dict
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

            # Free text placeholder
            elif isinstance(answer_val, str) and answer_val.strip().lower() in {"free_text", "freetext", "free text"}:
                user_val = st.text_input("Answer", key=key, label_visibility="collapsed")
                filled.append({"Answer": user_val})

            # Generated string answer (editable)
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


# -----------------------------
# UI Layout
# -----------------------------
st.set_page_config(page_title="Summary Prompt Lab", layout="centered")

# Password protection
# When running locally with password in config/.env, auto-verify (no prompt)
# On Streamlit Cloud, secrets are used and no password prompt is needed
if APP_PASSWORD and not PASSWORD_FROM_SECRETS:
    # Running locally with password from config/.env - auto-verify
    if "password_verified" not in st.session_state:
        st.session_state.password_verified = True  # Auto-verify when password is in .env

st.title("🧪 Summary Prompt Lab")

# Load files first (before sidebar)
files_ok = True
ex_struct = None
story_nodes = None
try:
    # Debug: Show path resolution in development
    if os.getenv("STREAMLIT_DEBUG"):
        with st.expander("🔍 Debug: Path Resolution"):
            st.write(f"**PROJECT_ROOT (from config):** {PROJECT_ROOT}")
            st.write(f"**Current working directory:** {os.getcwd()}")
            st.write(f"**STRUCT_JSON_PATH:** {STRUCT_JSON_PATH}")
            st.write(f"**SN_JSON_PATH:** {SN_JSON_PATH}")
            if STRUCT_JSON_PATH:
                st.write(f"**STRUCT_JSON_PATH exists:** {os.path.exists(STRUCT_JSON_PATH)}")
            if SN_JSON_PATH:
                st.write(f"**SN_JSON_PATH exists:** {os.path.exists(SN_JSON_PATH)}")
            # Check default paths
            default_struct = PROJECT_ROOT / "data" / "processed" / "kiso_app_merged_structured.json"
            default_sn = PROJECT_ROOT / "data" / "processed" / "kiso_app_storynodes_struct.json"
            st.write(f"**Default STRUCT path exists:** {default_struct.exists()} ({default_struct})")
            st.write(f"**Default SN path exists:** {default_sn.exists()} ({default_sn})")
    
    if not STRUCT_JSON_PATH or not os.path.exists(STRUCT_JSON_PATH):
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {STRUCT_JSON_PATH}")
    if not SN_JSON_PATH or not os.path.exists(SN_JSON_PATH):
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {SN_JSON_PATH}")
    
    ex_struct = load_json(STRUCT_JSON_PATH)
    story_nodes = load_json(SN_JSON_PATH)
except FileNotFoundError as e:
    files_ok = False
    st.error(str(e))
    with st.expander("🔍 Debug Information"):
        st.write(f"**PROJECT_ROOT:** {PROJECT_ROOT}")
        st.write(f"**Current working directory:** {os.getcwd()}")
        st.write(f"**STRUCT_JSON_PATH:** {STRUCT_JSON_PATH}")
        st.write(f"**SN_JSON_PATH:** {SN_JSON_PATH}")
        # Try to find files
        possible_paths = [
            PROJECT_ROOT / "data" / "processed" / "kiso_app_merged_structured.json",
            Path("data/processed/kiso_app_merged_structured.json"),
            Path("./data/processed/kiso_app_merged_structured.json"),
        ]
        st.write("**Checking possible paths:**")
        for path in possible_paths:
            exists = path.exists() if path else False
            st.write(f"- {path}: {'✅' if exists else '❌'}")
    if not STRUCT_JSON_PATH or not SN_JSON_PATH:
        st.info("""
        💡 Setze die Umgebungsvariablen in Streamlit Cloud Secrets:
        - `KISO_STRUCT_JSON` für die structured JSON-Datei
        - `KISO_SN_JSON` für die story nodes JSON-Datei
        
        Oder verwende die Standardpfade:
        - `data/processed/kiso_app_merged_structured.json`
        - `data/processed/kiso_app_storynodes_struct.json`
        """)
except Exception as e:
    files_ok = False
    st.error(f"Fehler beim Laden der Dateien: {e}")
    import traceback
    with st.expander("🔍 Error Details"):
        st.code(traceback.format_exc())

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
    
    st.markdown("---")
    st.header("Konfiguration")
    seed = st.number_input("Zufalls-Seed (für Branches)", min_value=0, value=42, step=1)
    
    if not GEMINI_API_KEY:
        st.warning("⚠️ GEMINI_API_KEY nicht gesetzt. Antwort-Generierung wird nicht funktionieren.")

if sel_uebung:
    # Store these in session state so sidebar buttons can access them
    session_key = f"generated_segments_{sel_uebung}_{seed}"
    if session_key not in st.session_state:
        st.session_state[session_key] = None
    # Build segments from your processing pipeline
    try:
        segments = get_prompt_segments_from_exercise(
            exercise_name=sel_uebung,
            json_struct_path=STRUCT_JSON_PATH,
            json_sn_struct_path=SN_JSON_PATH,
            seed=int(seed),
        )
    except Exception as e:
        st.error(f"Fehler bei der Segment-Generierung: {e}")
        st.stop()
    
    # Choose which segments to display
    has_generated = st.session_state[session_key] is not None
    segments_to_display = (
        st.session_state[session_key] if has_generated else segments
    )
    key_prefix = (
        f"{session_key}_generated" if has_generated else f"{session_key}_original"
    )

    # Add buttons to sidebar
    with st.sidebar:
        st.markdown("---")
        st.header("Aktionen")
        
        if st.button("✨ Generiere Antworten"):
            if not GEMINI_API_KEY:
                st.error("❌ GEMINI_API_KEY nicht gesetzt. Bitte setze die Umgebungsvariable GEMINI_API_KEY.")
            else:
                try:
                    with st.spinner("Generiere Antworten mit Gemini..."):
                        # Generate answers using Gemini
                        result = generate_answers_with_gemini(
                            segments=segments,
                            api_key=GEMINI_API_KEY,
                            debug=False,  # Set to True for console debugging
                            return_debug_info=True,
                        )
                        generated_segments, debug_info = result
                        
                        # Store in session state
                        st.session_state[session_key] = generated_segments
                        
                        # Debug: Store debug info for debugging
                        st.session_state[f"{session_key}_debug"] = {
                            "num_segments_input": len(segments),
                            "num_segments_output": len(generated_segments),
                            "has_questions": any("Question" in s for s in segments),
                            "gemini_debug": debug_info,
                        }
                    st.success("✅ Antworten generiert")
                    # Clear cached widget inputs so defaults update after rerun
                    prefixes_to_clear = [
                            f"{session_key}_original_ans_",
                        f"{session_key}_generated_ans_",
                    ]
                    for state_key in list(st.session_state.keys()):
                        if any(state_key.startswith(prefix) for prefix in prefixes_to_clear):
                            del st.session_state[state_key]
                        elif any(
                            state_key.startswith(prefix) and state_key.endswith(("_min", "_max"))
                            for prefix in prefixes_to_clear
                        ):
                            del st.session_state[state_key]
                    # Rerun to show updated UI
                    st.rerun()
                except ImportError as e:
                    st.error(f"❌ Fehler: {e}")
                    st.info("💡 Installiere das Paket mit: `pip install google-genai`")
                except Exception as e:
                    st.error(f"❌ Fehler bei der Antwort-Generierung: {e}")
                    import traceback
                    with st.expander("🔍 Fehlerdetails anzeigen"):
                        st.code(traceback.format_exc(), language="python")
        
        # Store segments in session state for summary button
        st.session_state[f"{session_key}_segments"] = segments
        st.session_state[f"{session_key}_segments_to_display"] = segments_to_display
        st.session_state[f"{session_key}_has_generated"] = has_generated

    st.markdown("---")
    st.subheader("Übung")
    filled_segments = render_segments_ui(segments_to_display, key_prefix=key_prefix)
    
    # Check if any answers were manually filled and update session state
    # Merge filled answers back into segment structure
    def merge_filled_answers_into_segments(
        original_segments: List[Dict[str, Any]], 
        filled_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge filled answers from UI back into the original segment structure."""
        result: List[Dict[str, Any]] = []
        filled_idx = 0

        def advance_to_next_answer(idx: int) -> int:
            """Advance idx to the next filled segment that contains an Answer."""
            while idx < len(filled_segments) and "Answer" not in filled_segments[idx]:
                idx += 1
            return idx

        # Ensure we start on an Answer item if present
        filled_idx = advance_to_next_answer(filled_idx)

        for seg in original_segments:
            if "Text" in seg or "Question" in seg:
                result.append(seg.copy())
                continue

            if "Answer" in seg:
                # Align to the next filled answer entry
                filled_idx = advance_to_next_answer(filled_idx)
                if filled_idx < len(filled_segments):
                    filled_entry = filled_segments[filled_idx]
                    if "Answer" in filled_entry:
                        new_seg = seg.copy()
                        # Apply updated Answer and propagate metadata
                        new_seg["Answer"] = filled_entry["Answer"]
                        if "AnswerOptions" in filled_entry:
                            new_seg["AnswerOptions"] = filled_entry["AnswerOptions"]
                        if "AllowMultiple" in filled_entry:
                            new_seg["AllowMultiple"] = filled_entry["AllowMultiple"]
                        result.append(new_seg)
                        filled_idx += 1
                        continue
                # Fallback: keep original if no filled answer aligned
                result.append(seg.copy())
                continue

            # Any other segment types
            result.append(seg.copy())

        return result
    
    # Check if filled segments have any non-empty answers
    has_manual_answers = any(
        "Answer" in filled and filled["Answer"] not in (None, "", [], {})
        for filled in filled_segments
    )
    
    # If there are manual answers, merge them into segments and update session state
    # This will be used on the next rerun
    merged_segments_for_debug = None
    if has_manual_answers:
        merged_segments = merge_filled_answers_into_segments(
            segments_to_display if has_generated else segments, 
            filled_segments
        )
        # Update session state if segments changed
        current_segments = st.session_state[session_key] if has_generated else segments
        if merged_segments != current_segments:
            st.session_state[session_key] = merged_segments
        merged_segments_for_debug = merged_segments
    
    # Build live-updating summary prompt from current state
    segments_for_prompt = merge_filled_answers_into_segments(
        segments_to_display if has_generated else segments,
        filled_segments
    )
    live_prompt_text = build_summary_prompt(segments_for_prompt, exercise_name=sel_uebung)
    
    # Helper function to extract system prompt from complete prompt
    def extract_system_prompt_from_complete(complete_prompt: str) -> str:
        """Extract the system prompt section from the complete prompt."""
        if "SYSTEM:\n\n" in complete_prompt:
            parts = complete_prompt.split("SYSTEM:\n\n", 1)
            if len(parts) > 1:
                system_part = parts[1].split("\n\n" + ("-" * 80) + "\nINHALT:\n\n", 1)[0]
                return system_part.strip()
        return ""
    
    # Helper function to rebuild complete prompt with new system prompt
    def rebuild_complete_prompt(system_prompt_text: str, segments_for_rebuild: List[Dict[str, Any]]) -> str:
        """Rebuild the complete prompt with a new system prompt."""
        content_lines: List[str] = []
        for segment in segments_for_rebuild:
            if "Text" in segment:
                clean_text = " ".join(segment["Text"].split())
                content_lines.append(f"TEXT: {clean_text}")
            elif "Question" in segment:
                question_text = segment["Question"]
                content_lines.append(f"FRAGE: {question_text}")
            elif "Answer" in segment:
                answer_val = segment.get("Answer")
                answer_text: Optional[str] = None
                if isinstance(answer_val, list):
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
                        answer_text = "; ".join(sub_parts)
                if answer_text:
                    content_lines.append(f"ANTWORT: {answer_text}")
        return (
            ("-" * 80)
            + "\nSYSTEM:\n\n"
            + system_prompt_text
            + "\n\n"
            + ("-" * 80)
            + "\nINHALT:\n\n"
            + "\n\n".join(content_lines)
        )
    
    st.markdown("---")
    st.subheader("Zusammenfassungs-Prompt")
    
    # Load current system prompt for this exercise
    exercise_prompts = load_exercise_prompts_json()
    system_prompt_state_key = f"{session_key}_system_prompt"
    system_prompt_baseline_key = f"{session_key}_system_prompt_baseline"
    
    # Initialize system prompt from JSON or extract from live prompt
    if system_prompt_state_key not in st.session_state:
        if sel_uebung and sel_uebung in exercise_prompts:
            initial_system_prompt = exercise_prompts[sel_uebung]
        else:
            initial_system_prompt = extract_system_prompt_from_complete(live_prompt_text)
        st.session_state[system_prompt_state_key] = initial_system_prompt
        st.session_state[system_prompt_baseline_key] = initial_system_prompt
    elif sel_uebung and sel_uebung in exercise_prompts:
        # If exercise changed, update system prompt from JSON
        if st.session_state.get(system_prompt_baseline_key) != exercise_prompts[sel_uebung]:
            st.session_state[system_prompt_state_key] = exercise_prompts[sel_uebung]
            st.session_state[system_prompt_baseline_key] = exercise_prompts[sel_uebung]
    
    # Display system prompt field
    st.markdown("**System-Prompt:**")
    system_prompt_input = st.text_area(
        "System-Prompt",
        value=st.session_state[system_prompt_state_key],
        key=system_prompt_state_key,
        height=660,
        label_visibility="collapsed",
    )
    
    # Save button for system prompt
    save_button_key = f"{session_key}_save_system_prompt"
    save_clicked = st.button("💾 Änderung speichern", key=save_button_key, use_container_width=True)
    if save_clicked:
        if sel_uebung:
            if save_exercise_prompt(sel_uebung, system_prompt_input):
                st.success(f"✅ System-Prompt für '{sel_uebung}' wurde gespeichert.")
                # Update baseline to reflect saved state
                st.session_state[system_prompt_baseline_key] = system_prompt_input
            else:
                st.error("❌ Fehler beim Speichern des System-Prompts.")
        else:
            st.warning("⚠️ Keine Übung ausgewählt.")
    
    # Rebuild complete prompt if system prompt changed
    current_system_prompt = system_prompt_input  # Use the current input value
    extracted_system = extract_system_prompt_from_complete(live_prompt_text)
    system_prompt_changed = current_system_prompt != extracted_system
    if system_prompt_changed:
        # System prompt was changed, rebuild complete prompt
        live_prompt_text = rebuild_complete_prompt(current_system_prompt, segments_for_prompt)
    
    # Preserve manual edits unless the underlying segments changed or system prompt changed.
    prompt_state_key = f"{session_key}_summary_prompt"
    prompt_baseline_key = f"{session_key}_summary_prompt_baseline"
    # Initialize on first render
    initial_value: Optional[str] = None
    if prompt_state_key not in st.session_state:
        # First render: initialize widget state and provide default via value
        st.session_state[prompt_state_key] = live_prompt_text
        st.session_state[prompt_baseline_key] = live_prompt_text
        initial_value = live_prompt_text
    else:
        # If the auto-generated prompt changed (due to changed answers or system prompt), refresh field and baseline
        if st.session_state.get(prompt_baseline_key) != live_prompt_text or system_prompt_changed:
            st.session_state[prompt_state_key] = live_prompt_text
            st.session_state[prompt_baseline_key] = live_prompt_text
    
    # Display complete prompt field
    st.markdown("**Vollständiger Prompt:**")
    # Create text area without conflicting state/value usage
    if initial_value is None:
        prompt_input = st.text_area(
            "Prompt für Zusammenfassung",
            key=prompt_state_key,
            height=640,
            label_visibility="collapsed",
        )
    else:
        prompt_input = st.text_area(
            "Prompt für Zusammenfassung",
            value=initial_value,
            key=prompt_state_key,
            height=640,
            label_visibility="collapsed",
        )
    # Initialize session state for storing recaps from different models
    recaps_key = f"{session_key}_recaps"
    if recaps_key not in st.session_state:
        st.session_state[recaps_key] = {}
    
    # Safety check function (shared across all models)
    def perform_safety_check() -> Tuple[bool, list]:
        """Perform safety check and return (is_safe, assessments)."""
        if not SUICIDE_LEXICON_PATH:
            st.error("❌ Sicherheitsprüfung nicht möglich: `SUICIDE_LEXICON_PATH` ist nicht gesetzt.")
            return False, []
        
        try:
            lexicon = load_self_harm_lexicon_cached(SUICIDE_LEXICON_PATH)
            assessments = assess_free_text_answers(segments_for_prompt, lexicon)
        except Exception as exc:
            st.error(f"❌ Sicherheitsprüfung fehlgeschlagen: {exc}")
            return False, []
        
        concerning_levels = {"mittel", "hoch"}
        concerning = [
            entry for entry in assessments
            if entry.get("analysis", {}).get("risk_level") in concerning_levels
        ]
        
        if concerning:
            st.error("⚠️ Sicherheitsprüfung: Auffällige Antworten gefunden. Zusammenfassung gestoppt.")
            for idx, entry in enumerate(concerning, start=1):
                analysis = entry.get("analysis", {})
                with st.container():
                    st.markdown(f"**Nutzer-Eingabe {idx}:** {entry.get('answer', '—')}")
                    st.markdown(f"Risikostufe: `{analysis.get('risk_level', 'unbekannt')}`")
                    with st.expander("Details anzeigen"):
                        st.json(analysis)
            st.info("Bitte prüfe die Eingaben, bevor die Zusammenfassung erneut gestartet wird.")
            return False, assessments
        else:
            if assessments:
                st.success("✅ Sicherheitsprüfung abgeschlossen: Keine Auffälligkeiten.")
            else:
                st.success("✅ Sicherheitsprüfung abgeschlossen: Keine Freitextantworten vorhanden.")
            return True, assessments
    
    # Gemini recap button (cloud)
    st.markdown("---")
    st.subheader("Zusammenfassung generieren")

    col_gemini = st.container()
    with col_gemini:
        if st.button("🧾 Gemini (Cloud)", key=f"{session_key}_summarize_gemini", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("❌ GEMINI_API_KEY nicht gesetzt. Bitte setze die Umgebungsvariable GEMINI_API_KEY.")
            else:
                safety_ok, _ = perform_safety_check()
                if safety_ok:
                    try:
                        from google import genai
                        with st.spinner("Generiere Zusammenfassung mit Gemini..."):
                            client = genai.Client(api_key=GEMINI_API_KEY)
                            resp = client.models.generate_content(
                                model="gemini-2.5-flash-lite",
                                contents=prompt_input,
                                config={"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 200},
                            )
                            if hasattr(resp, "text") and resp.text:
                                summary_text = resp.text.strip()
                            else:
                                summary_text = resp.candidates[0].content.parts[0].text.strip()
                            st.session_state[recaps_key]["Gemini"] = summary_text
                        st.success("✅ Zusammenfassung mit Gemini generiert")
                        st.rerun()
                    except ImportError as e:
                        st.error(f"❌ Fehler: {e}")
                        st.info("💡 Installiere das Paket mit: `pip install google-genai`")
                    except Exception as e:
                        st.error(f"❌ Fehler bei der Zusammenfassungs-Generierung mit Gemini: {e}")
                        import traceback
                        with st.expander("🔍 Fehlerdetails anzeigen"):
                            st.code(traceback.format_exc(), language="python")

    # Use Modal API endpoints for all models
    # All six models are available via Modal endpoints
    available_models = ["Gemma-3-12B", "Llama-3.1-8B", "Mistral-7B", "Qwen3-8B", "Teuken-7B", "Mistral-NeMo-12B"]
    
    # Display model buttons in columns
    num_cols = min(3, len(available_models))
    if available_models:
        cols = st.columns(num_cols)
        
        for idx, model_name in enumerate(available_models):
            col = cols[idx % num_cols]
            with col:
                button_key = f"{session_key}_summarize_{model_name}"
                if st.button(f"🧾 {model_name}", key=button_key, use_container_width=True):
                    safety_ok, assessments = perform_safety_check()
                    
                    if safety_ok:
                        try:
                            with st.spinner(f"Generiere Zusammenfassung mit {model_name}..."):
                                summary_text = generate_summary_with_model(
                                    prompt=prompt_input,
                                    model_name=model_name,
                                    backend_type="api",  # Use API backend (Modal endpoints)
                                    max_tokens=200,
                                    temperature=0.7,
                                )
                                # Store recap in session state
                                st.session_state[recaps_key][model_name] = summary_text
                            st.success(f"✅ Zusammenfassung mit {model_name} generiert")
                            st.rerun()
                        except ValueError as e:
                            st.error(f"❌ Konfigurationsfehler: {e}")
                            st.info("💡 Stelle sicher, dass die Modal-Endpoint-URLs konfiguriert sind.")
                        except Exception as e:
                            st.error(f"❌ Fehler bei der Zusammenfassungs-Generierung mit {model_name}: {e}")
                            import traceback
                            with st.expander("🔍 Fehlerdetails anzeigen"):
                                st.code(traceback.format_exc(), language="python")
    
    # Display all generated recaps for comparison
    if st.session_state[recaps_key]:
        st.markdown("---")
        st.subheader("Zusammenfassungen (Vergleich)")
        
        for model_name, recap_text in st.session_state[recaps_key].items():
            with st.expander(f"📝 {model_name}", expanded=True):
                st.markdown(recap_text)
    
    st.markdown("---")
    with st.expander("Debug / Rohdaten"):
        st.subheader("Original Segments (before any answers)")
        st.json(segments)
        
        st.subheader("Generated/Updated Segments")
        # Always compute current merged segments from filled_segments for immediate display
        # This ensures the debug view updates immediately when inputs change
        current_merged = merge_filled_answers_into_segments(
            segments_to_display if has_generated else segments,
            filled_segments
        )
        # Always show the current merged state (from filled_segments)
        # This will update immediately when any input changes
        segments_to_show = current_merged
        
        # Show a small indicator that this is live data
        import time
        st.caption(f"Last updated: {time.strftime('%H:%M:%S')} (updates automatically on input changes)")
        
        # Always show merged segments to ensure updates are visible
        # Streamlit reruns on widget changes, so this will reflect current state
        if segments_to_show:
            st.json(segments_to_show)
        elif st.session_state[session_key]:
            st.json(st.session_state[session_key])
        else:
            st.info("No generated segments yet.")

else:
    st.info("Bitte wähle eine Übung aus.")

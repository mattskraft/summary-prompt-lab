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
    
    # Get API keys
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
    
    from kiso_input import (
        assess_free_text_answers,
        build_summary_prompt,
        generate_answers_with_gemini,
        get_prompt_segments_from_exercise,
        load_self_harm_lexicon,
    )
    from kiso_input.processing.cloud_apis import (
        generate_summary_with_gemini_from_prompt,
        generate_summary_with_mistral,
    )
    from kiso_input.processing.local_models import (
        generate_summary_with_model,
    )
except ImportError as e:
    st.error(f"""
    ❌ Import error: {e}
    
    Please install the package in development mode:
    ```bash
    cd {PROJECT_ROOT}
    pip install -e .
    ```
    """)
    st.stop()

# -----------------------------
# Shared Helper Functions (reused from streamlit_app.py)
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
    """Ensure the exercise_system_prompts.json file exists with all exercises."""
    json_path = PROJECT_ROOT / "config" / "exercise_system_prompts.json"
    default_prompt_path = PROJECT_ROOT / "config" / "recap_system_prompt.txt"
    
    default_prompt_text = ""
    if default_prompt_path.exists():
        default_prompt_text = default_prompt_path.read_text(encoding="utf-8").strip()
    
    existing_data: Dict[str, str] = {}
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = {}
    
    updated = False
    for exercise_name in exercise_names:
        if exercise_name not in existing_data:
            existing_data[exercise_name] = default_prompt_text
            updated = True
    
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
        existing_data: Dict[str, str] = {}
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
        
        existing_data[exercise_name] = prompt_text.strip()
        
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
# New UI Components for App 2
# -----------------------------
def render_compact_segment(seg: Dict[str, Any], idx: int, key_prefix: str) -> Dict[str, Any]:
    """Render a segment in a more compact format."""
    filled = {}
    
    if "Text" in seg:
        st.markdown(f"**📄 Text:** {seg['Text']}")
        filled = {"Text": seg["Text"]}
    
    elif "Question" in seg:
        st.markdown(f"**❓ Frage:** {seg['Question']}")
        filled = {"Question": seg["Question"]}
    
    elif "Answer" in seg or "AnswerOptions" in seg:
        answer_val = seg.get("Answer")
        answer_options = seg.get("AnswerOptions")
        allow_multiple = bool(seg.get("AllowMultiple", True))
        key = f"{key_prefix}_ans_{idx}"
        
        if isinstance(answer_options, list):
            options = answer_options
            default_selection: List[str] = []
            if isinstance(answer_val, list):
                default_selection = [item for item in answer_val if item in options]
            elif isinstance(answer_val, str) and answer_val in options:
                default_selection = [answer_val]
            
            if allow_multiple:
                user_val = st.multiselect(
                    "Antwort",
                    options=options,
                    default=default_selection,
                    key=key,
                )
            else:
                default_value = default_selection[0] if default_selection else None
                if default_value is not None and default_value in options:
                    selected_value = st.selectbox(
                        "Antwort",
                        options=options,
                        index=options.index(default_value),
                        key=key,
                    )
                else:
                    selected_value = st.selectbox(
                        "Antwort",
                        options=options,
                        key=key,
                        placeholder="Option auswählen",
                        index=None,
                    )
                user_val = [selected_value] if selected_value else []
            
            filled = {
                "Answer": user_val,
                "AnswerOptions": options,
                "AllowMultiple": allow_multiple,
            }
        
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
            
            if min_numeric is not None and max_numeric is not None and min_numeric <= max_numeric:
                current_answer = answer_val
                default_value = None
                if isinstance(current_answer, (int, float)):
                    default_value = int(current_answer)
                elif isinstance(current_answer, dict) and isinstance(current_answer.get("value"), (int, float)):
                    default_value = int(current_answer["value"])
                
                if default_value is None or not (min_numeric <= default_value <= max_numeric):
                    default_value = min_numeric
                
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.caption(min_text if min_text else str(min_numeric))
                with col2:
                    slider_value = st.slider(
                        "Wert",
                        min_value=min_numeric,
                        max_value=max_numeric,
                        value=default_value,
                        format="%d",
                        key=f"{key}_slider",
                        label_visibility="collapsed",
                    )
                with col3:
                    st.caption(max_text if max_text else str(max_numeric))
                
                filled = {
                    "Answer": int(slider_value),
                    "AnswerOptions": slider_config,
                }
            else:
                st.warning("Slider-Konfiguration unvollständig")
                filled = {"Answer": answer_val, "AnswerOptions": slider_config}
        
        elif isinstance(answer_val, str) and answer_val.strip().lower() in {"free_text", "freetext", "free text"}:
            user_val = st.text_input("Antwort", key=key)
            filled = {"Answer": user_val}
        
        elif isinstance(answer_val, str):
            user_val = st.text_area("Antwort", value=answer_val, key=key, height=100)
            filled = {"Answer": user_val}
        
        else:
            st.markdown(f"**Antwort:** {str(answer_val)}")
            filled = {"Answer": answer_val}
    
    return filled


# -----------------------------
# Main UI Layout (Tab-based)
# -----------------------------
st.set_page_config(page_title="Summary Prompt Lab v2", layout="wide")

# Password protection
if APP_PASSWORD and not PASSWORD_FROM_SECRETS:
    if "password_verified" not in st.session_state:
        st.session_state.password_verified = True

st.title("🧪 Summary Prompt Lab v2")

# Load files
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

all_exercise_names = get_all_exercise_names(hier) if hier else []
if all_exercise_names:
    ensure_exercise_prompts_json(all_exercise_names)

# Top bar for exercise selection
col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    themen = sorted(hier.keys()) if hier else []
    thema_key = "app2_selected_thema"
    if themen and st.session_state.get(thema_key) not in themen:
        st.session_state[thema_key] = themen[0]
    sel_thema = st.selectbox("Thema", options=themen, key=thema_key) if themen else None

with col2:
    paths = sorted(hier.get(sel_thema, {}).keys()) if sel_thema else []
    path_key = "app2_selected_path"
    if paths and st.session_state.get(path_key) not in paths:
        st.session_state[path_key] = paths[0]
    elif not paths:
        st.session_state.pop(path_key, None)
    sel_path = st.selectbox("Pfad", options=paths, key=path_key) if paths else None

with col3:
    uebungen = sorted(hier.get(sel_thema, {}).get(sel_path, [])) if sel_path else []
    uebung_key = "app2_selected_uebung"
    if uebungen and st.session_state.get(uebung_key) not in uebungen:
        st.session_state[uebung_key] = uebungen[0]
    elif not uebungen:
        st.session_state.pop(uebung_key, None)
    sel_uebung = st.selectbox("Übung", options=uebungen, key=uebung_key) if uebungen else None

with col4:
    seed = st.number_input("Seed", min_value=0, value=42, step=1, key="app2_seed")
    if st.button("🎲 Zufällig", use_container_width=True, disabled=not hier):
        if themen:
            zufalls_thema = random.choice(themen)
            available_paths = sorted(hier.get(zufalls_thema, {}).keys())
            if available_paths:
                zufalls_path = random.choice(available_paths)
                available_uebungen = sorted(hier.get(zufalls_thema, {}).get(zufalls_path, []))
                if available_uebungen:
                    zufalls_uebung = random.choice(available_uebungen)
                    st.session_state[thema_key] = zufalls_thema
                    st.session_state[path_key] = zufalls_path
                    st.session_state[uebung_key] = zufalls_uebung
                    st.rerun()

st.markdown("---")

if sel_uebung:
    session_key = f"app2_{sel_uebung}_{seed}"
    
    # Load segments
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
    
    # Initialize session state
    if session_key not in st.session_state:
        st.session_state[session_key] = None
    
    has_generated = st.session_state[session_key] is not None
    segments_to_display = st.session_state[session_key] if has_generated else segments
    key_prefix = f"{session_key}_generated" if has_generated else f"{session_key}_original"
    
    # Tab-based interface
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Übung", "🤖 Antworten", "📋 Prompt", "📊 Zusammenfassungen"])
    
    with tab1:
        st.subheader(f"Übung: {sel_uebung}")
        
        # Render segments in compact format
        filled_segments = []
        for idx, seg in enumerate(segments_to_display):
            with st.container():
                filled = render_compact_segment(seg, idx, key_prefix)
                filled_segments.append(filled)
                st.markdown("---")
        
        # Generate answers button
        if st.button("✨ Antworten generieren", use_container_width=True, key=f"{session_key}_gen_answers"):
            if not GEMINI_API_KEY:
                st.error("❌ GEMINI_API_KEY nicht gesetzt")
            else:
                try:
                    with st.spinner("Generiere Antworten mit Gemini..."):
                        result = generate_answers_with_gemini(
                            segments=segments,
                            api_key=GEMINI_API_KEY,
                            debug=False,
                            return_debug_info=True,
                        )
                        generated_segments, debug_info = result
                        st.session_state[session_key] = generated_segments
                    st.success("✅ Antworten generiert")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Fehler: {e}")
    
    with tab2:
        st.subheader("Generierte Antworten")
        
        if has_generated:
            st.success("✅ Antworten wurden generiert")
            st.json(st.session_state[session_key])
        else:
            st.info("Klicke auf 'Antworten generieren' im Tab 'Übung'")
    
    with tab3:
        st.subheader("Zusammenfassungs-Prompt")
        
        # Merge filled answers
        segments_for_prompt = merge_filled_answers_into_segments(
            segments_to_display if has_generated else segments,
            filled_segments
        )
        
        # System prompt editor
        exercise_prompts = load_exercise_prompts_json()
        system_prompt_key = f"{session_key}_system_prompt"
        
        if system_prompt_key not in st.session_state:
            if sel_uebung and sel_uebung in exercise_prompts:
                st.session_state[system_prompt_key] = exercise_prompts[sel_uebung]
            else:
                live_prompt = build_summary_prompt(segments_for_prompt, exercise_name=sel_uebung)
                if "SYSTEM:\n\n" in live_prompt:
                    parts = live_prompt.split("SYSTEM:\n\n", 1)
                    if len(parts) > 1:
                        system_part = parts[1].split("\n\n" + ("-" * 80) + "\nINHALT:\n\n", 1)[0]
                        st.session_state[system_prompt_key] = system_part.strip()
                    else:
                        st.session_state[system_prompt_key] = ""
                else:
                    st.session_state[system_prompt_key] = ""
        
        system_prompt_input = st.text_area(
            "System-Prompt",
            value=st.session_state[system_prompt_key],
            key=system_prompt_key,
            height=300,
        )
        
        if st.button("💾 System-Prompt speichern", key=f"{session_key}_save_system"):
            if save_exercise_prompt(sel_uebung, system_prompt_input):
                st.success("✅ Gespeichert")
            else:
                st.error("❌ Fehler beim Speichern")
        
        st.markdown("---")
        
        # Complete prompt
        live_prompt_text = build_summary_prompt(segments_for_prompt, exercise_name=sel_uebung)
        
        # Rebuild with custom system prompt if changed
        if "SYSTEM:\n\n" in live_prompt_text:
            parts = live_prompt_text.split("SYSTEM:\n\n", 1)
            if len(parts) > 1:
                content_part = parts[1].split("\n\n" + ("-" * 80) + "\nINHALT:\n\n", 1)
                if len(content_part) > 1:
                    new_prompt = (
                        ("-" * 80)
                        + "\nSYSTEM:\n\n"
                        + system_prompt_input
                        + "\n\n"
                        + ("-" * 80)
                        + "\nINHALT:\n\n"
                        + content_part[1]
                    )
                    live_prompt_text = new_prompt
        
        prompt_state_key = f"{session_key}_summary_prompt"
        if prompt_state_key not in st.session_state:
            st.session_state[prompt_state_key] = live_prompt_text
        
        prompt_input = st.text_area(
            "Vollständiger Prompt",
            value=st.session_state[prompt_state_key],
            key=prompt_state_key,
            height=400,
        )
    
    with tab4:
        st.subheader("Zusammenfassungen generieren")
        
        # Safety check function
        def perform_safety_check() -> Tuple[bool, list]:
            if not SUICIDE_LEXICON_PATH:
                st.error("❌ Sicherheitsprüfung nicht möglich")
                return False, []
            
            try:
                lexicon = load_self_harm_lexicon_cached(SUICIDE_LEXICON_PATH)
                segments_for_check = merge_filled_answers_into_segments(
                    segments_to_display if has_generated else segments,
                    filled_segments
                )
                assessments = assess_free_text_answers(segments_for_check, lexicon)
            except Exception as exc:
                st.error(f"❌ Sicherheitsprüfung fehlgeschlagen: {exc}")
                return False, []
            
            concerning_levels = {"mittel", "hoch"}
            concerning = [
                entry for entry in assessments
                if entry.get("analysis", {}).get("risk_level") in concerning_levels
            ]
            
            if concerning:
                st.error("⚠️ Auffällige Antworten gefunden")
                return False, assessments
            else:
                if assessments:
                    st.success("✅ Sicherheitsprüfung OK")
                return True, assessments
        
        # Initialize recaps storage
        recaps_key = f"{session_key}_recaps"
        if recaps_key not in st.session_state:
            st.session_state[recaps_key] = {}
        
        # Model selection and generation
        st.markdown("#### Proprietäre APIs")
        prop_col1, prop_col2 = st.columns(2)
        
        with prop_col1:
            if st.button("🧾 Gemini", key=f"{session_key}_summarize_gemini", use_container_width=True):
                if not GEMINI_API_KEY:
                    st.error("❌ GEMINI_API_KEY nicht gesetzt")
                else:
                    safety_ok, _ = perform_safety_check()
                    if safety_ok:
                        try:
                            with st.spinner("Generiere mit Gemini..."):
                                summary_text = generate_summary_with_gemini_from_prompt(
                                    prompt=prompt_input,
                                    api_key=GEMINI_API_KEY,
                                    model="gemini-2.5-flash-lite",
                                    max_tokens=200,
                                    temperature=0.7,
                                )
                                st.session_state[recaps_key]["Gemini"] = summary_text
                            st.success("✅ Generiert")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Fehler: {e}")
        
        with prop_col2:
            if st.button("🧾 Mistral", key=f"{session_key}_summarize_mistral", use_container_width=True):
                if not MISTRAL_API_KEY:
                    st.error("❌ MISTRAL_API_KEY nicht gesetzt")
                else:
                    safety_ok, _ = perform_safety_check()
                    if safety_ok:
                        try:
                            with st.spinner("Generiere mit Mistral..."):
                                summary_text = generate_summary_with_mistral(
                                    prompt=prompt_input,
                                    api_key=MISTRAL_API_KEY,
                                    model="mistral-medium-latest",
                                    max_tokens=200,
                                    temperature=0.7,
                                )
                                st.session_state[recaps_key]["Mistral"] = summary_text
                            st.success("✅ Generiert")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Fehler: {e}")
        
        st.markdown("#### Self-hosted (Modal)")
        available_models = ["Gemma-3-12B", "Llama-3.1-8B", "Mistral-7B", "Qwen3-8B", "Teuken-7B", "Mistral-NeMo-12B"]
        
        model_cols = st.columns(3)
        for idx, model_name in enumerate(available_models):
            col = model_cols[idx % 3]
            with col:
                if st.button(f"🧾 {model_name}", key=f"{session_key}_summarize_{model_name}", use_container_width=True):
                    safety_ok, _ = perform_safety_check()
                    if safety_ok:
                        try:
                            with st.spinner(f"Generiere mit {model_name}..."):
                                summary_text = generate_summary_with_model(
                                    prompt=prompt_input,
                                    model_name=model_name,
                                    backend_type="api",
                                    max_tokens=200,
                                    temperature=0.7,
                                )
                                st.session_state[recaps_key][model_name] = summary_text
                            st.success(f"✅ Generiert")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Fehler: {e}")
        
        # Display generated recaps
        st.markdown("---")
        st.subheader("Generierte Zusammenfassungen")
        
        if st.session_state[recaps_key]:
            for model_name, recap_text in st.session_state[recaps_key].items():
                with st.expander(f"📝 {model_name}", expanded=True):
                    st.markdown(recap_text)
        else:
            st.info("Noch keine Zusammenfassungen generiert")

else:
    st.info("Bitte wähle eine Übung aus.")


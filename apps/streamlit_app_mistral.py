import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from github import Github, GithubException  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for cloud sync
    Github = None  # type: ignore
    GithubException = Exception  # type: ignore

import streamlit as st

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

pending_toast = st.session_state.pop("pending_toast", None)
if pending_toast:
    st.toast(pending_toast)

# Import from the kiso_input package
try:
    from kiso_input.config import (  # type: ignore
        STRUCT_JSON_PATH,
        SN_JSON_PATH,
        SAFETY_LEXICON_PATH,
    )
    import kiso_input.config as config_module  # type: ignore
    
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
    
    from kiso_input import (  # type: ignore
        assess_free_text_answers,
        generate_answers_with_gemini,
        generate_answers_with_mistral,
        get_prompt_segments_from_exercise,
        load_self_harm_lexicon,
    )
    from kiso_input.processing.cloud_apis import (  # type: ignore
        generate_summary_with_mistral,
        stream_summary_with_mistral,
    )
    from kiso_input.processing.recap_sections import (  # type: ignore
        assemble_system_prompt,
        get_exercise_sections,
        load_global_prompt,
        save_exercise_sections,
        save_global_prompt,
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


CONFIG_DIR = PROJECT_ROOT / "config"
EXERCISE_PROMPTS_STORE = CONFIG_DIR / "exercise_specific_prompts.json"
SEGMENT_SWITCH_PATH = CONFIG_DIR / "segment_switches.json"
GLOBAL_PROMPT_PATH = CONFIG_DIR / "system_prompt_global.txt"
MISTRAL_MAX_TOKENS = 120
# Note: mistral-medium-latest was deprecated in 2024
MISTRAL_MODEL_OPTIONS = [
    "mistral-small-latest",
    "mistral-large-latest",
]
MISTRAL_DEFAULT_MODEL = "mistral-small-latest"


def ensure_exercise_prompt_store() -> Path:
    EXERCISE_PROMPTS_STORE.parent.mkdir(parents=True, exist_ok=True)
    if not EXERCISE_PROMPTS_STORE.exists():
        EXERCISE_PROMPTS_STORE.write_text("{}", encoding="utf-8")
    return EXERCISE_PROMPTS_STORE


def ensure_segment_switch_store() -> Path:
    SEGMENT_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SEGMENT_SWITCH_PATH.exists():
        SEGMENT_SWITCH_PATH.write_text("{}", encoding="utf-8")
    return SEGMENT_SWITCH_PATH


def get_github_settings() -> Dict[str, Optional[str]]:
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("GITHUB_REPO")
    branch = os.environ.get("GITHUB_BRANCH", "main")
    try:
        secrets_obj = getattr(st, "secrets", None)
        if secrets_obj is not None:
            token = secrets_obj.get("GITHUB_TOKEN", token)
            repo_name = secrets_obj.get("GITHUB_REPO", repo_name)
            branch = secrets_obj.get("GITHUB_BRANCH", branch)
    except Exception:
        pass
    return {
        "token": token,
        "repo": repo_name,
        "branch": branch or "main",
    }


def commit_prompt_file(file_path: Path, message: str) -> bool:
    """Commit the given file to GitHub using PyGithub (if configured)."""
    settings = get_github_settings()
    token = settings["token"]
    repo_name = settings["repo"]
    branch = settings["branch"]
    if not token or not repo_name or Github is None:
        return False
    abs_path = file_path if isinstance(file_path, Path) else Path(file_path)
    try:
        rel_path = abs_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel_path = abs_path.as_posix()
    try:
        client = Github(token)
        repo = client.get_repo(repo_name)
        content = abs_path.read_text(encoding="utf-8")
        try:
            remote_file = repo.get_contents(rel_path, ref=branch)
            repo.update_file(rel_path, message, content, remote_file.sha, branch=branch)
            print(f"[GitHubSync] Updated {repo_name}:{branch}:{rel_path} – {message}")  # noqa: T201
        except GithubException as exc:
            if getattr(exc, "status", None) == 404:
                repo.create_file(rel_path, message, content, branch=branch)
                print(f"[GitHubSync] Created {repo_name}:{branch}:{rel_path} – {message}")  # noqa: T201
            else:
                raise
        return True
    except Exception as exc:
        st.warning(f"GitHub-Sync fehlgeschlagen: {exc}")
        return False


def load_segment_switch_config() -> Dict[str, Any]:
    """Load segment switch config. Each exercise entry is {"enabled": bool, "comment": str, "segment_toggles": [...]}."""
    ensure_segment_switch_store()
    try:
        raw = SEGMENT_SWITCH_PATH.read_text(encoding="utf-8").strip() or "{}"
        config = json.loads(raw)
        # Migrate old string format to new dict format
        for key, value in list(config.items()):
            if isinstance(value, str):
                # Old format: "summary" or "no questions" string
                enabled = value.strip().lower().startswith("summary")
                config[key] = {"enabled": enabled, "comment": "" if enabled else value}
        return config
    except json.JSONDecodeError as exc:
        st.warning(f"Ungültige segment_switches.json: {exc}")
        return {}


def save_segment_switch_config(config: Dict[str, Any]) -> None:
    ensure_segment_switch_store()
    SEGMENT_SWITCH_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summary_switch_is_on(value: Optional[Dict[str, Any]]) -> bool:
    """Check if summary is enabled for an exercise."""
    if not value:
        return False
    if isinstance(value, dict):
        return value.get("enabled", False)
    # Legacy string format fallback
    if isinstance(value, str):
        return value.strip().lower().startswith("summary")
    return False


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


def global_prompt_state_key() -> str:
    return "global_system_prompt"


def exercise_prompt_state_key(session_key: str) -> str:
    return f"{session_key}_exercise_prompt"


def save_exercise_payload(exercise_name: str, payload: Dict[str, str]) -> bool:
    try:
        save_exercise_sections(exercise_name, payload)
        print(f"[GitHubSync] Locally updated exercise data for '{exercise_name}'.")  # noqa: T201
        github_success = commit_prompt_file(
            EXERCISE_PROMPTS_STORE,
            f"Update exercise-specific prompts for {exercise_name}",
        )
        if github_success:
            st.session_state["pending_toast"] = f"✅ '{exercise_name}' gespeichert."
        else:
            st.session_state["pending_toast"] = (
                f"⚠️ '{exercise_name}' lokal gespeichert, aber GitHub-Sync fehlgeschlagen! "
                "Daten gehen beim Neustart verloren."
            )
        return github_success
    except Exception as exc:
        st.error(f"Fehler beim Speichern: {exc}")
        return False


def get_current_global_prompt(session_key: str) -> str:
    """Get global prompt from session state."""
    return st.session_state.get(global_prompt_state_key(), "")


def get_current_exercise_prompt(session_key: str) -> str:
    """Get exercise-specific prompt from session state."""
    return st.session_state.get(exercise_prompt_state_key(session_key), "")


def print_params_debug(context: str, params: Dict[str, Any]) -> None:
    print(f"[Recap:{context}] Params: {params}")  # noqa: T201


def initialize_prompt_states(
    sel_uebung: str,
    session_key: str,
    global_prompt_default: str,
    exercise_sections: Dict[str, str],
) -> None:
    """Initialize session state for global and exercise prompts."""
    # Global prompt - initialize once
    gp_key = global_prompt_state_key()
    if gp_key not in st.session_state:
        st.session_state[gp_key] = global_prompt_default

    # Exercise-specific prompt - update when exercise changes
    ep_key = exercise_prompt_state_key(session_key)
    tracker_key = f"{ep_key}_exercise"
    exercise_changed = st.session_state.get(tracker_key) != sel_uebung
    state_never_set = ep_key not in st.session_state
    if exercise_changed or state_never_set:
        st.session_state[ep_key] = exercise_sections.get("prompt", "")
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


def get_next_exercise(
    hierarchy: Dict[str, Dict[str, List[str]]], 
    current_uebung: Optional[str],
    skip_completed: bool = False,
    segment_switch_config: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[str, str, str]]:
    """Get the next exercise after the current one, wrapping around if at the end.
    
    Args:
        hierarchy: Exercise hierarchy
        current_uebung: Current exercise name
        skip_completed: If True, skip exercises marked as "config_complete"
        segment_switch_config: Config dict to check for completed exercises
    """
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
    
    # Get next exercise (wrap around), skipping completed if requested
    num_exercises = len(all_exercises)
    for offset in range(1, num_exercises + 1):
        next_idx = (current_idx + offset) % num_exercises
        candidate = all_exercises[next_idx]
        if skip_completed and segment_switch_config:
            exercise_config = segment_switch_config.get(candidate[2], {})
            if exercise_config.get("config_complete", False):
                continue  # Skip this exercise
        return candidate
    
    # All exercises are completed, return next anyway
    next_idx = (current_idx + 1) % num_exercises
    return all_exercises[next_idx]


def get_previous_exercise(
    hierarchy: Dict[str, Dict[str, List[str]]], 
    current_uebung: Optional[str],
    skip_completed: bool = False,
    segment_switch_config: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[str, str, str]]:
    """Get the previous exercise before the current one, wrapping around if at the beginning.
    
    Args:
        hierarchy: Exercise hierarchy
        current_uebung: Current exercise name
        skip_completed: If True, skip exercises marked as "config_complete"
        segment_switch_config: Config dict to check for completed exercises
    """
    all_exercises = get_all_exercises_with_paths(hierarchy)
    if not all_exercises:
        return None
    
    if not current_uebung:
        return all_exercises[-1]
    
    # Find current exercise index
    current_idx = None
    for idx, (_, _, uebung) in enumerate(all_exercises):
        if uebung == current_uebung:
            current_idx = idx
            break
    
    if current_idx is None:
        return all_exercises[-1]
    
    # Get previous exercise (wrap around), skipping completed if requested
    num_exercises = len(all_exercises)
    for offset in range(1, num_exercises + 1):
        prev_idx = (current_idx - offset) % num_exercises
        candidate = all_exercises[prev_idx]
        if skip_completed and segment_switch_config:
            exercise_config = segment_switch_config.get(candidate[2], {})
            if exercise_config.get("config_complete", False):
                continue  # Skip this exercise
        return candidate
    
    # All exercises are completed, return previous anyway
    prev_idx = (current_idx - 1) % num_exercises
    return all_exercises[prev_idx]


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


def calculate_text_area_height(text: str, min_height: int = 100, max_height: int = 1200, line_height: int = 24) -> int:
    """Calculate appropriate text area height based on content.
    
    Args:
        text: The text content
        min_height: Minimum height in pixels
        max_height: Maximum height in pixels  
        line_height: Approximate pixels per line
        
    Returns:
        Height in pixels that fits the content
    """
    if not text:
        return min_height
    
    # Count lines (including wrapped lines for very long lines)
    lines = text.split('\n')
    total_lines = 0
    chars_per_line = 80  # Approximate characters that fit in one line
    
    for line in lines:
        # Account for line wrapping on long lines
        wrapped_lines = max(1, (len(line) // chars_per_line) + 1)
        total_lines += wrapped_lines
    
    # Calculate height with some padding
    calculated_height = (total_lines * line_height) + 40  # 40px padding
    
    return max(min_height, min(calculated_height, max_height))


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


def render_segments_ui(
    segments: List[Dict[str, Any]], 
    key_prefix: str = "",
    segment_toggles: Optional[List[bool]] = None,
    show_toggles: bool = False,
) -> Tuple[List[Dict[str, Any]], List[bool]]:
    """Render the segments and collect user inputs in a parallel list.
    
    Args:
        segments: List of segment dictionaries
        key_prefix: Prefix for widget keys
        segment_toggles: Initial toggle states (default: all True)
        show_toggles: Whether to show toggle checkboxes
        
    Returns:
        Tuple of (filled_segments, toggle_states)
    """
    filled: List[Dict[str, Any]] = []
    toggles: List[bool] = []
    
    # Initialize toggles to all True if not provided
    if segment_toggles is None:
        segment_toggles = [True] * len(segments)
    # Ensure we have enough toggles
    while len(segment_toggles) < len(segments):
        segment_toggles.append(True)

    for idx, seg in enumerate(segments):
        toggle_key = f"{key_prefix}_toggle_{idx}"
        
        # Initialize toggle state
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = segment_toggles[idx]
        
        if "Text" in seg:
            if show_toggles:
                toggle_cols = st.columns([0.06, 0.94])
                with toggle_cols[0]:
                    st.checkbox("Toggle segment", key=toggle_key, label_visibility="collapsed")
                with toggle_cols[1]:
                    render_segment_header("TEXT", "#93C5FD")
                    st.markdown(seg["Text"])
            else:
                render_segment_header("TEXT", "#93C5FD")
                st.markdown(seg["Text"])
            filled.append({"Text": seg["Text"]})
            toggles.append(st.session_state.get(toggle_key, True))

        elif "Question" in seg:
            if show_toggles:
                toggle_cols = st.columns([0.06, 0.94])
                with toggle_cols[0]:
                    st.checkbox("Toggle segment", key=toggle_key, label_visibility="collapsed")
                with toggle_cols[1]:
                    render_segment_header("FRAGE", "#22C55E")
                    st.markdown(seg["Question"])
            else:
                render_segment_header("FRAGE", "#22C55E")
                st.markdown(seg["Question"])
            filled.append({"Question": seg["Question"]})
            toggles.append(st.session_state.get(toggle_key, True))

        elif "Answer" in seg or "AnswerOptions" in seg:
            answer_val = seg.get("Answer")
            answer_options = seg.get("AnswerOptions")
            allow_multiple = bool(seg.get("AllowMultiple", True))

            key = f"{key_prefix}_ans_{idx}"
            
            if show_toggles:
                toggle_cols = st.columns([0.06, 0.94])
                with toggle_cols[0]:
                    st.checkbox("Toggle segment", key=toggle_key, label_visibility="collapsed")
                content_container = toggle_cols[1]
            else:
                content_container = st.container()
            
            with content_container:
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
            
            toggles.append(st.session_state.get(toggle_key, True))

    return filled, toggles


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


def filter_segments_by_toggles(
    segments: List[Dict[str, Any]], 
    toggles: List[bool]
) -> List[Dict[str, Any]]:
    """Filter segments based on toggle states."""
    if not toggles:
        return segments
    return [seg for seg, enabled in zip(segments, toggles) if enabled]


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
    ensure_segment_switch_store()

with st.sidebar:
    st.header("Navigation")
    
    thema_key = "nav_selected_thema"
    path_key = "nav_selected_path"
    uebung_key = "nav_selected_uebung"
    
    themen = sorted(hier.keys())
    
    # Load segment switch config for navigation (to skip completed exercises)
    nav_segment_switch_config = load_segment_switch_config()
    
    # Navigation buttons
    nav_button_cols = st.columns(2)
    
    with nav_button_cols[0]:
        current_uebung = st.session_state.get(uebung_key)
        if st.button("⬅️ Previous", use_container_width=True, disabled=not hier):
            previous_exercise = get_previous_exercise(
                hier, current_uebung,
                skip_completed=True,
                segment_switch_config=nav_segment_switch_config,
            )
            if previous_exercise:
                previous_thema, previous_path, previous_uebung = previous_exercise
                st.session_state[thema_key] = previous_thema
                st.session_state[path_key] = previous_path
                st.session_state[uebung_key] = previous_uebung
                st.rerun()
    
    with nav_button_cols[1]:
        current_uebung = st.session_state.get(uebung_key)
        if st.button("➡️ Next", use_container_width=True, disabled=not hier):
            next_exercise = get_next_exercise(
                hier, current_uebung,
                skip_completed=True,
                segment_switch_config=nav_segment_switch_config,
            )
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
                        
                        # Store segment loading diagnostics in session state
                        st.session_state[f"{session_key}_segment_diag"] = {
                            "exercise_name": sel_uebung,
                            "struct_path": STRUCT_JSON_PATH,
                            "sn_path": SN_JSON_PATH,
                            "segments_loaded": len(current_segments),
                            "struct_exists": os.path.exists(STRUCT_JSON_PATH) if STRUCT_JSON_PATH else False,
                            "sn_exists": os.path.exists(SN_JSON_PATH) if SN_JSON_PATH else False,
                        }
                        
                    except Exception as e:
                        st.error(f"Fehler bei der Segment-Generierung: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.stop()
                    
                    # Create system prompt with max words
                    system_prompt_with_words = SYNTH_ANSWERS_PROMPT.replace("xxx", str(gemini_max_words))
                    
                    # Debug: verify segments right before calling Mistral
                    st.session_state[f"{session_key}_pre_mistral_count"] = len(current_segments)
                    st.session_state[f"{session_key}_pre_mistral_types"] = [
                        {"idx": i, "keys": list(seg.keys())} 
                        for i, seg in enumerate(current_segments[:5])  # First 5 only
                    ]
                    
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
                        
                        # Store debug info for display
                        st.session_state[f"{session_key}_debug_info"] = debug_info
                        
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
                    # Show debug info before rerun
                    st.success("✅ Antworten generiert")
                    with st.expander("🔍 Debug-Info", expanded=True):
                        st.json(debug_info)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Fehler bei der Antwort-Generierung: {e}")
                    import traceback
                    with st.expander("🔍 Fehlerdetails anzeigen"):
                        st.code(traceback.format_exc(), language="python")
        
        # Transfer buttons in sidebar
        st.subheader("Inhalte übertragen")
        
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
            if st.button("📋 TEST", key=transfer_main_key, use_container_width=True):
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
        
        # Show debug info from last answer generation
        debug_info_key = f"{session_key}_debug_info"
        segment_diag_key = f"{session_key}_segment_diag"
        
        if debug_info_key in st.session_state or segment_diag_key in st.session_state:
            with st.expander("🔍 Letzte Generierung - Debug Info", expanded=True):
                # Show segment loading diagnostics first
                if segment_diag_key in st.session_state:
                    seg_diag = st.session_state[segment_diag_key]
                    st.markdown("**📂 Segment-Laden:**")
                    st.text(f"  Übung: {seg_diag.get('exercise_name', 'N/A')}")
                    st.text(f"  STRUCT-Datei: {seg_diag.get('struct_path', 'N/A')}")
                    st.text(f"  STRUCT existiert: {seg_diag.get('struct_exists', False)}")
                    st.text(f"  SN-Datei: {seg_diag.get('sn_path', 'N/A')}")
                    st.text(f"  SN existiert: {seg_diag.get('sn_exists', False)}")
                    st.text(f"  ➡️ Segmente geladen: {seg_diag.get('segments_loaded', 0)}")
                
                # Show pre-Mistral segment count
                pre_mistral_key = f"{session_key}_pre_mistral_count"
                pre_mistral_types_key = f"{session_key}_pre_mistral_types"
                if pre_mistral_key in st.session_state:
                    st.text(f"  ➡️ Vor Mistral-Aufruf: {st.session_state[pre_mistral_key]}")
                if pre_mistral_types_key in st.session_state:
                    st.text(f"  Erste Segmente: {st.session_state[pre_mistral_types_key]}")
                
                st.markdown("---")
                
                if debug_info_key in st.session_state and st.session_state[debug_info_key]:
                    debug_info = st.session_state[debug_info_key]
                    st.markdown(f"""
**Code-Version:** {debug_info.get('code_version', 'UNKNOWN - OLD CODE!')}  
**Modell:** {debug_info.get('model_used', 'N/A')}  
**Segmente an Mistral:** {debug_info.get('total_segments', 0)}  
**Segment-Typ:** {debug_info.get('segments_received_type', 'N/A')}  
**MC-Antworten:** {debug_info.get('mc_questions_generated', 0)}  
**Slider-Antworten:** {debug_info.get('slider_questions_generated', 0)}  
**Freitext gefunden:** {debug_info.get('free_text_questions_found', 0)}  
**Freitext generiert:** {debug_info.get('free_text_answers_generated', 0)}
                    """)
                    # Show segments repr if available
                    segments_repr = debug_info.get('segments_repr')
                    if segments_repr:
                        st.text(f"Segments data: {segments_repr}")
                    
                    # Show answer types
                    answer_types = debug_info.get('answer_types', [])
                    if answer_types:
                        st.markdown("**Antwort-Typen:**")
                        for at in answer_types:
                            st.text(f"  Segment {at['segment']}: {at['type']}")
                    
                    # Show errors if any
                    errors = debug_info.get('errors', [])
                    if errors:
                        st.error(f"**{len(errors)} Fehler aufgetreten:**")
                        for err in errors:
                            st.text(f"  Frage: {err.get('question', 'N/A')}")
                            st.text(f"  Fehler: {err.get('error', 'N/A')}")
    
    
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
    segment_switch_config = load_segment_switch_config()
    segment_switch_value = segment_switch_config.get(sel_uebung)

    if segment_switch_value is None:
        # Default: enabled if has questions, disabled with comment if no questions
        segment_switch_value = {
            "enabled": has_questions,
            "comment": "" if has_questions else "Übung enthält keine Fragen",
            "segment_toggles": [],  # Will be populated when segments are rendered
        }
        segment_switch_config[sel_uebung] = segment_switch_value
        save_segment_switch_config(segment_switch_config)
    
    # Get segment toggles from config (default all True)
    saved_segment_toggles = segment_switch_value.get("segment_toggles", [])
    
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
    
    # If segment count changed, reset toggles to all True (avoids applying stale toggles to wrong segments)
    if saved_segment_toggles and len(saved_segment_toggles) != len(segments_to_display):
        st.warning(
            f"⚠️ Anzahl der Segmente hat sich geändert ({len(saved_segment_toggles)} → {len(segments_to_display)}). "
            "Segment-Toggles wurden zurückgesetzt."
        )
        saved_segment_toggles = []
    
    st.markdown("---")
    st.subheader("Übung")
    filled_segments, current_toggles = render_segments_ui(
        segments_to_display, 
        key_prefix=key_prefix,
        segment_toggles=saved_segment_toggles,
        show_toggles=True,
    )
    
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
    
    # Filter segments by current toggles for transfer
    filtered_segments = filter_segments_by_toggles(segments_for_prompt, current_toggles)
    
    # Handle transfer button clicks (using filtered segments)
    transfer_ex1_clicked = st.session_state.get(f"{sel_uebung}_transfer_ex1_clicked", False)
    if transfer_ex1_clicked:
        inhalt = segments_to_inhalt(filtered_segments)
        st.session_state[example1_state_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_ex1_clicked"] = False
    
    transfer_ex2_clicked = st.session_state.get(f"{sel_uebung}_transfer_ex2_clicked", False)
    if transfer_ex2_clicked:
        inhalt = segments_to_inhalt(filtered_segments)
        st.session_state[example2_state_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_ex2_clicked"] = False
    
    transfer_main_clicked = st.session_state.get(f"{sel_uebung}_transfer_main_clicked", False)
    if transfer_main_clicked:
        inhalt = segments_to_inhalt(filtered_segments)
        st.session_state[mainrecap_inhalt_key] = inhalt
        st.session_state[f"{sel_uebung}_transfer_main_clicked"] = False

    st.subheader("Summary Switch")
    
    # Extract current values from config
    switch_enabled = segment_switch_value.get("enabled", False) if isinstance(segment_switch_value, dict) else False
    switch_comment = segment_switch_value.get("comment", "") if isinstance(segment_switch_value, dict) else ""
    config_complete = segment_switch_value.get("config_complete", False) if isinstance(segment_switch_value, dict) else False
    
    # Session state keys
    switch_checkbox_key = f"{session_key}_segment_switch_enabled"
    switch_comment_key = f"{session_key}_segment_switch_comment"
    config_complete_key = f"{session_key}_config_complete"
    
    # Initialize session state if needed
    if switch_checkbox_key not in st.session_state:
        st.session_state[switch_checkbox_key] = switch_enabled
    if switch_comment_key not in st.session_state:
        st.session_state[switch_comment_key] = switch_comment
    if config_complete_key not in st.session_state:
        st.session_state[config_complete_key] = config_complete
    
    # Checkboxes in two columns
    switch_cols = st.columns(2)
    with switch_cols[0]:
        st.checkbox(
           "Summary aktiviert",
           key=switch_checkbox_key,
        )
    with switch_cols[1]:
        st.checkbox(
           "Konfiguration abgeschlossen",
           key=config_complete_key,
        )

    # Comment
    st.text_area(
        "Kommentar zur Übung",
        key=switch_comment_key,
        height=80,
        label_visibility="collapsed",
        placeholder="Comment for the exercise (optional)",
    )

    # Save button
    segment_switch_save_clicked = st.button(
        "💾 Speichern",
        key=f"{session_key}_segment_switch_save",
        use_container_width=True,
    )
    
    if segment_switch_save_clicked:
        new_enabled = st.session_state.get(switch_checkbox_key, False)
        new_comment = st.session_state.get(switch_comment_key, "").strip()
        new_config_complete = st.session_state.get(config_complete_key, False)
        segment_switch_value = {
            "enabled": new_enabled, 
            "comment": new_comment,
            "segment_toggles": current_toggles,
            "config_complete": new_config_complete,
        }
        segment_switch_config[sel_uebung] = segment_switch_value
        save_segment_switch_config(segment_switch_config)
        github_success = commit_prompt_file(
            SEGMENT_SWITCH_PATH,
            f"Update segment switch for {sel_uebung}",
        )
        if github_success:
            st.success("✅ Einstellungen gespeichert")
        else:
            st.warning(
                "⚠️ Einstellungen lokal gespeichert, aber GitHub-Sync fehlgeschlagen! "
                "Daten gehen beim Neustart verloren."
            )

    # React immediately to checkbox state (not just saved value)
    summary_mode_active = st.session_state.get(switch_checkbox_key, False)

    if not summary_mode_active:
        st.info(
            "Diese Übung ist deaktiviert. Aktiviere den Segment Switch, "
            "um die nachfolgenden Bereiche zu verwenden."
        )
        st.stop()
    
    # Load exercise sections
    forced_sections = st.session_state.pop(f"{session_key}_force_reload_sections", None)
    exercise_sections = forced_sections or get_exercise_sections(sel_uebung)
    if forced_sections:
        ep_key = exercise_prompt_state_key(session_key)
        st.session_state[ep_key] = forced_sections.get("prompt", "")
        st.session_state[f"{ep_key}_exercise"] = sel_uebung
    global_prompt_default = load_global_prompt()
    initialize_prompt_states(sel_uebung, session_key, global_prompt_default, exercise_sections)
    
    # Only use saved values, start empty otherwise (transfer buttons populate)
    default_example1 = exercise_sections.get("example1", "")
    default_example2 = exercise_sections.get("example2", "")
    
    # System Prompt Section - Simplified 2-section structure
    st.markdown("---")
    st.subheader("System-Prompt")
    
    # Global Prompt
    gp_state_key = global_prompt_state_key()
    gp_pending_load_key = f"{gp_state_key}_pending_load"
    gp_pending_value = st.session_state.pop(gp_pending_load_key, None)
    if gp_pending_value is not None:
        st.session_state[gp_state_key] = gp_pending_value
    
    with st.expander("Globaler System-Prompt", expanded=False):
        gp_current_value = st.session_state.get(gp_state_key, "")
        st.text_area(
            "Globaler System-Prompt",
            key=gp_state_key,
            height=calculate_text_area_height(gp_current_value),
            label_visibility="collapsed",
        )
        gp_button_cols = st.columns(2)
        with gp_button_cols[0]:
            if st.button("📥 Laden", key=f"{gp_state_key}_load", use_container_width=True):
                st.session_state[gp_pending_load_key] = load_global_prompt()
                st.rerun()
        with gp_button_cols[1]:
            gp_save_key = f"{gp_state_key}_save"
            if st.button("💾 Speichern", key=gp_save_key, use_container_width=True):
                st.session_state[f"{gp_save_key}_dialog"] = True
            def _save_global_prompt() -> None:
                save_global_prompt(st.session_state.get(gp_state_key, ""))
                github_success = commit_prompt_file(
                    GLOBAL_PROMPT_PATH,
                    "Update global system prompt",
                )
                if github_success:
                    st.session_state["pending_toast"] = "✅ Globaler Prompt gespeichert."
                else:
                    st.session_state["pending_toast"] = (
                        "⚠️ Globaler Prompt lokal gespeichert, aber GitHub-Sync fehlgeschlagen! "
                        "Daten gehen beim Neustart verloren."
                    )
            confirm_action(
                f"{gp_save_key}_dialog",
                "Möchten Sie den globalen System-Prompt wirklich überschreiben?",
                _save_global_prompt,
            )
    
    # Exercise-Specific Prompt
    ep_state_key = exercise_prompt_state_key(session_key)
    ep_pending_load_key = f"{ep_state_key}_pending_load"
    ep_pending_value = st.session_state.pop(ep_pending_load_key, None)
    if ep_pending_value is not None:
        st.session_state[ep_state_key] = ep_pending_value
    
    with st.expander(f"Übungsspezifischer Prompt ({sel_uebung})", expanded=False):
        st.text_area(
            f"Übungsspezifischer Prompt",
            key=ep_state_key,
            height=250,
            label_visibility="collapsed",
        )
        ep_button_cols = st.columns(2)
        with ep_button_cols[0]:
            if st.button("📥 Laden", key=f"{ep_state_key}_load", use_container_width=True):
                fresh_sections = get_exercise_sections(sel_uebung)
                st.session_state[ep_pending_load_key] = fresh_sections.get("prompt", "")
                st.session_state[f"{ep_state_key}_exercise"] = sel_uebung
                st.rerun()
        with ep_button_cols[1]:
            ep_save_key = f"{ep_state_key}_save"
            if st.button("💾 Speichern", key=ep_save_key, use_container_width=True):
                st.session_state[f"{ep_save_key}_dialog"] = True
            def _save_exercise_prompt() -> None:
                save_exercise_payload(sel_uebung, {"prompt": st.session_state.get(ep_state_key, "")})
                refreshed_sections = get_exercise_sections(sel_uebung)
                st.session_state[f"{session_key}_force_reload_sections"] = refreshed_sections
            confirm_action(
                f"{ep_save_key}_dialog",
                f"Möchten Sie den Prompt für '{sel_uebung}' wirklich speichern?",
                _save_exercise_prompt,
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
        example1_value = default_example1
        st.session_state[example1_state_key] = example1_value
    # Note: Widget with key=example1_state_key manages its own session state
    # We only set it explicitly during transfer or load operations
    
    # Check if we need to update Beispiel 1 from a previous recap generation
    update_ex1_key = f"{session_key}_update_ex1"
    if update_ex1_key in st.session_state:
        st.session_state[example1_state_key] = st.session_state[update_ex1_key]
        del st.session_state[update_ex1_key]
    
    pending_load_key = f"{example1_state_key}_pending_load"
    if pending_load_key in st.session_state:
        st.session_state[example1_state_key] = st.session_state[pending_load_key]
        del st.session_state[pending_load_key]
    
    # Create widget - Streamlit manages session state via key
    # Always create the widget the same way - Streamlit will use the value from session state
    example1_text = st.text_area(
        "Beispiel 1 INHALT",
        key=example1_state_key,
        height=300,
        label_visibility="collapsed",
    )
    
    # Model selector for Beispiel 1
    ex1_model_key = f"{session_key}_ex1_model"
    if ex1_model_key not in st.session_state:
        st.session_state[ex1_model_key] = MISTRAL_DEFAULT_MODEL
    st.selectbox(
        "Modell",
        options=MISTRAL_MODEL_OPTIONS,
        key=ex1_model_key,
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
                    
                    global_prompt = get_current_global_prompt(session_key)
                    exercise_prompt = get_current_exercise_prompt(session_key)
                    system_prompt_for_ex1 = assemble_system_prompt(global_prompt, exercise_prompt)
                    
                    # Build prompt: system prompt + INHALT
                    prompt = f"{system_prompt_for_ex1}\n\n# INHALT\n{example1_text}"
                    selected_model_ex1 = st.session_state.get(ex1_model_key, MISTRAL_DEFAULT_MODEL)
                    print_params_debug(
                        "beispiel1",
                        {"model": selected_model_ex1, "max_tokens": MISTRAL_MAX_TOKENS},
                    )
                    
                    with st.spinner("Generiere Recap..."):
                        recap = generate_summary_with_mistral(
                            prompt=prompt,
                            api_key=MISTRAL_API_KEY,
                            model=selected_model_ex1,
                            max_tokens=MISTRAL_MAX_TOKENS,
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
        confirm_ex1_key = f"{save_ex1_key}_dialog"
        if st.button("💾 Beispiel 1 speichern", key=save_ex1_key, use_container_width=True):
            st.session_state[confirm_ex1_key] = True

        def _save_example1() -> None:
            if save_exercise_payload(sel_uebung, {"example1": example1_text}):
                st.session_state[confirm_ex1_key] = False
                st.rerun()

        confirm_action(
            confirm_ex1_key,
            f"Möchten Sie Beispiel 1 für '{sel_uebung}' wirklich überschreiben?",
            _save_example1,
        )
    
    with example1_button_cols[2]:
        if st.button("📥 Beispiel 1 laden", key=f"{session_key}_load_ex1", use_container_width=True):
            example1_value = exercise_sections.get("example1", "")
            st.session_state[f"{example1_state_key}_pending_load"] = example1_value
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
        example2_value = default_example2
        st.session_state[example2_state_key] = example2_value
    # Note: Widget with key=example2_state_key manages its own session state
    # We only set it explicitly during transfer or load operations
    
    # Check if we need to update Beispiel 2 from a previous recap generation
    update_ex2_key = f"{session_key}_update_ex2"
    if update_ex2_key in st.session_state:
        st.session_state[example2_state_key] = st.session_state[update_ex2_key]
        del st.session_state[update_ex2_key]
    
    pending_load_key = f"{example2_state_key}_pending_load"
    if pending_load_key in st.session_state:
        st.session_state[example2_state_key] = st.session_state[pending_load_key]
        del st.session_state[pending_load_key]
    
    # Create widget - Streamlit manages session state via key
    # Always create the widget the same way - Streamlit will use the value from session state
    example2_text = st.text_area(
        "Beispiel 2 Inhalt",
        key=example2_state_key,
        height=300,
        label_visibility="collapsed",
    )
    
    # Model selector for Beispiel 2
    ex2_model_key = f"{session_key}_ex2_model"
    if ex2_model_key not in st.session_state:
        st.session_state[ex2_model_key] = MISTRAL_DEFAULT_MODEL
    st.selectbox(
        "Modell",
        options=MISTRAL_MODEL_OPTIONS,
        key=ex2_model_key,
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
                    
                    global_prompt = get_current_global_prompt(session_key)
                    exercise_prompt = get_current_exercise_prompt(session_key)
                    system_prompt_for_ex2 = assemble_system_prompt(global_prompt, exercise_prompt)
                    prompt = f"{system_prompt_for_ex2}\n\n# INHALT\n{example2_text}"
                    selected_model_ex2 = st.session_state.get(ex2_model_key, MISTRAL_DEFAULT_MODEL)
                    print_params_debug(
                        "beispiel2",
                        {"model": selected_model_ex2, "max_tokens": MISTRAL_MAX_TOKENS},
                    )
                    
                    with st.spinner("Generiere Recap..."):
                        recap = generate_summary_with_mistral(
                            prompt=prompt,
                            api_key=MISTRAL_API_KEY,
                            model=selected_model_ex2,
                            max_tokens=MISTRAL_MAX_TOKENS,
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
        confirm_ex2_key = f"{save_ex2_key}_dialog"
        if st.button("💾 Beispiel 2 speichern", key=save_ex2_key, use_container_width=True):
            st.session_state[confirm_ex2_key] = True

        def _save_example2() -> None:
            if save_exercise_payload(sel_uebung, {"example2": example2_text}):
                st.session_state[confirm_ex2_key] = False
                st.rerun()

        confirm_action(
            confirm_ex2_key,
            f"Möchten Sie Beispiel 2 für '{sel_uebung}' wirklich überschreiben?",
            _save_example2,
        )
    
    with example2_button_cols[2]:
        if st.button("📥 Beispiel 2 laden", key=f"{session_key}_load_ex2", use_container_width=True):
            example2_value = exercise_sections.get("example2", "")
            st.session_state[f"{example2_state_key}_pending_load"] = example2_value
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
        # Start empty - use transfer button to populate
        st.session_state[mainrecap_inhalt_key] = ""
    # Note: Widget with key=mainrecap_inhalt_key manages its own session state
    
    # Show TEST Inhalt text area - Streamlit manages session state via key
    # Always create the widget the same way - Streamlit will use the value from session state
    mainrecap_inhalt_text = st.text_area(
        "TEST Inhalt",
        key=mainrecap_inhalt_key,
        height=300,
        label_visibility="collapsed",
    )
    
    # Model selector for TEST
    main_model_key = f"{session_key}_main_model"
    if main_model_key not in st.session_state:
        st.session_state[main_model_key] = MISTRAL_DEFAULT_MODEL
    st.selectbox(
        "Modell",
        options=MISTRAL_MODEL_OPTIONS,
        key=main_model_key,
        label_visibility="collapsed",
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
                height=calculate_text_area_height(last_prompt),
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
                
                global_prompt = get_current_global_prompt(session_key)
                exercise_prompt = get_current_exercise_prompt(session_key)
                system_prompt_for_main = assemble_system_prompt(global_prompt, exercise_prompt)
                
                # Get the current text area values (including any generated recaps)
                example1_final = st.session_state.get(example1_state_key, exercise_sections.get("example1", "")).strip()
                example2_final = st.session_state.get(example2_state_key, exercise_sections.get("example2", "")).strip()
                
                # Build examples section only if at least one example has content
                examples_parts = []
                if example1_final:
                    examples_parts.append(f"## Beispiel 1\n{example1_final}")
                if example2_final:
                    examples_parts.append(f"## Beispiel 2\n{example2_final}")
                
                if examples_parts:
                    examples_section = f"# BEISPIELE\n\n" + "\n\n".join(examples_parts) + "\n\n"
                else:
                    examples_section = ""
                
                # Build complete Mistral prompt
                mistral_prompt = f"{system_prompt_for_main}\n\n{examples_section}# INHALT\n{mainrecap_inhalt_text}"
                selected_model_main = st.session_state.get(main_model_key, MISTRAL_DEFAULT_MODEL)
                print_params_debug(
                    "test",
                    {
                        "model": selected_model_main,
                        "max_tokens": MISTRAL_MAX_TOKENS,
                        "temperature": main_mistral_temperature,
                        "top_p": main_mistral_top_p,
                    },
                )
                
                # Store the exact prompt that we're about to send
                st.session_state[last_mistral_prompt_key] = mistral_prompt
                
                start_time = time.time()
                
                # Stream the response in real-time
                recap = st.write_stream(
                    stream_summary_with_mistral(
                        prompt=mistral_prompt,
                        api_key=MISTRAL_API_KEY,
                        model=selected_model_main,
                        max_tokens=MISTRAL_MAX_TOKENS,
                        temperature=main_mistral_temperature,
                        top_p=main_mistral_top_p,
                    )
                )
                
                latency_ms = (time.time() - start_time) * 1000
                st.session_state[r2_state_key] = recap.strip() if recap else ""
                st.session_state[f"{session_key}_latency"] = latency_ms
                
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
    
    # Display word count and latency for the generated summary
    summary_text = st.session_state.get(r2_state_key, "")
    latency_ms = st.session_state.get(f"{session_key}_latency")
    if summary_text.strip():
        word_count = len(summary_text.split())
        latency_str = f" | ⏱️ Latenz: {latency_ms:.0f} ms" if latency_ms else ""
        st.caption(f"📊 Anzahl Wörter: {word_count}{latency_str}")
    else:
        st.caption("📊 Anzahl Wörter: 0")

else:
    st.info("Bitte wähle eine Übung aus.")


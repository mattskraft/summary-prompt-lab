"""Configuration management for Summary Prompt Lab."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

def get_project_root() -> Path:
    """Get project root directory.
    
    Tries multiple strategies to find the project root:
    1. From environment variable (for Streamlit Cloud)
    2. From config.py location (2 levels up)
    3. From current working directory
    """
    # Strategy 1: Check if PROJECT_ROOT is set (useful for Streamlit Cloud)
    if os.getenv("PROJECT_ROOT"):
        root = Path(os.getenv("PROJECT_ROOT"))
        if root.exists():
            return root
    
    # Strategy 2: Calculate from this file's location
    # config.py is at: project_root/src/kiso_input/config.py
    config_file = Path(__file__).resolve()
    root_from_config = config_file.parents[2]
    if (root_from_config / "apps" / "streamlit_app.py").exists():
        return root_from_config
    
    # Strategy 3: Try to find project root from current working directory
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "apps" / "streamlit_app.py").exists():
            return parent
    
    # Fallback: use config.py location
    return root_from_config

# Get project root
PROJECT_ROOT = get_project_root()

# Load environment variables from config/.env
ENV_FILE = PROJECT_ROOT / "config" / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # Also try loading from current directory (for backward compatibility)
    load_dotenv()

# Paths from environment variables
# Support both naming conventions for flexibility
_struct_path_raw = os.getenv("KISO_STRUCT_JSON") or os.getenv("STRUCT_JSON_PATH")
_sn_path_raw = os.getenv("KISO_SN_JSON") or os.getenv("SN_JSON_PATH")
_lexicon_path_raw = os.getenv("KISO_SUICIDE_LEXICON") or os.getenv("SUICIDE_LEXICON_PATH")
_prompts_path_raw = os.getenv("KISO_PROMPTS_CONFIG") or os.getenv("PROMPTS_CONFIG_PATH")
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
APP_PASSWORD: Optional[str] = os.getenv("APP_PASSWORD")

# Resolve paths if they're set (convert relative to absolute)
STRUCT_JSON_PATH: Optional[str] = None
if _struct_path_raw:
    struct_path = Path(_struct_path_raw)
    if struct_path.is_absolute():
        STRUCT_JSON_PATH = str(struct_path) if struct_path.exists() else None
    else:
        # Try relative to PROJECT_ROOT first, then CWD
        for base in [PROJECT_ROOT, Path.cwd()]:
            full_path = base / struct_path
            if full_path.exists():
                STRUCT_JSON_PATH = str(full_path.resolve())
                break

SN_JSON_PATH: Optional[str] = None
if _sn_path_raw:
    sn_path = Path(_sn_path_raw)
    if sn_path.is_absolute():
        SN_JSON_PATH = str(sn_path) if sn_path.exists() else None
    else:
        # Try relative to PROJECT_ROOT first, then CWD
        for base in [PROJECT_ROOT, Path.cwd()]:
            full_path = base / sn_path
            if full_path.exists():
                SN_JSON_PATH = str(full_path.resolve())
                break

# Default paths relative to project root
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_STRUCT_JSON = DEFAULT_DATA_DIR / "kiso_app_merged_structured.json"
DEFAULT_SN_JSON = DEFAULT_DATA_DIR / "kiso_app_storynodes_struct.json"

# Use defaults if not set via environment
if not STRUCT_JSON_PATH:
    # Try multiple path strategies
    possible_paths = [
        DEFAULT_STRUCT_JSON,  # Relative to calculated PROJECT_ROOT
        Path("data/processed/kiso_app_merged_structured.json"),  # Relative to CWD
        Path("./data/processed/kiso_app_merged_structured.json"),  # Explicit relative
    ]
    for path in possible_paths:
        if path.exists():
            STRUCT_JSON_PATH = str(path.resolve())
            break
    else:
        # Fallback to shared data directory (only for local development)
        shared_data = Path.home() / "Kiso" / "data" / "processed" / "kiso_app_merged_structured.json"
        if shared_data.exists():
            STRUCT_JSON_PATH = str(shared_data)

if not SN_JSON_PATH:
    # Try multiple path strategies
    possible_paths = [
        DEFAULT_SN_JSON,  # Relative to calculated PROJECT_ROOT
        Path("data/processed/kiso_app_storynodes_struct.json"),  # Relative to CWD
        Path("./data/processed/kiso_app_storynodes_struct.json"),  # Explicit relative
    ]
    for path in possible_paths:
        if path.exists():
            SN_JSON_PATH = str(path.resolve())
            break
    else:
        # Fallback to shared data directory (only for local development)
        shared_data = Path.home() / "Kiso" / "data" / "processed" / "kiso_app_storynodes_struct.json"
        if shared_data.exists():
            SN_JSON_PATH = str(shared_data)

SUICIDE_LEXICON_PATH: Optional[str] = None
DEFAULT_LEXICON_PATH = PROJECT_ROOT / "config" / "safety_3tier_de.yaml"
if _lexicon_path_raw:
    lexicon_path = Path(_lexicon_path_raw)
    if lexicon_path.is_absolute():
        SUICIDE_LEXICON_PATH = str(lexicon_path) if lexicon_path.exists() else None
    else:
        for base in [PROJECT_ROOT, Path.cwd()]:
            full_path = base / lexicon_path
            if full_path.exists():
                SUICIDE_LEXICON_PATH = str(full_path.resolve())
                break

if not SUICIDE_LEXICON_PATH:
    if DEFAULT_LEXICON_PATH.exists():
        SUICIDE_LEXICON_PATH = str(DEFAULT_LEXICON_PATH.resolve())
    else:
        fallback_candidates = [
            PROJECT_ROOT / "config" / "suicide_lexicon_de.yaml",
            PROJECT_ROOT / "config" / "lexica" / "suicide_lexicon_de.yaml",
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                SUICIDE_LEXICON_PATH = str(candidate.resolve())
                break

PROMPTS_CONFIG_PATH: Optional[str] = None
DEFAULT_PROMPTS_CONFIG = PROJECT_ROOT / "config" / "prompts.yaml"
if _prompts_path_raw:
    prompts_path = Path(_prompts_path_raw)
    if prompts_path.is_absolute():
        PROMPTS_CONFIG_PATH = str(prompts_path) if prompts_path.exists() else None
    else:
        for base in [PROJECT_ROOT, Path.cwd()]:
            full_path = base / prompts_path
            if full_path.exists():
                PROMPTS_CONFIG_PATH = str(full_path.resolve())
                break

if not PROMPTS_CONFIG_PATH and DEFAULT_PROMPTS_CONFIG.exists():
    PROMPTS_CONFIG_PATH = str(DEFAULT_PROMPTS_CONFIG.resolve())

# Model configurations
GEMINI_MODEL_SUMMARY = os.getenv("GEMINI_MODEL_SUMMARY", "gemini-2.5-flash-lite")
GEMINI_MODEL_ANSWERS = os.getenv("GEMINI_MODEL_ANSWERS", "gemini-2.5-flash")
MISTRAL_MODEL_SUMMARY = os.getenv("MISTRAL_MODEL_SUMMARY", "mistral-medium-latest")
MISTRAL_MODEL_ANSWERS = os.getenv("MISTRAL_MODEL_ANSWERS", "mistral-medium-latest")


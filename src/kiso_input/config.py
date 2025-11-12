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
STRUCT_JSON_PATH: Optional[str] = os.getenv("KISO_STRUCT_JSON")
SN_JSON_PATH: Optional[str] = os.getenv("KISO_SN_JSON")
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

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


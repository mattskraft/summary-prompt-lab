"""Configuration management for Summary Prompt Lab."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from config/.env
# PROJECT_ROOT is 2 levels up from src/kiso_input/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    if DEFAULT_STRUCT_JSON.exists():
        STRUCT_JSON_PATH = str(DEFAULT_STRUCT_JSON)
    else:
        # Fallback to shared data directory
        shared_data = Path.home() / "Kiso" / "data" / "processed" / "kiso_app_merged_structured.json"
        if shared_data.exists():
            STRUCT_JSON_PATH = str(shared_data)

if not SN_JSON_PATH:
    if DEFAULT_SN_JSON.exists():
        SN_JSON_PATH = str(DEFAULT_SN_JSON)
    else:
        # Fallback to shared data directory
        shared_data = Path.home() / "Kiso" / "data" / "processed" / "kiso_app_storynodes_struct.json"
        if shared_data.exists():
            SN_JSON_PATH = str(shared_data)


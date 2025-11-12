# Summary Prompt Lab

Streamlit app for exploring and testing the language-processing pipeline for Kiso Mind.

## Project Structure

```
summary-prompt-lab/
├── apps/
│   └── streamlit_app.py      # Main Streamlit application
├── src/
│   └── kiso_input/           # Importable package
│       ├── __init__.py
│       ├── process_user_input.py
│       └── config.py
├── notebooks/
│   └── 01_user_input.ipynb   # Research/development notebooks
├── data/
│   └── processed/            # Sample data (symlinked to ~/Kiso/data/processed)
├── config/
│   ├── .env.example          # Example configuration
│   └── .env                  # Your actual config (gitignored)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install in development mode:
   ```bash
   pip install -e .
   ```

2. **Configure environment:**
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your actual values
   ```

3. **Run the app:**
   ```bash
   streamlit run apps/streamlit_app.py
   ```

## Configuration

The app uses environment variables from `config/.env`:

- `KISO_STRUCT_JSON`: Path to `kiso_app_merged_structured.json`
- `KISO_SN_JSON`: Path to `kiso_app_storynodes_struct.json`
- `GEMINI_API_KEY`: Your Gemini API key for answer generation

If not set, the app will try to use default paths in `data/processed/`.

## Development

The package can be imported after installation:

```python
from kiso_input import (
    get_prompt_segments_from_exercise,
    generate_answers_with_gemini,
    build_summary_prompt,
)
```


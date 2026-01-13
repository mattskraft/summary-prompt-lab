# Summary Prompt Lab

A prompt engineering workbench for developing and testing LLM-based summary generation for the **Kiso Mind** app. The prompts engineered here will be used in production to generate user input summaries.

## Purpose

The Kiso Mind app contains exercises where users answer questions (multiple choice, sliders, free text). This project provides a Streamlit interface to:

1. **Navigate exercises** from the Kiso app content structure
2. **Generate synthetic user answers** using Mistral API to simulate realistic input
3. **Test and refine summary prompts** by generating summaries with different models
4. **Configure which exercises need summaries** (some exercises have no questions)
5. **Save engineered prompts** to JSON with optional GitHub sync for collaboration

## Project Structure

```
summary-prompt-lab/
├── apps/
│   └── streamlit_app_mistral.py  # Main Streamlit application
├── src/
│   └── kiso_input/               # Core processing package
│       ├── __init__.py
│       ├── config.py
│       └── processing/           # Segment processing, API calls, prompts
├── config/
│   ├── system_prompt_global.txt       # Global system prompt template
│   ├── exercise_specific_prompts.json # Per-exercise prompt configurations
│   ├── segment_switches.json          # Exercise enable/disable settings
│   └── synth_answers_prompt.txt       # Prompt for synthetic answer generation
├── data/
│   └── processed/                # Kiso app exercise data (JSON)
├── notebooks/
│   └── 01_user_input.ipynb       # Research/development notebooks
├── scripts/
│   └── benchmark_mistral_latency.py
├── archive/
│   ├── api/                      # Archived FastAPI server for local llama.cpp inference
│   └── modal/                    # Archived Modal deployment experiment
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

2. **Configure secrets:**

   Create `.streamlit/secrets.toml` for local development:
   ```toml
   MISTRAL_API_KEY = "your-mistral-api-key"
   APP_PASSWORD = "optional-password-to-protect-app"
   ```

3. **Run the app:**
   ```bash
   streamlit run apps/streamlit_app_mistral.py
   ```

## Configuration

### Required

- `MISTRAL_API_KEY`: Your Mistral API key (for answer generation and summary testing)

### Optional

- `APP_PASSWORD`: Password to protect the app (recommended for cloud deployment)
- `GITHUB_TOKEN`, `GITHUB_REPO`, `GITHUB_BRANCH`: For automatic prompt sync to GitHub

## Streamlit Cloud Deployment

The JSON data files are included in the repository for Streamlit Cloud deployment:

1. Connect your GitHub repository to Streamlit Cloud
2. Set the main file path: `apps/streamlit_app_mistral.py`
3. Add secrets in Streamlit Cloud settings:
   - `MISTRAL_API_KEY` (required)
   - `APP_PASSWORD` (recommended)
   - GitHub sync secrets (optional)

## Workflow

1. **Select an exercise** using the sidebar navigation (Thema → Pfad → Übung)
2. **Generate synthetic answers** to simulate user input
3. **Transfer answers** to Example 1, Example 2, or TEST section
4. **Generate summaries** and iterate on the prompt
5. **Save configurations** when satisfied with results

## Archived: Local Inference API

The `archive/api/` folder contains a FastAPI server for running inference locally using llama.cpp backends. This was an early experiment as an alternative to the Mistral cloud API approach.

## Development

The package can be imported after installation:

```python
from kiso_input import (
    get_prompt_segments_from_exercise,
    generate_answers_with_mistral,
)
from kiso_input.processing.cloud_apis import (
    generate_summary_with_mistral,
    stream_summary_with_mistral,
)
```

# Modal Endpoints Setup

This document explains how to set up and use Modal endpoints for the five LLM models.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate
   ```bash
   pip install modal
   modal token new
   ```
3. **HuggingFace Secret**: Create a Modal secret named `huggingface` with your HuggingFace token
   ```bash
   modal secret create huggingface HF_TOKEN=your_huggingface_token_here
   ```

## Deployment

Deploy the Modal app:

```bash
cd /home/matthias/Kiso/code/projects/summary-prompt-lab
modal deploy modal_app.py
```

After deployment, Modal will provide URLs for each endpoint. The URLs will be in the format:
```
https://{username}--llama-inference-serve-{model}.modal.run
```

## Configuration

The code uses a priority system for endpoint URLs:
1. **Streamlit Secrets** (highest priority - for Streamlit Cloud overrides)
2. **Environment Variables** (for local development with custom endpoints)
3. **Default URLs** (committed to repo - shared endpoints for team)

**Important**: For Streamlit Cloud, you **must** use Streamlit Secrets if you want to override the defaults. Environment variables in `.env` files are **not** accessible on Streamlit Cloud.

### Option 1: Default URLs (Recommended for Team Sharing)

The easiest way to share endpoints with your team is to set them as defaults in the code. Edit `src/kiso_input/processing/local_models.py` and update `DEFAULT_MODAL_ENDPOINTS`:

```python
DEFAULT_MODAL_ENDPOINTS: Dict[str, Optional[str]] = {
    "Gemma-3-12B": "https://your-username--llama-inference-serve-gemma.modal.run",
    "Llama-3.1-8B": "https://your-username--llama-inference-serve-llama.modal.run",
    "Mistral-7B": "https://your-username--llama-inference-serve-mistral.modal.run",
    "Qwen3-8B": "https://your-username--llama-inference-serve-qwen.modal.run",
    "Teuken-7B": "https://your-username--llama-inference-serve-teuken.modal.run",
}
```

Then commit this file. Your colleagues will automatically use these endpoints without any configuration!

### Option 2: Streamlit Secrets (For Overrides on Streamlit Cloud)

The code automatically checks Streamlit secrets first, then falls back to environment variables.

#### For Streamlit Cloud:

1. Go to your Streamlit Cloud app settings
2. Navigate to "Secrets" section
3. Add the following secrets:

```toml
GEMMA_3_12B_MODAL_URL = "https://your-username--llama-inference-serve-gemma.modal.run"
LLAMA_3_1_8B_MODAL_URL = "https://your-username--llama-inference-serve-llama.modal.run"
MISTRAL_7B_MODAL_URL = "https://your-username--llama-inference-serve-mistral.modal.run"
QWEN3_8B_MODAL_URL = "https://your-username--llama-inference-serve-qwen.modal.run"
TEUKEN_7B_MODAL_URL = "https://your-username--llama-inference-serve-teuken.modal.run"
```

#### For Local Development (Optional):

You can also create `.streamlit/secrets.toml` for local testing (this file is gitignored):

```toml
GEMMA_3_12B_MODAL_URL = "https://your-username--llama-inference-serve-gemma.modal.run"
LLAMA_3_1_8B_MODAL_URL = "https://your-username--llama-inference-serve-llama.modal.run"
MISTRAL_7B_MODAL_URL = "https://your-username--llama-inference-serve-mistral.modal.run"
QWEN3_8B_MODAL_URL = "https://your-username--llama-inference-serve-qwen.modal.run"
TEUKEN_7B_MODAL_URL = "https://your-username--llama-inference-serve-teuken.modal.run"
```

### Option 3: Environment Variables (For Local Overrides)

For local development, you can also use environment variables (but these won't work on Streamlit Cloud):

Add to your `.env` file or export in your shell:

```bash
export GEMMA_3_12B_MODAL_URL="https://your-username--llama-inference-serve-gemma.modal.run"
export LLAMA_3_1_8B_MODAL_URL="https://your-username--llama-inference-serve-llama.modal.run"
export MISTRAL_7B_MODAL_URL="https://your-username--llama-inference-serve-mistral.modal.run"
export QWEN3_8B_MODAL_URL="https://your-username--llama-inference-serve-qwen.modal.run"
export TEUKEN_7B_MODAL_URL="https://your-username--llama-inference-serve-teuken.modal.run"
```

**Note**: The priority is: Streamlit secrets > Environment variables > Default URLs. This means:
- If you set defaults in code, everyone uses them automatically
- Individual users can override via secrets/env vars if needed
- Streamlit Cloud can override defaults via secrets

## Usage

Once configured, the Streamlit app will automatically use the Modal endpoints when you click the model buttons. The app uses the `api` backend type, which sends requests to the Modal endpoints via HTTP.

## Model Endpoints

Each model is served on a separate endpoint:

- **Gemma-3-12B**: Port 8000
- **Llama-3.1-8B**: Port 8001
- **Mistral-7B**: Port 8002
- **Qwen3-8B**: Port 8003
- **Teuken-7B**: Port 8004

All endpoints use vLLM's OpenAI-compatible API at `/v1/completions`.

## Troubleshooting

1. **Endpoint not found**: Make sure you've deployed the Modal app and copied the correct URLs
2. **Authentication errors**: Ensure your HuggingFace secret is set up correctly in Modal
3. **Model loading errors**: Check that the HuggingFace model names are correct and accessible
4. **Timeout errors**: Increase the timeout in `ApiBackend.generate_summary()` if needed

## GPU Configuration

By default, all endpoints use `T4` GPUs. You can change this in `modal_app.py`:

```python
gpu="A10G"  # or "H100" for better performance
```

Note: Different GPU types have different costs and availability.

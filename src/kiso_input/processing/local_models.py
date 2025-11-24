"""Local model inference using llama_cpp with modular design for future API support."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any

# Model configuration mapping model names to file paths
# Can be overridden via MODEL_DIR environment variable (for Cloud Run)
MODEL_DIR = Path(os.getenv("MODEL_DIR", os.path.expanduser("~/Kiso/data/models")))

MODEL_MAP: Dict[str, Path] = {
    "Gemma-3-12B": MODEL_DIR / "gemma-3-12b-it-q4_0.gguf",
    "Llama-3.1-8B": MODEL_DIR / "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
    "Mistral-7B": MODEL_DIR / "mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    "Qwen3-8B": MODEL_DIR / "Qwen3-8B-Q5_K_M.gguf",
    "Teuken-7B": MODEL_DIR / "Teuken-7B-instruct-commercial-v0.4.i1-Q4_K_M.gguf",
    "Mistral-NeMo-12B": MODEL_DIR / "mistral-nemo-12b-instruct.gguf",
}

# Default Modal endpoint URLs (shared endpoints - can be committed to repo)
# These can be overridden via Streamlit secrets or environment variables
# Format: https://{username}--{app-name}-{function-name}.modal.run
DEFAULT_MODAL_ENDPOINTS: Dict[str, Optional[str]] = {
    "Gemma-3-12B": "https://mattskraft--llama-inference-serve-gemma.modal.run",
    "Llama-3.1-8B": "https://mattskraft--llama-inference-serve-llama.modal.run",
    "Mistral-7B": "https://mattskraft--llama-inference-serve-mistral.modal.run",
    "Qwen3-8B": "https://mattskraft--llama-inference-serve-qwen.modal.run",
    "Teuken-7B": "https://mattskraft--llama-inference-serve-teuken.modal.run",
    "Mistral-NeMo-12B": "https://mattskraft--llama-inference-serve-mistral-nemo.modal.run",
}


def _get_modal_endpoint_url(model_name: str) -> Optional[str]:
    """Get Modal endpoint URL for a model.
    
    Priority order:
    1. Streamlit secrets (for Streamlit Cloud)
    2. Environment variables (for local development with custom endpoints)
    3. Default URLs from DEFAULT_MODAL_ENDPOINTS (shared endpoints, committed to repo)
    
    Args:
        model_name: Name of the model
        
    Returns:
        Modal endpoint URL or None if not configured
    """
    # Map model names to environment variable names
    env_var_map = {
        "Gemma-3-12B": "GEMMA_3_12B_MODAL_URL",
        "Llama-3.1-8B": "LLAMA_3_1_8B_MODAL_URL",
        "Mistral-7B": "MISTRAL_7B_MODAL_URL",
        "Qwen3-8B": "QWEN3_8B_MODAL_URL",
        "Teuken-7B": "TEUKEN_7B_MODAL_URL",
        "Mistral-NeMo-12B": "MISTRAL_NEMO_12B_MODAL_URL",
    }
    
    env_var_name = env_var_map.get(model_name)
    if not env_var_name:
        return None
    
    # Priority 1: Try Streamlit secrets first (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            try:
                # Check if secrets is accessible
                _ = len(st.secrets)
                # If we get here, secrets exist
                url = st.secrets.get(env_var_name)
                if url:
                    return url
            except Exception:
                # No secrets file found, fall through to env vars
                pass
    except (ImportError, AttributeError, KeyError, TypeError):
        # streamlit not available or secrets not accessible
        pass
    
    # Priority 2: Fallback to environment variable (for local development with custom endpoints)
    env_url = os.getenv(env_var_name)
    if env_url:
        return env_url
    
    # Priority 3: Use default URL from DEFAULT_MODAL_ENDPOINTS (shared endpoints)
    return DEFAULT_MODAL_ENDPOINTS.get(model_name)


# Modal endpoint URLs for API backend
# These are loaded dynamically with priority: secrets > env vars > defaults
MODAL_ENDPOINTS: Dict[str, Optional[str]] = {
    "Gemma-3-12B": _get_modal_endpoint_url("Gemma-3-12B"),
    "Llama-3.1-8B": _get_modal_endpoint_url("Llama-3.1-8B"),
    "Mistral-7B": _get_modal_endpoint_url("Mistral-7B"),
    "Qwen3-8B": _get_modal_endpoint_url("Qwen3-8B"),
    "Teuken-7B": _get_modal_endpoint_url("Teuken-7B"),
    "Mistral-NeMo-12B": _get_modal_endpoint_url("Mistral-NeMo-12B"),
}


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""
    
    @abstractmethod
    def generate_summary(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate a summary from a prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated summary text
        """
        pass


class LlamaCppBackend(InferenceBackend):
    """Local inference using llama_cpp."""
    
    def __init__(self, model_path: Path, n_ctx: int = 2048, n_threads: int = 8):
        """Initialize llama_cpp backend.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_threads: Number of threads for inference
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python package not installed. Install it with: pip install llama-cpp-python"
            ) from exc
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self._llm: Optional[Llama] = None
        self.n_ctx = n_ctx
        self.n_threads = n_threads
    
    def _get_llm(self) -> Any:
        """Lazy load the Llama model."""
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
                seed=-1,  # -1 = true random seed per run
            )
        return self._llm
    
    def generate_summary(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate summary using llama_cpp."""
        llm = self._get_llm()
        result = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        summary = result["choices"][0]["text"].strip()
        return summary


class ApiBackend(InferenceBackend):
    """API-based inference backend for Modal vLLM endpoints."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """Initialize API backend.
        
        Args:
            api_url: Base URL for the inference API (Modal endpoint URL)
            api_key: Optional API key for authentication (not used for vLLM)
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
    
    def generate_summary(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate summary using vLLM API endpoint (OpenAI-compatible).
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated summary text
        """
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "requests package not installed. Install it with: pip install requests"
            ) from exc
        
        # vLLM uses OpenAI-compatible API endpoint
        url = f"{self.api_url}/v1/completions"
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": [],  # No stop tokens
        }
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            # Increased timeout to 600 seconds (10 minutes) for model inference
            # This accounts for cold starts and longer generation times
            import logging
            import time
            
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            # Log request details (without full prompt to avoid spam)
            prompt_preview = payload["prompt"][:100] + "..." if len(payload["prompt"]) > 100 else payload["prompt"]
            logger.info(f"Sending request to {url}")
            logger.info(f"Request details: max_tokens={payload['max_tokens']}, temperature={payload['temperature']}, prompt_length={len(payload['prompt'])}")
            
            start_time = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=600)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Received response status: {response.status_code} after {elapsed_time:.2f} seconds")
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Response JSON keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Extract text from OpenAI-compatible response
            if "choices" in result and len(result["choices"]) > 0:
                text = result["choices"][0].get("text", "").strip()
                logger.info(f"Generated text length: {len(text)} characters")
                return text
            else:
                logger.error(f"Unexpected API response format: {result}")
                raise ValueError(f"Unexpected API response format: {result}")
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out after 600 seconds: {e}")
            raise RuntimeError(f"API request timed out after 600 seconds. The model may be taking too long to respond.") from e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise RuntimeError(f"Could not connect to Modal endpoint {self.api_url}. Check if the endpoint is running and accessible.") from e
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}, Response: {e.response.text if hasattr(e, 'response') and e.response else 'No response'}")
            raise RuntimeError(f"API request failed with HTTP error: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            raise RuntimeError(f"API request failed: {e}") from e


def get_backend_for_model(model_name: str, backend_type: str = "local") -> InferenceBackend:
    """Get an inference backend for a specific model.
    
    Args:
        model_name: Name of the model (must be in MODEL_MAP)
        backend_type: Type of backend ("local" for llama_cpp, "api" for API)
        
    Returns:
        InferenceBackend instance
        
    Raises:
        KeyError: If model_name is not in MODEL_MAP
        ValueError: If backend_type is not supported
    """
    if model_name not in MODEL_MAP:
        available = ", ".join(MODEL_MAP.keys())
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available}")
    
    if backend_type == "local":
        model_path = MODEL_MAP[model_name]
        return LlamaCppBackend(model_path)
    elif backend_type == "api":
        # Get Modal endpoint URL - try dynamic lookup first (supports Streamlit secrets)
        api_url = _get_modal_endpoint_url(model_name)
        
        # Fallback to static MODAL_ENDPOINTS dict (for backwards compatibility)
        if not api_url:
            api_url = MODAL_ENDPOINTS.get(model_name)
        
        # Final fallback to environment variable with standard naming
        if not api_url:
            env_var_name = f"{model_name.upper().replace('-', '_')}_MODAL_URL"
            api_url = os.getenv(env_var_name)
        
        if not api_url:
            raise ValueError(
                f"Modal endpoint URL not configured for model {model_name}. "
                f"For Streamlit Cloud: Add {model_name.upper().replace('-', '_')}_MODAL_URL to Streamlit secrets. "
                f"For local development: Set {model_name.upper().replace('-', '_')}_MODAL_URL environment variable."
            )
        api_key = os.getenv(f"{model_name.upper().replace('-', '_')}_API_KEY")
        return ApiBackend(api_url, api_key)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def generate_summary_with_model(
    prompt: str,
    model_name: str,
    backend_type: str = "local",
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Generate a summary using a specified model.
    
    Args:
        prompt: The input prompt
        model_name: Name of the model to use
        backend_type: Type of backend ("local" or "api")
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated summary text
    """
    backend = get_backend_for_model(model_name, backend_type)
    return backend.generate_summary(prompt, max_tokens=max_tokens, temperature=temperature)


def get_available_models() -> list[str]:
    """Get list of available model names that have existing model files."""
    available = []
    for model_name, model_path in MODEL_MAP.items():
        if model_path.exists():
            available.append(model_name)
    return available


def is_local_models_available() -> bool:
    """Check if local models are available (models exist and llama_cpp is installed)."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    
    # Check if at least one model file exists
    return any(model_path.exists() for model_path in MODEL_MAP.values())


__all__ = [
    "InferenceBackend",
    "LlamaCppBackend",
    "ApiBackend",
    "get_backend_for_model",
    "generate_summary_with_model",
    "get_available_models",
    "is_local_models_available",
    "MODEL_MAP",
]


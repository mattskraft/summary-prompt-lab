"""Local model inference using llama_cpp with modular design for future API support."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any

# Model configuration mapping model names to file paths
MODEL_DIR = Path(os.path.expanduser("~/Kiso/data/models"))

MODEL_MAP: Dict[str, Path] = {
    "Gemma-3-12B": MODEL_DIR / "gemma-3-12b-it-q4_0.gguf",
    "Llama-3.1-8B": MODEL_DIR / "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
    "Mistral-7B": MODEL_DIR / "mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    "Qwen3-8B": MODEL_DIR / "Qwen3-8B-Q5_K_M.gguf",
    "Teuken-7B": MODEL_DIR / "Teuken-7B-instruct-commercial-v0.4.i1-Q4_K_M.gguf",
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
    """API-based inference backend (placeholder for future implementation)."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """Initialize API backend.
        
        Args:
            api_url: Base URL for the inference API
            api_key: Optional API key for authentication
        """
        self.api_url = api_url
        self.api_key = api_key
    
    def generate_summary(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate summary using API (to be implemented).
        
        This is a placeholder for future API integration.
        """
        # TODO: Implement API call
        # Example structure:
        # import requests
        # response = requests.post(
        #     f"{self.api_url}/generate",
        #     json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        #     headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # )
        # return response.json()["text"]
        raise NotImplementedError("API backend not yet implemented")


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
        # For API backend, you would configure the API URL per model
        # This is a placeholder - adjust based on your API structure
        api_url = os.getenv(f"{model_name.upper().replace('-', '_')}_API_URL", "")
        api_key = os.getenv(f"{model_name.upper().replace('-', '_')}_API_KEY")
        if not api_url:
            raise ValueError(f"API URL not configured for model {model_name}")
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
    """Get list of available model names."""
    return list(MODEL_MAP.keys())


__all__ = [
    "InferenceBackend",
    "LlamaCppBackend",
    "ApiBackend",
    "get_backend_for_model",
    "generate_summary_with_model",
    "get_available_models",
    "MODEL_MAP",
]


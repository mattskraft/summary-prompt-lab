"""FastAPI server for model inference on Cloud Run."""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kiso_input.processing.local_models import (
    LlamaCppBackend,
    MODEL_MAP,
    get_backend_for_model,
)

app = FastAPI(title="Model Inference API", version="1.0.0")

# Initialize backends lazily (per model)
_backends: dict[str, LlamaCppBackend] = {}


def get_backend(model_name: str) -> LlamaCppBackend:
    """Get or create backend for a model."""
    if model_name not in _backends:
        if model_name not in MODEL_MAP:
            raise ValueError(f"Unknown model: {model_name}")
        model_path = MODEL_MAP[model_name]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _backends[model_name] = LlamaCppBackend(model_path)
    return _backends[model_name]


class GenerateRequest(BaseModel):
    """Request model for generation."""
    prompt: str
    model_name: str
    max_tokens: int = 200
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    """Response model for generation."""
    text: str
    model_name: str


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "model-inference-api"}


@app.get("/models")
def list_models():
    """List available models."""
    available = [name for name, path in MODEL_MAP.items() if path.exists()]
    return {"models": available}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate text using a specified model."""
    try:
        backend = get_backend(request.model_name)
        summary = backend.generate_summary(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return GenerateResponse(text=summary, model_name=request.model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


"""Modal app for serving six LLM models via vLLM endpoints."""

import modal

app = modal.App("llama-inference")

# Base image with vLLM
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("vllm==0.11.0", "huggingface-hub==0.36.0", "requests==2.31.0")
)

# Common vLLM arguments
# These settings optimize memory usage for 7B-9B models
VLLM_ARGS_DEFAULT = (
    "--gpu-memory-utilization 0.97 "
    "--max-model-len 2048 "
    "--tensor-parallel-size 1 "
    "--disable-custom-all-reduce "
    "--enforce-eager "
)

# Mistral-specific args (includes tokenizer mode for correct encoding/decoding)
VLLM_ARGS_MISTRAL = (
    "--gpu-memory-utilization 0.97 "
    "--max-model-len 2048 "
    "--tensor-parallel-size 1 "
    "--disable-custom-all-reduce "
    "--tokenizer-mode mistral "
    "--enforce-eager "
)

# Model configuration
MODEL_CONFIGS = {
    "Gemma-2-9B": {
        "hf_name": "google/gemma-2-9b-it",
        "port": 8000,
        "vllm_args": VLLM_ARGS_DEFAULT,
    },
    "Llama-3.1-8B": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "port": 8001,
        "vllm_args": VLLM_ARGS_DEFAULT,
    },
    "Mistral-7B": {
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "port": 8002,
        "vllm_args": VLLM_ARGS_MISTRAL,
    },
    "Qwen3-8B": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "port": 8003,
        "vllm_args": VLLM_ARGS_DEFAULT,
    },
    "Teuken-7B": {
        "hf_name": "openGPT-X/Teuken-7B-instruct-commercial-v0.4",
        "port": 8004,
        "vllm_args": VLLM_ARGS_DEFAULT + "--trust-remote-code ",
    },
    "Mistral-NeMo-12B": {
        "hf_name": "casperhansen/mistral-nemo-instruct-2407-awq",
        "port": 8005,
        "vllm_args": VLLM_ARGS_DEFAULT + "--quantization awq ",
    },
}

# Common Modal function configuration
FUNCTION_CONFIG = {
    "image": vllm_image,
    "gpu": "L4",
    "secrets": [modal.Secret.from_name("huggingface")],
    "timeout": 600,  # 10 minutes for cold start
    "scaledown_window": 600,
    "min_containers": 1,  # Keep at least 1 container warm at all times
}

WEB_SERVER_STARTUP_TIMEOUT = 600


def start_vllm_server(model_key: str):
    """Start vLLM server for a given model configuration.
    
    Args:
        model_key: Key from MODEL_CONFIGS dictionary
        
    Raises:
        SystemExit: If the server process fails to start or exits with an error
    """
    import subprocess
    import sys
    import time
    
    config = MODEL_CONFIGS[model_key]
    model_name = config["hf_name"]
    port = config["port"]
    vllm_args = config["vllm_args"]
    
    cmd = f"vllm serve {model_name} --port {port} --host 0.0.0.0 {vllm_args}"
    
    # Start the server process
    # Modal's @web_server decorator will keep the container alive and handle routing
    process = subprocess.Popen(cmd, shell=True)
    
    # Give process a moment to start, then check for immediate failures
    time.sleep(2)
    
    return_code = process.poll()
    if return_code is not None and return_code != 0:
        sys.exit(return_code)  # Stop container on error
    
    # Don't block here - let Modal's web_server decorator handle the container lifecycle
    # The process will keep running in the background, and Modal will route requests to it


# Gemma endpoint
@app.function(**FUNCTION_CONFIG)
@modal.web_server(port=MODEL_CONFIGS["Gemma-2-9B"]["port"], startup_timeout=WEB_SERVER_STARTUP_TIMEOUT)
def serve_gemma():
    """Serve Gemma-2-9B model."""
    start_vllm_server("Gemma-2-9B")


# Llama endpoint
@app.function(**FUNCTION_CONFIG)
@modal.web_server(port=MODEL_CONFIGS["Llama-3.1-8B"]["port"], startup_timeout=WEB_SERVER_STARTUP_TIMEOUT)
def serve_llama():
    """Serve Llama-3.1-8B model."""
    start_vllm_server("Llama-3.1-8B")


# Mistral endpoint
@app.function(**FUNCTION_CONFIG)
@modal.web_server(port=MODEL_CONFIGS["Mistral-7B"]["port"], startup_timeout=WEB_SERVER_STARTUP_TIMEOUT)
def serve_mistral():
    """Serve Mistral-7B model."""
    start_vllm_server("Mistral-7B")


# Qwen endpoint
@app.function(**FUNCTION_CONFIG)
@modal.web_server(port=MODEL_CONFIGS["Qwen3-8B"]["port"], startup_timeout=WEB_SERVER_STARTUP_TIMEOUT)
def serve_qwen():
    """Serve Qwen2.5-7B model (internally named Qwen3-8B)."""
    start_vllm_server("Qwen3-8B")


# Teuken endpoint
@app.function(**FUNCTION_CONFIG)
@modal.web_server(port=MODEL_CONFIGS["Teuken-7B"]["port"], startup_timeout=WEB_SERVER_STARTUP_TIMEOUT)
def serve_teuken():
    """Serve Teuken-7B model."""
    start_vllm_server("Teuken-7B")


# Mistral-NeMo endpoint
@app.function(**FUNCTION_CONFIG)
@modal.web_server(port=MODEL_CONFIGS["Mistral-NeMo-12B"]["port"], startup_timeout=WEB_SERVER_STARTUP_TIMEOUT)
def serve_mistral_nemo():
    """Serve Mistral-NeMo-12B model."""
    start_vllm_server("Mistral-NeMo-12B")

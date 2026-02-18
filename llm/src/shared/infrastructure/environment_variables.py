import os
from dotenv import load_dotenv

# Load environment variables from .env.dev
load_dotenv(".env.dev")

# // ─────────────────────────────────────
# Ollama configuration
# // ─────────────────────────────────────
OLLAMA_SERVICE_HOST = os.getenv(
    "OLLAMA-SERVICE-HOST",
    "http://localhost:11435"
)
OLLAMA_SERVICE_MODEL = os.getenv(
    "OLLAMA-SERVICE-MODEL",
    "qwen3-vl:4b"
)

# // ─────────────────────────────────────
# Ollama configuration
# // ─────────────────────────────────────

__all__ = ["OLLAMA_SERVICE_HOST", "OLLAMA_SERVICE_MODEL"]

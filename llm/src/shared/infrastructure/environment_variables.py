import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env.dev
load_dotenv(".env.dev")


# // ─────────────────────────────────────
# Environment Configuration Model
# // ─────────────────────────────────────
class EnvironmentConfig(BaseModel):
    """Validated environment configuration."""

    # // ─────────────────────────────────────
    # OLLAMA Configuration
    # // ─────────────────────────────────────
    ollama_service_host: str = Field(
        default="http://localhost:11435",
        alias="OLLAMA-SERVICE-HOST",
        description="The base URL of the Ollama service"
    )
    ollama_service_model_qwen3vl4b: str = Field(
        default="qwen3-vl:4b",
        alias="OLLAMA_SERVICE_MODEL_QWEN3VL4B",
        description="The Qwen3-VL:4B model name"
    )

    # // ─────────────────────────────────────
    # Error Handling & Retries
    # // ─────────────────────────────────────
    max_retries: int = Field(
        default=3,
        alias="MAX_RETRIES",
        description="Maximum number of retries for API calls"
    )

    class Config:
        populate_by_name = True


# // ─────────────────────────────────────
# Load and validate environment variables
# // ─────────────────────────────────────
environment_config = EnvironmentConfig(
    **{
        "OLLAMA-SERVICE-HOST": os.getenv(
            "OLLAMA-SERVICE-HOST",
            "http://localhost:11435"
        ),
        "OLLAMA_SERVICE_MODEL_QWEN3VL4B": os.getenv(
            "OLLAMA_SERVICE_MODEL_QWEN3VL4B",
            "qwen3-vl:4b"
        ),
        "MAX_RETRIES": os.getenv(
            "MAX_RETRIES",
            "3"
        ),
    }
)

__all__ = ["EnvironmentConfig", "environment_config"]

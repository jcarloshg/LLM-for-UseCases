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

    ollama_service_host: str = Field(
        default="http://localhost:11435",
        alias="OLLAMA-SERVICE-HOST",
        description="The base URL of the Ollama service"
    )
    ollama_service_model_qwen3vl4b: str = Field(
        default="qwen3-vl:4b",
        alias="OLLAMA-SERVICE-MODEL",
        description="The Qwen3-VL:4B model name"
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
        "OLLAMA-SERVICE-MODEL": os.getenv(
            "OLLAMA-SERVICE-MODEL",
            "qwen3-vl:4b"
        ),
    }
)

__all__ = ["EnvironmentConfig", "environment_config"]

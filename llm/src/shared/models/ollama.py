from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM

from src.shared.infrastructure import OLLAMA_SERVICE_HOST, OLLAMA_SERVICE_MODEL


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM service."""

    host: str = Field(
        default="http://localhost:11435",
        description="The base URL of the Ollama service"
    )
    model: str = Field(
        default="qwen3-vl:4b",
        description="The model name to use for Ollama"
    )

    class Config:
        frozen = True


class OllamaService:
    """Service wrapper for Ollama LLM."""

    def __init__(self, config: OllamaConfig):
        self.config = config
        self.llm = OllamaLLM(
            model=config.model,
            base_url=config.host
        )

    def get_llm(self) -> OllamaLLM:
        """Get the OllamaLLM instance."""
        return self.llm

    def get_qwen3vl4b() -> OllamaLLM:
        if not self.qwen3vl4b:
            self.qwen3vl4b = OllamaLLM(
                model=OLLAMA_SERVICE_MODEL,
                base_url=OLLAMA_SERVICE_HOST,
            )
        return self.qwen3vl4b

from langchain_ollama import OllamaLLM

from src.shared.infrastructure import environment_config


class OllamaService:
    """Service wrapper for Ollama LLM (Singleton)."""

    __instance: "OllamaService | None" = None
    __qwen3vl4b_instance: OllamaLLM | None = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self):
        if self.__initialized:
            return
        self.llm = OllamaLLM(
            model=environment_config.ollama_service_model_qwen3vl4b,
            base_url=environment_config.ollama_service_host
        )
        self.__initialized = True

    @classmethod
    def get_qwen3vl4b(cls) -> OllamaLLM:
        """Get or create a singleton instance of Qwen3-VL:4B model."""
        if cls.__qwen3vl4b_instance is None:
            cls.__qwen3vl4b_instance = OllamaLLM(
                model=environment_config.ollama_service_model_qwen3vl4b,
                base_url=environment_config.ollama_service_host,
            )
        return cls.__qwen3vl4b_instance

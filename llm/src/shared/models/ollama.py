from langchain_ollama import OllamaLLM

from src.shared.infrastructure import environment_config


class OllamaCallError(Exception):
    """Exception raised when Ollama call fails after retries."""

    def __init__(self, message: str, user_message: str | None = None):
        """
        Initialize OllamaCallError.

        Args:
            message: Technical error message for developers
            user_message: User-friendly error message (defaults to generic message)
        """
        super().__init__(message)
        self._dev_message = message
        self._user_message = user_message or (
            "Unable to process your request. Please try again later."
        )

    @property
    def user_message(self) -> str:
        """Get the user-friendly error message."""
        return self._user_message

    @property
    def dev_message(self) -> str:
        """Get the developer/technical error message."""
        return self._dev_message


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
            model=environment_config.OLLAMA_SERVICE_MODEL_QWEN3VL4B,
            base_url=environment_config.OLLAMA_SERVICE_HOST
        )
        self.__initialized = True

    def safe_call(self, prompt: str) -> str:
        """
        Execute an LLM call with retry logic.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response as a string

        Raises:
            OllamaCallError: If all retry attempts fail
        """
        max_retries = environment_config.MAX_RETRIES
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return response
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    continue
                else:
                    raise OllamaCallError(
                        message=f"{environment_config.MAX_RETRIES_DEV_MSG}{attempt}",
                        user_message=environment_config.MAX_RETRIES_USER_MSG,
                    ) from e

    @classmethod
    def get_qwen3vl4b(cls) -> OllamaLLM:
        """Get or create a singleton instance of Qwen3-VL:4B model."""
        if cls.__qwen3vl4b_instance is None:
            cls.__qwen3vl4b_instance = OllamaLLM(
                model=environment_config.OLLAMA_SERVICE_MODEL_QWEN3VL4B,
                base_url=environment_config.OLLAMA_SERVICE_HOST,
            )
        return cls.__qwen3vl4b_instance

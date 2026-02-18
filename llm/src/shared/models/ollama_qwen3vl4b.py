from langchain_ollama import OllamaLLM

from src.shared.infrastructure import environment_config
from .llm_model_base import LLMModelBase
from .llm_exeptions import OllamaCallError


class OllamaQwen3vl4b(LLMModelBase):
    """Qwen3-VL:4B model implementation using Ollama."""

    def __init__(self):
        self.llm = OllamaLLM(
            model=environment_config.ollama_service_model_qwen3vl4b,
            base_url=environment_config.ollama_service_host
        )

    def get_llm(self) -> OllamaLLM:
        """
        Get the underlying OllamaLLM instance.

        Returns:
            The OllamaLLM instance
        """
        return self.llm

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
        max_retries = environment_config.max_retries

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    dev_message = (
                        f"{environment_config.max_retries_dev_msg}"
                        f"{max_retries}. Last error: {str(e)}"
                    )
                    user_message = environment_config.max_retries_user_msg
                    raise OllamaCallError(
                        message=dev_message,
                        user_message=user_message,
                    ) from e

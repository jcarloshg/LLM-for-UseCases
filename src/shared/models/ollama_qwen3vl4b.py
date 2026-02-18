import re
from langchain_ollama import OllamaLLM

from src.shared.infrastructure import ENVIRONMENT_CONFIG
from .llm_model_base import LLMModelBase, TokenUsage
from .llm_exeptions import OllamaCallError


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text using word-based approximation.
    Approximately 1.3 tokens per word on average.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated number of tokens
    """
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    # Split by whitespace and count words
    words = text.split()
    # Estimate: ~1.3 tokens per word (common approximation)
    estimated_tokens = max(1, int(len(words) * 1.3))
    return estimated_tokens


class OllamaQwen3vl4b(LLMModelBase):
    """Qwen3-VL:4B model implementation using Ollama."""

    def __init__(self):
        self.llm = OllamaLLM(
            model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL4B,
            base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST
        )

    def get_llm(self) -> OllamaLLM:
        """
        Get the underlying OllamaLLM instance.

        Returns:
            The OllamaLLM instance
        """
        return self.llm

    def safe_call(self, prompt: str, max_retries: int = 3) -> str:
        """
        Execute an LLM call with retry logic.

        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retry attempts (default from config if not provided)

        Returns:
            The LLM response as a string

        Raises:
            OllamaCallError: If all retry attempts fail
        """
        # Use provided max_retries or fall back to config value
        if max_retries == 3:
            max_retries = ENVIRONMENT_CONFIG.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    dev_message = (
                        f"{ENVIRONMENT_CONFIG.MAX_RETRIES_DEV_MSG}"
                        f"{max_retries}. Last error: {str(e)}"
                    )
                    user_message = ENVIRONMENT_CONFIG.MAX_RETRIES_USER_MSG
                    raise OllamaCallError(
                        message=dev_message,
                        user_message=user_message,
                    ) from e

    def safe_call_with_tokens(
        self, prompt: str, max_retries: int = 3
    ) -> tuple[str, TokenUsage]:
        """
        Execute an LLM call with retry logic and token usage tracking.

        Estimates tokens based on word count approximation (~1.3 tokens per word).

        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retry attempts

        Returns:
            A tuple of (response, token_usage)

        Raises:
            OllamaCallError: If all retry attempts fail
        """
        # Get the response using safe_call
        response = self.safe_call(prompt, max_retries)

        # Estimate tokens for prompt and response
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(response)
        total_tokens = prompt_tokens + completion_tokens

        # Create TokenUsage object
        token_usage = TokenUsage(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost=0.0,  # Ollama runs locally, no cost
        )

        return response, token_usage

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenUsage:
    """Token usage statistics for LLM calls."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0

    def __str__(self) -> str:
        """String representation of token usage."""
        return (
            f"Tokens used: {self.total_tokens}\n"
            f"Prompt tokens: {self.prompt_tokens}\n"
            f"Completion tokens: {self.completion_tokens}\n"
            f"Total cost: ${self.total_cost:.4f}"
        )


class LLMModelBase(ABC):
    """Abstract base class for Language Model implementations."""

    @abstractmethod
    def safe_call(self, prompt: str, max_retries: int = 3) -> str:
        """
        Execute an LLM call with retry logic.

        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retry attempts

        Raises:
            Exception: Implementation-specific exceptions for call failures
        """
        pass

    @abstractmethod
    def get_llm(self):
        """
        Get the underlying LLM instance.

        Returns:
            The LLM instance used by this service
        """
        pass

    @abstractmethod
    def safe_call_with_tokens(
        self, prompt: str, max_retries: int = 3
    ) -> tuple[str, TokenUsage]:
        """
        Execute an LLM call with retry logic and token tracking.

        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retry attempts

        Returns:
            A tuple of (response, token_usage)

        Raises:
            Exception: Implementation-specific exceptions for call failures

        Note:
            Each implementation must provide its own token tracking mechanism
            appropriate for the LLM provider (OpenAI, Ollama, etc.)
        """
        pass

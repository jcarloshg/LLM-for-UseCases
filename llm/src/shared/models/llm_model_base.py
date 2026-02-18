from abc import ABC, abstractmethod


class LLMModelBase(ABC):
    """Abstract base class for Language Model implementations."""

    @abstractmethod
    def safe_call(self, prompt: str, max_retries: int = 3) -> str:
        """
        Raises:
            Exception: Implementation-specific exceptions for call failures
        """
        pass

    @abstractmethod
    def get_llm(self):
        """
        Returns:
            The LLM instance used by this service
        """
        pass

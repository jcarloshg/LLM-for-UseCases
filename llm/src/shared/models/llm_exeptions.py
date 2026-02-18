
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

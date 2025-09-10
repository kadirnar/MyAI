"""Exceptions for the Multi-LLM Client."""


class MultiLLMError(Exception):
    """Base exception for Multi-LLM Client."""
    pass


class ProviderError(MultiLLMError):
    """Error from a specific provider."""
    def __init__(self, provider: str, message: str, status_code: int = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider}: {message}")


class AuthenticationError(ProviderError):
    """Authentication error with a provider."""
    pass


class RateLimitError(ProviderError):
    """Rate limit error from a provider."""
    pass


class ModelNotFoundError(ProviderError):
    """Model not found error."""
    pass


class InvalidRequestError(ProviderError):
    """Invalid request error."""
    pass


class ConfigurationError(MultiLLMError):
    """Configuration error."""
    pass
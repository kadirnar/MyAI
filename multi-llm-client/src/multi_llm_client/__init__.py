"""Multi-LLM Client: A unified interface for multiple LLM/VLM providers."""

from .client import MultiLLMClient
from .config import MultiLLMConfig
from .models import (
    Provider, ChatMessage, ChatResponse, ModelInfo, ProviderConfig,
    StreamingChatResponse, MessageRole, TextContent, ImageContent
)
from .exceptions import (
    MultiLLMError, ProviderError, AuthenticationError, RateLimitError,
    ModelNotFoundError, InvalidRequestError, ConfigurationError
)

__version__ = "0.1.0"
__all__ = [
    "MultiLLMClient",
    "MultiLLMConfig",
    "Provider",
    "ChatMessage",
    "ChatResponse", 
    "ModelInfo",
    "ProviderConfig",
    "StreamingChatResponse",
    "MessageRole",
    "TextContent",
    "ImageContent",
    "MultiLLMError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "InvalidRequestError",
    "ConfigurationError"
]
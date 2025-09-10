"""Base provider interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional

from .models import ChatMessage, ChatResponse, ModelInfo, ProviderConfig, StreamingChatResponse


class BaseProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatResponse:
        """Generate a chat completion."""
        pass
    
    @abstractmethod
    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[StreamingChatResponse]:
        """Generate a streaming chat completion."""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List available models for this provider."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass
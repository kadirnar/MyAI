"""Core data models for the Multi-LLM Client."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class Provider(str, Enum):
    """Supported LLM providers."""
    HYPERBOLIC = "hyperbolic"
    TOGETHER = "together"
    GROQ = "groq"
    CEREBRAS = "cerebras"


class MessageRole(str, Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ImageContent(BaseModel):
    """Image content for vision models."""
    type: str = "image_url"
    image_url: Dict[str, str]


class TextContent(BaseModel):
    """Text content for messages."""
    type: str = "text"
    text: str


class ChatMessage(BaseModel):
    """A chat message with role and content."""
    role: MessageRole
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ModelInfo(BaseModel):
    """Information about a model."""
    id: str
    provider: Provider
    name: str
    description: Optional[str] = None
    context_length: Optional[int] = None
    supports_vision: bool = False
    supports_streaming: bool = True


class ChatResponse(BaseModel):
    """Response from a chat completion."""
    id: str
    model: str
    provider: Provider
    content: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class StreamingChatResponse(BaseModel):
    """Streaming response chunk from a chat completion."""
    id: str
    model: str
    provider: Provider
    delta: str
    finish_reason: Optional[str] = None


class ProviderConfig(BaseModel):
    """Configuration for a specific provider."""
    api_key: str
    base_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
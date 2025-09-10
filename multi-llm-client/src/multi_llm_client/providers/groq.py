"""Groq provider implementation."""

import json
from typing import AsyncIterator, List, Optional, Dict, Any

from groq import AsyncGroq

from ..base_provider import BaseProvider
from ..models import (
    ChatMessage, ChatResponse, ModelInfo, Provider, ProviderConfig,
    StreamingChatResponse, ImageContent, TextContent
)
from ..exceptions import ProviderError, AuthenticationError, RateLimitError


class GroqProvider(BaseProvider):
    """Groq provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncGroq(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Format messages for Groq API."""
        formatted = []
        for msg in messages:
            if isinstance(msg.content, str):
                formatted.append({"role": msg.role.value, "content": msg.content})
            else:
                # Handle multimodal content
                content = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        content.append({"type": "text", "text": item.text})
                    elif isinstance(item, ImageContent):
                        content.append({
                            "type": "image_url", 
                            "image_url": item.image_url
                        })
                formatted.append({"role": msg.role.value, "content": content})
        return formatted
    
    def _handle_groq_error(self, error: Exception) -> None:
        """Handle Groq SDK errors."""
        from groq import AuthenticationError as GroqAuthError, RateLimitError as GroqRateError
        
        if isinstance(error, GroqAuthError):
            raise AuthenticationError("groq", str(error))
        elif isinstance(error, GroqRateError):
            raise RateLimitError("groq", str(error))
        else:
            raise ProviderError("groq", str(error))
    
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
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=self._format_messages(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                **kwargs
            )
            
            return ChatResponse(
                id=response.id,
                model=response.model,
                provider=Provider.GROQ,
                content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else None,
                finish_reason=response.choices[0].finish_reason
            )
        
        except Exception as e:
            self._handle_groq_error(e)
    
    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[StreamingChatResponse]:
        """Generate a streaming chat completion."""
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=self._format_messages(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield StreamingChatResponse(
                        id=chunk.id,
                        model=chunk.model,
                        provider=Provider.GROQ,
                        delta=chunk.choices[0].delta.content,
                        finish_reason=chunk.choices[0].finish_reason
                    )
        
        except Exception as e:
            self._handle_groq_error(e)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models for Groq."""
        try:
            models_response = await self.client.models.list()
            models = []
            
            for model_data in models_response.data:
                # Check for vision capabilities based on model name patterns
                model_id = model_data.id
                supports_vision = any(pattern in model_id.lower() for pattern in [
                    "vision", "vlm", "llava", "scout", "llama-4"
                ])
                
                models.append(ModelInfo(
                    id=model_id,
                    provider=Provider.GROQ,
                    name=model_id,
                    description=f"Groq model: {model_id}",
                    supports_vision=supports_vision,
                    supports_streaming=True
                ))
            
            return models
        
        except Exception as e:
            self._handle_groq_error(e)
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "groq"
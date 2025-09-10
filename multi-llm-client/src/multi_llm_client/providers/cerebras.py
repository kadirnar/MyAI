"""Cerebras provider implementation."""

import json
from typing import AsyncIterator, List, Optional, Dict, Any

from cerebras.cloud.sdk import AsyncCerebras

from ..base_provider import BaseProvider
from ..models import (
    ChatMessage, ChatResponse, ModelInfo, Provider, ProviderConfig,
    StreamingChatResponse, ImageContent, TextContent
)
from ..exceptions import ProviderError, AuthenticationError, RateLimitError


class CerebrasProvider(BaseProvider):
    """Cerebras provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncCerebras(
            api_key=config.api_key,
            timeout=config.timeout
        )
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Format messages for Cerebras API."""
        formatted = []
        for msg in messages:
            if isinstance(msg.content, str):
                formatted.append({"role": msg.role.value, "content": msg.content})
            else:
                # Handle multimodal content (though Cerebras primarily focuses on text models)
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
    
    def _handle_cerebras_error(self, error: Exception) -> None:
        """Handle Cerebras SDK errors."""
        error_str = str(error).lower()
        
        if "authentication" in error_str or "unauthorized" in error_str:
            raise AuthenticationError("cerebras", str(error))
        elif "rate limit" in error_str or "429" in error_str:
            raise RateLimitError("cerebras", str(error))
        else:
            raise ProviderError("cerebras", str(error))
    
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
                provider=Provider.CEREBRAS,
                content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else None,
                finish_reason=response.choices[0].finish_reason
            )
        
        except Exception as e:
            self._handle_cerebras_error(e)
    
    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[StreamingChatResponse]:
        """Generate a streaming chat completion."""
        payload = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": True
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        
        payload.update(kwargs)
        
        try:
            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                if not response.is_success:
                    self._handle_error_response(response)
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            
                            if "content" in delta:
                                yield StreamingChatResponse(
                                    id=data["id"],
                                    model=data["model"],
                                    provider=Provider.CEREBRAS,
                                    delta=delta["content"],
                                    finish_reason=choice.get("finish_reason")
                                )
                        except json.JSONDecodeError:
                            continue
        
        except httpx.HTTPError as e:
            raise ProviderError("cerebras", f"HTTP error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models for Cerebras."""
        try:
            models_response = await self.client.models.list()
            models = []
            
            # Known Cerebras models with accurate information
            known_models = {
                # Production Models
                "llama-4-scout-17b-16e-instruct": {
                    "name": "Llama 4 Scout",
                    "description": "109B parameter model with ~2600 tokens/s speed",
                    "supports_vision": True,  # Scout supports vision
                    "parameters": "109 billion"
                },
                "llama3.1-8b": {
                    "name": "Llama 3.1 8B",
                    "description": "8B parameter model with ~2200 tokens/s speed",
                    "supports_vision": False,
                    "parameters": "8 billion"
                },
                "llama-3.3-70b": {
                    "name": "Llama 3.3 70B", 
                    "description": "70B parameter model with ~2100 tokens/s speed",
                    "supports_vision": False,
                    "parameters": "70 billion"
                },
                "gpt-oss-120b": {
                    "name": "OpenAI GPT OSS",
                    "description": "120B parameter model with ~2800 tokens/s speed",
                    "supports_vision": False,
                    "parameters": "120 billion"
                },
                "qwen-3-32b": {
                    "name": "Qwen 3 32B",
                    "description": "32B parameter model with ~2600 tokens/s speed", 
                    "supports_vision": False,
                    "parameters": "32 billion"
                },
                # Preview Models
                "llama-4-maverick-17b-128e-instruct": {
                    "name": "Llama 4 Maverick",
                    "description": "400B parameter model with ~2400 tokens/s speed (Preview)",
                    "supports_vision": True,  # Maverick supports vision
                    "parameters": "400 billion"
                },
                "qwen-3-235b-a22b-instruct-2507": {
                    "name": "Qwen 3 235B Instruct",
                    "description": "235B parameter model with ~1400 tokens/s speed (Preview)",
                    "supports_vision": False,
                    "parameters": "235 billion"
                },
                "qwen-3-235b-a22b-thinking-2507": {
                    "name": "Qwen 3 235B Thinking", 
                    "description": "235B parameter model with ~1700 tokens/s speed (Preview)",
                    "supports_vision": False,
                    "parameters": "235 billion"
                },
                "qwen-3-coder-480b": {
                    "name": "Qwen 3 480B Coder",
                    "description": "480B parameter model with ~2000 tokens/s speed (Preview)",
                    "supports_vision": False,
                    "parameters": "480 billion"
                }
            }
            
            for model_data in models_response.data:
                model_id = model_data.id
                model_info = known_models.get(model_id, {})
                
                models.append(ModelInfo(
                    id=model_id,
                    provider=Provider.CEREBRAS,
                    name=model_info.get("name", model_id),
                    description=model_info.get("description", f"Cerebras model: {model_id}"),
                    supports_vision=model_info.get("supports_vision", False),
                    supports_streaming=True
                ))
            
            return models
        
        except Exception as e:
            self._handle_cerebras_error(e)
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "cerebras"
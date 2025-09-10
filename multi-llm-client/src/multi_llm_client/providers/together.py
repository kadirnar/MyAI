"""Together AI provider implementation."""

import json
from typing import AsyncIterator, List, Optional, Dict, Any

from together import AsyncTogether

from ..base_provider import BaseProvider
from ..models import (
    ChatMessage, ChatResponse, ModelInfo, Provider, ProviderConfig,
    StreamingChatResponse, ImageContent, TextContent
)
from ..exceptions import ProviderError, AuthenticationError, RateLimitError


class TogetherProvider(BaseProvider):
    """Together AI provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncTogether(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Format messages for Together API."""
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
    
    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle API error responses."""
        if response.status_code == 401:
            raise AuthenticationError("together", "Invalid API key", response.status_code)
        elif response.status_code == 429:
            raise RateLimitError("together", "Rate limit exceeded", response.status_code)
        elif response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("error", {}).get("message", "Unknown error")
            except:
                message = f"HTTP {response.status_code}: {response.text}"
            raise ProviderError("together", message, response.status_code)
    
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
        payload = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": stream
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        
        payload.update(kwargs)
        
        try:
            response = await self.client.post("/chat/completions", json=payload)
            
            if not response.is_success:
                self._handle_error_response(response)
            
            data = response.json()
            choice = data["choices"][0]
            
            return ChatResponse(
                id=data["id"],
                model=data["model"],
                provider=Provider.TOGETHER,
                content=choice["message"]["content"],
                usage=data.get("usage"),
                finish_reason=choice.get("finish_reason")
            )
        
        except httpx.HTTPError as e:
            raise ProviderError("together", f"HTTP error: {str(e)}")
    
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
                                    provider=Provider.TOGETHER,
                                    delta=delta["content"],
                                    finish_reason=choice.get("finish_reason")
                                )
                        except json.JSONDecodeError:
                            continue
        
        except httpx.HTTPError as e:
            raise ProviderError("together", f"HTTP error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models for Together AI."""
        try:
            response = await self.client.get("/models")
            
            if not response.is_success:
                self._handle_error_response(response)
            
            data = response.json()
            models = []
            
            for model_data in data.get("data", []):
                # Check for vision capabilities based on model name patterns
                model_id = model_data["id"]
                supports_vision = any(pattern in model_id.lower() for pattern in [
                    "vision", "vlm", "llava", "pixtral", "qwen2-vl", "llama-4", "maverick"
                ])
                
                models.append(ModelInfo(
                    id=model_id,
                    provider=Provider.TOGETHER,
                    name=model_data.get("display_name", model_id),
                    description=model_data.get("description"),
                    context_length=model_data.get("context_length"),
                    supports_vision=supports_vision,
                    supports_streaming=True
                ))
            
            return models
        
        except httpx.HTTPError as e:
            raise ProviderError("together", f"HTTP error: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "together"
"""Hyperbolic provider implementation."""

import json
from typing import AsyncIterator, List, Optional, Dict, Any

from openai import AsyncOpenAI

from ..base_provider import BaseProvider
from ..models import (
    ChatMessage, ChatResponse, ModelInfo, Provider, ProviderConfig,
    StreamingChatResponse, ImageContent, TextContent
)
from ..exceptions import ProviderError, AuthenticationError, RateLimitError


class HyperbolicProvider(BaseProvider):
    """Hyperbolic AI provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.hyperbolic.xyz/v1",
            timeout=config.timeout
        )
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Format messages for Hyperbolic API."""
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
            raise AuthenticationError("hyperbolic", "Invalid API key", response.status_code)
        elif response.status_code == 429:
            raise RateLimitError("hyperbolic", "Rate limit exceeded", response.status_code)
        elif response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("error", {}).get("message", "Unknown error")
            except:
                message = f"HTTP {response.status_code}: {response.text}"
            raise ProviderError("hyperbolic", message, response.status_code)
    
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
                provider=Provider.HYPERBOLIC,
                content=choice["message"]["content"],
                usage=data.get("usage"),
                finish_reason=choice.get("finish_reason")
            )
        
        except httpx.HTTPError as e:
            raise ProviderError("hyperbolic", f"HTTP error: {str(e)}")
    
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
                                    provider=Provider.HYPERBOLIC,
                                    delta=delta["content"],
                                    finish_reason=choice.get("finish_reason")
                                )
                        except json.JSONDecodeError:
                            continue
        
        except httpx.HTTPError as e:
            raise ProviderError("hyperbolic", f"HTTP error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models for Hyperbolic."""
        try:
            response = await self.client.get("/models")
            
            if not response.is_success:
                self._handle_error_response(response)
            
            data = response.json()
            models = []
            
            for model_data in data["data"]:
                models.append(ModelInfo(
                    id=model_data["id"],
                    provider=Provider.HYPERBOLIC,
                    name=model_data["id"],
                    description=model_data.get("description"),
                    supports_vision="vision" in model_data["id"].lower() or "vlm" in model_data["id"].lower(),
                    supports_streaming=True
                ))
            
            return models
        
        except httpx.HTTPError as e:
            raise ProviderError("hyperbolic", f"HTTP error: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "hyperbolic"
"""Main client class for Multi-LLM Client."""

from typing import AsyncIterator, Dict, List, Optional, Union

from .base_provider import BaseProvider
from .config import MultiLLMConfig
from .models import (
    ChatMessage, ChatResponse, ModelInfo, Provider, ProviderConfig,
    StreamingChatResponse
)
from .providers import HyperbolicProvider, TogetherProvider, GroqProvider, CerebrasProvider
from .exceptions import ConfigurationError, ProviderError


class MultiLLMClient:
    """Unified client for multiple LLM providers."""
    
    def __init__(self, config: Optional[MultiLLMConfig] = None):
        """Initialize the client with configuration."""
        self.config = config or MultiLLMConfig.from_env()
        self._providers: Dict[Provider, BaseProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize provider instances."""
        provider_classes = {
            Provider.HYPERBOLIC: HyperbolicProvider,
            Provider.TOGETHER: TogetherProvider,
            Provider.GROQ: GroqProvider,
            Provider.CEREBRAS: CerebrasProvider
        }
        
        for provider, provider_config in self.config.providers.items():
            provider_class = provider_classes[provider]
            self._providers[provider] = provider_class(provider_config)
    
    def add_provider(self, provider: Provider, api_key: str, **kwargs) -> None:
        """Add a new provider or update existing one."""
        provider_config = ProviderConfig(api_key=api_key, **kwargs)
        self.config.add_provider(provider, provider_config)
        
        # Initialize the provider
        provider_classes = {
            Provider.HYPERBOLIC: HyperbolicProvider,
            Provider.TOGETHER: TogetherProvider,
            Provider.GROQ: GroqProvider,
            Provider.CEREBRAS: CerebrasProvider
        }
        
        if provider in provider_classes:
            provider_class = provider_classes[provider]
            self._providers[provider] = provider_class(provider_config)
    
    def _get_provider(self, provider: Optional[Provider] = None) -> BaseProvider:
        """Get provider instance."""
        target_provider = provider or self.config.default_provider
        
        if not target_provider:
            raise ConfigurationError("No provider specified and no default provider configured")
        
        if target_provider not in self._providers:
            raise ConfigurationError(f"Provider {target_provider} is not configured")
        
        return self._providers[target_provider]
    
    async def chat_completion(
        self,
        messages: Union[List[ChatMessage], str],
        model: str,
        provider: Optional[Provider] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatResponse:
        """Generate a chat completion."""
        # Convert string message to ChatMessage list
        if isinstance(messages, str):
            from .models import MessageRole
            messages = [ChatMessage(role=MessageRole.USER, content=messages)]
        
        provider_instance = self._get_provider(provider)
        return await provider_instance.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs
        )
    
    async def stream_chat_completion(
        self,
        messages: Union[List[ChatMessage], str],
        model: str,
        provider: Optional[Provider] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[StreamingChatResponse]:
        """Generate a streaming chat completion."""
        # Convert string message to ChatMessage list
        if isinstance(messages, str):
            from .models import MessageRole
            messages = [ChatMessage(role=MessageRole.USER, content=messages)]
        
        provider_instance = self._get_provider(provider)
        async for chunk in provider_instance.stream_chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        ):
            yield chunk
    
    async def list_models(
        self, 
        provider: Optional[Provider] = None
    ) -> List[ModelInfo]:
        """List available models for a provider."""
        if provider:
            provider_instance = self._get_provider(provider)
            return await provider_instance.list_models()
        else:
            # List models from all configured providers
            all_models = []
            for provider in self._providers:
                try:
                    models = await self._providers[provider].list_models()
                    all_models.extend(models)
                except Exception:
                    # Skip providers that fail to list models
                    continue
            return all_models
    
    def list_providers(self) -> List[Provider]:
        """List configured providers."""
        return list(self._providers.keys())
    
    def get_provider_info(self, provider: Provider) -> Dict[str, str]:
        """Get information about a provider."""
        if provider not in self._providers:
            raise ConfigurationError(f"Provider {provider} is not configured")
        
        provider_instance = self._providers[provider]
        return {
            "name": provider_instance.get_provider_name(),
            "provider": provider.value,
            "configured": True
        }
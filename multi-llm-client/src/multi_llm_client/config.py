"""Configuration management for Multi-LLM Client."""

import os
from typing import Dict, Optional
from pydantic import BaseModel, Field

from .models import Provider, ProviderConfig
from .exceptions import ConfigurationError


class MultiLLMConfig(BaseModel):
    """Configuration for all providers."""
    providers: Dict[Provider, ProviderConfig] = Field(default_factory=dict)
    default_provider: Optional[Provider] = None
    
    @classmethod
    def from_env(cls) -> "MultiLLMConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Load API keys from environment
        env_mappings = {
            Provider.HYPERBOLIC: "HYPERBOLIC_API_KEY",
            Provider.TOGETHER: "TOGETHER_API_KEY", 
            Provider.GROQ: "GROQ_API_KEY",
            Provider.CEREBRAS: "CEREBRAS_API_KEY"
        }
        
        for provider, env_var in env_mappings.items():
            api_key = os.getenv(env_var)
            if api_key:
                config.providers[provider] = ProviderConfig(api_key=api_key)
        
        # Set default provider
        if config.providers:
            config.default_provider = next(iter(config.providers.keys()))
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MultiLLMConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        for provider_name, provider_config in config_dict.get("providers", {}).items():
            try:
                provider = Provider(provider_name)
                config.providers[provider] = ProviderConfig(**provider_config)
            except ValueError:
                raise ConfigurationError(f"Unknown provider: {provider_name}")
        
        default_provider = config_dict.get("default_provider")
        if default_provider:
            try:
                config.default_provider = Provider(default_provider)
            except ValueError:
                raise ConfigurationError(f"Unknown default provider: {default_provider}")
        
        return config
    
    def add_provider(self, provider: Provider, config: ProviderConfig) -> None:
        """Add or update a provider configuration."""
        self.providers[provider] = config
        if not self.default_provider:
            self.default_provider = provider
    
    def get_provider_config(self, provider: Provider) -> ProviderConfig:
        """Get configuration for a specific provider."""
        if provider not in self.providers:
            raise ConfigurationError(f"No configuration found for provider: {provider}")
        return self.providers[provider]
    
    def has_provider(self, provider: Provider) -> bool:
        """Check if provider is configured."""
        return provider in self.providers
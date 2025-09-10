"""Tests for configuration management."""

import pytest
from unittest.mock import patch
from multi_llm_client import (
    MultiLLMConfig, Provider, ProviderConfig, ConfigurationError
)


class TestMultiLLMConfig:
    """Test MultiLLMConfig functionality."""
    
    def test_empty_config(self):
        """Test creating empty configuration."""
        config = MultiLLMConfig()
        assert len(config.providers) == 0
        assert config.default_provider is None
    
    def test_add_provider(self):
        """Test adding a provider."""
        config = MultiLLMConfig()
        provider_config = ProviderConfig(api_key="test-key", timeout=30.0)
        
        config.add_provider(Provider.GROQ, provider_config)
        
        assert Provider.GROQ in config.providers
        assert config.providers[Provider.GROQ] == provider_config
        assert config.default_provider == Provider.GROQ
    
    def test_add_multiple_providers(self):
        """Test adding multiple providers."""
        config = MultiLLMConfig()
        
        config.add_provider(Provider.GROQ, ProviderConfig(api_key="groq-key"))
        config.add_provider(Provider.TOGETHER, ProviderConfig(api_key="together-key"))
        
        assert len(config.providers) == 2
        assert Provider.GROQ in config.providers
        assert Provider.TOGETHER in config.providers
        assert config.default_provider == Provider.GROQ  # First one added
    
    def test_get_provider_config(self):
        """Test getting provider configuration."""
        config = MultiLLMConfig()
        provider_config = ProviderConfig(api_key="test-key")
        config.add_provider(Provider.GROQ, provider_config)
        
        retrieved_config = config.get_provider_config(Provider.GROQ)
        assert retrieved_config == provider_config
    
    def test_get_provider_config_not_found(self):
        """Test getting configuration for non-existent provider."""
        config = MultiLLMConfig()
        
        with pytest.raises(ConfigurationError, match="No configuration found for provider"):
            config.get_provider_config(Provider.GROQ)
    
    def test_has_provider(self):
        """Test checking if provider exists."""
        config = MultiLLMConfig()
        
        assert not config.has_provider(Provider.GROQ)
        
        config.add_provider(Provider.GROQ, ProviderConfig(api_key="test-key"))
        assert config.has_provider(Provider.GROQ)
        assert not config.has_provider(Provider.TOGETHER)
    
    def test_from_env_empty(self):
        """Test loading configuration from empty environment."""
        with patch.dict('os.environ', {}, clear=True):
            config = MultiLLMConfig.from_env()
            assert len(config.providers) == 0
            assert config.default_provider is None
    
    def test_from_env_with_keys(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            'GROQ_API_KEY': 'test-groq-key',
            'TOGETHER_API_KEY': 'test-together-key',
            'CEREBRAS_API_KEY': 'test-cerebras-key'
        }
        
        with patch.dict('os.environ', env_vars, clear=True):
            config = MultiLLMConfig.from_env()
            
            assert len(config.providers) == 3
            assert Provider.GROQ in config.providers
            assert Provider.TOGETHER in config.providers
            assert Provider.CEREBRAS in config.providers
            assert config.providers[Provider.GROQ].api_key == 'test-groq-key'
            assert config.default_provider is not None
    
    def test_from_dict_valid(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "providers": {
                "groq": {
                    "api_key": "groq-key",
                    "timeout": 45.0
                },
                "together": {
                    "api_key": "together-key",
                    "base_url": "https://custom.url",
                    "max_retries": 5
                }
            },
            "default_provider": "groq"
        }
        
        config = MultiLLMConfig.from_dict(config_dict)
        
        assert len(config.providers) == 2
        assert Provider.GROQ in config.providers
        assert Provider.TOGETHER in config.providers
        assert config.default_provider == Provider.GROQ
        
        groq_config = config.providers[Provider.GROQ]
        assert groq_config.api_key == "groq-key"
        assert groq_config.timeout == 45.0
        
        together_config = config.providers[Provider.TOGETHER]
        assert together_config.api_key == "together-key"
        assert together_config.base_url == "https://custom.url"
        assert together_config.max_retries == 5
    
    def test_from_dict_unknown_provider(self):
        """Test error handling for unknown provider in dictionary."""
        config_dict = {
            "providers": {
                "unknown_provider": {
                    "api_key": "test-key"
                }
            }
        }
        
        with pytest.raises(ConfigurationError, match="Unknown provider: unknown_provider"):
            MultiLLMConfig.from_dict(config_dict)
    
    def test_from_dict_unknown_default_provider(self):
        """Test error handling for unknown default provider."""
        config_dict = {
            "providers": {
                "groq": {"api_key": "test-key"}
            },
            "default_provider": "unknown_provider"
        }
        
        with pytest.raises(ConfigurationError, match="Unknown default provider: unknown_provider"):
            MultiLLMConfig.from_dict(config_dict)
    
    def test_from_dict_minimal(self):
        """Test loading minimal configuration from dictionary."""
        config_dict = {
            "providers": {
                "groq": {"api_key": "test-key"}
            }
        }
        
        config = MultiLLMConfig.from_dict(config_dict)
        
        assert len(config.providers) == 1
        assert Provider.GROQ in config.providers
        assert config.default_provider is None  # Not specified


class TestProviderConfig:
    """Test ProviderConfig functionality."""
    
    def test_basic_config(self):
        """Test basic provider configuration."""
        config = ProviderConfig(api_key="test-key")
        
        assert config.api_key == "test-key"
        assert config.base_url is None
        assert config.timeout == 30.0
        assert config.max_retries == 3
    
    def test_full_config(self):
        """Test provider configuration with all parameters."""
        config = ProviderConfig(
            api_key="test-key",
            base_url="https://custom.api.url",
            timeout=60.0,
            max_retries=5
        )
        
        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.api.url"
        assert config.timeout == 60.0
        assert config.max_retries == 5
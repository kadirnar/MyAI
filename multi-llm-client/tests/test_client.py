"""Tests for the main MultiLLMClient."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from multi_llm_client import (
    MultiLLMClient, MultiLLMConfig, Provider, ChatMessage, MessageRole,
    ChatResponse, ConfigurationError
)


class TestMultiLLMClient:
    """Test MultiLLMClient functionality."""
    
    def test_initialization_with_config(self, mock_config):
        """Test client initialization with configuration."""
        client = MultiLLMClient(mock_config)
        assert client.config == mock_config
        assert len(client._providers) == 2
        assert Provider.GROQ in client._providers
        assert Provider.TOGETHER in client._providers
    
    def test_initialization_without_config(self):
        """Test client initialization without configuration."""
        with patch.dict('os.environ', {}, clear=True):
            client = MultiLLMClient()
            assert isinstance(client.config, MultiLLMConfig)
            assert len(client.config.providers) == 0
    
    def test_add_provider(self, client):
        """Test adding a new provider."""
        client.add_provider(Provider.CEREBRAS, "test-cerebras-key", timeout=45.0)
        
        assert Provider.CEREBRAS in client.config.providers
        assert Provider.CEREBRAS in client._providers
        assert client.config.providers[Provider.CEREBRAS].api_key == "test-cerebras-key"
        assert client.config.providers[Provider.CEREBRAS].timeout == 45.0
    
    def test_get_provider_with_explicit_provider(self, client):
        """Test getting provider with explicit provider specified."""
        provider = client._get_provider(Provider.GROQ)
        assert provider is not None
        assert provider.get_provider_name() == "groq"
    
    def test_get_provider_with_default(self, client):
        """Test getting provider with default provider."""
        provider = client._get_provider()
        assert provider is not None
        assert provider.get_provider_name() == "groq"  # Default provider
    
    def test_get_provider_not_configured(self, client):
        """Test error when provider is not configured."""
        with pytest.raises(ConfigurationError, match="Provider hyperbolic is not configured"):
            client._get_provider(Provider.HYPERBOLIC)
    
    def test_get_provider_no_default(self):
        """Test error when no default provider and none specified."""
        config = MultiLLMConfig()
        client = MultiLLMClient(config)
        
        with pytest.raises(ConfigurationError, match="No provider specified and no default"):
            client._get_provider()
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_string(self, client, mock_chat_response):
        """Test chat completion with string input."""
        with patch.object(client._providers[Provider.GROQ], 'chat_completion') as mock_method:
            mock_method.return_value = ChatResponse(
                id="test-123",
                model="test-model",
                provider=Provider.GROQ,
                content="Test response",
                finish_reason="stop"
            )
            
            response = await client.chat_completion(
                messages="Hello",
                model="llama-3.1-8b-instant"
            )
            
            assert response.content == "Test response"
            assert response.provider == Provider.GROQ
            
            # Check that the method was called with converted ChatMessage
            args, kwargs = mock_method.call_args
            messages = kwargs['messages']
            assert len(messages) == 1
            assert messages[0].role == MessageRole.USER
            assert messages[0].content == "Hello"
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_messages(self, client, sample_messages):
        """Test chat completion with ChatMessage list."""
        with patch.object(client._providers[Provider.GROQ], 'chat_completion') as mock_method:
            mock_method.return_value = ChatResponse(
                id="test-123",
                model="test-model", 
                provider=Provider.GROQ,
                content="Test response",
                finish_reason="stop"
            )
            
            response = await client.chat_completion(
                messages=sample_messages,
                model="llama-3.1-8b-instant",
                provider=Provider.GROQ,
                max_tokens=100,
                temperature=0.7
            )
            
            assert response.content == "Test response"
            
            # Check method call arguments
            args, kwargs = mock_method.call_args
            assert kwargs['messages'] == sample_messages
            assert kwargs['model'] == "llama-3.1-8b-instant"
            assert kwargs['max_tokens'] == 100
            assert kwargs['temperature'] == 0.7
    
    @pytest.mark.asyncio
    async def test_stream_chat_completion(self, client):
        """Test streaming chat completion."""
        from multi_llm_client import StreamingChatResponse
        
        mock_responses = [
            StreamingChatResponse(
                id="test-123",
                model="test-model",
                provider=Provider.GROQ,
                delta="Hello",
                finish_reason=None
            ),
            StreamingChatResponse(
                id="test-123",
                model="test-model",
                provider=Provider.GROQ,
                delta=" world",
                finish_reason="stop"
            )
        ]
        
        async def mock_stream(*args, **kwargs):
            for response in mock_responses:
                yield response
        
        with patch.object(client._providers[Provider.GROQ], 'stream_chat_completion', side_effect=mock_stream):
            chunks = []
            async for chunk in client.stream_chat_completion(
                messages="Hello",
                model="llama-3.1-8b-instant"
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0].delta == "Hello"
            assert chunks[1].delta == " world"
            assert chunks[1].finish_reason == "stop"
    
    @pytest.mark.asyncio
    async def test_list_models_specific_provider(self, client, mock_models_response):
        """Test listing models for specific provider."""
        from multi_llm_client import ModelInfo
        
        mock_models = [
            ModelInfo(
                id="llama-3.1-8b-instant",
                provider=Provider.GROQ,
                name="llama-3.1-8b-instant",
                supports_vision=False
            )
        ]
        
        with patch.object(client._providers[Provider.GROQ], 'list_models') as mock_method:
            mock_method.return_value = mock_models
            
            models = await client.list_models(Provider.GROQ)
            
            assert len(models) == 1
            assert models[0].id == "llama-3.1-8b-instant"
            assert models[0].provider == Provider.GROQ
    
    @pytest.mark.asyncio
    async def test_list_models_all_providers(self, client):
        """Test listing models from all providers."""
        from multi_llm_client import ModelInfo
        
        groq_models = [
            ModelInfo(id="groq-model", provider=Provider.GROQ, name="Groq Model", supports_vision=False)
        ]
        together_models = [
            ModelInfo(id="together-model", provider=Provider.TOGETHER, name="Together Model", supports_vision=True)
        ]
        
        with patch.object(client._providers[Provider.GROQ], 'list_models', return_value=groq_models), \
             patch.object(client._providers[Provider.TOGETHER], 'list_models', return_value=together_models):
            
            models = await client.list_models()
            
            assert len(models) == 2
            provider_models = {model.provider: model for model in models}
            assert Provider.GROQ in provider_models
            assert Provider.TOGETHER in provider_models
    
    def test_list_providers(self, client):
        """Test listing configured providers."""
        providers = client.list_providers()
        assert len(providers) == 2
        assert Provider.GROQ in providers
        assert Provider.TOGETHER in providers
    
    def test_get_provider_info(self, client):
        """Test getting provider information."""
        info = client.get_provider_info(Provider.GROQ)
        
        assert info["name"] == "groq"
        assert info["provider"] == "groq"
        assert info["configured"] is True
    
    def test_get_provider_info_not_configured(self, client):
        """Test getting info for non-configured provider."""
        with pytest.raises(ConfigurationError, match="Provider hyperbolic is not configured"):
            client.get_provider_info(Provider.HYPERBOLIC)
"""Tests for data models."""

import pytest
from multi_llm_client import (
    Provider, MessageRole, ChatMessage, ChatResponse, StreamingChatResponse,
    ModelInfo, ProviderConfig, TextContent, ImageContent
)


class TestEnums:
    """Test enum definitions."""
    
    def test_provider_values(self):
        """Test Provider enum values."""
        assert Provider.GROQ.value == "groq"
        assert Provider.TOGETHER.value == "together"
        assert Provider.CEREBRAS.value == "cerebras"
        assert Provider.HYPERBOLIC.value == "hyperbolic"
    
    def test_message_role_values(self):
        """Test MessageRole enum values."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"


class TestChatMessage:
    """Test ChatMessage model."""
    
    def test_text_message(self):
        """Test creating text message."""
        message = ChatMessage(role=MessageRole.USER, content="Hello, world!")
        
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
    
    def test_multimodal_message(self):
        """Test creating multimodal message."""
        content = [
            TextContent(text="What do you see?"),
            ImageContent(image_url={"url": "https://example.com/image.jpg"})
        ]
        
        message = ChatMessage(role=MessageRole.USER, content=content)
        
        assert message.role == MessageRole.USER
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextContent)
        assert isinstance(message.content[1], ImageContent)
        assert message.content[0].text == "What do you see?"
        assert message.content[1].image_url["url"] == "https://example.com/image.jpg"


class TestContent:
    """Test content models."""
    
    def test_text_content(self):
        """Test TextContent model."""
        content = TextContent(text="Hello, world!")
        
        assert content.type == "text"
        assert content.text == "Hello, world!"
    
    def test_image_content(self):
        """Test ImageContent model."""
        image_url = {"url": "https://example.com/image.jpg"}
        content = ImageContent(image_url=image_url)
        
        assert content.type == "image_url"
        assert content.image_url == image_url


class TestChatResponse:
    """Test ChatResponse model."""
    
    def test_basic_response(self):
        """Test basic chat response."""
        response = ChatResponse(
            id="test-123",
            model="llama-3.1-8b",
            provider=Provider.GROQ,
            content="Hello! How can I help you?"
        )
        
        assert response.id == "test-123"
        assert response.model == "llama-3.1-8b"
        assert response.provider == Provider.GROQ
        assert response.content == "Hello! How can I help you?"
        assert response.usage is None
        assert response.finish_reason is None
    
    def test_full_response(self):
        """Test chat response with all fields."""
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
        
        response = ChatResponse(
            id="test-123",
            model="llama-3.1-8b",
            provider=Provider.GROQ,
            content="Hello! How can I help you?",
            usage=usage,
            finish_reason="stop"
        )
        
        assert response.usage == usage
        assert response.finish_reason == "stop"


class TestStreamingChatResponse:
    """Test StreamingChatResponse model."""
    
    def test_streaming_response(self):
        """Test streaming chat response."""
        response = StreamingChatResponse(
            id="test-123",
            model="llama-3.1-8b",
            provider=Provider.GROQ,
            delta="Hello"
        )
        
        assert response.id == "test-123"
        assert response.model == "llama-3.1-8b"
        assert response.provider == Provider.GROQ
        assert response.delta == "Hello"
        assert response.finish_reason is None
    
    def test_final_streaming_chunk(self):
        """Test final streaming response chunk."""
        response = StreamingChatResponse(
            id="test-123",
            model="llama-3.1-8b",
            provider=Provider.GROQ,
            delta="",
            finish_reason="stop"
        )
        
        assert response.delta == ""
        assert response.finish_reason == "stop"


class TestModelInfo:
    """Test ModelInfo model."""
    
    def test_basic_model_info(self):
        """Test basic model information."""
        model = ModelInfo(
            id="llama-3.1-8b-instant",
            provider=Provider.GROQ,
            name="Llama 3.1 8B Instant"
        )
        
        assert model.id == "llama-3.1-8b-instant"
        assert model.provider == Provider.GROQ
        assert model.name == "Llama 3.1 8B Instant"
        assert model.description is None
        assert model.context_length is None
        assert model.supports_vision is False
        assert model.supports_streaming is True
    
    def test_full_model_info(self):
        """Test model information with all fields."""
        model = ModelInfo(
            id="llama-4-vision",
            provider=Provider.TOGETHER,
            name="Llama 4 Vision",
            description="Vision-enabled language model",
            context_length=32768,
            supports_vision=True,
            supports_streaming=False
        )
        
        assert model.description == "Vision-enabled language model"
        assert model.context_length == 32768
        assert model.supports_vision is True
        assert model.supports_streaming is False


class TestProviderConfig:
    """Test ProviderConfig model."""
    
    def test_minimal_config(self):
        """Test minimal provider configuration."""
        config = ProviderConfig(api_key="test-key")
        
        assert config.api_key == "test-key"
        assert config.base_url is None
        assert config.timeout == 30.0
        assert config.max_retries == 3
    
    def test_full_config(self):
        """Test full provider configuration."""
        config = ProviderConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            timeout=60.0,
            max_retries=5
        )
        
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 60.0
        assert config.max_retries == 5
    
    def test_config_validation(self):
        """Test provider configuration validation."""
        # Test that api_key is required
        with pytest.raises(ValueError):
            ProviderConfig()
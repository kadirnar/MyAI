"""Test configuration and fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from multi_llm_client import (
    MultiLLMClient, MultiLLMConfig, Provider, ProviderConfig,
    ChatMessage, MessageRole
)


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MultiLLMConfig()
    config.add_provider(
        Provider.GROQ,
        ProviderConfig(api_key="test-groq-key", timeout=30.0)
    )
    config.add_provider(
        Provider.TOGETHER,
        ProviderConfig(api_key="test-together-key", timeout=30.0)
    )
    config.default_provider = Provider.GROQ
    return config


@pytest.fixture
def client(mock_config):
    """Create a client with mock configuration."""
    return MultiLLMClient(mock_config)


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello, world!")
    ]


@pytest.fixture
def mock_chat_response():
    """Mock chat response data."""
    return {
        "id": "test-id-123",
        "model": "test-model",
        "choices": [{
            "message": {"content": "Hello! How can I help you?"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }


@pytest.fixture
def mock_stream_response():
    """Mock streaming response chunks."""
    return [
        {
            "id": "test-id-123",
            "model": "test-model", 
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        },
        {
            "id": "test-id-123",
            "model": "test-model",
            "choices": [{
                "delta": {"content": " there!"},
                "finish_reason": None
            }]
        },
        {
            "id": "test-id-123",
            "model": "test-model",
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }]
        }
    ]


@pytest.fixture
def mock_models_response():
    """Mock models list response."""
    return {
        "data": [
            {
                "id": "llama-3.1-8b-instant",
                "description": "Fast Llama model"
            },
            {
                "id": "llama-3.2-vision",
                "description": "Vision-enabled model"
            }
        ]
    }
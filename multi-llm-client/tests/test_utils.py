"""Tests for utility functions."""

import pytest
from multi_llm_client.utils import (
    validate_api_key, validate_model_name, validate_temperature,
    validate_max_tokens, sanitize_content, format_error_message,
    get_provider_rate_limits, estimate_tokens, truncate_messages
)
from multi_llm_client import InvalidRequestError, ChatMessage, MessageRole


class TestValidation:
    """Test validation functions."""
    
    def test_validate_api_key_valid_groq(self):
        """Test valid Groq API key."""
        key = "gsk_" + "a" * 32
        validate_api_key(key, "groq")  # Should not raise
    
    def test_validate_api_key_valid_cerebras(self):
        """Test valid Cerebras API key."""
        key = "csk-" + "a" * 32
        validate_api_key(key, "cerebras")  # Should not raise
    
    def test_validate_api_key_valid_together(self):
        """Test valid Together API key."""
        key = "a" * 64  # 64 hex characters
        validate_api_key(key, "together")  # Should not raise
    
    def test_validate_api_key_empty(self):
        """Test empty API key."""
        with pytest.raises(InvalidRequestError, match="API key must be a non-empty string"):
            validate_api_key("", "groq")
    
    def test_validate_api_key_none(self):
        """Test None API key."""
        with pytest.raises(InvalidRequestError, match="API key must be a non-empty string"):
            validate_api_key(None, "groq")
    
    def test_validate_api_key_invalid_format(self):
        """Test invalid API key format."""
        with pytest.raises(InvalidRequestError, match="Invalid API key format"):
            validate_api_key("invalid-key", "groq")
    
    def test_validate_model_name_valid(self):
        """Test valid model name."""
        validate_model_name("llama-3.1-8b-instant", "groq")  # Should not raise
        validate_model_name("meta-llama/Llama-3.1-8B", "together")  # Should not raise
    
    def test_validate_model_name_empty(self):
        """Test empty model name."""
        with pytest.raises(InvalidRequestError, match="Model name must be a non-empty string"):
            validate_model_name("", "groq")
    
    def test_validate_model_name_invalid_chars(self):
        """Test model name with invalid characters."""
        with pytest.raises(InvalidRequestError, match="Invalid model name format"):
            validate_model_name("model with spaces!", "groq")
    
    def test_validate_temperature_valid(self):
        """Test valid temperature values."""
        validate_temperature(0.0, "groq")  # Should not raise
        validate_temperature(0.7, "groq")  # Should not raise
        validate_temperature(1.5, "groq")  # Should not raise
        validate_temperature(2.0, "groq")  # Should not raise
        validate_temperature(None, "groq")  # Should not raise
    
    def test_validate_temperature_invalid_type(self):
        """Test invalid temperature type."""
        with pytest.raises(InvalidRequestError, match="Temperature must be a number"):
            validate_temperature("0.7", "groq")
    
    def test_validate_temperature_out_of_range(self):
        """Test temperature out of valid range."""
        with pytest.raises(InvalidRequestError, match="Temperature must be between 0.0 and 2.0"):
            validate_temperature(-0.1, "groq")
        
        with pytest.raises(InvalidRequestError, match="Temperature must be between 0.0 and 2.0"):
            validate_temperature(2.1, "groq")
    
    def test_validate_max_tokens_valid(self):
        """Test valid max_tokens values."""
        validate_max_tokens(100, "groq")  # Should not raise
        validate_max_tokens(1000, "groq")  # Should not raise
        validate_max_tokens(None, "groq")  # Should not raise
    
    def test_validate_max_tokens_invalid_type(self):
        """Test invalid max_tokens type."""
        with pytest.raises(InvalidRequestError, match="max_tokens must be an integer"):
            validate_max_tokens(100.5, "groq")
    
    def test_validate_max_tokens_negative(self):
        """Test negative max_tokens."""
        with pytest.raises(InvalidRequestError, match="max_tokens must be positive"):
            validate_max_tokens(-1, "groq")
        
        with pytest.raises(InvalidRequestError, match="max_tokens must be positive"):
            validate_max_tokens(0, "groq")
    
    def test_validate_max_tokens_too_large(self):
        """Test max_tokens too large."""
        with pytest.raises(InvalidRequestError, match="max_tokens too large"):
            validate_max_tokens(200000, "groq")


class TestUtilities:
    """Test utility functions."""
    
    def test_sanitize_content_basic(self):
        """Test basic content sanitization."""
        content = "Hello, world!"
        result = sanitize_content(content)
        assert result == "Hello, world!"
    
    def test_sanitize_content_with_null_bytes(self):
        """Test sanitizing content with null bytes."""
        content = "Hello\x00world"
        result = sanitize_content(content)
        assert result == "Helloworld"
    
    def test_sanitize_content_line_endings(self):
        """Test normalizing line endings."""
        content = "Line 1\r\nLine 2\rLine 3\nLine 4"
        result = sanitize_content(content)
        assert result == "Line 1\nLine 2\nLine 3\nLine 4"
    
    def test_sanitize_content_invalid_type(self):
        """Test sanitizing non-string content."""
        with pytest.raises(ValueError, match="Content must be a string"):
            sanitize_content(123)
    
    def test_format_error_message_dict(self):
        """Test formatting error message from dictionary."""
        error_data = {"message": "Invalid request"}
        result = format_error_message(error_data)
        assert result == "Invalid request"
    
    def test_format_error_message_nested(self):
        """Test formatting nested error message."""
        error_data = {"error": {"message": "API key invalid"}}
        result = format_error_message(error_data)
        assert result == "API key invalid"
    
    def test_format_error_message_detail(self):
        """Test formatting error with detail field."""
        error_data = {"detail": "Rate limit exceeded"}
        result = format_error_message(error_data)
        assert result == "Rate limit exceeded"
    
    def test_format_error_message_fallback(self):
        """Test formatting error message fallback."""
        error_data = {"code": 400, "status": "error"}
        result = format_error_message(error_data)
        assert "code" in result and "400" in result
    
    def test_get_provider_rate_limits_known(self):
        """Test getting rate limits for known providers."""
        limits = get_provider_rate_limits("groq")
        assert "requests_per_minute" in limits
        assert "tokens_per_minute" in limits
        assert limits["requests_per_minute"] == 30
    
    def test_get_provider_rate_limits_unknown(self):
        """Test getting rate limits for unknown provider."""
        limits = get_provider_rate_limits("unknown")
        assert limits["requests_per_minute"] == 60
        assert limits["tokens_per_minute"] == 60000
    
    def test_estimate_tokens_basic(self):
        """Test basic token estimation."""
        text = "Hello, world!"
        tokens = estimate_tokens(text)
        assert tokens == max(1, len(text) // 4)
    
    def test_estimate_tokens_empty(self):
        """Test token estimation for empty text."""
        tokens = estimate_tokens("")
        assert tokens == 1  # Minimum 1 token
    
    def test_estimate_tokens_long_text(self):
        """Test token estimation for long text."""
        text = "A" * 1000
        tokens = estimate_tokens(text)
        assert tokens == 250  # 1000 / 4
    
    def test_truncate_messages_empty(self):
        """Test truncating empty message list."""
        result = truncate_messages([], 100)
        assert result == []
    
    def test_truncate_messages_single_fits(self):
        """Test truncating single message that fits."""
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello")
        ]
        
        result = truncate_messages(messages, 100)
        assert len(result) == 1
        assert result[0].content == "Hello"
    
    def test_truncate_messages_single_too_large(self):
        """Test truncating single message that's too large."""
        large_content = "A" * 1000  # ~250 tokens
        messages = [
            ChatMessage(role=MessageRole.USER, content=large_content)
        ]
        
        result = truncate_messages(messages, 100)  # Only 100 tokens allowed
        assert len(result) == 1  # Should still include the last message
    
    def test_truncate_messages_multiple(self):
        """Test truncating multiple messages."""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),  # ~4 tokens
            ChatMessage(role=MessageRole.USER, content="Hello"),  # ~2 tokens
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),  # ~3 tokens
            ChatMessage(role=MessageRole.USER, content="How are you?")  # ~4 tokens
        ]
        
        # Allow enough tokens for last 2 messages (~7 tokens)
        result = truncate_messages(messages, 7)
        assert len(result) >= 1  # Should at least include the last message
        assert result[-1].content == "How are you?"  # Last message should be preserved
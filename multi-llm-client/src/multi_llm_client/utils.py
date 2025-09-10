"""Utility functions for Multi-LLM Client."""

import re
from typing import Dict, Any, Optional
from .exceptions import InvalidRequestError


def validate_api_key(api_key: str, provider: str) -> None:
    """Validate API key format for different providers."""
    if not api_key or not isinstance(api_key, str):
        raise InvalidRequestError(provider, "API key must be a non-empty string")
    
    # Provider-specific validation patterns
    patterns = {
        "groq": r"^gsk_[a-zA-Z0-9]{32,}$",
        "cerebras": r"^csk-[a-zA-Z0-9_-]{32,}$", 
        "together": r"^[a-f0-9]{64}$",  # Together uses 64-char hex strings
        "hyperbolic": r"^[a-zA-Z0-9_-]{32,}$"  # More flexible pattern for Hyperbolic
    }
    
    pattern = patterns.get(provider.lower())
    if pattern and not re.match(pattern, api_key):
        raise InvalidRequestError(
            provider, 
            f"Invalid API key format for {provider}. Please check your API key."
        )


def validate_model_name(model: str, provider: str) -> None:
    """Validate model name for different providers."""
    if not model or not isinstance(model, str):
        raise InvalidRequestError(provider, "Model name must be a non-empty string")
    
    # Basic validation - model names should not contain invalid characters
    if not re.match(r"^[a-zA-Z0-9_/.-]+$", model):
        raise InvalidRequestError(
            provider,
            f"Invalid model name format: {model}"
        )


def validate_temperature(temperature: Optional[float], provider: str) -> None:
    """Validate temperature parameter."""
    if temperature is not None:
        if not isinstance(temperature, (int, float)):
            raise InvalidRequestError(provider, "Temperature must be a number")
        
        # Most providers support 0.0 to 2.0, some allow up to 1.5
        if temperature < 0.0 or temperature > 2.0:
            raise InvalidRequestError(
                provider,
                f"Temperature must be between 0.0 and 2.0, got {temperature}"
            )


def validate_max_tokens(max_tokens: Optional[int], provider: str) -> None:
    """Validate max_tokens parameter."""
    if max_tokens is not None:
        if not isinstance(max_tokens, int):
            raise InvalidRequestError(provider, "max_tokens must be an integer")
        
        if max_tokens <= 0:
            raise InvalidRequestError(
                provider,
                f"max_tokens must be positive, got {max_tokens}"
            )
        
        # Set reasonable upper limits based on typical context windows
        if max_tokens > 128000:
            raise InvalidRequestError(
                provider,
                f"max_tokens too large ({max_tokens}), maximum is 128000"
            )


def sanitize_content(content: str) -> str:
    """Sanitize text content for API requests."""
    if not isinstance(content, str):
        raise ValueError("Content must be a string")
    
    # Remove null bytes and other problematic characters
    content = content.replace("\x00", "")
    
    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    
    return content


def format_error_message(error_data: Dict[str, Any]) -> str:
    """Format error message from provider response."""
    if isinstance(error_data, dict):
        # Try different common error message fields
        message = (
            error_data.get("message") or
            error_data.get("error", {}).get("message") if isinstance(error_data.get("error"), dict) else
            error_data.get("detail") or
            str(error_data)
        )
        return str(message)
    return str(error_data)


def get_provider_rate_limits(provider: str) -> Dict[str, int]:
    """Get known rate limits for providers."""
    limits = {
        "groq": {
            "requests_per_minute": 30,
            "tokens_per_minute": 30000,
        },
        "together": {
            "requests_per_minute": 60,
            "tokens_per_minute": 60000,
        },
        "cerebras": {
            "requests_per_minute": 60, 
            "tokens_per_minute": 60000,
        },
        "hyperbolic": {
            "requests_per_minute": 60,
            "tokens_per_minute": 60000,
        }
    }
    
    return limits.get(provider.lower(), {
        "requests_per_minute": 60,
        "tokens_per_minute": 60000,
    })


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for text."""
    # Simple estimation: ~4 characters per token on average
    return max(1, len(text) // 4)


def truncate_messages(messages: list, max_tokens: int) -> list:
    """Truncate messages to fit within token limit."""
    if not messages:
        return messages
    
    # Keep system message if present
    truncated = []
    total_tokens = 0
    
    # Always keep the last user message
    if messages:
        last_message = messages[-1]
        last_tokens = estimate_tokens(str(last_message.content))
        if last_tokens < max_tokens:
            truncated.append(last_message)
            total_tokens += last_tokens
        
        # Add previous messages if they fit
        for message in reversed(messages[:-1]):
            message_tokens = estimate_tokens(str(message.content))
            if total_tokens + message_tokens > max_tokens:
                break
            truncated.insert(0, message)
            total_tokens += message_tokens
    
    return truncated
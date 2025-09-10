"""Basic usage examples for Multi-LLM Client."""

import asyncio
import os
from multi_llm_client import MultiLLMClient, Provider, ChatMessage, MessageRole

async def basic_text_completion():
    """Basic text completion example."""
    # Initialize client with API keys from environment variables
    client = MultiLLMClient()
    
    # Simple string message
    response = await client.chat_completion(
        messages="What is the capital of France?",
        model="llama-3.1-8b-instant",  # Groq model
        provider=Provider.GROQ,
        max_tokens=100,
        temperature=0.7
    )
    
    print(f"Response: {response.content}")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")

async def structured_chat_completion():
    """Structured chat completion with multiple messages."""
    client = MultiLLMClient()
    
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Explain quantum computing in simple terms.")
    ]
    
    response = await client.chat_completion(
        messages=messages,
        model="llama-3.3-70b",  # Cerebras model (70B params, ~2100 tokens/s)
        provider=Provider.CEREBRAS,
        max_tokens=500,
        temperature=0.3
    )
    
    print(f"Response: {response.content}")

async def streaming_completion():
    """Streaming completion example."""
    client = MultiLLMClient()
    
    print("Streaming response:")
    async for chunk in client.stream_chat_completion(
        messages="Write a short poem about artificial intelligence.",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # Together model
        provider=Provider.TOGETHER,
        max_tokens=200
    ):
        print(chunk.delta, end="", flush=True)
    print("\n")

async def vision_example():
    """Vision model example with image."""
    from multi_llm_client import TextContent, ImageContent
    
    client = MultiLLMClient()
    
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[
                TextContent(text="What do you see in this image?"),
                ImageContent(image_url={"url": "https://example.com/image.jpg"})
            ]
        )
    ]
    
    # Example using Cerebras vision model
    response = await client.chat_completion(
        messages=messages,
        model="llama-4-scout-17b-16e-instruct",  # Cerebras vision model (109B params)
        provider=Provider.CEREBRAS,
        max_tokens=300
    )
    
    print(f"Vision response: {response.content}")

async def list_models_example():
    """List available models from providers."""
    client = MultiLLMClient()
    
    # List models from all providers
    print("All available models:")
    models = await client.list_models()
    for model in models[:5]:  # Show first 5
        print(f"- {model.name} ({model.provider}) - Vision: {model.supports_vision}")
    
    # List models from specific provider
    print(f"\nGroq models:")
    groq_models = await client.list_models(Provider.GROQ)
    for model in groq_models:
        print(f"- {model.id}")

async def provider_management():
    """Provider management example."""
    client = MultiLLMClient()
    
    # Add a new provider or update existing
    client.add_provider(
        Provider.HYPERBOLIC, 
        api_key="your-hyperbolic-api-key",
        timeout=60.0
    )
    
    # List configured providers
    providers = client.list_providers()
    print(f"Configured providers: {[p.value for p in providers]}")
    
    # Get provider info
    if Provider.GROQ in providers:
        info = client.get_provider_info(Provider.GROQ)
        print(f"Groq info: {info}")

async def main():
    """Run all examples."""
    print("=== Basic Text Completion ===")
    await basic_text_completion()
    
    print("\n=== Structured Chat ===")
    await structured_chat_completion()
    
    print("\n=== Streaming ===")
    await streaming_completion()
    
    print("\n=== Vision Model ===")
    await vision_example()
    
    print("\n=== List Models ===")
    await list_models_example()
    
    print("\n=== Provider Management ===")
    await provider_management()

if __name__ == "__main__":
    # Set up environment variables first:
    # export GROQ_API_KEY="your-groq-api-key"
    # export TOGETHER_API_KEY="your-together-api-key" 
    # export CEREBRAS_API_KEY="your-cerebras-api-key"
    # export HYPERBOLIC_API_KEY="your-hyperbolic-api-key"
    
    asyncio.run(main())
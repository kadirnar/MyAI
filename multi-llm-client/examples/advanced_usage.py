"""Advanced usage examples for Multi-LLM Client."""

import asyncio
import json
from multi_llm_client import (
    MultiLLMClient, MultiLLMConfig, Provider, ChatMessage, MessageRole,
    ProviderConfig, TextContent, ImageContent
)

async def custom_configuration():
    """Example with custom configuration."""
    # Create configuration manually
    config = MultiLLMConfig()
    
    # Add providers with custom settings
    config.add_provider(
        Provider.GROQ,
        ProviderConfig(
            api_key="your-groq-key",
            timeout=45.0,
            max_retries=5
        )
    )
    
    config.add_provider(
        Provider.TOGETHER,
        ProviderConfig(
            api_key="your-together-key",
            base_url="https://api.together.xyz/v1",  # Custom base URL
            timeout=60.0
        )
    )
    
    config.default_provider = Provider.GROQ
    
    # Initialize client with custom config
    client = MultiLLMClient(config)
    
    response = await client.chat_completion(
        messages="Hello, world!",
        model="llama-3.1-8b-instant"  # Will use default provider (Groq)
    )
    
    print(f"Custom config response: {response.content}")

async def config_from_dict():
    """Load configuration from dictionary."""
    config_dict = {
        "providers": {
            "groq": {
                "api_key": "your-groq-key",
                "timeout": 30.0
            },
            "cerebras": {
                "api_key": "your-cerebras-key", 
                "timeout": 45.0
            }
        },
        "default_provider": "groq"
    }
    
    config = MultiLLMConfig.from_dict(config_dict)
    client = MultiLLMClient(config)
    
    response = await client.chat_completion(
        messages="What is machine learning?",
        model="llama-3.3-70b",
        provider=Provider.CEREBRAS
    )
    
    print(f"Dict config response: {response.content}")

async def error_handling():
    """Error handling examples."""
    from multi_llm_client import (
        AuthenticationError, RateLimitError, ProviderError, ConfigurationError
    )
    
    client = MultiLLMClient()
    
    try:
        # This will fail if no providers are configured
        response = await client.chat_completion(
            messages="Test message",
            model="nonexistent-model",
            provider=Provider.GROQ
        )
    except AuthenticationError as e:
        print(f"Authentication error: {e}")
    except RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
    except ProviderError as e:
        print(f"Provider error: {e}")
    except ConfigurationError as e:
        print(f"Configuration error: {e}")

async def batch_requests():
    """Process multiple requests in parallel."""
    client = MultiLLMClient()
    
    questions = [
        "What is Python?",
        "Explain machine learning",
        "What is quantum computing?",
        "How does the internet work?"
    ]
    
    # Create tasks for parallel processing
    tasks = []
    for i, question in enumerate(questions):
        task = client.chat_completion(
            messages=question,
            model="llama-3.1-8b-instant",
            provider=Provider.GROQ,
            max_tokens=100
        )
        tasks.append(task)
    
    # Execute all requests in parallel
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Q{i+1}: {questions[i]}")
        print(f"A{i+1}: {response.content[:100]}...")
        print()

async def multi_provider_comparison():
    """Compare responses from different providers."""
    client = MultiLLMClient()
    
    question = "Explain the concept of recursion in programming."
    providers_models = [
        (Provider.GROQ, "llama-3.1-8b-instant"),
        (Provider.CEREBRAS, "llama-3.3-70b"),
        (Provider.TOGETHER, "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    ]
    
    print(f"Question: {question}\n")
    
    for provider, model in providers_models:
        try:
            response = await client.chat_completion(
                messages=question,
                model=model,
                provider=provider,
                max_tokens=200,
                temperature=0.3
            )
            print(f"{provider.value.upper()} ({model}):")
            print(f"{response.content}\n")
            print("-" * 80)
        except Exception as e:
            print(f"Error with {provider.value}: {e}\n")

async def streaming_with_callback():
    """Streaming with custom processing."""
    client = MultiLLMClient()
    
    full_response = ""
    word_count = 0
    
    print("Streaming response with live word count:")
    async for chunk in client.stream_chat_completion(
        messages="Write a detailed explanation of artificial neural networks.",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        provider=Provider.TOGETHER,
        max_tokens=400
    ):
        full_response += chunk.delta
        word_count += len(chunk.delta.split())
        
        print(chunk.delta, end="", flush=True)
        
        # Show word count every 10 words
        if word_count % 10 == 0:
            print(f" [Words: {word_count}]", end="", flush=True)
    
    print(f"\n\nFinal word count: {len(full_response.split())} words")

async def vision_analysis():
    """Advanced vision model usage."""
    client = MultiLLMClient()
    
    # Analyze multiple images
    image_urls = [
        "https://example.com/chart.png",
        "https://example.com/code_screenshot.png"
    ]
    
    for i, url in enumerate(image_urls):
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    TextContent(text="Analyze this image and describe what you see in detail. If it contains text, transcribe it."),
                    ImageContent(image_url={"url": url})
                ]
            )
        ]
        
        try:
            response = await client.chat_completion(
                messages=messages,
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                provider=Provider.TOGETHER,
                max_tokens=500
            )
            
            print(f"Image {i+1} Analysis:")
            print(response.content)
            print("-" * 80)
            
        except Exception as e:
            print(f"Error analyzing image {i+1}: {e}")

async def main():
    """Run advanced examples."""
    print("=== Custom Configuration ===")
    await custom_configuration()
    
    print("\n=== Configuration from Dict ===")
    await config_from_dict()
    
    print("\n=== Error Handling ===")
    await error_handling()
    
    print("\n=== Batch Requests ===")
    await batch_requests()
    
    print("\n=== Multi-Provider Comparison ===")
    await multi_provider_comparison()
    
    print("\n=== Streaming with Callback ===")
    await streaming_with_callback()
    
    print("\n=== Vision Analysis ===")
    await vision_analysis()

if __name__ == "__main__":
    asyncio.run(main())
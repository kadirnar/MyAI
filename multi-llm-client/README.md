# Multi-LLM Client

A unified Python library for interacting with multiple LLM/VLM providers including Hyperbolic, Together AI, Groq, and Cerebras. Built with uv for fast, reliable dependency management.

## Features

- 🚀 **Unified Interface**: Single API for multiple providers
- 🎯 **Provider Support**: Hyperbolic, Together AI, Groq, Cerebras
- 🖼️ **Vision Models**: Support for multimodal LLM/VLM models
- 🔄 **Streaming**: Real-time streaming responses
- ⚙️ **Flexible Configuration**: Environment variables or programmatic setup
- 🛡️ **Type Safety**: Full type hints with Pydantic models
- 🔧 **Error Handling**: Comprehensive error types and handling
- ⚡ **Async/Await**: Modern async Python support

## Installation

```bash
# Using uv (recommended)
uv add multi-llm-client

# Using pip
pip install multi-llm-client
```

This automatically installs the required official SDKs:
- `groq` - Official Groq Python SDK
- `together` - Official Together AI SDK  
- `cerebras-cloud-sdk` - Official Cerebras Cloud SDK
- `openai` - For Hyperbolic (OpenAI-compatible API)

## Quick Start

### 1. Set up API Keys

```bash
export GROQ_API_KEY="gsk_your_groq_key"          # Get from console.groq.com
export TOGETHER_API_KEY="your_64_char_key"       # Get from api.together.xyz  
export CEREBRAS_API_KEY="csk_your_cerebras_key"  # Get from inference.cerebras.ai
export HYPERBOLIC_API_KEY="your_hyperbolic_key"  # Get from app.hyperbolic.xyz
```

> 📋 **Detailed Setup**: See [SETUP.md](SETUP.md) for provider-specific setup instructions with official SDK examples.

### 2. Basic Usage

```python
import asyncio
from multi_llm_client import MultiLLMClient, Provider

async def main():
    # Initialize client (loads API keys from environment)
    client = MultiLLMClient()
    
    # Simple chat completion
    response = await client.chat_completion(
        messages="What is the capital of France?",
        model="llama-3.1-8b-instant",
        provider=Provider.GROQ,
        max_tokens=100
    )
    
    print(response.content)

asyncio.run(main())
```

### 3. Streaming Response

```python
async def streaming_example():
    client = MultiLLMClient()
    
    async for chunk in client.stream_chat_completion(
        messages="Write a poem about AI",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        provider=Provider.TOGETHER
    ):
        print(chunk.delta, end="", flush=True)

asyncio.run(streaming_example())
```

### 4. Vision Models

```python
from multi_llm_client import ChatMessage, MessageRole, TextContent, ImageContent

async def vision_example():
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
    
    response = await client.chat_completion(
        messages=messages,
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        provider=Provider.TOGETHER
    )
    
    print(response.content)

asyncio.run(vision_example())
```

## Supported Providers

### Groq
- **Base URL**: `https://api.groq.com/openai/v1`
- **Models**: Llama 3.1, Llama 4 Scout (vision), Mixtral, Gemma
- **Features**: Ultra-fast inference, vision support

### Together AI  
- **Base URL**: `https://api.together.xyz/v1`
- **Models**: 200+ models including Llama 4 Maverick, Qwen, Code models
- **Features**: Vision models, fine-tuning, extensive model selection

### Cerebras
- **Base URL**: `https://api.cerebras.ai/v1`
- **Production Models**: Llama 4 Scout (109B, vision), Llama 3.3 70B, Llama 3.1 8B, GPT-OSS 120B, Qwen 3 32B
- **Preview Models**: Llama 4 Maverick (400B, vision), Qwen 3 235B/480B variants
- **Features**: Ultra-fast inference (1400-2800+ tokens/sec depending on model)

### Hyperbolic
- **Base URL**: `https://api.hyperbolic.xyz/v1`  
- **Models**: Latest open-source models, vision models
- **Features**: Single-tenant GPUs, private endpoints

## Configuration

### Environment Variables (Recommended)

```bash
export GROQ_API_KEY="gsk_..."
export TOGETHER_API_KEY="..."  
export CEREBRAS_API_KEY="csk-..."
export HYPERBOLIC_API_KEY="..."
```

### Programmatic Configuration

```python
from multi_llm_client import MultiLLMClient, MultiLLMConfig, Provider, ProviderConfig

# Custom configuration
config = MultiLLMConfig()
config.add_provider(
    Provider.GROQ,
    ProviderConfig(
        api_key="your-api-key",
        timeout=60.0,
        max_retries=3
    )
)

client = MultiLLMClient(config)
```

### Configuration from Dictionary

```python
config_dict = {
    "providers": {
        "groq": {
            "api_key": "your-groq-key",
            "timeout": 30.0
        },
        "together": {
            "api_key": "your-together-key"
        }
    },
    "default_provider": "groq"
}

config = MultiLLMConfig.from_dict(config_dict)
client = MultiLLMClient(config)
```

## Advanced Usage

### List Available Models

```python
# List all models from all providers
models = await client.list_models()

# List models from specific provider  
groq_models = await client.list_models(Provider.GROQ)
```

### Error Handling

```python
from multi_llm_client import AuthenticationError, RateLimitError, ProviderError

try:
    response = await client.chat_completion(
        messages="Hello",
        model="llama-3.1-8b-instant",
        provider=Provider.GROQ
    )
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded") 
except ProviderError as e:
    print(f"Provider error: {e}")
```

### Batch Processing

```python
import asyncio

questions = ["What is AI?", "Explain ML", "Define NLP"]

tasks = [
    client.chat_completion(
        messages=question,
        model="llama-3.1-8b-instant", 
        provider=Provider.GROQ
    )
    for question in questions
]

responses = await asyncio.gather(*tasks)
```

## API Reference

### MultiLLMClient

Main client class providing unified interface to all providers.

#### Methods

- `chat_completion()` - Generate chat completion
- `stream_chat_completion()` - Generate streaming completion  
- `list_models()` - List available models
- `add_provider()` - Add/update provider configuration
- `list_providers()` - List configured providers

### Models

- `ChatMessage` - Chat message with role and content
- `ChatResponse` - Chat completion response
- `StreamingChatResponse` - Streaming response chunk
- `ModelInfo` - Model information
- `ProviderConfig` - Provider configuration

### Enums

- `Provider` - Supported providers (GROQ, TOGETHER, CEREBRAS, HYPERBOLIC)
- `MessageRole` - Message roles (SYSTEM, USER, ASSISTANT)

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.py` - Basic features and usage patterns
- `advanced_usage.py` - Advanced configuration and error handling

## Development

### Using uv

```bash
# Clone repository
git clone <repo-url>
cd multi-llm-client

# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
black src/ --check

# Type checking
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Changelog

### v0.1.0
- Initial release
- Support for Groq, Together AI, Cerebras, and Hyperbolic
- Chat completions and streaming
- Vision model support
- Comprehensive configuration system
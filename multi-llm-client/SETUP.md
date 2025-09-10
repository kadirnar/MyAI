# Multi-LLM Client - Setup Guide

This guide provides detailed setup instructions for each provider with their official SDKs.

## Installation

### Core Library
```bash
# Using uv (recommended)
uv add multi-llm-client

# Using pip
pip install multi-llm-client
```

The library automatically installs the required SDKs:
- `groq` - Official Groq Python SDK
- `together` - Official Together AI Python SDK  
- `cerebras-cloud-sdk` - Official Cerebras Cloud SDK
- `openai` - For Hyperbolic (OpenAI-compatible)

## Provider Setup

### 1. Groq Setup

**Installation:**
```bash
pip install groq>=0.4.0
```

**Get API Key:**
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up/login and create an API key
3. Set environment variable:
```bash
export GROQ_API_KEY="gsk_your_api_key_here"
```

**Standalone Usage:**
```python
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ],
    model="mixtral-8x7b-32768",
)
print(chat_completion.choices[0].message.content)
```

### 2. Together AI Setup

**Installation:**
```bash
pip install together>=1.0.0
```

**Get API Key:**
1. Visit [api.together.xyz](https://api.together.xyz)
2. Sign up/login and get your API key from settings
3. Set environment variable:
```bash
export TOGETHER_API_KEY="your_64_character_api_key"
```

**Standalone Usage:**
```python
import os
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-chat-hf",
    messages=[{"role": "user", "content": "tell me about new york"}],
)
print(response.choices[0].message.content)
```

### 3. Cerebras Setup

**Installation:**
```bash
pip install cerebras-cloud-sdk>=1.0.0
```

**Get API Key:**
1. Visit [inference.cerebras.ai](https://inference.cerebras.ai)
2. Sign up and get your API key
3. Set environment variable:
```bash
export CEREBRAS_API_KEY="csk_your_api_key_here"
```

**Standalone Usage:**
```python
import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Why is fast inference important?",
        }
    ],
    model="llama3.1-8b",  # 8B param, ~2200 tokens/s
    # Other available models:
    # "llama-3.3-70b"                     # 70B param, ~2100 tokens/s  
    # "llama-4-scout-17b-16e-instruct"    # 109B param, ~2600 tokens/s (Vision)
    # "gpt-oss-120b"                      # 120B param, ~2800 tokens/s
    # "qwen-3-32b"                        # 32B param, ~2600 tokens/s
)
print(chat_completion.choices[0].message.content)
```

### 4. Hyperbolic Setup

**Installation:**
```bash
pip install openai>=1.0.0  # Hyperbolic uses OpenAI-compatible API
```

**Get API Key:**
1. Visit [app.hyperbolic.xyz](https://app.hyperbolic.xyz/settings)
2. Sign up/login and generate an API key
3. Set environment variable:
```bash
export HYPERBOLIC_API_KEY="your_hyperbolic_api_key"
```

**Standalone Usage:**
```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("HYPERBOLIC_API_KEY"),
    base_url="https://api.hyperbolic.xyz/v1"
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Your question here"}
    ],
    temperature=0.7,
    max_tokens=1024
)
print(response.choices[0].message.content)
```

## Multi-LLM Client Usage

After setting up environment variables for one or more providers:

```python
import asyncio
from multi_llm_client import MultiLLMClient, Provider

async def main():
    # Auto-loads from environment variables
    client = MultiLLMClient()
    
    # Chat with any provider
    response = await client.chat_completion(
        messages="What is the capital of France?",
        model="llama-3.1-8b-instant",  # Groq
        provider=Provider.GROQ
    )
    print(response.content)
    
    # Or use Cerebras
    response = await client.chat_completion(
        messages="Explain quantum computing",
        model="llama3.1-8b",  # Cerebras
        provider=Provider.CEREBRAS
    )
    print(response.content)

asyncio.run(main())
```

## Environment Variables Summary

Set one or more of these environment variables:

```bash
# Groq
export GROQ_API_KEY="gsk_..."

# Together AI  
export TOGETHER_API_KEY="..."

# Cerebras
export CEREBRAS_API_KEY="csk-..."

# Hyperbolic
export HYPERBOLIC_API_KEY="..."
```

## Verification

Run the included quickstart script to verify your setup:

```bash
cd multi-llm-client
python quickstart.py
```

This will:
- ✅ Check which API keys are configured
- 🤖 Test a simple completion
- 📋 List available models
- ✨ Confirm everything is working

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all SDK dependencies are installed
2. **Authentication Errors**: Verify API keys are correct and environment variables are set
3. **Network Issues**: Check internet connection and firewall settings
4. **Rate Limits**: Each provider has different rate limits - the library handles retries automatically

### Getting Help

- Check the [examples/](examples/) directory for usage patterns
- Review provider documentation for model-specific requirements
- File issues on GitHub for library-specific problems

## Next Steps

- Explore [examples/basic_usage.py](examples/basic_usage.py) for common patterns
- Check [examples/advanced_usage.py](examples/advanced_usage.py) for complex scenarios
- Read the main [README.md](README.md) for complete API documentation
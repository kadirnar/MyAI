#!/usr/bin/env python3
"""Quick start example for Multi-LLM Client."""

import asyncio
import os
from multi_llm_client import MultiLLMClient, Provider

async def main():
    """Quick demonstration of the Multi-LLM Client."""
    print("🚀 Multi-LLM Client - Quick Start Demo")
    print("=" * 50)
    
    # Check if any API keys are available
    api_keys = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY"),
        "CEREBRAS_API_KEY": os.getenv("CEREBRAS_API_KEY"),
        "HYPERBOLIC_API_KEY": os.getenv("HYPERBOLIC_API_KEY")
    }
    
    available_keys = {k: v for k, v in api_keys.items() if v}
    
    if not available_keys:
        print("❌ No API keys found in environment variables.")
        print("\nTo use this library, set one or more of these environment variables:")
        print(f"  export GROQ_API_KEY='gsk_your_groq_key'          # Get from console.groq.com")
        print(f"  export TOGETHER_API_KEY='your_64_char_key'       # Get from api.together.xyz")
        print(f"  export CEREBRAS_API_KEY='csk_your_cerebras_key'  # Get from inference.cerebras.ai")
        print(f"  export HYPERBOLIC_API_KEY='your_hyperbolic_key'  # Get from app.hyperbolic.xyz")
        print("\nSee SETUP.md for detailed provider setup instructions.")
        print("Then run this script again.")
        return
    
    print("✅ Found API keys for:")
    for key in available_keys.keys():
        provider_name = key.replace("_API_KEY", "").title()
        print(f"  - {provider_name}")
    
    # Initialize client
    client = MultiLLMClient()
    
    print(f"\n📋 Configured providers: {[p.value for p in client.list_providers()]}")
    
    # Try a simple completion
    if client.list_providers():
        print("\n💬 Testing chat completion...")
        try:
            # Choose appropriate model based on available provider
            if Provider.GROQ in client.list_providers():
                model = "llama-3.1-8b-instant"  # Groq
            elif Provider.CEREBRAS in client.list_providers():
                model = "llama3.1-8b"  # Cerebras (~2200 tokens/s)
            elif Provider.TOGETHER in client.list_providers():
                model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  # Together
            else:
                model = "llama-3.1-8b-instant"  # Default
            
            response = await client.chat_completion(
                messages="Say hello and introduce yourself as a multi-provider AI assistant!",
                model=model,
                provider=client.list_providers()[0],
                max_tokens=150
            )
            
            print(f"🤖 Response from {response.provider.value}:")
            print(f"   {response.content}")
            
        except Exception as e:
            print(f"❌ Error during completion: {e}")
            print("   This might be due to invalid API keys or network issues.")
    
    # Try listing models if we have a working provider
    if client.list_providers():
        print(f"\n🔍 Available models from {client.list_providers()[0].value}:")
        try:
            models = await client.list_models(client.list_providers()[0])
            for model in models[:5]:  # Show first 5
                vision_indicator = "🖼️" if model.supports_vision else "📝"
                print(f"   {vision_indicator} {model.id}")
            if len(models) > 5:
                print(f"   ... and {len(models) - 5} more models")
                
        except Exception as e:
            print(f"❌ Error listing models: {e}")
    
    print(f"\n✨ Multi-LLM Client is ready to use!")
    print("   Check out examples/basic_usage.py for more examples.")

if __name__ == "__main__":
    asyncio.run(main())
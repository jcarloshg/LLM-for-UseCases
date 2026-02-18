#!/usr/bin/env python3
"""Debug script to check Ollama response structure"""

import json
import sys
from pathlib import Path

# Handle imports
try:
    from src.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from src.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG


def test_ollama_direct():
    """Test Ollama API directly"""
    try:
        from ollama import Client

        print("=" * 60)
        print("Testing Ollama Connection")
        print("=" * 60)
        print(f"\nConfig: {ENVIRONMENT_CONFIG}\n")

        client = Client(host=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST)

        # Test 1: List available models
        print("Step 1: Fetching available models...")
        try:
            response = client.list()
            print(f"✅ Available models: {response}")
        except Exception as e:
            print(f"❌ Failed to list models: {e}")
            return

        # Test 2: Simple chat request
        print("\nStep 2: Testing simple chat request...")
        print(f"Model: {ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL4B}")

        try:
            response = client.chat(
                model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL4B,
                messages=[
                    {"role": "user", "content": "Say hello"}
                ],
                stream=False  # Important: no streaming
            )

            print(f"\n✅ Raw Response Type: {type(response)}")
            print(f"✅ Response Keys: {response.keys() if hasattr(response, 'keys') else 'N/A'}")
            print(f"\n✅ Full Response:")
            print(json.dumps(response, indent=2, default=str))

            # Extract content
            if isinstance(response, dict):
                content = response.get('message', {}).get('content', '')
                print(f"\n✅ Extracted Content: {content[:100]}")

        except Exception as e:
            print(f"❌ Chat failed: {e}")
            import traceback
            traceback.print_exc()

    except ImportError:
        print("❌ ollama package not installed")
        print("Run: pip install ollama")


if __name__ == "__main__":
    test_ollama_direct()

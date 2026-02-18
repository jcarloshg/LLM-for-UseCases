# tests/test_prompts.py
import json
import sys
from pathlib import Path

# Handle both module and direct execution
try:
    from src.llm.client import LLMClient, LLMConfig
    from src.llm.prompts import PromptBuilder
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.llm.client import LLMClient, LLMConfig
    from src.llm.prompts import PromptBuilder


def check_service_health(config):
    """Check if Ollama service is running"""
    import requests
    try:
        response = requests.get(
            f"{config.ollama_base_url}/api/tags", timeout=5)
        # print(f"response {response.json()}")
        return response.status_code == 200
    except:
        return False


def test_prompt_quality():
    """Test prompt on sample user stories"""

    config = LLMConfig()

    # Check service health
    print("Checking Ollama service...")
    if not check_service_health(config):
        print(f"❌ Ollama service not available at {config.ollama_base_url}")
        print("   Make sure to start Ollama with: docker-compose up")
        return []

    print(f"✅ Ollama service is running at {config.ollama_base_url}\n")

    llm_client = LLMClient(config)
    prompt_builder = PromptBuilder()

    test_stories = [
        "As a user, I want to reset my password so that I can regain access",
        # "As an admin, I want to export user data so that I can analyze trends",
        # "As a customer, I want to track my order so that I know when it arrives"
    ]

    results = []

    for story in test_stories:
        print(f"\n{'='*60}")
        print(f"Testing: {story}")
        print(f"{'='*60}")

        prompts = prompt_builder.build(story)
        print(f"="*60)
        print(prompts)
        print(f"="*60)
        response = llm_client.generate(
            prompts['user'],
            prompts['system']
        )

        print(f"\nLatency: {response.get('latency', 'N/A')}")
        print(f"Tokens: {response.get('tokens', 'N/A')}")

        # Check for errors in response
        if response.get('error'):
            print(f"❌ Error: {response['error']}")
            results.append({
                "story": story,
                "success": False,
                "error": response['error']
            })
            continue

        # Debug: Print response structure
        print(f"Response keys: {list(response.keys())}")
        text_length = len(response.get('text', ''))
        print(f"Text length: {text_length}")

        # Try to parse JSON
        try:
            output_text = response.get('text', '').strip()

            if not output_text:
                print(
                    f"⚠️  DEBUG - Full response: {json.dumps(response, indent=2)}")
                raise ValueError("Empty response from LLM")

            # Extract JSON if wrapped in markdown
            if '```json' in output_text:
                output_text = output_text.split('```json')[1].split('```')[0]
            elif '```' in output_text:
                output_text = output_text.split('```')[1].split('```')[0]

            test_cases = json.loads(output_text)

            print(f"✅ Valid JSON")
            print(
                f"Test cases generated: {len(test_cases.get('test_cases', []))}")

            # Display first test case
            if test_cases.get('test_cases'):
                tc = test_cases['test_cases'][0]
                print(f"\nSample test case:")
                print(f"  Title: {tc.get('title')}")
                print(f"  Priority: {tc.get('priority')}")
                print(f"  Given: {tc.get('given')}")

            results.append({
                "story": story,
                "success": True,
                "count": len(test_cases.get('test_cases', []))
            })

        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Parse Error: {e}")
            raw_output = response.get('text', '')
            if raw_output:
                print(f"Raw output preview: {raw_output[:200]}")
            else:
                print("Response text is empty - LLM may not be running or responding")
            results.append({
                "story": story,
                "success": False,
                "error": str(e)
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r.get('success'))
    print(f"Success rate: {successful}/{len(results)}")

    return results


if __name__ == "__main__":
    test_prompt_quality()

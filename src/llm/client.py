from typing import Optional, Dict, Any
from pydantic import BaseModel
import time
import os
import sys
from pathlib import Path

# Handle both module and direct execution
try:
    from ..shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL4B
    temperature: float = 0.3  # Low for consistency
    max_tokens: int = 2000
    api_key: Optional[str] = None
    ollama_base_url: str = ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST
    max_retries: int = ENVIRONMENT_CONFIG.MAX_RETRIES


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        if self.config.provider == "openai":
            import openai
            openai.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")

    def generate(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """Generate response from LLM"""
        start = time.time()

        try:
            if self.config.provider == "ollama":
                response = self._call_ollama(prompt, system_prompt)
            elif self.config.provider == "openai":
                response = self._call_openai(prompt, system_prompt)
            else:
                raise ValueError(f"Unknown provider: {self.config.provider}")

            print(f"response {response}")

            return {
                "text": response["text"],
                "latency": time.time() - start,
                "tokens": response.get("tokens", 0),
                "model": self.config.model,
                "provider": self.config.provider
            }
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "latency": time.time() - start,
                "tokens": 0
            }

    def _call_ollama(self, prompt: str, system_prompt: str) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Create Ollama client with configured base URL
        from ollama import Client
        client = Client(host=self.config.ollama_base_url)

        response = client.chat(
            model=self.config.model,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                # "num_predict": self.config.max_tokens
            }
        )

        # Debug: Check response structure
        text_content = response.get('message', {}).get('content', '')

        return {
            "text": text_content,
            "tokens": response.get('eval_count', 0) + response.get('prompt_eval_count', 0)
        }

    def _call_openai(self, prompt: str, system_prompt: str) -> dict:
        import openai

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = openai.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            # max_tokens=self.config.max_tokens
        )

        return {
            "text": response.choices[0].message.content,
            "tokens": response.usage.total_tokens
        }


# Test script
if __name__ == "__main__":
    config = LLMConfig()
    client = LLMClient(config)

    result = client.generate("Say hello")
    print(f"✅ Model test: {result['text']}")
    print(f"Latency: {result['latency']:.2f}s")

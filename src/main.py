import sys
from pathlib import Path
# from src.llm.prompts import PromptBuilder
# from src.llm.client import LLMClient, LLMConfig
from langchain_core.prompts import ChatPromptTemplate


try:
    from src.llm.client import LLMClient, LLMConfig
    from src.llm.prompts import PromptBuilder
    from src.llm.prompts import PromptBuilder
    from src.llm.client import LLMClient, LLMConfig
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.llm.client import LLMClient, LLMConfig
    from src.llm.prompts import PromptBuilder
    from src.llm.prompts import PromptBuilder
    from src.llm.client import LLMClient, LLMConfig

# Create a simple prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question: {question}"
)


# Example usage
if __name__ == "__main__":
    question = "What is the capital of France?"
    formatted_prompt = prompt.format(question=question)

    print(f"Question: {question}")
    print("-" * 50)

    try:
        # Initialize the Ollama Qwen3-VL:4B model with retry logic
        qwen3vl4b_model = OllamaQwen3vl4b()

        # Get response from the LLM with safe_call (includes retry logic)
        # response = qwen3vl4b_model.safe_call(formatted_prompt)
        response = qwen3vl4b_model.safe_call_with_tokens(
            max_retries=ENVIRONMENT_CONFIG.MAX_RETRIES,
            prompt=formatted_prompt
        )

        test_stories = [
            "As a user, I want to reset my password so that I can regain access",
            # "As an admin, I want to export user data so that I can analyze trends",
            # "As a customer, I want to track my order so that I know when it arrives"
        ]

        for story in test_stories:
            print(f"\n{'='*60}")
            print(f"Testing: {story}")
            print(f"{'='*60}")

            config = LLMConfig()
            llm_client = LLMClient(config)
            prompt_builder = PromptBuilder()
            print(f"="*60)
            print(prompt_builder)
            print(f"="*60)
            # prompts = prompt_builder.build(story)
            # response = llm_client.generate(
            #     prompts['user'],
            #     prompts['system']
            # )
            # print(response)

        print(f"Answer: {response}")
    except OllamaCallError as e:
        # Show user-friendly message to end users
        print(f"Error: {e.user_message}")
        # Optionally log developer message for debugging
        print(f"[DEBUG] {e.dev_message}")

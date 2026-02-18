from langchain_core.prompts import ChatPromptTemplate
from src.shared.models import OllamaQwen3vl4b, OllamaCallError
from src.shared.infrastructure import ENVIRONMENT_CONFIG

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
        print(f"Answer: {response}")
    except OllamaCallError as e:
        # Show user-friendly message to end users
        print(f"Error: {e.user_message}")
        # Optionally log developer message for debugging
        print(f"[DEBUG] {e.dev_message}")

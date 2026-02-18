from langchain_core.prompts import ChatPromptTemplate
from src.shared.models import OllamaService, OllamaCallError

# Initialize the Ollama service singleton with validated environment variables
ollama_service = OllamaService()

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
        # Get response from the LLM with retry logic
        response = ollama_service.safe_call(formatted_prompt)
        print(f"Answer: {response}")
    except OllamaCallError as e:
        # Show user-friendly message to end users
        print(f"Error: {e._user_message}")
        # Optionally log developer message for debugging
        print(f"[DEBUG] {e.dev_message}")

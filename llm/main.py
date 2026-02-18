from langchain_core.prompts import ChatPromptTemplate
from src.shared.models import OllamaService

# Initialize the Ollama service singleton with validated environment variables
ollama_service = OllamaService()
llm = ollama_service.get_qwen3vl4b()

# Create a simple prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question: {question}"
)

# Create a simple chain
chain = prompt | llm

# Example usage
if __name__ == "__main__":
    question = "What is the capital of France?"
    print(f"Question: {question}")
    print("-" * 50)

    # Get response from the LLM
    response = chain.invoke({"question": question})
    print(f"Answer: {response}")

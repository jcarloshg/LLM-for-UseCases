from langchain_core.prompts import ChatPromptTemplate
from src.shared.infrastructure import OLLAMA_SERVICE_HOST, OLLAMA_SERVICE_MODEL
from src.shared.models import OllamaConfig, OllamaService

# Initialize the Ollama service with configuration from .env.dev
ollama_config = OllamaConfig(
    host=OLLAMA_SERVICE_HOST,
    model=OLLAMA_SERVICE_MODEL
)
ollama_service = OllamaService(ollama_config)
llm = ollama_service.get_llm()

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

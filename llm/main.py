from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Ollama LLM with local host
# Note: Make sure ollama is running via docker-compose up
llm = OllamaLLM(model="qwen3-vl:4b", base_url="http://localhost:11435")

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

import os
from langchain_ollama import OllamaEmbeddings, ChatOllama

class Models:
    def __init__(self):
        # Embedding model: Switch to a more robust option for better document representation
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # Better for semantic understanding than mxbai-embed-large
            # Optional: Adjust embedding dimensionality if supported by the model
        )

        # Chat model: Enhanced configuration for accuracy and speed
        self.llm = ChatOllama(
            model="llama3.2",  # Keeping llama3.2 for now; consider alternatives below
            temperature=0.0,   # Lowered from 0.1 to reduce hallucination
            top_p=0.9,         # Added to focus on high-probability tokens
            max_tokens=4096,   # Increased to ensure full context processing
            num_ctx=8192,      # Increased context window for better document handling
            num_threads=4,     # Adjust based on your CPU cores for parallelism
            # Optional: Enable streaming for faster perceived response time
            # stream=True,
        )

        self.multimodal_llm = ChatOllama(
            model="llava-phi3",
            temperature=0.0,      # Minimize randomness
            max_tokens=512,       # Concise summaries
            num_ctx=2048,
        )

# Example usage
if __name__ == "__main__":
    models = Models()
    print("Embeddings model:", models.embeddings_ollama.model)
    print("Chat model:", models.chat_ollama.model)
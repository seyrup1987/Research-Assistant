from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models
import threading
import time
import sys
import itertools

# Initialize models and embeddings
models = Models()
embeddings = models.embeddings
llm = models.llm

# Set up the Chroma vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db"
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise assistant tasked with answering questions based solely on the documents in the database. Do not generate information beyond what is explicitly provided in the context. If the context does not contain enough information to answer, respond with 'I do not know.'"),
    ("user", "Answer the question '{input}' using only the provided {context}. Do not speculate, extrapolate, or add details not present in the context. If the answer is unclear or unavailable, say 'I do not know.'")
])

# Create the retrieval chain
retriever = vector_store.as_retriever(kwargs={"k": 10})
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(
    retriever,
    combine_docs_chain
)

# Create a threading event for graceful shutdown
shutdown_event = threading.Event()

def run_query(query, result_container):
    """
    Run the query in a separate thread and store the result in result_container.
    Check for shutdown signal to exit early if needed.
    """
    if shutdown_event.is_set():
        return
    try:
        result = retrieval_chain.invoke({"input": query})
        if not shutdown_event.is_set():  # Only append result if not shutting down
            result_container.append(result)
    except Exception as e:
        if not shutdown_event.is_set():
            result_container.append({"error": f"Error processing query: {str(e)}"})

def spinner_animation():
    """
    Generator for spinner animation.
    """
    spinner = itertools.cycle(["-", "/", "|", "\\"])
    while True:
        yield next(spinner)

def Iris(query):
    """
    Function to execute a query provided by the Desktop Research Assistant
    """
    result_container = []

    user_input = query
    query_thread = threading.Thread(
        target=run_query,
        args=(user_input, result_container)
    )
    query_thread.start()
    query_thread.join()

    if result_container:
        result = result_container[0]
        if "error" in result:
            return {result["error"]}
        else:
            print(result["answer"])
            return {
                'answer': result["answer"],
                'context': result["context"]
            }
    else:
        return {
                'answer': "No result returned. The query may have been interrupted.",
                'context': "N/A"
            }


def main():
    """
    Main function to handle user input, run queries, and display results.
    """
    while True:
        # Get user input
        user_input = input("Enter a question (or type 'q' to quit): ")
        if user_input.lower() == 'q':
            print("E X I T I N G . . . .")
            shutdown_event.set()  # Signal threads to stop
            break

        # Reset shutdown event if this is a new query
        shutdown_event.clear()
        result_container = []

        # Start the query in a separate thread
        query_thread = threading.Thread(
            target=run_query,
            args=(user_input, result_container)
        )
        query_thread.start()
        query_thread.join()

        # Show spinner animation while the query is running
        spinner = spinner_animation()
        print("Ollama is thinking ", end="", flush=True)
        while query_thread.is_alive():
            sys.stdout.write(next(spinner))
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b')

        # Wait for the thread to complete
        query_thread.join()
        print("\n")

        # Check if the program is shutting down
        if shutdown_event.is_set():
            continue

        # Display the result if available
        if result_container:
            result = result_container[0]
            if "error" in result:
                print(result["error"])
            else:
                print(result["answer"])
                if "context" in result:
                    print("\nSources: ")
                    for i, doc in enumerate(result["context"]):
                        print(f"Source {i+1}:")
                        print(doc.page_content[:200] + " . . ." if len(doc.page_content) > 200 else doc.page_content)
                        print()
        else:
            print("No result returned. The query may have been interrupted.")

if __name__ == '__main__':
    try:
        main()
    finally:
        # Ensure shutdown event is set when the program exits
        shutdown_event.set()
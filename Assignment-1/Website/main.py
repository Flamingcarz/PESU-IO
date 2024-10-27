from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import requests
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure the embedding model and LLM
try:
    Settings.embed_model = JinaEmbedding(
        api_key=os.getenv("JINA_API_KEY"),
        model="jina-embeddings-v3",
        task="retrieval.passage",
    )
except Exception as e:
    print("Error using JinaEmbedding, switching to CohereEmbedding. Error:", e)
    Settings.embed_model = CohereEmbedding(
        api_key=os.getenv("COHERE_API_KEY"),
        model_name="embed-english-v3.0",
        input_type="search_query"
    )

Settings.llm = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-70b-versatile",
    temperature=0.7
)

def create_rag_system(data_dir="./web_data"):
    """Initialize the RAG (Retrieval-Augmented Generation) system using Qdrant for vector storage."""
    client = QdrantClient(path="./qdrant_web_data")

    # Configure Qdrant vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="my_documents",
        dimension=1024  # Ensure this matches the embedding dimension
    )
    
    # Load documents from the specified directory
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    # Create vector store index with Qdrant
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store
    )
    
    # Create query engine from the index
    query_engine = index.as_query_engine()
    
    return query_engine

def query_rag(query_engine, question: str):
    """Query the RAG system."""
    response = query_engine.query(question)
    return response

def fetch_and_save_webpage(url: str, output_dir: str = "./web_data") -> str:
    """Fetch webpage content and save as markdown file."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Fetch markdown content using the Jina service
    response = requests.get(f"https://r.jina.ai/{url}")
    if response.status_code != 200:
        raise Exception(f"Failed to fetch content: {response.status_code}")
    
    # Format filename from URL
    filename = url.replace("/", "_").replace(":", "_") + ".md"
    filepath = os.path.join(output_dir, filename)
    
    # Save content to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    return filepath

def main():
    # Get URL from user
    url = input("Enter the URL to analyze: ")
    
    # Fetch and save webpage content
    filepath = fetch_and_save_webpage(url)
    print(f"Webpage content saved to {filepath}")
    
    # Initialize the RAG system with web_data directory
    query_engine = create_rag_system("./web_data")
    
    # Interactive query loop
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        response = query_rag(query_engine, question)
        print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()

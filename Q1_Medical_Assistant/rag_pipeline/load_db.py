from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

# Load the existing database without re-embedding
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = Chroma(
    collection_name="research_data",
    embedding_function=embedder,
    persist_directory="./chroma_db"
)

# Check if documents exist in the database
print(f"Number of documents in DB: {vector_store._collection.count()}") 
from langchain_chroma import Chroma
from langchain_nomic import NomicEmbeddings
from loader import chunks

from dotenv import load_dotenv

load_dotenv()

# Load the existing database without re-embedding
embedder = NomicEmbeddings(model="nomic-embed-text-v1.5")

vector_store = Chroma(
    collection_name="finance_data",
    embedding_function=embedder,
    persist_directory="./chroma_db"
)

vector_store.add_documents(chunks)

# Check if documents exist in the database
print(f"Number of documents in DB: {vector_store._collection.count()}") 
from langchain_chroma import Chroma
from langchain_nomic import NomicEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load existing database ONLY (no document addition)
embedder = NomicEmbeddings(model="nomic-embed-text-v1.5")

vector_store = Chroma(
    collection_name="finance_data",
    embedding_function=embedder,
    persist_directory="./chroma_db"
)

# Verify database has content
doc_count = vector_store._collection.count()
print(f"✅ Loaded database with {doc_count} documents") 
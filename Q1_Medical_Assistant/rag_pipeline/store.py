from langchain_chroma import Chroma
from embedd import embedder
from loader import chunks

vector_store = Chroma(
    collection_name="research_data",
    embedding_function=embedder,
    persist_directory="./chroma_db"  # optional, for local persistence
)

# Add your chunks (list of Document objects) to the vector store
vector_store.add_documents(chunks)



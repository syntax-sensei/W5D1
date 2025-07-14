from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader("../docs", glob="**/*.pdf", show_progress=True, loader_cls=PyMuPDFLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True )

chunks = text_splitter.split_documents(docs)

print(f"Number of chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks[:2]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)

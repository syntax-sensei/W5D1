from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def separate_prose_and_tables(page_content):
    """Split content into prose (text) and tables (markdown)"""
    lines = page_content.split('\n')
    
    prose_lines = []
    table_lines = []
    in_table = False
    
    for line in lines:
        # Detect start of table (line with |)
        if '|' in line:
            if not in_table:
                in_table = True
            table_lines.append(line)
        elif in_table and line.strip() == '':
            # Empty line might be part of table formatting
            table_lines.append(line)
        elif in_table and '|' not in line and line.strip():
            # Non-empty line without | after table = end of table
            in_table = False
            prose_lines.append(line)
        else:
            prose_lines.append(line)
    
    return '\n'.join(prose_lines).strip(), '\n'.join(table_lines).strip()

def chunk_with_table_preservation(docs):
    """Chunk documents while keeping tables intact"""
    
    prose_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    all_chunks = []
    
    for doc in docs:
        prose_text, table_text = separate_prose_and_tables(doc.page_content)
        
        # Chunk prose normally
        if prose_text:
            prose_chunks = prose_splitter.split_text(prose_text)
            for i, chunk_text in enumerate(prose_chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata, 
                        'content_type': 'prose',
                        'chunk_id': f"prose_{doc.metadata.get('page', 0)}_{i}"
                    }
                )
                all_chunks.append(chunk_doc)
        
        # Keep entire table as one chunk
        if table_text:
            table_doc = Document(
                page_content=table_text,
                metadata={
                    **doc.metadata, 
                    'content_type': 'table',
                    'chunk_id': f"table_{doc.metadata.get('page', 0)}"
                }
            )
            all_chunks.append(table_doc)
    
    return all_chunks

print("Loading all PDF documents from ./data directory...")
loader = DirectoryLoader(
    "../data", 
    glob="**/*.pdf", 
    show_progress=True,
    loader_cls=PyMuPDFLoader,
    loader_kwargs={
        "mode": "page",
        "images_parser": TesseractBlobParser(),
        "extract_tables": "markdown"
    }
)

docs = loader.load()
print(f"Loaded {len(docs)} pages from all PDFs")

# Use table-aware chunking
chunks = chunk_with_table_preservation(docs)
print(f"Created {len(chunks)} chunks total")

# Show breakdown
prose_chunks = [c for c in chunks if c.metadata.get('content_type') == 'prose']
table_chunks = [c for c in chunks if c.metadata.get('content_type') == 'table']
print(f"  - {len(prose_chunks)} prose chunks")
print(f"  - {len(table_chunks)} table chunks")



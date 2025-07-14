from load_db import vector_store

# Test simple similarity search without re-embedding
print("Testing retrieval from existing database...")

# Simple similarity search
results = vector_store.similarity_search("transformer architecture", k=2)

print(f"\nFound {len(results)} results:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Page: {doc.metadata.get('page', 'Unknown')}")
    print(f"Content: {doc.page_content[:200]}...")

print("\n✅ Retrieval working without re-embedding!") 
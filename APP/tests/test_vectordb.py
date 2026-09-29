from services.embeddings import generate_embeddings
from services.vectordb import store_embeddings

# Sample chunks
chunks = [
    "Artificial Intelligence is transforming the world.",
    "Machine Learning is a subset of Artificial Intelligence.",
    "Deep Learning uses neural networks.",
    "RAG combines retrieval with generation."
]

# Generate embeddings
embeddings = generate_embeddings(chunks)

# Store in ChromaDB
total = store_embeddings(
    chunks=chunks,
    embeddings=embeddings,
    filename="sample_document"
)

print(f"Successfully stored {total} chunks in ChromaDB.")
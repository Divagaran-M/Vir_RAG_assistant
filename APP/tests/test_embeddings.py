from services.embeddings import generate_embeddings

sample_chunks = [
    "Artificial Intelligence is transforming the world.",
    "Machine Learning is a subset of Artificial Intelligence.",
    "Deep Learning uses neural networks.",
    "RAG combines retrieval with generation."
]

embeddings = generate_embeddings(sample_chunks)

print(f"Total Vectors: {len(embeddings)}")
print()

print(f"Dimensions of Vector 1: {len(embeddings[0])}")
print()

print("First 10 values of Vector 1:")
print(embeddings[0][:10])
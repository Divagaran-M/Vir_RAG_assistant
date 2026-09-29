from services.chunker import chunk_text

sample_text = """
Artificial Intelligence is transforming the world.

Machine Learning is a subset of Artificial Intelligence.

Deep Learning is a subset of Machine Learning.

RAG stands for Retrieval Augmented Generation.

Embeddings convert text into vectors.

Vector databases store embeddings for semantic search.
""" * 20

chunks = chunk_text(sample_text)

print(f"Total Chunks: {len(chunks)}")

print()

for i, chunk in enumerate(chunks, start=1):
    print("=" * 50)
    print(f"Chunk {i}")
    print("=" * 50)
    print(chunk)
    print()
from services.embeddings import generate_embeddings
from services.vectordb import search_embeddings

question = "What projects has Divagaran done?"

query_embedding = generate_embeddings([question])[0]

results = search_embeddings(query_embedding)

print("Retrieved Chunks:\n")

for doc in results["documents"][0]:
    print("=" * 50)
    print(doc)
    print()
import re
from services.embeddings import generate_query_embedding
from services.vectordb import search_embeddings
from services.context_filter import remove_duplicate_chunks


def retrieve_context(question: str, filename: str = None, question_type: str = "general", max_chars: int = 6000) -> str:
    """
    Retrieve relevant context from the selected document or all documents
    using vector search + entity/keyword-aware re-ranking.
    """

    # ------------------------------------
    # Decide Retrieval Strategy
    # ------------------------------------
    top_k = 8

    print("\n========== RETRIEVAL ==========")
    print(f"Question Type : {question_type}")
    print(f"Top K         : {top_k}")

    # ------------------------------------
    # Generate Query Embedding
    # ------------------------------------
    query_embedding = generate_query_embedding(question)

    # ------------------------------------
    # Search Vector Database
    # ------------------------------------
    results = search_embeddings(
        query_embedding=query_embedding,
        filename=filename,
        top_k=top_k
    )

    chunks = results["documents"][0] if results.get("documents") else []

    print("\n================ RAW CHUNKS ================\n")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}")
        print("-" * 50)
        print(chunk[:250])
    print("\n============================================\n")

    print(f"Retrieved Chunks : {len(chunks)}")

    # ------------------------------------
    # Remove Duplicate Chunks
    # ------------------------------------
    unique_chunks = remove_duplicate_chunks(chunks)
    print(f"Unique Chunks    : {len(unique_chunks)}")

    # ------------------------------------
    # Keyword/Entity-Aware Re-ranking
    # ------------------------------------
    # Give priority to chunks that explicitly mention specific query entities
    STOPWORDS = {"who", "what", "where", "when", "which", "tell", "about", "the", "and", "for", "with", "from", "list", "show", "give", "how", "are", "you"}
    q_tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", question) if len(t) > 1 and t.lower() not in STOPWORDS]

    def _chunk_score(ch: str) -> int:
        ch_lower = ch.lower()
        return sum(len(t) * 10 for t in q_tokens if t in ch_lower)

    unique_chunks.sort(key=_chunk_score, reverse=True)


    # ------------------------------------
    # Assemble Context within Budget
    # ------------------------------------
    accumulated = []
    current_len = 0
    for chunk in unique_chunks:
        if current_len + len(chunk) > max_chars and accumulated:
            break
        accumulated.append(chunk)
        current_len += len(chunk) + 2

    context = "\n\n".join(accumulated)

    print(f"Final Context Chars: {len(context)}")
    print("===============================\n")

    return context

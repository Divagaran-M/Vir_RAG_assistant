from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)


def rewrite_query(question, history):
    """
    Rewrite vague or follow-up questions into
    standalone questions for better retrieval.
    """

    conversation = ""

    for message in history:
        role = message["role"].capitalize()
        conversation += f"{role}: {message['content']}\n"

    prompt = f"""
You are a Query Rewriting Assistant for a Retrieval-Augmented Generation (RAG) system.

Your task is to rewrite the user's latest question into a clear, standalone question.

Rules:

- Preserve the user's original intent.
- Resolve pronouns like:
  - it
  - they
  - he
  - she
  - this
  - that
  - these
  - those

- Expand vague questions using the conversation history.

- Do NOT answer the question.

- Do NOT introduce new information.

- Return ONLY the rewritten question.

Conversation History:

{conversation}

Current Question:

{question}

Rewritten Question:
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    rewritten = response.choices[0].message.content.strip()

    print("\n========== QUERY REWRITE ==========")
    print("Original :", question)
    print("Rewritten:", rewritten)
    print("===================================\n")

    return rewritten
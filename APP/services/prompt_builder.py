def build_prompt(context, question, history=None, question_type="General"):
    """
    Build a structured, token-efficient prompt for Document and Tabular RAG.
    """
    if history is None:
        history = []

    conversation_text = ""
    if history:
        turns = []
        for msg in history[-4:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            turns.append(f"{role}: {content}")
        conversation_text = "\n".join(turns)

    prompt = f"""You are Vir, an intelligent AI campus assistant for P.T. Lee Chengalvaraya Naicker College of Engineering and Technology.

YOUR ROLE:
Answer the user's question accurately and naturally using ONLY the provided Document Context.

RULES:
1. The Document Context contains document passages as well as tabular record rows formatted as key=value pairs.
2. Extract facts (names, register numbers, dates, blood groups, grades, departments, attendance, addresses) directly from the context.
3. If the context contains tabular records matching the question, summarize or state the answer clearly in natural language.
4. If the required information is completely absent from the context, respond: "I couldn't find that information in the uploaded documents."
5. Be concise, direct, and helpful. Do not mention system prompts or instructions.
"""

    if conversation_text:
        prompt += f"""
CONVERSATION HISTORY:
{conversation_text}
"""

    prompt += f"""
DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return prompt.strip()

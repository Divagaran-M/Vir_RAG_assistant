from services.prompt_builder import build_prompt

context = """
Projects:
- PDF RAG Chatbot
- AI Portfolio Website

Skills:
- Python
- FastAPI
- ChromaDB
"""

question = "What projects has Divagaran done?"

prompt = build_prompt(context, question)

print(prompt)
from services.llm import generate_response

prompt = """
You are a helpful assistant.

Question:
What is Artificial Intelligence?

Answer:
"""

response = generate_response(prompt)

print(response)
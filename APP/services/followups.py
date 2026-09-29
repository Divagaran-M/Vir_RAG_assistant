from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

# we might need to change this 
def generate_followup_questions(question, answer):

    prompt = f"""
You are an expert AI document assistant.

The user asked:

{question}

The assistant answered:

{answer}

Generate exactly THREE intelligent follow-up questions.

Rules:

- Continue the conversation naturally.
- Questions must relate to the previous answer.
- Questions should be answerable using the uploaded document.
- Maximum 10 words.
- No numbering.
- One question per line.
- Return questions only.
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

    questions = response.choices[0].message.content.strip().split("\n")

    questions = [
        q.strip("- ").strip()
        for q in questions
        if q.strip()
    ]


    # Keep only the first four valid questions
    questions = questions[:4]

    return questions



from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)


def generate_suggested_questions(text): #we need to change the generate the followup and change the prompt

    print("Generating suggested questions...")

    prompt = f"""
You are an expert AI document analyst.

A user has just uploaded a document.

First understand what type of document this is
(e.g. Resume, Research Paper, Report, Notes, Book, Manual, Legal Document, Invoice).

Then generate EXACTLY FOUR questions that would help a user understand and explore this document.

Requirements:

- Questions MUST be answerable using ONLY the uploaded document.
- Avoid generic questions.
- Never ask whether it is a research paper, technical document or PDF.
- Cover different aspects of the document.
- Generate:
  1. One summary question.
  2. One key concept or important information question.
  3. One deeper explanation question.
  4. One analytical or insight question.
- Make the questions natural and engaging.
- Maximum 10 words per question.
- One question per line.
- No numbering.
- No bullet points.
- Return ONLY the questions.

Document:

{text[:6000]}
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
        q.strip().lstrip("0123456789.-? ")
        for q in questions
        if q.strip()
    ]

    return questions[:4]

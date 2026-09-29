import json

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)


def analyze_document(text):

    print("Analyzing document...")

    prompt = f"""
You are an expert AI document analyst.

Analyze the uploaded document.

First identify its document type.

Choose EXACTLY ONE:

- Research Paper
- Resume
- Report
- Book
- Notes
- User Manual
- Legal Document
- Invoice
- Presentation
- Other

Then generate EXACTLY FOUR intelligent questions.

Requirements:

- Questions must be answerable using ONLY this document.
- Avoid generic questions.
- Cover different aspects.
- Maximum 10 words.
- No numbering.

Return ONLY valid JSON.

Example:

{{
  "document_type": "Research Paper",
  "suggested_questions": [
    "Summarize this paper.",
    "What problem does it solve?",
    "Explain the proposed method.",
    "What are the key contributions?"
  ]
}}

Document:

{text[:5000]}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(
        response.choices[0].message.content
    )

    print(result)

    return result

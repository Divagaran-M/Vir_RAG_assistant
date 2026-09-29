from fastapi import APIRouter
from pydantic import BaseModel

from services.vectordb import get_document_preview
from services.question_generator import generate_suggested_questions

router = APIRouter()


class SuggestionRequest(BaseModel):
    filename: str


@router.post("/suggestions") # we need to change this entirely
async def get_suggestions(request: SuggestionRequest):

    preview = get_document_preview(
        filename=request.filename,
        limit=5
    )

    questions = generate_suggested_questions(preview)

    return {
        "suggested_questions": questions
    }

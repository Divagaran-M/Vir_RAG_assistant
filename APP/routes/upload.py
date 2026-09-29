from fastapi import APIRouter, UploadFile, File

from services import embeddings
from services.validator import validate_file
from services.saver import save_file
from services.extractor import extract_text
from services.document_analyzer import analyze_document
from services.chunker import chunk_text
from services.embeddings import generate_document_embeddings
from services.vectordb import store_embeddings

router = APIRouter()


@router.post("/upload")



async def upload_document(file: UploadFile = File(...)):

    # ----------------------------------
    # Validate
    # ----------------------------------
    await validate_file(file)

    # ----------------------------------
    # Save File
    # ----------------------------------

    saved_path = await save_file(file)

    # ----------------------------------
    # Extract Pages
    # ----------------------------------

    pages = extract_text(saved_path)


    print(f"\nTotal Pages : {len(pages)}")

    # ----------------------------------
    # Merge Pages (Temporary)
    # ----------------------------------

    full_text = "\n\n".join(
        page["text"] for page in pages
    )

    # ----------------------------------
    # Analyze Document
    # ----------------------------------

    analysis = analyze_document(full_text)

    document_type = analysis["document_type"]
    suggested_questions = analysis["suggested_questions"]

    print("\n========== DOCUMENT ANALYSIS ==========\n")
    print(f"Document Type : {document_type}")
    print(f"Suggested Questions : {suggested_questions}")
    print("\n=======================================\n")

    # ----------------------------------
    # Chunk Text
    # ----------------------------------

    chunks = chunk_text(pages)

    print(f"Total Chunks : {len(chunks)}")

    # ----------------------------------
    # Generate Embeddings
    # ----------------------------------
    
    embeddings = generate_document_embeddings(
    [chunk["text"] for chunk in chunks]
    )

    print("Chunks before store:", len(chunks))
    print("Embeddings before store:", len(embeddings))

    # ----------------------------------
    # Store Embeddings
    # ----------------------------------

    total_stored = store_embeddings(
        chunks=chunks,
        embeddings=embeddings,
        filename=file.filename
    )


    # ----------------------------------
    # Response
    # ----------------------------------

    return {
        "success": True,
        "message": "Document indexed successfully.",
        "filename": file.filename,
        "document_type": document_type,
        "chunks_stored": total_stored,
        "suggested_questions": suggested_questions
    }


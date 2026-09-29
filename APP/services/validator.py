# services/validator.py

from fastapi import HTTPException, UploadFile
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE


async def validate_file(file: UploadFile):
    """
    Validate uploaded file.
    """

    # -------------------------
    # Validate File Extension
    # -------------------------
    extension = file.filename.split(".")[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # -------------------------
    # Validate File Size
    # -------------------------
    content = await file.read()

    if len(content) > MAX_FILE_SIZE: #200mb max
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 20 MB limit."
        )

    # -------------------------
    # Validate Empty File
    # -------------------------
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # Reset pointer after reading
    file.file.seek(0) # why do we need this line

    return True

# services/saver.py

import os
import shutil
import uuid

from fastapi import UploadFile


from config import UPLOAD_FOLDER


async def save_file(file: UploadFile):
    """
    Save the uploaded file to data/uploads
    """

    # Create uploads folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Generate unique filename
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"

    # Full file path
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path
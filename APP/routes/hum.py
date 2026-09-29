"""
routes/hum.py — HUM model file server

Serves the face-api.js model weight files from App/HUM/models/
so the browser component can load them without CORS issues.
"""

import os
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

HUM_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "HUM", "models")


@router.get("/hum/models/{filename}")
async def serve_hum_model(filename: str):
    """Serve face-api.js model weight files."""
    # Basic path-traversal guard
    safe_name = os.path.basename(filename)
    file_path = os.path.join(HUM_MODELS_DIR, safe_name)
    if not os.path.isfile(file_path):
        return JSONResponse({"error": f"Model file not found: {safe_name}"}, status_code=404)
    return FileResponse(file_path)

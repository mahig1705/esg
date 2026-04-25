"""
api/upload.py
-------------
POST /api/upload  — Accept PDF, TXT, DOCX, CSV files.
Returns file_id for inclusion in subsequent /api/analyse requests.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/api")

UPLOAD_DIR = Path(os.getenv("ESG_UPLOAD_DIR", "uploads"))
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts PDF, TXT, DOCX, CSV files.
    Saves to uploads/ folder and returns a file_id.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // 1024 // 1024} MB). Max 50 MB."
        )

    file_id = str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(save_path, "wb") as f:
        f.write(content)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "ext": ext,
        "status": "ready",
        "path": str(save_path),
    }

import logging
import sys
import json
import uuid
import shutil
import os
from pathlib import Path
from fastapi import APIRouter, Request, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from app.resources.database import Database
from app.models.query import Query
from app.resources.LLM import LLM

FILES_FOLDER = "files"

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

router = APIRouter()


@router.post("/upload_document")
async def upload_document(request: Request, document: UploadFile) -> JSONResponse:
    # Check file type
    if (filename := document.filename) is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Missing input document")
    if not filename.endswith("pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Only PDF files are supported")

    # Delete all existing files
    for path in Path(FILES_FOLDER).glob("*.pdf"):
        os.remove(path)

    # Persist incoming file
    document_path = f"{FILES_FOLDER}/{uuid.uuid4()}.pdf"
    with open(document_path, "wb") as file:
        shutil.copyfileobj(document.file, file)

    # Store in vector DB
    try:
        db: Database = request.app.db
        db.store(document_path)
        return JSONResponse(content="Document successfully received")
    except Exception as exc:
        logging.error(exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Could not store document in database")


@router.post("/related_excerpts")
async def query_document(request: Request, query: Query) -> list[str]:
    db: Database = request.app.db
    return db.retrieve(query.query)


@router.post("/query_document")
async def query_document(request: Request, query: Query) -> str:
    db: Database = request.app.db
    texts = db.retrieve(query.query)
    llm: LLM = request.app.llm
    result = llm.predict(query=query, texts=texts)
    return result

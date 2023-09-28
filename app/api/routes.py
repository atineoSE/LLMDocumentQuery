import logging
import sys
import json
from fastapi import APIRouter, Request, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from app.resources.database import Database
from app.models.query import Query
from app.resources.LLM import LLM

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

    try:
        db: Database = request.app.state.db
        db.store(document.file)
        return JSONResponse(content="Document successfully received")
    except Exception as exc:
        logging.error(exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Could not store document in database")


@router.post("/related_excerpts")
async def query_document(request: Request, query: Query) -> list[str]:
    db: Database = request.app.state.db
    return db.retrieve(query)


@router.post("/query_document")
async def query_document(request: Request, query: Query) -> str:
    db: Database = request.app.state.db
    texts = db.retrieve(query)
    llm: LLM = request.app.llm
    result = llm.predict(query=query.text, texts=texts)
    return result

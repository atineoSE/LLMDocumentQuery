import logging
import sys
import json
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse
from app.resources.database import Database

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

router = APIRouter()


@router.post("/upload_document/")
async def upload_document(request: Request, document_file: UploadFile):
    db: Database = request.app.db
    db.store(document_file.file)


@router.post("/related_excerpts")
async def query_document(request: Request, query: str) -> JSONResponse:
    db: Database = request.app.db
    results = db.retrieve(query)
    return JSONResponse(content=json.dump(results), status_code=200)

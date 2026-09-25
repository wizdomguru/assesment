from functools import lru_cache
import logging

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from app.config import get_settings
from app.models.api import DeleteResponse, IngestResponse, QueryRequest, QueryResponse
from app.services.application import ApplicationServices, build_application_services


logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {".md", ".markdown", ".pdf"}


@lru_cache
def get_services() -> ApplicationServices:
    return build_application_services(settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@app.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...), services: ApplicationServices = Depends(get_services)
) -> IngestResponse:
    filename = file.filename or ""
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only PDF and Markdown files are supported")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 10 MB limit")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        document_id, chunks_indexed = services.ingest(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document ingestion failed for filename=%s", filename)
        raise HTTPException(status_code=503, detail="Document ingestion failed") from exc
    return IngestResponse(document_id=document_id, filename=filename, chunks_indexed=chunks_indexed)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, services: ApplicationServices = Depends(get_services)) -> QueryResponse:
    try:
        llm_enabled = settings.llm_enabled if request.llm_enabled is None else request.llm_enabled
        result, cached, chunks = services.query(request.question, llm_enabled=llm_enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Question answering failed")
        raise HTTPException(status_code=503, detail="Question answering is unavailable") from exc
    return QueryResponse.from_result(
        result, chunks=chunks, cached=cached, llm_enabled=llm_enabled
    )


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(
    document_id: str, services: ApplicationServices = Depends(get_services)
) -> DeleteResponse:
    if not document_id.strip():
        raise HTTPException(status_code=400, detail="Document ID cannot be empty")
    try:
        services.delete(document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Document deletion failed") from exc
    return DeleteResponse(document_id=document_id)

import logging

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.services.documents import DocumentError, index_document
from app.services.rag import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    session_id: str = Field(min_length=1, max_length=256)


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description", min_length=1),
) -> dict[str, bool]:
    content = await file.read()
    try:
        await run_in_threadpool(
            index_document,
            filename=file.filename or "",
            content=content,
            description=description,
        )
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(
            status_code=502,
            detail="The document could not be indexed because an upstream service failed.",
        ) from exc
    return {"success": True}


@router.post("/query")
async def query(request: QueryRequest) -> dict[str, str]:
    question = request.query.strip()
    session_id = request.session_id.strip()
    if not question:
        raise HTTPException(status_code=422, detail="query must not be blank")
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id must not be blank")
    try:
        result = await run_in_threadpool(answer_question, question, session_id)
    except Exception as exc:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=502,
            detail="The question could not be answered because a configured service failed.",
        ) from exc
    return {"result": result}

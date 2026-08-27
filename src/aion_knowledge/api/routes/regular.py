"""文件上传和知识摄入端点。"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from aion_knowledge.common.config import settings
from aion_knowledge.common.suffix import infer_suffix
from aion_knowledge.infrastructure.db import get_db
from aion_knowledge.ingestion.kb_guard import KnowledgeBaseNotFoundError
from aion_knowledge.ingestion.postproc_rerun import (
    DocumentNotFoundError,
    ModuleValidationError,
    enqueue_postproc_rerun,
)
from aion_knowledge.ingestion.strategy.registry import get_strategy
from aion_knowledge.models.enums import StrategyName

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/knowledge/{kb_id}/documents/upload")
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    chunk_strategy: str = Form("auto"),
    creator: str = Form("system"),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """上传文件并触发知识摄入管道。"""
    content = await file.read()
    max_bytes = settings.upload_max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超过上限（{settings.upload_max_size_mb}MB）",
        )
    suffix = infer_suffix(filename=file.filename, content_type=file.content_type)

    if chunk_strategy != "no_split":
        chunk_strategy = "auto"

    strategy = get_strategy(StrategyName.regular, suffix=suffix)
    try:
        result = await strategy.execute(
            kb_id=kb_id,
            content=content,
            file_name=file.filename or "unnamed",
            creator=creator,
            chunk_strategy=chunk_strategy,
        )
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


class PostProcRerunRequest(BaseModel):
    """后处理重跑请求体。"""
    modules: list[str]


@router.post("/knowledge/{kb_id}/documents/{document_id}/postproc/run", status_code=201)
async def rerun_postproc(
    kb_id: str,
    document_id: str,
    body: PostProcRerunRequest,
) -> dict[str, Any]:
    """对已入库文档单独执行指定的二批后处理模块（异步入队，返回即成功）。

    Body: {"modules": ["raptor", "graph_extract"]}
    """
    try:
        task = await enqueue_postproc_rerun(kb_id, document_id, body.modules)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModuleValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "queued",
        "document_id": str(task.document_id),
        "modules": task.modules,
    }



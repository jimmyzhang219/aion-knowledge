"""FAQ 导入与管理 API 端点。"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from aion_knowledge.common.config import settings
from aion_knowledge.common.suffix import infer_suffix
from aion_knowledge.ingestion.kb_guard import KnowledgeBaseNotFoundError
from aion_knowledge.ingestion.strategy.registry import get_strategy
from aion_knowledge.models.enums import StrategyName

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["faq"])


@router.post("/knowledge/{kb_id}/faq/import")
async def faq_import(
    kb_id: str,
    file: UploadFile = File(...),
    mode: str = Form("append"),
    creator: str = Form("system"),
) -> dict[str, Any]:
    """批量导入 FAQ 条目。"""
    content = await file.read()
    max_bytes = settings.upload_max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超过上限（{settings.upload_max_size_mb}MB）",
        )
    suffix = infer_suffix(filename=file.filename, default="csv")
    file_name = f"faq_import.{suffix}"

    strategy = get_strategy(StrategyName.faq, suffix=suffix)
    try:
        result = await strategy.execute(
            kb_id=kb_id,
            content=content,
            file_name=file_name,
            creator=creator,
            mode=mode,
        )
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result

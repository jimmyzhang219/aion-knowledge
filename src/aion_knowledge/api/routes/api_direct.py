"""API Direct — 上游系统直接推送数据的 RESTful 端点。"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from aion_knowledge.ingestion.kb_guard import KnowledgeBaseNotFoundError
from aion_knowledge.ingestion.strategy.registry import get_strategy
from aion_knowledge.models.enums import StrategyName

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


class ApiDirectRequest(BaseModel):
    """上游系统直接推送数据的请求体。"""
    content: str            # 原始数据内容（文本/JSON 序列化等）
    doc_name: str           # 文档名称（含扩展名）
    suffix: str = "txt"     # 文档后缀（文件扩展名）
    chunk_strategy: str = "auto"


class ApiDirectResponse(BaseModel):
    """API Direct 接入响应。"""
    status: str
    context_id: str
    document_id: str


@router.post("/knowledge/{kb_id}/api-direct", response_model=ApiDirectResponse)
async def ingest_api_direct(kb_id: str, body: ApiDirectRequest) -> ApiDirectResponse:
    """上游系统直接推送数据接入知识库（无需文件上传）。

    Request body (JSON):
    - content: 原始数据内容，将写入 UnifiedContext.content
    - doc_name: 文档名称
    - suffix: 文档后缀（默认 txt）
    - chunk_strategy: 分块策略（默认 auto）
    """
    chunk_strategy = "no_split" if body.chunk_strategy == "no_split" else "auto"

    strategy = get_strategy(StrategyName.api_direct, suffix=body.suffix)
    try:
        result = await strategy.execute(
            kb_id=kb_id,
            content=body.content.encode("utf-8"),
            file_name=body.doc_name,
            chunk_strategy=chunk_strategy,
        )
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ApiDirectResponse(**result)

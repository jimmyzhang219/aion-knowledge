"""知识库（KnowledgeBase）CRUD 路由。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from aion_knowledge.common.trace import get_trace_id
from aion_knowledge.ingestion.deletion import delete_document, delete_kb
from aion_knowledge.models.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
)
from aion_knowledge.storage.relational.kb_repo import KnowledgeBaseRepo

router = APIRouter(prefix="/api/v1", tags=["knowledge-base"])


@router.post("/knowledge", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(body: KnowledgeBaseCreate) -> KnowledgeBaseResponse:
    repo = KnowledgeBaseRepo()
    kb = await repo.create(
        name=body.name,
        tags=body.tags,
        description=body.description,
        trace_id=get_trace_id(),
    )
    return KnowledgeBaseResponse.model_validate(kb)


@router.get("/knowledge", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases() -> KnowledgeBaseListResponse:
    repo = KnowledgeBaseRepo()
    items = await repo.list_all()
    return KnowledgeBaseListResponse(
        items=[KnowledgeBaseResponse.model_validate(kb) for kb in items],
        total=len(items),
    )


@router.get("/knowledge/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(kb_id: uuid.UUID) -> KnowledgeBaseResponse:
    repo = KnowledgeBaseRepo()
    kb = await repo.get_by_id(kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail=f"KnowledgeBase {kb_id} not found")
    return KnowledgeBaseResponse.model_validate(kb)


@router.delete("/knowledge/{kb_id}")
async def remove_knowledge_base(kb_id: uuid.UUID) -> dict[str, str | bool]:
    """删除整个知识库（逻辑删除；AION_DELETION_LOGICAL=false 时异步真实删除）。"""
    ok = await delete_kb(str(kb_id))
    if not ok:
        raise HTTPException(status_code=404, detail=f"KnowledgeBase {kb_id} not found")
    return {"kb_id": str(kb_id), "deleted": True}


@router.delete("/knowledge/{kb_id}/documents/{document_id}")
async def remove_document(kb_id: uuid.UUID, document_id: uuid.UUID) -> dict[str, str | bool]:
    """删除单个文档（逻辑删除；AION_DELETION_LOGICAL=false 时异步真实删除）。"""
    ok = await delete_document(str(kb_id), str(document_id))
    if not ok:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return {"document_id": str(document_id), "deleted": True}

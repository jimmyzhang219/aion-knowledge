"""文档与入库任务 DB 操作。

提供 KnowledgeDocument 和 IngestionTask 的创建/更新/查询。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aion_knowledge.models.orm import IngestionTask, KnowledgeDocument

logger = logging.getLogger(__name__)


async def create_document(
    session: AsyncSession,
    kb_id: str,
    doc_name: str,
    suffix: str,
    file_hash: str,
    size: int,
    chunk_strategy: str = "auto",
    creator: str = "system",
    file_path: str = "",
    trace_id: str | None = None,
) -> KnowledgeDocument:
    """创建文档记录，status 默认 pending。"""
    doc = KnowledgeDocument(
        kb_id=uuid.UUID(kb_id),
        doc_name=doc_name,
        suffix=suffix,
        hash=file_hash,
        size=size,
        chunk_strategy=chunk_strategy,
        creator=creator,
        file_path=file_path,
        trace_id=trace_id,
    )
    session.add(doc)
    await session.flush()
    logger.info("Document created: id=%s name=%s status=pending", doc.id, doc_name)
    return doc


async def update_document_status(
    session: AsyncSession,
    document_id: str | uuid.UUID,
    status: str,
) -> None:
    """更新文档处理状态。"""
    if isinstance(document_id, str):
        document_id = uuid.UUID(document_id)
    stmt = (
        update(KnowledgeDocument)
        .where(KnowledgeDocument.id == document_id)
        .values(status=status)
    )
    await session.execute(stmt)
    logger.info("Document status updated: id=%s status=%s", document_id, status)


async def create_ingestion_task(
    session: AsyncSession,
    document_id: str | uuid.UUID,
    pipeline_id: str = "pipeline_v1",
) -> IngestionTask:
    """创建入库任务记录，status 默认 pending。"""
    if isinstance(document_id, str):
        document_id = uuid.UUID(document_id)
    task = IngestionTask(
        document_id=document_id,
        pipeline_id=pipeline_id,
    )
    session.add(task)
    await session.flush()
    logger.info("IngestionTask created: id=%s doc=%s status=pending", task.id, document_id)
    return task


async def update_ingestion_task_status(
    session: AsyncSession,
    task_id: str | uuid.UUID,
    status: str,
    error_info: dict[str, Any] | None = None,
) -> None:
    """更新入库任务状态。"""
    if isinstance(task_id, str):
        task_id = uuid.UUID(task_id)
    values: dict[str, Any] = {"status": status}
    if error_info is not None:
        values["error_info"] = error_info
    stmt = update(IngestionTask).where(IngestionTask.id == task_id).values(**values)
    await session.execute(stmt)
    logger.info("IngestionTask status updated: id=%s status=%s", task_id, status)


async def get_document_by_hash(
    session: AsyncSession,
    kb_id: str,
    file_hash: str,
) -> KnowledgeDocument | None:
    """按 hash + kb_id 查重（仅未删除文档；同内容改名重传同样命中）。"""
    stmt = select(KnowledgeDocument).where(
        KnowledgeDocument.hash == file_hash,
        KnowledgeDocument.kb_id == uuid.UUID(kb_id),
        KnowledgeDocument.deleted == False,  # noqa: E712
    )
    result = await session.execute(stmt)
    # 查重只需"存在任一"：历史数据可能有多条未删除同 hash 行（旧规则遗留），
    # scalar_one_or_none 会抛 MultipleResultsFound，故取首条即可。
    return result.scalars().first()


async def get_document_by_id(
    session: AsyncSession,
    document_id: str | uuid.UUID,
) -> KnowledgeDocument | None:
    """按 ID 查询文档记录（仅未删除文档；已删视为不存在）。"""
    if isinstance(document_id, str):
        document_id = uuid.UUID(document_id)
    stmt = select(KnowledgeDocument).where(
        KnowledgeDocument.id == document_id,
        KnowledgeDocument.deleted == False,  # noqa: E712
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

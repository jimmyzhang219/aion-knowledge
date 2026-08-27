"""KnowledgeBaseRepo — KnowledgeBase 表的基本 CRUD 操作。"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.models.orm import KnowledgeBase


class KnowledgeBaseRepo:
    """知识库（KnowledgeBase）的数据库操作。"""

    async def create(
        self,
        name: str,
        tags: list[str] | None = None,
        description: str = "",
        trace_id: str | None = None,
    ) -> KnowledgeBase:
        async with get_session() as session:
            kb = KnowledgeBase(
                name=name,
                tags=tags or [],
                description=description,
                trace_id=trace_id,
            )
            session.add(kb)
            await session.flush()
            return kb

    async def list_all(self) -> list[KnowledgeBase]:
        async with get_session() as session:
            stmt = (
                select(KnowledgeBase)
                .where(KnowledgeBase.deleted == False)  # noqa: E712
                .order_by(KnowledgeBase.created_at.desc())
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_by_id(self, kb_id: uuid.UUID) -> KnowledgeBase | None:
        async with get_session() as session:
            stmt = select(KnowledgeBase).where(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.deleted == False,  # noqa: E712
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

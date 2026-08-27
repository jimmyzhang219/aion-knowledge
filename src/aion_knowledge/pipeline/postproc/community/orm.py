"""Community ORM — 表名 chunk_community。"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.models.orm import Base


class ChunkCommunity(Base):
    """社区发现结果 ORM：Leiden 社区分组与 LLM 报告的持久化存储。"""

    __tablename__ = "chunk_community"
    __table_args__ = (
        Index("idx_diskann_community_embedding", "embedding",
              postgresql_using="diskann",
              postgresql_ops={"embedding": "vector_cosine_ops"}),
        {"comment": "社区发现结果表"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7, comment="社区记录ID")
    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True, comment="关联切片ID")
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True, comment="所属知识库ID")
    community_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="社区ID")
    community_level: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="社区层级")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="", comment="社区摘要")
    findings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, comment="社区发现的关键发现")
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1024), nullable=True, comment="社区摘要向量（title+summary+findings 拼接，pgvector）"
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, comment="扩展元数据")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间")

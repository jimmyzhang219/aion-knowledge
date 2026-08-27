"""ChunkVector ORM — 表名 chunk_vector。

列定义对齐实际数据库 schema（含 embedding_questions / questions / embedding_summary）。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Index, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.models.orm import Base


class ChunkVector(Base):
    """向量嵌入元数据存储。"""

    __tablename__ = "chunk_vector"
    __table_args__ = (
        Index("idx_diskann_embedding", "embedding",
              postgresql_using="diskann",
              postgresql_ops={"embedding": "vector_cosine_ops"}),
        {"comment": "向量嵌入元数据表"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7, comment="切片向量ID")
    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True, comment="关联切片ID")
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True, comment="所属知识库ID")
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False, comment="向量嵌入（1024维，pgvector）")
    embedding_questions: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1024), nullable=True, comment="问题生成向量"
    )
    questions: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="原始问题文本（调试用）")
    embedding_summary: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1024), nullable=True, comment="摘要向量"
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict, comment="负载元数据（chunk_type/seq_num 等）")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间"
    )

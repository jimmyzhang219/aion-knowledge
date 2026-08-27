"""Disambiguation ORM — 表名 chunk_disambiguation。"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.models.orm import Base


class ChunkDisambiguation(Base):
    """实体消歧记录 ORM：记录实体消歧决策与合并历史。"""

    __tablename__ = "chunk_disambiguation"
    __table_args__ = {"comment": "实体消歧记录表"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7, comment="消歧记录ID")
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True, comment="关联切片ID（KB 级消歧决策为 NULL）")
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True, comment="所属知识库ID")
    entity_name: Mapped[str] = mapped_column(String(512), nullable=False, comment="实体名称")
    resolved_id: Mapped[str] = mapped_column(String(64), nullable=False, default="", comment="消歧后标准ID")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, comment="消歧置信度")
    merged_into: Mapped[str | None] = mapped_column(String(512), nullable=True, comment="合并到的目标实体名")
    merge_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, comment="合并置信度")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, comment="扩展元数据")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间")

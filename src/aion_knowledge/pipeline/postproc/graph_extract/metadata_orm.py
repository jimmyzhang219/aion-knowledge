"""GraphMetadata ORM — 表名 graph_metadata。"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.models.orm import Base


class GraphMetadata(Base):
    """知识图谱元数据 ORM：KB 级图谱统计与处理断点（checkpoints）存储。"""

    __tablename__ = "graph_metadata"
    __table_args__ = {"comment": "知识图谱元数据统计表"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7, comment="主键ID",
    )
    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True,
        comment="知识库ID（图谱与知识库 1:1，一个 KB 至多一个图）",
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active", comment="图谱状态")
    doc_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="关联文档数")
    entity_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="实体总数")
    relation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="关系总数")
    community_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="社区总数")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="版本号")
    checkpoints: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, comment="处理断点状态")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment="更新时间")

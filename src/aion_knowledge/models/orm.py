"""aion_knowledge 数据库的 SQLAlchemy ORM 模型。

所有表使用 ``knowledge`` 命名约定。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Enum, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.models.enums import (
    ChunkStrategy,
    DocumentStatus,
    IngestionTaskStatus,
)


class Base(DeclarativeBase):
    pass


class KnowledgeDocument(Base):
    __tablename__ = "doc_knowledge_documents"
    __table_args__ = {"comment": "文档元数据表"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7, comment="文档ID",
    )
    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True, comment="所属知识库ID",
    )
    doc_name: Mapped[str] = mapped_column(
        String(512), nullable=False, comment="文档名称（含扩展名）",
    )
    suffix: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", comment="文件后缀（pdf/docx/md 等）",
    )
    hash: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="文件 SHA256 哈希（去重用）",
    )
    size: Mapped[int] = mapped_column(Integer, nullable=False, comment="文件大小（字节）")
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), nullable=False, default=DocumentStatus.pending, comment="处理状态"
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list, comment="标签列表",
    )
    source_label: Mapped[str] = mapped_column(
        String(128), nullable=False, default="", comment="来源标签",
    )
    creator: Mapped[str] = mapped_column(String(128), nullable=False, comment="创建者")
    file_path: Mapped[str] = mapped_column(
        String(1024), nullable=False, default="", comment="对象存储路径",
    )
    chunk_strategy: Mapped[ChunkStrategy] = mapped_column(
        Enum(ChunkStrategy), nullable=False, default=ChunkStrategy.auto, comment="切片策略"
    )
    trace_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="创建请求 trace_id",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False, comment="更新时间",
    )
    deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        server_default="false", comment="逻辑删除标记",
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
        server_default="true", comment="是否启用（预留，检索路可用）",
    )



class IngestionTask(Base):
    __tablename__ = "task_ingestion_tasks"
    __table_args__ = {"comment": "文档处理任务表"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7, comment="任务ID",
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True, comment="关联文档ID",
    )
    pipeline_id: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="管道流水线标识",
    )
    status: Mapped[IngestionTaskStatus] = mapped_column(
        Enum(IngestionTaskStatus), nullable=False,
        default=IngestionTaskStatus.pending, comment="任务状态",
    )
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="重试次数",
    )
    checkpoint: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, comment="断点续传状态",
    )
    error_info: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, comment="错误信息",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
        nullable=False, comment="更新时间",
    )


class KnowledgeBase(Base):
    __tablename__ = "kb_knowledge_bases"
    __table_args__ = {"comment": "知识库信息表"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7, comment="知识库ID",
    )
    trace_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="创建请求 trace_id",
    )
    name: Mapped[str] = mapped_column(
        String(256), nullable=False, comment="知识库名称",
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list, comment="标签列表",
    )
    description: Mapped[str] = mapped_column(
        Text, nullable=False, default="", comment="知识库描述",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        nullable=False, comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        nullable=False, comment="更新时间",
    )
    deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        server_default="false", comment="逻辑删除标记",
    )

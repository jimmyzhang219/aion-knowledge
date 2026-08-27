"""ChunkText ORM — 表名 chunk_text。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.models.orm import Base


class ChunkText(Base):
    """明文 chunk + parent chunk 统一存储。"""

    __tablename__ = "chunk_text"
    __table_args__ = (
        Index("idx_bm25_content_tokens", "content_tokens",
              postgresql_using="bm25",
              postgresql_with={"text_config": "public.zh_cfg"},
              postgresql_ops={"content_tokens": "text_array_bm25_ops"}),
        Index("idx_bm25_summary_tokens", "summary_tokens",
              postgresql_using="bm25",
              postgresql_with={"text_config": "public.zh_cfg"},
              postgresql_ops={"summary_tokens": "text_array_bm25_ops"}),
        {"comment": "文档切片存储表（text/table/image/parent 等类型）"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7, comment="切片ID")
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True, comment="所属文档ID")
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True, comment="所属知识库ID")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="切片内容（Markdown / 图片上下文 / VLM 描述）")
    context_header: Mapped[str] = mapped_column(Text, nullable=False, default="", comment="上下文标题路径")
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list, comment="关键词列表")
    seq_num: Mapped[int] = mapped_column(Integer, nullable=False, comment="切片序号（文档内唯一）")
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=False, default="text", comment="切片类型：text/table/image/parent 等（见 ChunkType 枚举）")
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, comment="父切片ID（RAPTOR 层级检索用）")
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="Token 估算数")
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict, comment="元数据 JSON（table_caption/heading_path/context_above 等）")
    image_refs: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list, comment="关联图片 S3 路径列表")
    summary_text: Mapped[str] = mapped_column(Text, nullable=False, default="", comment="摘要文本")
    content_tokens: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, comment="内容 zhparser 分词结果"
    )
    summary_tokens: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, comment="摘要 zhparser 分词结果"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间"
    )

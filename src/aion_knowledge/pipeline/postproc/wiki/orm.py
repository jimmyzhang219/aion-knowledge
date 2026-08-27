"""Wiki ORM — 表名 chunk_wiki（KB 级页面池，一页引用多 chunk）。"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.models.orm import Base


class ChunkWiki(Base):
    """Wiki 百科页面 ORM：KB 级页面池，page_slug 在 KB 内唯一。"""

    __tablename__ = "chunk_wiki"
    __table_args__ = (
        UniqueConstraint("kb_id", "page_slug", name="uq_chunk_wiki_kb_slug"),
        {"comment": "百科数据表（KB 级页面池）"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7, comment="页面ID")
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True, comment="所属知识库ID")
    page_slug: Mapped[str] = mapped_column(String(512), nullable=False, comment="页面 slug（KB 内唯一）")
    page_title: Mapped[str] = mapped_column(String(512), nullable=False, default="", comment="页面标题")
    content: Mapped[str] = mapped_column(Text, nullable=False, default="", comment="页面内容（Markdown，含 [[slug]] wikilink）")
    chunk_refs: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list, comment="引用 chunk UUID 列表")
    source_refs: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list, comment="贡献文档 UUID 列表")
    out_links: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list, comment="本页链接出去的 slug 列表")
    in_links: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list, comment="反向链接（被哪些 slug 链向）")
    taxonomy_path: Mapped[str] = mapped_column(String(512), nullable=False, default="", comment="分类路径")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft", comment="状态（draft/published）")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, comment="扩展元数据")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")

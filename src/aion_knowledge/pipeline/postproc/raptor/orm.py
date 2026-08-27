"""Raptor ORM — 表名 chunk_raptor。

RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）
递归摘要树。每行可以是一条 per-summary 摘要节点，也可以是 tree 模式下
根节点的一整棵序列化树。

Embedding 直接存在本表（vector 列），不依赖 chunk_vector，做到自包含。
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.models.orm import Base


class ChunkRaptor(Base):
    """RAPTOR 递归摘要树节点 ORM：存储每层摘要节点或整棵序列化树。"""

    __tablename__ = "chunk_raptor"
    __table_args__ = (
        CheckConstraint("tree_builder IN ('raptor', 'psi')", name="valid_tree_builder"),
        CheckConstraint("clustering_method IN ('gmm', 'ahc')", name="valid_clustering_method"),
        CheckConstraint("output_mode IN ('flat', 'tree')", name="valid_output_mode"),
        {"comment": "RAPTOR 递归摘要树节点表"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7, comment="节点ID",
    )

    # —— 归属 ——
    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True, comment="所属知识库ID",
    )
    doc_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
        comment="文档ID（NULL 表示 dataset 级别）",
    )

    # —— 摘要内容 ——
    title: Mapped[str] = mapped_column(Text, nullable=False, default="", comment="摘要标题")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="", comment="摘要内容")

    # —— 树位置 ——
    layer: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="树层级（0=叶子，越大越靠近根）",
    )
    cluster_id: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", comment="所属聚类ID",
    )
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
        comment="父节点ID（NULL=根节点）",
    )
    children_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list,
        server_default="{}",
        comment="子节点ID列表（软聚类下子节点可属多个父，遍历下钻用）",
    )

    # —— 溯源 ——
    source_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list,
        comment="原始 chunk_text ID 列表（溯源链）",
    )

    # —— Tree 模式（仅 root 节点有值） ——
    tree_json: Mapped[Optional[dict[str, object]]] = mapped_column(
        JSONB, nullable=True,
        comment="整棵树序列化 dict（output_mode=tree 时使用）",
    )

    # —— Embedding ——
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1024), nullable=True,
        comment="摘要向量（直接存本表，不依赖 chunk_vector）",
    )

    # —— 算法元信息 ——
    tree_builder: Mapped[str] = mapped_column(
        String(10), nullable=False, default="raptor",
        comment="树构建算法（raptor）",
    )
    clustering_method: Mapped[str] = mapped_column(
        String(10), nullable=False, default="gmm",
        comment="聚类算法（gmm/ahc）",
    )
    output_mode: Mapped[str] = mapped_column(
        String(10), nullable=False, default="flat",
        comment="输出模式（flat/tree）",
    )

    # —— 扩展 ——
    payload: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict, comment="扩展元数据",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )

"""API 请求/响应校验的 Pydantic 模型。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from aion_knowledge.models.enums import ChunkStrategy

# ── 知识文档 ──

class KnowledgeDocumentCreate(BaseModel):
    kb_id: uuid.UUID
    doc_name: str = Field(..., max_length=512)
    suffix: str = Field(..., max_length=64)
    size: int = Field(..., ge=0)
    tags: list[str] = []
    source_label: str = ""
    creator: str = Field(..., max_length=128)
    file_path: str = ""
    content: bytes | None = None
    chunk_strategy: ChunkStrategy = ChunkStrategy.auto


class KnowledgeDocumentResponse(BaseModel):
    id: uuid.UUID
    kb_id: uuid.UUID
    doc_name: str
    suffix: str
    hash: str
    size: int
    status: str
    tags: list[str]
    source_label: str
    creator: str
    file_path: str
    created_at: datetime
    updated_at: datetime
    deleted: bool = False
    enabled: bool = True

    model_config = {"from_attributes": True}


# ── 切片 ──

class ChunkResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    chunk_type: str
    content: str
    context_header: str
    seq_num: int
    start_at: int
    end_at: int
    chunk_metadata: dict[str, Any]
    parent_chunk_id: uuid.UUID | None = None
    image_refs: list[str]

    model_config = {"from_attributes": True}


# ── 摄入任务 ──

class IngestionTaskResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    pipeline_id: str
    status: str
    retry_count: int
    error_info: dict[str, Any] | None

    model_config = {"from_attributes": True}


# ── 搜索 ──

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    kb_id: str = Field(..., description="知识库 ID")
    top_k: int = Field(default=10, ge=1, le=100)
    path_top_k: int = Field(default=20, ge=1, le=200, description="每路预取数")
    enabled_paths: list[str] | None = Field(default=None, description="启用路径列表，默认全部")
    generate_answer: bool = Field(default=False, description="是否 LLM 生成回答")
    stream: bool = Field(default=False, description="是否流式输出")

    model_config = {"extra": "forbid"}


class SearchResultItem(BaseModel):
    """单条搜索结果。"""
    chunk_id: str
    document_id: str
    content: str
    score: float
    source_paths: list[str] = Field(default_factory=list)
    chunk_type: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class PathStat(BaseModel):
    """单路检索统计。"""
    results: int = 0          # 该路返回结果数
    in_final: int = 0         # 该路在最终 top_k 中的命中数


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    answer: str | None = None
    source_breakdown: dict[str, int] = Field(default_factory=dict)
    path_stats: dict[str, PathStat] = Field(default_factory=dict)
    total_fused: int = 0
    query: str


# ── 知识库 ──


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    tags: list[str] = Field(default_factory=list)
    description: str = ""


class KnowledgeBaseResponse(BaseModel):
    id: uuid.UUID
    name: str
    tags: list[str]
    description: str
    created_at: datetime
    updated_at: datetime
    deleted: bool = False

    model_config = {"from_attributes": True}


class KnowledgeBaseListResponse(BaseModel):
    items: list[KnowledgeBaseResponse]
    total: int

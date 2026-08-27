"""事件驱动插件链中使用的事件结构。

定义了在摄入和检索管道执行期间通过 EventManager 总线传递的数据负载。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class DocumentIngested:
    """文档完成摄入管道时发布。"""
    document_id: uuid.UUID
    kb_id: uuid.UUID
    status: str
    error: str | None = None


@dataclass
class ChunkIndexed:
    """切片写入数据库和向量存储后发布。"""
    document_id: uuid.UUID
    chunk_count: int


@dataclass
class GraphUpdated:
    """文档的图提取完成时发布。"""
    document_id: uuid.UUID
    entity_count: int
    relationship_count: int


@dataclass
class SearchCompleted:
    """搜索管道完成时发布。"""
    session_id: str
    query: str
    result_count: int
    elapsed_ms: int

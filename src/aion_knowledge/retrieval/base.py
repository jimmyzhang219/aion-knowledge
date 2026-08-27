"""多路召回基础抽象类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkResult:
    """多路召回统一输出格式。"""
    chunk_id: str
    kb_id: str
    document_id: str
    content: str
    score: float = 0.0
    source_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_type: str = "text"


@dataclass
class RetrieverContext:
    """检索上下文，贯穿所有检索路径。"""
    query: str
    kb_id: str
    original_query: str | None = None
    top_k: int = 20
    enabled_paths: set[str] | None = None
    # 以下由调用方填充后传入
    query_embedding: list[float] | None = None
    expansion_keywords: list[str] | None = None
    entities: list[str] | None = None


class BaseRetriever(ABC):
    """检索器抽象基类。

    每个子类实现一种检索路径，name 作为唯一标识参与 RRF 权重匹配。
    """

    name: str = ""
    weight: float = 0.0

    @abstractmethod
    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        """执行检索，返回 ChunkResult 列表（按 score 降序）。"""

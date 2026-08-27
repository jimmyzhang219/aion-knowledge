"""Embedding 基础设施模块 — 统一 Embedding 调用层。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Embedding 生成器统一接口。

    所有 Embedding provider 实现均需满足此 Protocol。
    """

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本 embedding。

        Args:
            texts: 待编码的文本列表。

        Returns:
            每个文本对应的向量列表。
        """

    async def embed_query(self, text: str) -> list[float]:
        """生成单个查询 embedding。

        Args:
            text: 查询文本。

        Returns:
            查询向量。
        """


__all__ = ["Embedder", "create_embedder"]


def create_embedder() -> Embedder:
    """快捷导入，实际实现在 factory.py。"""
    from aion_knowledge.infrastructure.embedder.factory import create_embedder as _cf
    return _cf()

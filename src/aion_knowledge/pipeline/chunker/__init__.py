"""Chunker —— 文档分块模块。

提供统一的分块入口 ``Chunker``，支持多种分块策略自动适配或手动指定。
策略选择逻辑见 ``chunker.py`` 的 ``_select_splitters`` 方法。
"""

from aion_knowledge.pipeline.chunker.base import ChunkConfig, ChunkResult
from aion_knowledge.pipeline.chunker.chunker import Chunker

__all__ = [
    "ChunkConfig",
    "ChunkResult",
    "Chunker",
]

"""Embedder 工厂 —— 根据配置创建统一的 Embedder 实例。

当前仅支持 Ollama，未来在此处添加新 provider 的 match 分支。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aion_knowledge.infrastructure.embedder import Embedder

from aion_knowledge.common.config import settings


def create_embedder() -> Embedder:
    """根据 settings 创建嵌入器实例。

    Returns:
        满足 Embedder Protocol 的实例。

    Raises:
        ValueError: 配置不完整（如 base_url 为空）。
    """
    if not settings.embedding_base_url:
        raise ValueError("embedding_base_url is not configured")

    from aion_knowledge.infrastructure.embedder.ollama import OllamaEmbeddings

    return OllamaEmbeddings()

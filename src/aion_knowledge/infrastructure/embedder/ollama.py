"""Ollama Embedding 实现 —— 直接调用 Ollama 原生 ``/api/embed`` API。

使用 ``httpx`` 直连 Ollama，避免 ``langchain-openai`` 的 tokenize 不兼容问题。

环境变量（通过 ``settings`` 读取）：
    - ``AION_EMBEDDING_BASE_URL``: Ollama 服务地址，如 ``http://localhost:11434``
    - ``AION_EMBEDDING_MODEL``: 模型名，如 ``bge-m3``
    - ``AION_EMBEDDING_DIMENSIONS``: 向量维度，如 ``1024``
"""

from __future__ import annotations

from typing import cast

import httpx

from aion_knowledge.common.config import settings


class OllamaEmbeddings:
    """Ollama Embedding 封装。

    直接调用 Ollama 原生 ``/api/embed`` API，支持批量输入。
    """

    def __init__(self) -> None:
        if not settings.embedding_base_url:
            raise ValueError("embedding_base_url is not configured")

        self.model = settings.embedding_model
        self.base_url = settings.embedding_base_url.rstrip("/")
        self.dimensions = settings.embedding_dimensions
        self._client = httpx.AsyncClient(base_url=self.base_url, trust_env=False, timeout=300.0)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """将文档列表转为向量。

        Args:
            texts: 待编码的文本列表。

        Returns:
            每个文本对应的向量列表。
        """
        if not texts:
            return []

        resp = await self._client.post(
            "/api/embed",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        return cast(list[list[float]], resp.json()["embeddings"])

    async def embed_query(self, text: str) -> list[float]:
        """将单个查询文本转为向量。

        Args:
            text: 查询文本。

        Returns:
            查询向量。
        """
        if not text:
            return []

        resp = await self._client.post(
            "/api/embed",
            json={"model": self.model, "input": [text]},
        )
        resp.raise_for_status()
        return cast(list[float], resp.json()["embeddings"][0])

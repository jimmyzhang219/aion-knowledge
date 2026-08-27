"""向量检索引擎接口。"""

from __future__ import annotations

import abc
from typing import Any


class VectorStore(abc.ABC):
    @abc.abstractmethod
    async def upsert(self, entries: list[dict[str, Any]]) -> int:
        ...

    @abc.abstractmethod
    async def search(
        self, embedding: list[float], top_k: int, **filters: Any
    ) -> list[dict[str, Any]]:
        ...

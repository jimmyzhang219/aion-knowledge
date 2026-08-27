"""切片存储接口。"""

from __future__ import annotations

import abc
import uuid
from typing import Any


class ChunkStore(abc.ABC):
    @abc.abstractmethod
    async def bulk_insert(
        self,
        chunks: list[dict[str, Any]],
        document_id: str | None = None,
        kb_id: str | None = None,
    ) -> int:
        ...

    @abc.abstractmethod
    async def get_by_document(self, document_id: uuid.UUID) -> list[dict[str, Any]]:
        ...

    @abc.abstractmethod
    async def get_by_id(self, chunk_id: uuid.UUID) -> dict[str, Any] | None:
        ...

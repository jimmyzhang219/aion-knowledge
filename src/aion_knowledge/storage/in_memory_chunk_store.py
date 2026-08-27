"""InMemoryChunkStore — ChunkStore 的内存 dict 实现（开发/测试用）。"""

from __future__ import annotations

import uuid
from typing import Any

from aion_knowledge.storage.interfaces.chunk_store import ChunkStore


class InMemoryChunkStore(ChunkStore):
    """基于内存 dict 的 ChunkStore 实现，替代 PostgreSQL。

    生产环境中应替换为 PostgreSQL 实现的 ChunkStore。
    """

    def __init__(self) -> None:
        self._chunks: dict[str, list[dict[str, Any]]] = {}

    async def bulk_insert(
        self,
        chunks: list[dict[str, Any]],
        document_id: str | None = None,
        kb_id: str | None = None,
    ) -> int:
        for c in chunks:
            if document_id:
                c["document_id"] = document_id
            if kb_id:
                c["kb_id"] = kb_id
        if document_id:
            self._chunks[document_id] = chunks
        return len(chunks)

    async def get_by_document(self, document_id: uuid.UUID) -> list[dict[str, Any]]:
        return self._chunks.get(str(document_id), [])

    async def get_by_id(self, chunk_id: uuid.UUID) -> dict[str, Any] | None:
        for chunks in self._chunks.values():
            for c in chunks:
                if c.get("chunk_id") == str(chunk_id):
                    return c
        return None

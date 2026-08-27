"""文档元数据存储接口。"""

from __future__ import annotations

import abc
import uuid
from typing import Any


class DocumentStore(abc.ABC):
    @abc.abstractmethod
    async def create(
        self, kb_id: uuid.UUID, doc_name: str, suffix: str, **kwargs: Any
    ) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def get(self, doc_id: uuid.UUID) -> dict[str, Any] | None:
        ...

    @abc.abstractmethod
    async def update_status(self, doc_id: uuid.UUID, status: str) -> None:
        ...

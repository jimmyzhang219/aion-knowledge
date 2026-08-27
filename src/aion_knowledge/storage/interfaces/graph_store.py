"""图数据库接口。"""

from __future__ import annotations

import abc
from typing import Any


class GraphStore(abc.ABC):
    @abc.abstractmethod
    async def upsert_entity(self, label: str, name: str, **props: Any) -> str:
        ...

    @abc.abstractmethod
    async def upsert_relationship(
        self, src: str, dst: str, rel_type: str, **props: Any
    ) -> None:
        ...

    @abc.abstractmethod
    async def search_entities(self, query: str) -> list[dict[str, Any]]:
        ...

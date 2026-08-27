"""对象/文件存储接口（S3/OSS）。"""

from __future__ import annotations

import abc


class FileStore(abc.ABC):
    @abc.abstractmethod
    async def upload(self, key: str, data: bytes, content_type: str = "") -> str:
        ...

    @abc.abstractmethod
    async def download(self, key: str) -> bytes:
        ...

    @abc.abstractmethod
    async def delete(self, key: str) -> None:
        ...

    @abc.abstractmethod
    async def exists(self, key: str) -> bool:
        ...

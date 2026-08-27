"""StorageBackend — 对象存储抽象基类（S3/Minio/OSS/本地）。"""

from __future__ import annotations

import abc


class StorageBackend(abc.ABC):
    """对象存储统一接口。"""

    @abc.abstractmethod
    async def upload(self, key: str, data: bytes, content_type: str = "") -> str:
        """上传数据，返回引用路径。"""
        ...

    @abc.abstractmethod
    async def download(self, key: str) -> bytes:
        """下载数据。"""
        ...

    @abc.abstractmethod
    async def delete(self, key: str) -> None:
        """删除对象。"""
        ...

    @abc.abstractmethod
    async def exists(self, key: str) -> bool:
        """检查对象是否存在。"""
        ...

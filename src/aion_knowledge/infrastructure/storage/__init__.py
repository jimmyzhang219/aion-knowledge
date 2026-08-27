"""存储后端 — 统一文件存储接口。

用法：
    from aion_knowledge.infrastructure.storage import resolve_storage

    store = resolve_storage()
    url = await store.upload("path/to/file", data)
    data = await store.download(url)
"""

from aion_knowledge.common.config import settings
from aion_knowledge.infrastructure.storage.base import StorageBackend
from aion_knowledge.infrastructure.storage.local_file_store import LocalFileStore
from aion_knowledge.infrastructure.storage.s3_file_store import S3FileStore


def resolve_storage() -> StorageBackend:
    """根据 settings 返回对应的存储后端。

    - 配置了 ``s3_access_key`` → S3FileStore（兼容 Minio/OSS）
    - 未配置 → LocalFileStore（开发/测试用）
    """
    if settings.s3_access_key:
        return S3FileStore()
    return LocalFileStore()


__all__ = [
    "StorageBackend",
    "resolve_storage",
    "S3FileStore",
    "LocalFileStore",
]

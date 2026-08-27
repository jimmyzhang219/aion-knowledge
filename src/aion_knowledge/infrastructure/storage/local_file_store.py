"""LocalFileStore -- FileStore 的本地文件系统实现（开发/测试用）。"""

from __future__ import annotations

import logging
from pathlib import Path

from aion_knowledge.infrastructure.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class LocalFileStore(StorageBackend):
    """将文件存储到本地文件系统，替代 S3/MinIO。

    生产环境中应替换为 S3FileStore。
    """

    def __init__(self, base_dir: str = "/tmp/aion_storage") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    async def upload(self, key: str, data: bytes, content_type: str = "") -> str:
        full_path = self._base_dir / key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)
        logger.info(
            "LocalFileStore upload: %s (%d bytes, type=%s)",
            full_path, len(data), content_type,
        )
        return str(full_path)

    async def download(self, key: str) -> bytes:
        full_path = self._base_dir / key
        data = full_path.read_bytes()
        logger.info("LocalFileStore download: %s (%d bytes)", full_path, len(data))
        return data

    async def delete(self, key: str) -> None:
        full_path = self._base_dir / key
        if full_path.exists():
            full_path.unlink()
            logger.info("LocalFileStore delete: %s", full_path)

    async def exists(self, key: str) -> bool:
        full_path = self._base_dir / key
        return full_path.exists()

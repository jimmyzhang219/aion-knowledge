"""Downloader — 文档原始文件下载。

从文件引用（本地路径 / S3 URI / OSS URI）下载到本地临时文件，
供后续 Parser / Cleaner / Chunker 处理。
"""

from __future__ import annotations

import logging
import os

from aion_knowledge.infrastructure.storage import resolve_storage

logger = logging.getLogger(__name__)


class Downloader:
    """文件下载器。将远程文件引用解析为本地临时路径。"""

    async def download(self, file_ref: str, doc_name: str) -> str:
        """下载 raw 文件到本地临时路径。

        Args:
            file_ref: 文件引用（本地路径 / s3://... / oss://...）。
            doc_name: 文档名，用于生成临时文件名。

        Returns:
            本地临时文件路径。
        """
        if os.path.isfile(file_ref):
            logger.info("Download: local file %s", file_ref)
            return file_ref

        store = resolve_storage()
        data = await store.download(file_ref)

        tmp_path = f"/tmp/aion_{doc_name}"
        with open(tmp_path, "wb") as f:
            f.write(data)
        logger.info("Download: %s -> local %s (%d bytes)", file_ref, tmp_path, len(data))
        return tmp_path

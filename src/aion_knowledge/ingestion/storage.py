"""共享对象存储工具函数：文件 hash、S3/本地存储、存储路径提取。"""

from __future__ import annotations

import hashlib
import logging
import mimetypes

from aion_knowledge.infrastructure.storage import resolve_storage

logger = logging.getLogger(__name__)


def _infer_content_type(file_ext: str) -> str:
    """根据文件扩展名推断 MIME type，未知类型返回 application/octet-stream。"""
    return mimetypes.guess_type(f"file.{file_ext}")[0] or "application/octet-stream"


def hash_file(content: bytes) -> str:
    """计算文件 SHA256 哈希。"""
    return hashlib.sha256(content).hexdigest()


async def save_to_storage(content: bytes, key: str, content_type: str | None = None) -> str:
    """保存文件到对象存储（S3/本地），返回引用路径。

    Args:
        content: 文件字节内容。
        key: 存储路径（如 ``docs/xxx/original.pdf``）。
        content_type: MIME type，为 None 时从 key 扩展名推断。

    Returns:
        引用路径（如 ``s3://bucket/docs/xxx/original.pdf`` 或本地路径）。
    """
    if not content_type:
        ext = key.rsplit(".", 1)[-1] if "." in key else ""
        content_type = _infer_content_type(ext)

    store = resolve_storage()
    ref = await store.upload(key, content, content_type=content_type)
    logger.info("File saved to storage: %s (%d bytes)", ref, len(content))
    return ref


def extract_storage_key(ref: str) -> str:
    """从 S3 ref 或本地路径中提取对象存储 key。"""
    if ref.startswith("s3://"):
        # s3://bucket/docs/... → docs/...
        return ref.split("/", 3)[3]
    if ref.startswith("/tmp/aion_storage/"):
        return ref[len("/tmp/aion_storage/"):]
    return ref

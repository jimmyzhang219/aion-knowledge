"""S3FileStore — 基于 boto3 的标准 S3 协议实现。

兼容阿里云 OSS、MinIO 及所有 S3 兼容对象存储。
boto3 同步调用通过 run_in_executor 托管到线程池，不阻塞 asyncio 事件循环。
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial

import boto3
from botocore.config import Config as BotoConfig

from aion_knowledge.common.config import settings
from aion_knowledge.infrastructure.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class S3FileStore(StorageBackend):
    """标准 S3 协议对象存储实现。

    通过 settings 中的 S3 配置连接，兼容：
    - 阿里云 OSS（endpoint=oss-cn-{region}.aliyuncs.com）
    - MinIO（endpoint=http://localhost:9000）
    - AWS S3 及其他 S3 兼容服务
    """

    def __init__(self) -> None:
        self._bucket = settings.s3_bucket
        self._prefix = settings.s3_prefix

        boto_config = BotoConfig(
            region_name=settings.s3_region,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            s3={"addressing_style": settings.s3_addressing_style},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            config=boto_config,
        )
        self._loop = asyncio.get_event_loop()

    def _s3_key(self, key: str) -> str:
        """补全 prefix 前缀，构成完整 S3 key。"""
        return f"{self._prefix}/{key}" if self._prefix else key

    def _parse_s3_ref(self, ref: str) -> str:
        """从 s3://bucket/key 中提取 key 部分。"""
        return "/".join(ref.split("/")[3:])

    async def upload(self, key: str, data: bytes, content_type: str = "") -> str:
        s3_key = self._s3_key(key)
        extra = {"ContentType": content_type} if content_type else {}
        await self._loop.run_in_executor(
            None,
            partial(self._client.put_object, Bucket=self._bucket, Key=s3_key, Body=data, **extra),
        )
        ref = f"s3://{self._bucket}/{s3_key}"
        logger.info(
            "S3 upload: bucket=%s key=%s (%d bytes, type=%s)",
            self._bucket, s3_key, len(data), content_type or "application/octet-stream",
        )
        return ref

    async def download(self, key: str) -> bytes:
        """从 S3 下载对象。

        key 参数可以是完整 s3://bucket/key 引用，也可以是相对路径。
        """
        s3_key = self._parse_s3_ref(key) if key.startswith("s3://") else self._s3_key(key)
        resp = await self._loop.run_in_executor(
            None,
            partial(self._client.get_object, Bucket=self._bucket, Key=s3_key),
        )
        data: bytes = resp["Body"].read()
        logger.info("S3 download: bucket=%s key=%s size=%d bytes", self._bucket, s3_key, len(data))
        return data

    async def delete(self, key: str) -> None:
        s3_key = self._s3_key(key)
        await self._loop.run_in_executor(
            None,
            partial(self._client.delete_object, Bucket=self._bucket, Key=s3_key),
        )
        logger.info("S3 delete: bucket=%s key=%s", self._bucket, s3_key)

    async def exists(self, key: str) -> bool:
        s3_key = self._s3_key(key)
        try:
            await self._loop.run_in_executor(
                None,
                partial(self._client.head_object, Bucket=self._bucket, Key=s3_key),
            )
            return True
        except Exception:
            return False

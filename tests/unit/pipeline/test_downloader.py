"""Downloader 单元测试。"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.pipeline.downloader import Downloader


class TestDownloader:
    """验证 Downloader.download() 的两种路径。"""

    @pytest.mark.asyncio
    async def test_local_file_returns_directly(self):
        """本地文件应直接返回路径，不调用 resolve_storage。"""
        downloader = Downloader()
        result = await downloader.download(__file__, "test_file")
        assert result == __file__

    @pytest.mark.asyncio
    async def test_s3_ref_downloads_via_storage(self):
        """S3 引用应调用 resolve_storage().download() 并写入临时文件。"""
        mock_store = AsyncMock()
        mock_store.download.return_value = b"fake content"

        try:
            with patch(
                "aion_knowledge.pipeline.downloader.resolve_storage",
                return_value=mock_store,
            ):
                downloader = Downloader()
                result = await downloader.download(
                    "s3://bucket/path/to/doc.pdf", "test_s3"
                )

            assert result == "/tmp/aion_test_s3"
            assert os.path.isfile("/tmp/aion_test_s3")
            with open("/tmp/aion_test_s3", "rb") as f:
                assert f.read() == b"fake content"
        finally:
            if os.path.exists("/tmp/aion_test_s3"):
                os.remove("/tmp/aion_test_s3")

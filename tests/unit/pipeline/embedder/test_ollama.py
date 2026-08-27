"""Tests for OllamaEmbeddings."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from aion_knowledge.infrastructure.embedder.ollama import OllamaEmbeddings


@pytest.fixture
def mock_settings():
    """Mock Ollama embedding settings."""
    with patch("aion_knowledge.infrastructure.embedder.ollama.settings") as ms:
        ms.embedding_model = "bge-m3"
        ms.embedding_base_url = "http://localhost:11434"
        ms.embedding_dimensions = 1024
        yield ms


class TestOllamaEmbeddings:
    """OllamaEmbeddings 初始化与接口测试。"""

    def test_init_defaults(self, mock_settings):
        """从 settings 读取配置创建实例。"""
        emb = OllamaEmbeddings()
        assert emb.model == "bge-m3"
        assert emb.base_url == "http://localhost:11434"
        assert emb.dimensions == 1024

    def test_init_empty_base_url_raises(self):
        """base_url 为空时抛出 ValueError。"""
        with patch(
            "aion_knowledge.infrastructure.embedder.ollama.settings"
        ) as mock_settings:
            mock_settings.embedding_base_url = ""
            with pytest.raises(ValueError, match="embedding_base_url"):
                OllamaEmbeddings()

    def test_init_trailing_slash_stripped(self, mock_settings):
        """base_url 结尾的斜杠被去除。"""
        mock_settings.embedding_base_url = "http://localhost:11434/"
        emb = OllamaEmbeddings()
        assert emb.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_embed_documents_success(self, mock_settings):
        """embed_documents 成功返回向量。"""
        emb = OllamaEmbeddings()
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"model": "bge-m3", "embeddings": [[0.1, 0.2], [0.3, 0.4]]}

        with patch.object(emb._client, "post", AsyncMock(return_value=resp)):
            result = await emb.embed_documents(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_embed_documents_empty(self, mock_settings):
        """空列表返回空列表。"""
        emb = OllamaEmbeddings()
        result = await emb.embed_documents([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_query_success(self, mock_settings):
        """embed_query 成功返回向量。"""
        emb = OllamaEmbeddings()
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"model": "bge-m3", "embeddings": [[0.1, 0.2, 0.3]]}

        with patch.object(emb._client, "post", AsyncMock(return_value=resp)):
            result = await emb.embed_query("test")

        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_query_empty(self, mock_settings):
        """空字符串返回空列表。"""
        emb = OllamaEmbeddings()
        result = await emb.embed_query("")
        assert result == []

    @pytest.mark.asyncio
    async def test_http_error_propagated(self, mock_settings):
        """Ollama 服务错误时异常向上传播。"""
        emb = OllamaEmbeddings()
        resp = Mock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 error", request=Mock(), response=Mock(status_code=500)
        )

        with patch.object(emb._client, "post", AsyncMock(return_value=resp)):
            with pytest.raises(httpx.HTTPStatusError):
                await emb.embed_documents(["test"])

"""Tests for infrastructure/reranker/client.py — 真实 HTTP 调用路径。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aion_knowledge.infrastructure.reranker.client import rerank


class TestRerankerClient:
    """rerank() HTTP 客户端，覆盖两种响应格式 + 异常场景。"""

    # ── 成功场景 ──────────────────────────────────────────────

    def _mock_response(self, json_data, status_code=200):
        """构建一个同步响应 mock（resp.json() 是同步调用）。"""
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.raise_for_status = MagicMock()
        return resp

    async def test_returns_scores_list_response(self):
        """服务返回纯分数列表 [0.99, 0.13, ...]。"""
        mock_post = AsyncMock(return_value=self._mock_response([0.99, 0.13, 0.02]))

        with patch.object(httpx, "AsyncClient", return_value=AsyncMock()) as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.__aenter__.return_value.post = mock_post

            scores = await rerank("query", ["a", "b", "c"])

        assert scores == [0.99, 0.13, 0.02]
        # 验证请求体
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"] == {"query": "query", "texts": ["a", "b", "c"]}

    async def test_returns_object_list_response(self):
        """服务返回对象列表 [{"index":0,"score":0.99}, ...]（可能乱序）。"""
        mock_post = AsyncMock(return_value=self._mock_response([
            {"index": 2, "score": 0.02},
            {"index": 0, "score": 0.99},
            {"index": 1, "score": 0.13},
        ]))

        with patch.object(httpx, "AsyncClient", return_value=AsyncMock()) as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.__aenter__.return_value.post = mock_post

            scores = await rerank("query", ["a", "b", "c"])

        # 按 index 还原顺序
        assert scores == [0.99, 0.13, 0.02]

    async def test_has_results_key(self):
        """服务返回 {"results": [0.99, 0.13]} 格式。"""
        mock_post = AsyncMock(return_value=self._mock_response({"results": [0.99, 0.13]}))

        with patch.object(httpx, "AsyncClient", return_value=AsyncMock()) as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.__aenter__.return_value.post = mock_post

            scores = await rerank("query", ["a", "b"])
        assert scores == [0.99, 0.13]

    async def test_empty_texts_returns_zeros(self):
        """空 texts 不应发 HTTP 请求，直接返回全零。"""
        scores = await rerank("query", [])
        assert scores == []

    async def test_empty_query_returns_zeros(self):
        scores = await rerank("", ["a", "b"])
        assert scores == [0.0, 0.0]

    # ── 分批与截断 ───────────────────────────────────────────

    async def test_batched_all_scored(self, monkeypatch):
        """超过 max_batch 的候选分批发送，全部打分，无补零丢弃（并发，不依赖调用顺序）。"""
        monkeypatch.setattr("aion_knowledge.infrastructure.reranker.client.settings.reranker_max_batch", 2)
        monkeypatch.setattr("aion_knowledge.infrastructure.reranker.client.settings.reranker_max_concurrency", 4)

        def _by_len(*args, **kwargs):
            batch = kwargs["json"]["texts"]
            return self._mock_response([0.5 + 0.01 * i for i in range(len(batch))])
        mock_post = AsyncMock(side_effect=_by_len)

        with patch.object(httpx, "AsyncClient", return_value=AsyncMock()) as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.__aenter__.return_value.post = mock_post

            scores = await rerank("query", ["a", "b", "c", "d"])

        assert mock_post.call_count == 2
        assert len(scores) == 4 and all(s > 0 for s in scores)

    async def test_batch_preserves_global_order(self, monkeypatch):
        """跨批分数按原始候选顺序拼回（并发 gather + 全局 start 偏移，不依赖调用顺序）。"""
        monkeypatch.setattr("aion_knowledge.infrastructure.reranker.client.settings.reranker_max_batch", 2)
        monkeypatch.setattr("aion_knowledge.infrastructure.reranker.client.settings.reranker_max_concurrency", 4)

        score_map = {"a": 0.9, "b": 0.2, "c": 0.8, "d": 0.1, "e": 0.7}

        def _by_content(*args, **kwargs):
            batch = kwargs["json"]["texts"]
            entries = [{"index": i, "score": score_map[t]} for i, t in enumerate(batch)]
            entries.reverse()
            return self._mock_response(entries)
        mock_post = AsyncMock(side_effect=_by_content)

        with patch.object(httpx, "AsyncClient", return_value=AsyncMock()) as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.__aenter__.return_value.post = mock_post

            scores = await rerank("query", ["a", "b", "c", "d", "e"])

        assert scores == [0.9, 0.2, 0.8, 0.1, 0.7]

    async def test_token_truncation(self, monkeypatch):
        """超过 max_tokens 的文本按 ×1.5 字符防护截断（payload 防护，非 token 预算）。"""
        monkeypatch.setattr("aion_knowledge.infrastructure.reranker.client.settings.reranker_max_tokens", 2)

        mock_post = AsyncMock(return_value=self._mock_response([0.5]))

        with patch.object(httpx, "AsyncClient", return_value=AsyncMock()) as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.__aenter__.return_value.post = mock_post

            await rerank("query", ["a" * 20])

        sent_texts = mock_post.call_args.kwargs["json"]["texts"]
        # 2 tokens × 1.5 chars/token = 3 chars
        assert len(sent_texts[0]) == 3

    async def test_trust_env_disabled(self):
        """httpx.AsyncClient 必须禁用环境代理（trust_env=False），避免系统代理超时。"""
        with patch.object(httpx, "AsyncClient", return_value=AsyncMock()) as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.__aenter__.return_value.post = AsyncMock(
                return_value=self._mock_response([0.5])
            )
            await rerank("query", ["a"])

        _, kwargs = mock_client.call_args
        assert kwargs.get("trust_env") is False

    async def test_concurrency_one_degrades_to_serial(self, monkeypatch):
        """max_concurrency=1 时退化为串行，side_effect 顺序可用（验证 Semaphore 生效）。"""
        monkeypatch.setattr("aion_knowledge.infrastructure.reranker.client.settings.reranker_max_batch", 1)
        monkeypatch.setattr("aion_knowledge.infrastructure.reranker.client.settings.reranker_max_concurrency", 1)

        mock_post = AsyncMock(side_effect=[
            self._mock_response([0.1]),
            self._mock_response([0.2]),
            self._mock_response([0.3]),
        ])
        with patch.object(httpx, "AsyncClient", return_value=AsyncMock()) as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.__aenter__.return_value.post = mock_post
            scores = await rerank("query", ["a", "b", "c"])

        assert scores == [0.1, 0.2, 0.3]
        assert mock_post.call_count == 3

    # ── 异常场景 ──────────────────────────────────────────────

    async def test_http_error_raises(self):
        """HTTP 非 200 应抛出异常（由 httpx 的 raise_for_status 处理）。"""
        resp = self._mock_response([], status_code=503)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=httpx.Request("POST", "http://localhost/rerank"),
            response=httpx.Response(503),
        )
        mock_post = AsyncMock(return_value=resp)

        with patch.object(httpx, "AsyncClient", return_value=AsyncMock()) as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.__aenter__.return_value.post = mock_post

            with pytest.raises(httpx.HTTPStatusError):
                await rerank("query", ["a"])

    async def test_timeout_raises(self):
        """连接超时应抛出 httpx.TimeoutException。"""
        mock_post = AsyncMock()
        mock_post.side_effect = httpx.TimeoutException("Connection timeout")

        with patch.object(httpx, "AsyncClient", return_value=AsyncMock()) as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.__aenter__.return_value.post = mock_post

            with pytest.raises(httpx.TimeoutException):
                await rerank("query", ["a"])

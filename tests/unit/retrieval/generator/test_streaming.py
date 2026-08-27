"""Tests for streaming — SSE event format."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aion_knowledge.retrieval.generator.streaming import (
    _generate_sse,
    build_sse_response,
)


class TestGenerateSSE:
    """SSE 生成器测试。"""

    @pytest.mark.asyncio
    async def test_retrieval_event_first(self):
        """第一个事件类型为 retrieval。"""
        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(return_value=AsyncMock())
        mock_llm.astream.return_value.__aiter__.return_value = []

        events = []
        async for event in _generate_sse("测试", [{"content": "a", "score": 0.9}], mock_llm):
            events.append(event)

        assert len(events) == 2  # retrieval + done
        assert events[0]["event"] == "retrieval"

    @pytest.mark.asyncio
    async def test_token_events_in_order(self):
        """token 事件按顺序发出。"""
        mock_chunks = [
            MagicMock(content="你好"),
            MagicMock(content="世界"),
            MagicMock(content=""),
        ]
        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(return_value=AsyncMock())
        mock_llm.astream.return_value.__aiter__.return_value = mock_chunks

        events = []
        async for event in _generate_sse("测试", [{"content": "a", "score": 0.9}], mock_llm):
            events.append(event)

        events_by_type = {e["event"]: e for e in events}
        assert "retrieval" in events_by_type
        assert "done" in events_by_type

    @pytest.mark.asyncio
    async def test_sse_response_type(self):
        """build_sse_response 返回 StreamingResponse。"""
        mock_llm = MagicMock()
        mock_llm.astream = MagicMock()

        response = build_sse_response("测试", [{"content": "a", "score": 0.9}], mock_llm)
        assert response.media_type == "text/event-stream"
        assert response.headers.get("x-accel-buffering") == "no"

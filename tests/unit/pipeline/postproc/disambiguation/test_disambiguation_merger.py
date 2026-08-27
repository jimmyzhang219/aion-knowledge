"""Disambiguation 图节点合并测试（mock DB 调用，验证合并逻辑）。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.pipeline.postproc.disambiguation.merger import DisambiguationMerger

KB_UUID = "00000000-0000-0000-0000-000000000002"


class TestDisambiguationMerger:
    @pytest.mark.asyncio
    async def test_merge_entities(self):
        merger = DisambiguationMerger()
        canonical = "Apple Inc."
        aliases = ["Apple", "App1e"]
        kb_id = "kb_test"

        merger._record_merge_history = AsyncMock()

        with patch("aion_knowledge.infrastructure.graph.merge_aliases", AsyncMock(return_value=None)):
            await merger.merge_entities(canonical, aliases, kb_id)

        merger._record_merge_history.assert_awaited_once_with(canonical, aliases, kb_id)

    @pytest.mark.asyncio
    async def test_merge_empty_aliases(self):
        merger = DisambiguationMerger()
        merger._record_merge_history = AsyncMock()

        await merger.merge_entities("Apple Inc.", [], "kb_test")

        merger._record_merge_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_merge(self):
        merger = DisambiguationMerger()
        merger.merge_entities = AsyncMock()
        groups = [
            ("Apple Inc.", ["Apple", "App1e"]),
            ("Microsoft Corp.", ["Microsoft", "MS"]),
        ]
        await merger.batch_merge(groups, "kb_test")
        assert merger.merge_entities.await_count == 2

    @pytest.mark.asyncio
    async def test_merge_entities_writes_null_chunk_id(self):
        """KB 级合并历史应写 chunk_id=None（而非零值 UUID）。"""
        merger = DisambiguationMerger()
        session = AsyncMock()
        session.add = MagicMock(return_value=None)  # add 同步调用，避免未 await 的 coroutine

        @asynccontextmanager
        async def mock_get_session():
            yield session

        with (
            patch("aion_knowledge.infrastructure.graph.merge_aliases", AsyncMock()) as mock_merge,
            patch("aion_knowledge.pipeline.postproc.disambiguation.merger.get_session", mock_get_session),
        ):
            await merger.merge_entities("Apple", ["App1e"], KB_UUID)

        mock_merge.assert_awaited_once_with(KB_UUID, "Apple", ["App1e"])
        assert session.add.call_count == 1
        added = session.add.call_args[0][0]
        assert added.chunk_id is None
        assert added.kb_id is not None

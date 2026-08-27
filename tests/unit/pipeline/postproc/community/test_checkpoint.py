"""Community 检查点逻辑测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from aion_knowledge.pipeline.postproc.community.checkpoint import (
    CommunityCheckpointManager,
    compute_graph_hash,
)


class TestGraphHash:
    def test_identical_entities_same_hash(self):
        h1 = compute_graph_hash(["A", "B"], [("A", "B", "link")])
        h2 = compute_graph_hash(["A", "B"], [("A", "B", "link")])
        assert h1 == h2

    def test_different_order_same_hash(self):
        h1 = compute_graph_hash(["B", "A"], [("B", "A", "link")])
        h2 = compute_graph_hash(["A", "B"], [("A", "B", "link")])
        assert h1 == h2

    def test_different_entities_different_hash(self):
        h1 = compute_graph_hash(["A", "B"], [("A", "B", "link")])
        h2 = compute_graph_hash(["A", "C"], [("A", "C", "link")])
        assert h1 != h2


class TestCommunityCheckpointManager:
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        mgr = CommunityCheckpointManager()
        h = compute_graph_hash(["A", "B"], [("A", "B", "link")])
        with patch("aion_knowledge.pipeline.postproc.community.checkpoint.get_session") as mock_gs:
            mock_session = AsyncMock()
            mock_execute = AsyncMock()
            mock_session.execute = mock_execute
            mock_gs.return_value.__aenter__.return_value = mock_session
            await mgr.save("kb_test", h)

            # verify execute was called
            assert mock_execute.await_count >= 1

    @pytest.mark.asyncio
    async def test_load_missing(self):
        mgr = CommunityCheckpointManager()
        with patch("aion_knowledge.pipeline.postproc.community.checkpoint.get_session") as mock_gs:
            mock_session = AsyncMock()
            mock_result = Mock()
            mock_result.scalar.return_value = None

            async def mock_execute(*args, **kwargs):
                return mock_result
            mock_session.execute = mock_execute
            mock_gs.return_value.__aenter__.return_value = mock_session
            result = await mgr.load("kb_nonexistent")
            assert result is None

    @pytest.mark.asyncio
    async def test_should_skip_true_when_same_hash(self):
        mgr = CommunityCheckpointManager()
        h = compute_graph_hash(["A", "B"], [("A", "B", "link")])
        with patch.object(mgr, "load", AsyncMock(return_value=h)):
            with patch.object(mgr, "_count_community_reports", AsyncMock(return_value=5)):
                assert await mgr.should_skip("kb_test", h) is True

    @pytest.mark.asyncio
    async def test_should_skip_false_when_different_hash(self):
        mgr = CommunityCheckpointManager()
        h1 = compute_graph_hash(["A", "B"], [("A", "B", "link")])
        h2 = compute_graph_hash(["A", "C"], [("A", "C", "link")])
        with patch.object(mgr, "load", AsyncMock(return_value=h1)):
            assert await mgr.should_skip("kb_test", h2) is False

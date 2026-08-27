"""IndexingExecutor 编排流程测试。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.indexing.executor import IndexingExecutor


class TestIndexingExecutor:
    """验证 IndexingExecutor 编排流程的正确性。"""

    @pytest.mark.asyncio
    async def test_run_calls_strategy_then_first_batch(self):
        """run() 应调用 strategy.execute() → _run_first_batch() → _mark_completed()。"""
        executor = IndexingExecutor()

        mock_strategy = AsyncMock()
        mock_strategy.execute.return_value = [{"seq_num": 0, "content": "test"}]
        mock_strategy.__class__.__name__ = "MockStrategy"

        ctx = MagicMock()
        ctx.source = "regular"
        ctx.ext_metadata = {}

        with (
            patch.object(executor, '_get_strategy', return_value=mock_strategy),
            patch.object(executor, '_run_first_batch', AsyncMock()) as mock_first_batch,
            patch.object(executor, '_mark_completed', AsyncMock()) as mock_mark,
            patch.object(executor, '_build_postproc_task', return_value="task"),
        ):
            result = await executor.run(ctx)

        mock_strategy.execute.assert_awaited_once_with(ctx)
        mock_first_batch.assert_awaited_once_with(ctx, None, [{"seq_num": 0, "content": "test"}])
        mock_mark.assert_awaited_once()
        assert result == "task"

    @pytest.mark.asyncio
    async def test_run_empty_chunks_skips_first_batch(self):
        """如果策略返回空列表，则跳过 _run_first_batch。"""
        executor = IndexingExecutor()

        mock_strategy = AsyncMock()
        mock_strategy.execute.return_value = []

        ctx = MagicMock()
        ctx.source = "regular"
        ctx.ext_metadata = {}

        with (
            patch.object(executor, '_get_strategy', return_value=mock_strategy),
            patch.object(executor, '_run_first_batch', AsyncMock()) as mock_first_batch,
            patch.object(executor, '_mark_completed', AsyncMock()),
            patch.object(executor, '_build_postproc_task', return_value="task"),
        ):
            result = await executor.run(ctx)

        mock_first_batch.assert_not_called()
        assert result == "task"

    def test_build_postproc_task_carries_trace_id(self):
        """PostProcTask.trace_id 显式穿透 UnifiedContext.trace_id（Queue2 关联）。"""
        from aion_knowledge.infrastructure.models import UnifiedContext

        executor = IndexingExecutor()
        ctx = UnifiedContext(
            source="regular", kb_id="kb-1", doc_name="d.md", suffix="md",
            original_file_ref="s3://x",
            ext_metadata={"document_id": "doc-1"},
            trace_id="trace-xyz",
        )
        task = executor._build_postproc_task(ctx, [{"seq_num": 0}])
        assert task.trace_id == "trace-xyz"

    @pytest.mark.asyncio
    async def test_get_strategy_uses_registry(self):
        """_get_strategy 应通过注册表获取策略。"""
        from aion_knowledge.indexing.strategy.base import ChunkingStrategy
        from aion_knowledge.indexing.strategy.registry import _registry, register_strategy

        saved = _registry.copy()
        _registry.clear()
        try:
            @register_strategy("test_source")
            class TestStrategy(ChunkingStrategy):
                strategy_key = "test_source"
                async def execute(self, ctx):
                    return []

            executor = IndexingExecutor()
            ctx = MagicMock()
            ctx.source = "test_source"

            strategy = executor._get_strategy(ctx)
            assert isinstance(strategy, TestStrategy)
        finally:
            _registry.clear()
            _registry.update(saved)

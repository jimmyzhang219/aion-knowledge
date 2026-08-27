"""MultiRetriever 单元测试。"""
from __future__ import annotations

import asyncio

from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext
from aion_knowledge.retrieval.orchestrator.multi_retriever import MultiRetriever


class FakeRetriever(BaseRetriever):
    def __init__(self, name: str, weight: float, delay: float = 0):
        self.name = name
        self.weight = weight
        self._delay = delay

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        if self._delay:
            await asyncio.sleep(self._delay)
        return [ChunkResult(
            chunk_id=f"{self.name}_1", kb_id=ctx.kb_id, document_id="",
            content=f"From {self.name}", score=1.0, source_paths=[self.name],
        )]


class FailingRetriever(BaseRetriever):
    name = "failing"
    weight = 0.1

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        raise RuntimeError("Simulated failure")


class SlowRetriever(BaseRetriever):
    name = "slow"
    weight = 0.1

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        await asyncio.sleep(10)  # 超时
        return []


class TestMultiRetriever:
    def test_all_paths_concurrent(self):
        """多路并发，结果应包含所有路径的输出。"""
        mr = MultiRetriever(timeout=5.0)
        mr.register(FakeRetriever("a", 0.5))
        mr.register(FakeRetriever("b", 0.5))
        ctx = RetrieverContext(query="test", kb_id="kb1", top_k=10)
        results = asyncio.run(mr.search(ctx))
        assert "a" in results
        assert "b" in results
        assert len(results["a"]) == 1
        assert len(results["b"]) == 1

    def test_single_path_failure_does_not_block_others(self):
        """单路失败不影响其他路。"""
        mr = MultiRetriever(timeout=5.0)
        mr.register(FakeRetriever("good", 0.5))
        mr.register(FailingRetriever())
        ctx = RetrieverContext(query="test", kb_id="kb1", top_k=10)
        results = asyncio.run(mr.search(ctx))
        assert "good" in results
        assert len(results["good"]) == 1
        assert "failing" not in results  # 失败的路径不出现

    def test_timeout_skips_path(self):
        """超时路径被跳过，不阻塞其他路。"""
        mr = MultiRetriever(timeout=0.5)
        mr.register(FakeRetriever("fast", 0.5))
        mr.register(SlowRetriever())
        ctx = RetrieverContext(query="test", kb_id="kb1", top_k=10)
        results = asyncio.run(mr.search(ctx))
        assert "fast" in results
        assert "slow" not in results

    def test_enabled_paths_filtering(self):
        """enabled_paths 仅执行指定的路径。"""
        mr = MultiRetriever(timeout=5.0)
        mr.register(FakeRetriever("keep", 0.5))
        mr.register(FakeRetriever("skip", 0.5))
        ctx = RetrieverContext(query="test", kb_id="kb1", top_k=10, enabled_paths={"keep"})
        results = asyncio.run(mr.search(ctx))
        assert "keep" in results
        assert "skip" not in results

    def test_empty_paths(self):
        """无可用路径返回空 dict。"""
        mr = MultiRetriever(timeout=5.0)
        ctx = RetrieverContext(query="test", kb_id="kb1", top_k=10)
        results = asyncio.run(mr.search(ctx))
        assert results == {}

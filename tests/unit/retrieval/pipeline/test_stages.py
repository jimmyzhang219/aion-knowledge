"""Tests for retrieval pipeline stages."""
from __future__ import annotations

import pytest

from aion_knowledge.retrieval.base import ChunkResult, RetrieverContext
from aion_knowledge.retrieval.orchestrator.rrf_fuser import RRFFuser
from aion_knowledge.retrieval.pipeline import PipelineContext
from aion_knowledge.retrieval.pipeline.stages import (
    MultiRecallStage,
    RRFFusionStage,
    TopKTruncateStage,
)


class _FakeMultiRetriever:
    def __init__(self, results: dict[str, list[ChunkResult]]):
        self._results = results

    async def search(self, ctx: RetrieverContext) -> dict[str, list[ChunkResult]]:
        return self._results


class _CapturingMultiRetriever:
    def __init__(self) -> None:
        self.captured: RetrieverContext | None = None

    async def search(self, ctx: RetrieverContext) -> dict[str, list[ChunkResult]]:
        self.captured = ctx
        return {}


class TestMultiRecallStage:
    async def test_sets_path_results(self):
        fake = _FakeMultiRetriever({
            "bm25": [ChunkResult(chunk_id="c1", kb_id="kb1", document_id="d1", content="x", score=0.9)],
        })
        stage = MultiRecallStage(fake)
        ctx = PipelineContext(query="q", kb_id="kb1")
        await stage.run(ctx)
        assert "bm25" in ctx.path_results
        assert len(ctx.path_results["bm25"]) == 1
        assert ctx.path_results["bm25"][0].chunk_id == "c1"

    async def test_propagates_original_query(self):
        fake = _CapturingMultiRetriever()
        stage = MultiRecallStage(fake)
        ctx = PipelineContext(query="改写后", kb_id="kb1", original_query="原始查询")
        await stage.run(ctx)
        assert fake.captured is not None
        assert fake.captured.original_query == "原始查询"


class TestRRFFusionStage:
    def _make_rrf_fuser(self, weights: dict[str, float] | None = None, k: int = 60):
        return RRFFuser(k=k, weights=weights)

    async def test_fuses_and_filters(self):
        fuser = self._make_rrf_fuser({"bm25": 1.0, "vector": 1.0})
        stage = RRFFusionStage(fuser)
        ctx = PipelineContext(query="q", kb_id="kb1", top_k=5)
        ctx.path_results = {
            "bm25": [ChunkResult(chunk_id="c1", kb_id="kb1", document_id="d1", content="a", score=0.9)],
            "vector": [ChunkResult(chunk_id="c2", kb_id="kb1", document_id="d2", content="b", score=0.8)],
        }
        await stage.run(ctx)
        assert len(ctx.results) == 2
        assert ctx.results[0].score > 0

    async def test_empty_path_results(self):
        fuser = self._make_rrf_fuser()
        stage = RRFFusionStage(fuser)
        ctx = PipelineContext(query="q", kb_id="kb1")
        ctx.path_results = {}
        await stage.run(ctx)
        assert ctx.results == []

    async def test_threshold_filter(self):
        # 使用小 k 值让 RRF 分数有足够分散度，低排名项会被阈值过滤
        fuser = self._make_rrf_fuser({"path": 1.0}, k=1)
        stage = RRFFusionStage(fuser)
        ctx = PipelineContext(query="q", kb_id="kb1", top_k=10)
        ctx.path_results = {
            "path": [
                ChunkResult(chunk_id=f"c{i}", kb_id="kb1", document_id="d1",
                            content=f"doc{i}", score=1.0 - i * 0.1)
                for i in range(10)
            ],
        }
        await stage.run(ctx)
        # RRF(k=1) 分数：rank 1-10 → 1/2~1/11 ≈ [0.5~0.091]
        # 阈值 30% → max*0.3=0.15，rank 6+ (score≤0.143) 被过滤
        assert len(ctx.results) == 5


class TestTopKTruncateStage:
    async def test_truncates_to_top_k(self):
        stage = TopKTruncateStage()
        ctx = PipelineContext(query="q", kb_id="kb1", top_k=3)
        ctx.results = [
            ChunkResult(chunk_id=f"c{i}", kb_id="kb1", document_id="d1", content=f"x{i}")
            for i in range(10)
        ]
        await stage.run(ctx)
        assert len(ctx.results) == 3

    async def test_empty_results(self):
        stage = TopKTruncateStage()
        ctx = PipelineContext(query="q", kb_id="kb1", top_k=5)
        ctx.results = []
        await stage.run(ctx)
        assert ctx.results == []


# ── RerankerStage ──


class TestRerankerStage:
    async def test_skip_when_disabled(self, monkeypatch):
        from aion_knowledge.retrieval.pipeline.stages.reranker import RerankerStage
        monkeypatch.setattr("aion_knowledge.common.config.settings.reranker_enabled", False)
        stage = RerankerStage()
        ctx = PipelineContext(query="q", kb_id="kb1")
        ctx.results = [ChunkResult(chunk_id="c1", kb_id="kb1", document_id="d1", content="x", score=0.5)]
        await stage.run(ctx)
        assert ctx.results[0].score == 0.5  # 未修改

    async def test_skip_when_no_query(self):
        from aion_knowledge.retrieval.pipeline.stages.reranker import RerankerStage
        stage = RerankerStage()
        ctx = PipelineContext(query="", kb_id="kb1")
        ctx.results = [ChunkResult(chunk_id="c1", kb_id="kb1", document_id="d1", content="x", score=0.5)]
        await stage.run(ctx)
        assert ctx.results[0].score == 0.5

    async def test_skip_when_no_results(self):
        from aion_knowledge.retrieval.pipeline.stages.reranker import RerankerStage
        stage = RerankerStage()
        ctx = PipelineContext(query="q", kb_id="kb1")
        ctx.results = []
        await stage.run(ctx)
        assert ctx.results == []

    # ── 实际调用路径 ──────────────────────────────────────────

    async def test_applies_scores_and_reorders(self, monkeypatch):
        """reranker 返回分数后，score 被替换，并按新分数降序排列。"""
        async def fake_rerank(query: str, texts: list[str]) -> list[float]:
            # "c" 最相关 → 排第一
            return [0.1, 0.9, 0.5]

        monkeypatch.setattr(
            "aion_knowledge.infrastructure.reranker.client.rerank",
            fake_rerank,
        )
        from aion_knowledge.retrieval.pipeline.stages.reranker import RerankerStage
        stage = RerankerStage()
        ctx = PipelineContext(query="test", kb_id="kb1")
        ctx.results = [
            ChunkResult(chunk_id="c1", kb_id="kb1", document_id="d1", content="a", score=0.5),
            ChunkResult(chunk_id="c2", kb_id="kb1", document_id="d1", content="b", score=0.6),
            ChunkResult(chunk_id="c3", kb_id="kb1", document_id="d1", content="c", score=0.4),
        ]
        await stage.run(ctx)

        # score 已被替换
        scores = [r.score for r in ctx.results]
        assert scores == pytest.approx([0.9, 0.5, 0.1])
        # 按 reranker 分数降序排列
        assert [r.chunk_id for r in ctx.results] == ["c2", "c3", "c1"]

    async def test_reranker_error_does_not_modify_scores(self, monkeypatch):
        """reranker 抛出异常时 scores 保持不变（except Exception: pass）。"""
        async def fake_rerank(query: str, texts: list[str]) -> list[float]:
            msg = "Service unavailable"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "aion_knowledge.infrastructure.reranker.client.rerank",
            fake_rerank,
        )
        from aion_knowledge.retrieval.pipeline.stages.reranker import RerankerStage
        stage = RerankerStage()
        ctx = PipelineContext(query="test", kb_id="kb1")
        ctx.results = [
            ChunkResult(chunk_id="c1", kb_id="kb1", document_id="d1", content="a", score=0.5),
            ChunkResult(chunk_id="c2", kb_id="kb1", document_id="d1", content="b", score=0.6),
        ]
        await stage.run(ctx)

        # 异常被 except Exception: pass 吃掉，分数不变
        assert [r.score for r in ctx.results] == [0.5, 0.6]
        assert [r.chunk_id for r in ctx.results] == ["c1", "c2"]

    async def test_reranker_error_logs_warning(self, monkeypatch, caplog):
        """reranker 调用失败时应记录 WARNING 日志（含异常原因），便于排查，且不中断主线。"""
        import logging

        async def fake_rerank(query: str, texts: list[str]) -> list[float]:
            raise RuntimeError("connection refused")

        monkeypatch.setattr(
            "aion_knowledge.infrastructure.reranker.client.rerank",
            fake_rerank,
        )
        from aion_knowledge.retrieval.pipeline.stages.reranker import RerankerStage
        stage = RerankerStage()
        ctx = PipelineContext(query="test", kb_id="kb1")
        ctx.results = [
            ChunkResult(chunk_id="c1", kb_id="kb1", document_id="d1", content="a", score=0.5),
        ]
        with caplog.at_level(logging.WARNING, logger="aion_knowledge.retrieval.pipeline.stages.reranker"):
            await stage.run(ctx)

        # 分数保持 RRF 原始值（降级语义不变）
        assert ctx.results[0].score == 0.5
        # 有 WARNING 日志且包含异常信息
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "reranker 失败时应记录 WARNING 日志"
        assert any("connection refused" in (w.message or "") for w in warnings)


# ── MMRStage ──


class TestMMRStage:
    async def test_skip_when_no_results(self):
        from aion_knowledge.retrieval.pipeline.stages.mmr import MMRStage
        stage = MMRStage()
        ctx = PipelineContext(query="q", kb_id="kb1")
        ctx.results = []
        await stage.run(ctx)
        assert ctx.results == []

class TestRewriteStageSkip:
    """RewriteStage 在 pipeline 中配置关闭时的行为测试。"""

    async def test_skip_when_disabled(self, monkeypatch):
        """query_rewrite_enabled=False 时 RewriteStage 跳过。"""
        from aion_knowledge.retrieval.pipeline.stages.rewrite_stage import RewriteStage
        monkeypatch.setattr("aion_knowledge.common.config.settings.query_rewrite_enabled", False)
        stage = RewriteStage(llm_client=None, enabled=False)
        ctx = PipelineContext(query="原始查询", kb_id="kb1")
        await stage.run(ctx)
        assert ctx.query == "原始查询"
        assert ctx.expansion_keywords is None

    async def test_skip_when_no_llm(self):
        """llm_client=None 时 RewriteStage 跳过。"""
        from aion_knowledge.retrieval.pipeline.stages.rewrite_stage import RewriteStage
        stage = RewriteStage(llm_client=None, enabled=True)
        ctx = PipelineContext(query="原始查询", kb_id="kb1")
        await stage.run(ctx)
        assert ctx.query == "原始查询"

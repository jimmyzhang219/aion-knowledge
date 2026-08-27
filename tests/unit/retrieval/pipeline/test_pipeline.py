"""Tests for RetrievalPipeline orchestration."""
from __future__ import annotations

from aion_knowledge.retrieval.base import ChunkResult
from aion_knowledge.retrieval.pipeline import PipelineContext, RetrievalPipeline, Stage


class _OrderRecorder:
    def __init__(self) -> None:
        self.order: list[str] = []


class _RecordStage(Stage):
    def __init__(self, name: str, recorder: _OrderRecorder):
        self._name = name
        self._recorder = recorder

    async def run(self, ctx: PipelineContext) -> None:
        self._recorder.order.append(self._name)


class TestRetrievalPipeline:
    async def test_runs_stages_in_order(self):
        recorder = _OrderRecorder()
        stages = [
            _RecordStage("A", recorder),
            _RecordStage("B", recorder),
            _RecordStage("C", recorder),
        ]
        pipeline = RetrievalPipeline(stages)
        ctx = PipelineContext(query="test", kb_id="kb1")
        await pipeline.run(ctx)
        assert recorder.order == ["A", "B", "C"]

    async def test_stage_receives_context(self):
        class CheckStage(Stage):
            async def run(self, ctx: PipelineContext) -> None:
                assert ctx.query == "hello"
                assert ctx.kb_id == "kb1"
                assert ctx.top_k == 5

        pipeline = RetrievalPipeline([CheckStage()])
        ctx = PipelineContext(query="hello", kb_id="kb1", top_k=5)
        await pipeline.run(ctx)

    async def test_stage_can_modify_results(self):
        class AddResultStage(Stage):
            async def run(self, ctx: PipelineContext) -> None:
                ctx.results = [ChunkResult(chunk_id="c1", kb_id="kb1", document_id="d1", content="x")]

        pipeline = RetrievalPipeline([AddResultStage()])
        ctx = PipelineContext(query="q", kb_id="kb1")
        await pipeline.run(ctx)
        assert len(ctx.results) == 1
        assert ctx.results[0].chunk_id == "c1"

    async def test_empty_stages(self):
        pipeline = RetrievalPipeline([])
        ctx = PipelineContext(query="q", kb_id="kb1")
        await pipeline.run(ctx)
        assert ctx.results == []


class TestRewriteStageIntegration:
    """RewriteStage 在 Pipeline 中的集成测试。"""

    async def test_rewrite_then_multi_recall_uses_rewritten_query(self):
        """RewriteStage 改写 query 后，后续 Stage 收到的 ctx.query 已改写。"""
        from aion_knowledge.retrieval.pipeline.stages.rewrite_stage import RewriteStage

        class _MockLLM:
            async def generate_structured(self, **kwargs):
                return {
                    "rewritten_query": "RAG 架构 传统搜索 区别",
                    "keywords": ["RAG", "检索增强生成", "传统搜索"],
                    "entities": [],
                }

        class _CheckStage(Stage):
            async def run(self, ctx: PipelineContext) -> None:
                assert ctx.query == "RAG 架构 传统搜索 区别"
                assert ctx.expansion_keywords == ["RAG", "检索增强生成", "传统搜索"]

        pipeline = RetrievalPipeline([RewriteStage(llm_client=_MockLLM(), enabled=True), _CheckStage()])
        ctx = PipelineContext(query="请问RAG和传统搜索有啥区别", kb_id="kb1")
        await pipeline.run(ctx)

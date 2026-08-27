"""Tests for RewriteStage."""
from __future__ import annotations

from aion_knowledge.retrieval.pipeline import PipelineContext
from aion_knowledge.retrieval.pipeline.stages.rewrite_stage import RewriteStage


class _FakeLLMClient:
    """模拟 LLMClient，返回预设的 JSON 响应。"""
    def __init__(self, response: dict | None = None):
        self._response = response
        self.last_prompt = None
        self.last_system_prompt = None

    async def generate_structured(
        self,
        prompt: str,
        output_schema: dict | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return self._response or {
            "rewritten_query": "RAG 架构 传统搜索 区别",
            "keywords": ["RAG", "检索增强生成", "传统搜索"],
            "entities": [],
        }


class TestRewriteStage:
    async def test_rewrites_query_and_sets_keywords(self):
        llm = _FakeLLMClient()
        stage = RewriteStage(llm_client=llm, enabled=True)
        ctx = PipelineContext(query="请问RAG和传统搜索有啥区别", kb_id="kb1")
        await stage.run(ctx)
        assert ctx.query == "RAG 架构 传统搜索 区别"
        assert ctx.expansion_keywords == ["RAG", "检索增强生成", "传统搜索"]

    async def test_preserves_original_query(self):
        llm = _FakeLLMClient()
        stage = RewriteStage(llm_client=llm, enabled=True)
        ctx = PipelineContext(query="请问RAG和传统搜索有啥区别", kb_id="kb1")
        await stage.run(ctx)
        assert ctx.original_query == "请问RAG和传统搜索有啥区别"
        assert ctx.query == "RAG 架构 传统搜索 区别"

    async def test_preserves_original_when_disabled(self):
        llm = _FakeLLMClient()
        stage = RewriteStage(llm_client=llm, enabled=False)
        ctx = PipelineContext(query="原始查询", kb_id="kb1")
        await stage.run(ctx)
        assert ctx.query == "原始查询"
        assert ctx.expansion_keywords is None

    async def test_preserves_original_when_no_llm(self):
        stage = RewriteStage(llm_client=None, enabled=True)
        ctx = PipelineContext(query="原始查询", kb_id="kb1")
        await stage.run(ctx)
        assert ctx.query == "原始查询"
        assert ctx.expansion_keywords is None

    async def test_preserves_original_on_llm_error(self):
        class _BrokenLLM:
            async def generate_structured(self, **kwargs):
                raise RuntimeError("LLM unavailable")

        stage = RewriteStage(llm_client=_BrokenLLM(), enabled=True)
        ctx = PipelineContext(query="原始查询", kb_id="kb1")
        await stage.run(ctx)
        assert ctx.query == "原始查询"
        assert ctx.expansion_keywords is None

    async def test_preserves_original_on_missing_fields(self):
        llm = _FakeLLMClient(response={"rewritten_query": "只改写了查询"})
        stage = RewriteStage(llm_client=llm, enabled=True)
        ctx = PipelineContext(query="原始查询", kb_id="kb1")
        await stage.run(ctx)
        assert ctx.query == "只改写了查询"
        assert ctx.expansion_keywords is None

    async def test_handles_entities_field(self):
        llm = _FakeLLMClient(response={
            "rewritten_query": "SpaceX 发射 时间",
            "keywords": ["SpaceX", "火箭发射"],
            "entities": ["SpaceX"],
        })
        stage = RewriteStage(llm_client=llm, enabled=True)
        ctx = PipelineContext(query="SpaceX什么时候发射", kb_id="kb1")
        await stage.run(ctx)
        assert ctx.query == "SpaceX 发射 时间"
        assert ctx.expansion_keywords == ["SpaceX", "火箭发射"]
        assert ctx.entities == ["SpaceX"]

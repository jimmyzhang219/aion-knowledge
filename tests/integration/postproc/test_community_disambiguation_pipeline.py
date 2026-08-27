"""Community + Disambiguation 管道集成测试（mock LLM，验证模块间交互）。"""
from __future__ import annotations

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule
from aion_knowledge.pipeline.postproc.dispatcher import PostProcDispatcher


class TestCommunityDisambiguationPipeline:
    """验证依赖链 graph_extract → disambiguation → community 正确执行。"""

    @pytest.mark.asyncio
    async def test_dependency_order(self):
        """验证调度器按正确顺序执行三个模块。"""
        d = PostProcDispatcher({})
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="test.md")
        chunks = [{"chunk_uuid": "c1", "content": "test"}]

        execution_order: list[str] = []

        class MockGraph(PostProcModule):
            always_on = False
            depends_on = ["text"]
            async def process(self, ctx, chunks):
                execution_order.append("graph_extract")
                return 2

        class MockDisambig(PostProcModule):
            always_on = False
            depends_on = ["text", "graph_extract"]
            async def process(self, ctx, chunks):
                execution_order.append("disambiguation")
                return 1

        class MockCommunity(PostProcModule):
            always_on = False
            depends_on = ["text", "disambiguation"]
            async def process(self, ctx, chunks):
                execution_order.append("community")
                return 3

        d._modules = {
            "graph_extract": MockGraph(),
            "disambiguation": MockDisambig(),
            "community": MockCommunity(),
        }
        d._settings = {
            "graph_extract": True,
            "disambiguation": True,
            "community": True,
        }

        await d.run_second_batch(ctx, chunks)
        assert execution_order == ["graph_extract", "disambiguation", "community"]

    @pytest.mark.asyncio
    async def test_failure_does_not_block_upstream(self):
        """验证一个模块失败后，后续批次模块仍可运行。"""
        d = PostProcDispatcher({})
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="test.md")
        chunks = [{"chunk_uuid": "c1", "content": "test"}]

        class FailingModule(PostProcModule):
            always_on = False
            depends_on = []
            async def process(self, ctx, chunks):
                raise RuntimeError("Intentional failure")

        class NormalModule(PostProcModule):
            always_on = False
            depends_on = []
            async def process(self, ctx, chunks):
                return 1

        d._modules = {"failing": FailingModule(), "normal": NormalModule()}
        d._settings = {"failing": True, "normal": True}

        # 不应抛出异常
        await d.run_second_batch(ctx, chunks)

    def test_real_module_topological_order(self):
        """使用真实模块验证拓扑顺序。"""
        d = PostProcDispatcher({})
        from aion_knowledge.pipeline.postproc.community.processor import CommunityModule
        from aion_knowledge.pipeline.postproc.disambiguation.processor import DisambiguationModule
        from aion_knowledge.pipeline.postproc.graph_extract.processor import GraphExtractModule

        modules = [
            ("graph_extract", GraphExtractModule()),
            ("disambiguation", DisambiguationModule()),
            ("community", CommunityModule()),
        ]
        batches = d._topological_sort(modules)
        all_sorted = [n for batch in batches for n, _ in batch]
        assert all_sorted.index("graph_extract") < all_sorted.index("disambiguation")
        assert all_sorted.index("disambiguation") < all_sorted.index("community")

    @pytest.mark.asyncio
    async def test_disabled_module_skipped(self):
        """未启用的模块应被跳过，不影响已启用的模块。"""
        d = PostProcDispatcher({})
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="test.md")
        chunks = [{"chunk_uuid": "c1", "content": "test"}]

        executed: list[str] = []

        class ModA(PostProcModule):
            always_on = False
            depends_on = []
            async def process(self, ctx, chunks):
                executed.append("a")
                return 1

        class ModB(PostProcModule):
            always_on = False
            depends_on = []
            async def process(self, ctx, chunks):
                executed.append("b")
                return 1

        d._modules = {"mod_a": ModA(), "mod_b": ModB()}
        d._settings = {"mod_a": True, "mod_b": False}  # B disabled

        await d.run_second_batch(ctx, chunks)
        assert executed == ["a"]

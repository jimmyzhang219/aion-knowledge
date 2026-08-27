"""PostProcDispatcher 拓扑排序测试。"""
from __future__ import annotations

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcModule
from aion_knowledge.pipeline.postproc.dispatcher import PostProcDispatcher


class FakeModuleA(PostProcModule):
    always_on = False
    depends_on = []
    async def process(self, ctx, chunks): return 1

class FakeModuleB(PostProcModule):
    always_on = False
    depends_on = ["module_a"]
    async def process(self, ctx, chunks): return 1

class FakeModuleC(PostProcModule):
    always_on = False
    depends_on = ["module_b"]
    async def process(self, ctx, chunks): return 1

class FakeModuleD(PostProcModule):
    always_on = False
    depends_on = []       # 与 A 同级
    async def process(self, ctx, chunks): return 1


class TestTopologicalSort:
    def test_simple_chain(self):
        """A → B → C 链式依赖应分为 3 批。"""
        d = PostProcDispatcher({})
        modules = [
            ("module_a", FakeModuleA()),
            ("module_b", FakeModuleB()),
            ("module_c", FakeModuleC()),
        ]
        batches = d._topological_sort(modules)
        assert len(batches) == 3
        assert len(batches[0]) == 1 and batches[0][0][0] == "module_a"
        assert len(batches[1]) == 1 and batches[1][0][0] == "module_b"
        assert len(batches[2]) == 1 and batches[2][0][0] == "module_c"

    def test_parallel_levels(self):
        """A 和 D（均无依赖）应在同一批次并发。"""
        d = PostProcDispatcher({})
        modules = [
            ("module_a", FakeModuleA()),
            ("module_d", FakeModuleD()),
            ("module_b", FakeModuleB()),
        ]
        batches = d._topological_sort(modules)
        assert len(batches[0]) == 2
        names_0 = {n for n, _ in batches[0]}
        assert names_0 == {"module_a", "module_d"}

    def test_diamond_dependency(self):
        """A → {B, C} → D 菱形依赖：B 和 C 并发，D 在最后。"""
        class ModuleE(PostProcModule):
            always_on = False
            depends_on = ["module_a"]
            async def process(self, ctx, chunks): return 1

        class ModuleF(PostProcModule):
            always_on = False
            depends_on = ["module_e", "module_b"]
            async def process(self, ctx, chunks): return 1

        d = PostProcDispatcher({})
        modules = [
            ("module_a", FakeModuleA()),
            ("module_b", FakeModuleB()),
            ("module_e", ModuleE()),
            ("module_f", ModuleF()),
        ]
        batches = d._topological_sort(modules)
        # batch 0: A
        # batch 1: B, E (B depends on A, E depends on A)
        # batch 2: F (F depends on B and E)
        assert len(batches) == 3
        assert [n for n, _ in batches[0]] == ["module_a"]
        assert {n for n, _ in batches[1]} == {"module_b", "module_e"}
        assert [n for n, _ in batches[2]] == ["module_f"]

    def test_circular_dependency_raises(self):
        """循环依赖应抛出 ValueError。"""
        class ModCircularB(PostProcModule):
            always_on = False
            depends_on = ["mod_circular_c"]
            async def process(self, ctx, chunks): return 1

        class ModCircularC(PostProcModule):
            always_on = False
            depends_on = ["mod_circular_b"]
            async def process(self, ctx, chunks): return 1

        d = PostProcDispatcher({})
        modules = [
            ("mod_circular_b", ModCircularB()),
            ("mod_circular_c", ModCircularC()),
        ]
        with pytest.raises(ValueError, match="Circular dependency"):
            d._topological_sort(modules)

    def test_disabled_deps_ignored(self):
        """依赖的模块不在启用列表中时应忽略（不影响排序）。"""
        class ModuleWithMissingDep(PostProcModule):
            always_on = False
            depends_on = ["not_enabled"]
            async def process(self, ctx, chunks): return 1

        d = PostProcDispatcher({})
        modules = [("module_a", FakeModuleA()), ("test_mod", ModuleWithMissingDep())]
        batches = d._topological_sort(modules)
        assert len(batches) == 1  # 都在同一层（因为 not_enabled 不在列表中）
        assert len(batches[0]) == 2

    def test_empty_modules(self):
        d = PostProcDispatcher({})
        assert d._topological_sort([]) == []


class TestRunSecondBatchIntegration:
    """验证 run_second_batch 按 DAG 顺序执行模块。"""

    def test_execution_order_chain(self):
        """A → B → C，验证 run_second_batch 的执行顺序。"""
        execution_order = []

        class LogA(PostProcModule):
            always_on = False
            depends_on = []
            async def process(self, ctx, chunks):
                execution_order.append("A")
                return 1

        class LogB(PostProcModule):
            always_on = False
            depends_on = ["log_a"]
            async def process(self, ctx, chunks):
                execution_order.append("B")
                return 1

        class LogC(PostProcModule):
            always_on = False
            depends_on = ["log_b"]
            async def process(self, ctx, chunks):
                execution_order.append("C")
                return 1

        import asyncio

        from aion_knowledge.pipeline.postproc.base import PostProcContext

        d = PostProcDispatcher({"log_a": True, "log_b": True, "log_c": True})
        d._modules = {"log_a": LogA(), "log_b": LogB(), "log_c": LogC()}
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="test.md")

        asyncio.run(d.run_second_batch(ctx, [{"chunk_uuid": "c1", "content": "test"}]))
        assert execution_order == ["A", "B", "C"]

    def test_module_failure_caught(self):
        """模块失败应通过日志捕获，不阻断其他模块。"""
        class FailingModule(PostProcModule):
            always_on = False
            depends_on = []
            async def process(self, ctx, chunks):
                raise RuntimeError("intentional failure")

        class NormalModule(PostProcModule):
            always_on = False
            depends_on = []
            async def process(self, ctx, chunks):
                return 1

        import asyncio

        from aion_knowledge.pipeline.postproc.base import PostProcContext

        d = PostProcDispatcher({"fail": True, "normal": True})
        d._modules = {"fail": FailingModule(), "normal": NormalModule()}
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="test.md")

        # 不应抛出异常
        asyncio.run(d.run_second_batch(ctx, [{"chunk_uuid": "c1", "content": "test"}]))


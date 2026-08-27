"""PostProcDispatcher 调度器测试。"""

from __future__ import annotations

import asyncio

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule
from aion_knowledge.pipeline.postproc.dispatcher import PostProcDispatcher


class _TextModule(PostProcModule):
    always_on = True
    depends_on = []

    async def process(self, ctx, chunks):
        return len(chunks)


class _VectorModule(PostProcModule):
    always_on = True
    depends_on = ["text"]

    async def process(self, ctx, chunks):
        return len(chunks)


class _OptionalModule(PostProcModule):
    always_on = False
    depends_on = ["text"]

    async def process(self, ctx, chunks):
        return len(chunks)


class _SecondOptionalModule(PostProcModule):
    always_on = False
    depends_on = ["text"]

    async def process(self, ctx, chunks):
        return len(chunks)


class TestPostProcDispatcher:
    """验证调度器扁平并发执行。"""

    @pytest.mark.asyncio
    async def test_run_only_always_on_when_none_enabled(self):
        """未启用任何可选模块时，只执行 always_on 模块。"""
        dispatcher = PostProcDispatcher({})
        dispatcher._modules = {
            "text": _TextModule(),
            "optional": _OptionalModule(),
        }

        results = {"text": False, "optional": False}

        async def track_text(ctx, chunks):
            results["text"] = True
            return 1

        async def track_opt(ctx, chunks):
            results["optional"] = True
            return 0

        dispatcher._modules["text"].process = track_text
        dispatcher._modules["optional"].process = track_opt

        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        await dispatcher.run(ctx, [{"content": "hello"}])

        assert results["text"] is True, "always_on 模块应被执行"
        assert results["optional"] is False, "未启用可选模块不应被执行"

    @pytest.mark.asyncio
    async def test_run_second_batch_skips_always_on(self):
        """run_second_batch 应只执行非 always_on 且 enabled 的模块。"""
        dispatcher = PostProcDispatcher({"optional": True})
        dispatcher._modules = {
            "text": _TextModule(),
            "optional": _OptionalModule(),
        }

        results = {"text": False, "optional": False}

        async def track_text(ctx, chunks):
            results["text"] = True
            return 1

        async def track_opt(ctx, chunks):
            results["optional"] = True
            return 1

        dispatcher._modules["text"].process = track_text
        dispatcher._modules["optional"].process = track_opt

        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        await dispatcher.run_second_batch(ctx, [])

        assert results["text"] is False, "always_on 模块不应在 second_batch 执行"
        assert results["optional"] is True, "enabled 的可选模块应在 second_batch 执行"

    @pytest.mark.asyncio
    async def test_run_first_batch_skips_optional(self):
        """run_first_batch 应执行 always_on 模块，跳过可选模块。"""
        dispatcher = PostProcDispatcher({})
        dispatcher._modules = {
            "optional": _OptionalModule(),
            "text": _TextModule(),
        }

        results = {"text": False, "optional": False}

        async def track_text(ctx, chunks):
            results["text"] = True
            return 1

        async def track_opt(ctx, chunks):
            results["optional"] = True
            return 1

        dispatcher._modules["text"].process = track_text
        dispatcher._modules["optional"].process = track_opt

        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        await dispatcher.run_first_batch(ctx, [])

        assert results["text"] is True, "always_on 模块应被执行"
        assert results["optional"] is False, "可选模块不应在 first_batch 执行"

    @pytest.mark.asyncio
    async def test_run_second_batch_concurrent(self):
        """run_second_batch 应并发执行所有启用的可选模块。"""
        dispatcher = PostProcDispatcher({"opt1": True, "opt2": True})
        dispatcher._modules = {
            "text": _TextModule(),
            "opt1": _OptionalModule(),
            "opt2": _SecondOptionalModule(),
        }

        # 用 Event 验证两个模块确实在同时运行
        started = asyncio.Event()
        can_finish = asyncio.Event()
        order = []

        async def concurrent_opt(ctx, chunks):
            order.append("started")
            if len(order) == 2:  # 两个模块都已启动
                started.set()
            await can_finish.wait()  # 等待主测试释放
            order.append("finished")
            return 1

        async def concurrent_opt2(ctx, chunks):
            order.append("started")
            if len(order) == 2:  # 两个模块都已启动
                started.set()
            await can_finish.wait()
            order.append("finished")
            return 1

        dispatcher._modules["opt1"].process = concurrent_opt
        dispatcher._modules["opt2"].process = concurrent_opt2

        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        # 并发运行 second_batch
        task = asyncio.create_task(dispatcher.run_second_batch(ctx, []))

        # 等待两个模块都已启动
        await asyncio.wait_for(started.wait(), timeout=2)
        assert len(order) == 2  # 两个模块都应已启动
        assert "finished" not in order  # 尚未完成

        # 释放完成信号
        can_finish.set()
        await asyncio.wait_for(task, timeout=2)
        assert order.count("finished") == 2

    @pytest.mark.asyncio
    async def test_one_module_failure_does_not_block_others(self):
        """单个可选模块失败不应影响其他模块执行。"""
        dispatcher = PostProcDispatcher({"mod_a": True, "mod_b": True})

        executed = {"B": False}

        class _FailingModule(PostProcModule):
            always_on = False
            depends_on = []

            async def process(self, ctx, chunks):
                raise ValueError("模块处理失败")

        class _SucceedingModule(PostProcModule):
            always_on = False
            depends_on = []

            async def process(self, ctx, chunks):
                executed["B"] = True
                return 1

        dispatcher._modules = {
            "mod_a": _FailingModule(),
            "mod_b": _SucceedingModule(),
        }

        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        await dispatcher.run_second_batch(ctx, [])

        assert executed["B"] is True, "模块 B 应不受模块 A 失败的影响"


class TestOnlyWhitelist:
    """only 白名单参数验证。"""

    @pytest.mark.asyncio
    async def test_only_filters_second_batch(self):
        """only 白名单外已启用模块不执行，白名单内执行。"""
        dispatcher = PostProcDispatcher({"keyword_extract": True, "raptor": True}, only=["raptor"])
        dispatcher._modules = {
            "text": _TextModule(),
            "keyword_extract": _OptionalModule(),
            "raptor": _OptionalModule(),
        }

        results = {"keyword_extract": False, "raptor": False}

        async def track_keyword(ctx, chunks):
            results["keyword_extract"] = True
            return 1

        async def track_raptor(ctx, chunks):
            results["raptor"] = True
            return 1

        dispatcher._modules["keyword_extract"].process = track_keyword
        dispatcher._modules["raptor"].process = track_raptor

        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        await dispatcher.run_second_batch(ctx, [{"content": "hello"}])

        assert results["keyword_extract"] is False, "白名单外模块不应执行"
        assert results["raptor"] is True, "白名单内模块应执行"

    @pytest.mark.asyncio
    async def test_only_none_runs_all_enabled(self):
        """only=None 时行为不变（回归）。"""
        dispatcher = PostProcDispatcher({"optional": True})
        dispatcher._modules = {
            "text": _TextModule(),
            "optional": _OptionalModule(),
        }

        results = {"optional": False}

        async def track_opt(ctx, chunks):
            results["optional"] = True
            return 1

        dispatcher._modules["optional"].process = track_opt

        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        await dispatcher.run_second_batch(ctx, [{"content": "hello"}])

        assert results["optional"] is True

    @pytest.mark.asyncio
    async def test_only_does_not_affect_first_batch(self):
        """only 不影响首批 always_on 模块（run_first_batch 不经过 _is_enabled）。"""
        dispatcher = PostProcDispatcher({}, only=["raptor"])
        dispatcher._modules = {
            "text": _TextModule(),
            "raptor": _OptionalModule(),
        }

        results = {"text": False}

        async def track_text(ctx, chunks):
            results["text"] = True
            return 1

        dispatcher._modules["text"].process = track_text

        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        await dispatcher.run_first_batch(ctx, [{"content": "hello"}])

        assert results["text"] is True

    @pytest.mark.asyncio
    async def test_only_empty_list_runs_nothing(self):
        """only=[] 空白名单：不执行任何二批模块（fail-safe）。"""
        dispatcher = PostProcDispatcher({"raptor": True}, only=[])
        dispatcher._modules = {
            "text": _TextModule(),
            "raptor": _OptionalModule(),
        }

        results = {"raptor": False}

        async def track_raptor(ctx, chunks):
            results["raptor"] = True
            return 1

        dispatcher._modules["raptor"].process = track_raptor

        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        await dispatcher.run_second_batch(ctx, [{"content": "hello"}])

        assert results["raptor"] is False, "空白名单不应执行任何模块"


class TestEnabledModuleNames:
    """enabled_module_names：重跑白名单的合法模块集合。"""

    @pytest.mark.asyncio
    async def test_filters_first_batch_and_gates(self, monkeypatch):
        """过滤首批模块、.env 关闭模块、出厂硬控关闭模块。"""
        from aion_knowledge.common.config import settings
        from aion_knowledge.infrastructure.models import PostProcConfig
        from aion_knowledge.pipeline.postproc.dispatcher import enabled_module_names

        # 确定性基线：所有 .env 门控全开
        for name in ("keyword_extract", "question_gen", "summarizer", "raptor",
                     "graph_extract", "community", "disambiguation", "wiki"):
            monkeypatch.setattr(settings, f"postproc_{name}", True)

        # 图谱系出厂硬控当前默认开启（commit adb155a2 改回），
        # 此处显式关闭以验证出厂硬控过滤确实生效
        class _GraphOffConfig(PostProcConfig):
            enable_graph_extract: bool = False

        monkeypatch.setattr(
            "aion_knowledge.infrastructure.models.PostProcConfig", _GraphOffConfig
        )

        names = enabled_module_names()
        assert "text" not in names, "首批模块不可重跑"
        assert "vector" not in names, "首批模块不可重跑"
        assert "vlm_caption" not in names, "首批模块不可重跑"
        assert "graph_extract" not in names, "出厂硬控关闭的图谱模块不可重跑"
        assert "raptor" in names
        assert "wiki" in names

        # .env 门控关闭后不在集合内
        monkeypatch.setattr(settings, "postproc_raptor", False)
        assert "raptor" not in enabled_module_names()

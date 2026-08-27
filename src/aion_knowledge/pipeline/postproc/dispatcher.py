"""后处理调度器 — 自动发现模块、DAG 拓扑序分批执行。

核心机制：
  1. 自动扫描 postproc/ 下所有子包，通过每个子包的 processor.module() 工厂函数注册
  2. always_on 模块串行运行（首批），确保 text/vector/vlm_caption 等基础处理先完成
  3. 二次模块按 depends_on 构建 DAG，用 Kahn 算法拓扑排序，将无依赖的模块分到同一批次
  4. 同批次内并发执行，单模块失败不影响同批其他模块
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import pkgutil
from typing import Any

from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule

logger = logging.getLogger(__name__)


class PostProcDispatcher:
    """后处理调度器。"""

    def __init__(self, settings: dict[str, bool], only: list[str] | None = None) -> None:
        """初始化调度器。

        only：重跑白名单（模块名列表）。None = 全部已启用模块；空列表 = 不执行任何模块（fail-safe）；
        非空列表 = 在已启用模块之上做最终白名单过滤。
        """
        self._settings = settings
        self._only = set(only) if only is not None else None
        self._modules: dict[str, PostProcModule] = self._discover_modules()

    def _discover_modules(self) -> dict[str, PostProcModule]:
        """扫描 postproc/ 下所有子包，按约定注册模块。"""
        modules: dict[str, PostProcModule] = {}
        from aion_knowledge.pipeline import postproc as pkg

        for _, name, is_pkg in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
            if not is_pkg:
                continue
            try:
                processor_mod = importlib.import_module(f"{name}.processor")
                factory = getattr(processor_mod, "module", None)
                if factory is None:
                    factory = getattr(processor_mod, "create_module", None)
                module = factory() if factory else None
                if isinstance(module, PostProcModule):
                    modules[name.split(".")[-1]] = module
                    logger.debug("Discovered postproc module: %s", name)
            except (ImportError, AttributeError) as exc:
                logger.debug("Skipping %s: %s", name, exc)
        return modules

    def _is_enabled(self, name: str) -> bool:
        """检查模块是否通过配置启用（only 白名单在其上做最终过滤）。"""
        if self._only is not None and name not in self._only:
            return False
        key = name.lower()
        return self._settings.get(key, False)

    def _topological_sort(
        self, modules: list[tuple[str, PostProcModule]]
    ) -> list[list[tuple[str, PostProcModule]]]:
        """根据 depends_on 构建 DAG，按 Kahn 算法拓扑排序，返回按层级分组的批次。"""
        names = {name for name, _ in modules}
        graph: dict[str, set[str]] = {}
        in_degree: dict[str, int] = {}

        for name, mod in modules:
            graph.setdefault(name, set())
            in_degree.setdefault(name, 0)
            for dep in mod.depends_on:
                if dep in names and dep != name:  # 仅考虑在启用列表中的依赖
                    graph.setdefault(dep, set()).add(name)
                    in_degree[name] = in_degree.get(name, 0) + 1

        queue = [n for n, d in in_degree.items() if d == 0]
        batches: list[list[tuple[str, PostProcModule]]] = []
        visited: set[str] = set()
        mod_map = dict(modules)

        while queue:
            _queue = sorted(queue)  # 稳定排序，便于测试
            batches.append([(n, mod_map[n]) for n in _queue])
            visited.update(_queue)
            next_queue: list[str] = []
            for n in _queue:
                for neighbor in graph.get(n, set()):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue

        if len(visited) != len(names):
            raise ValueError(
                f"Circular dependency detected among postproc modules: "
                f"{names - visited}"
            )
        return batches

    async def run(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> None:
        """先执行 always_on 模块，再并发执行已启用的可选模块。"""
        await self.run_first_batch(ctx, chunks)
        await self.run_second_batch(ctx, chunks)

    async def run_first_batch(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> None:
        """按依赖拓扑序串行执行 always_on 的首批模块。

        使用与二批相同的 DAG 拓扑排序，确保模块按依赖关系逐批执行：
        text → vlm_caption → vector（而非按模块发现顺序）。
        """
        logger.info("首批模块开始执行：文档=%s", ctx.document_id)

        always_on = [(name, mod) for name, mod in self._modules.items() if mod.always_on]
        batches = self._topological_sort(always_on)
        for batch in batches:
            for name, module in batch:
                logger.info("「%s」开始执行：文档=%s", name, ctx.document_id)
                try:
                    count = await module.process(ctx, chunks)
                except Exception:
                    logger.exception("「%s」执行失败：文档=%s", name, ctx.document_id)
                    raise
                logger.info("「%s」执行完成：%d 条记录", name, count)

        logger.info("首批模块全部执行完毕：文档=%s", ctx.document_id)

    async def run_second_batch(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> None:
        """按 DAG 拓扑序分批执行已启用的可选模块，批内并发，失败互不影响。"""
        enabled: list[tuple[str, PostProcModule]] = []
        for name, module in self._modules.items():
            if module.always_on:
                continue
            if not self._is_enabled(name):
                logger.debug("Postproc '%s' disabled, skipping", name)
                continue
            enabled.append((name, module))

        logger.info("二批模块开始执行：文档=%s，启用模块=%s", ctx.document_id, [n for n, _ in enabled])

        if not enabled:
            logger.info("二批：未启用任何模块")
            return

        batches = self._topological_sort(enabled)
        for i, batch in enumerate(batches):
            logger.info("二批第%d/%d批：%s", i + 1, len(batches),
                         [n for n, _ in batch])
            tasks = [self._run_one(name, mod, ctx, chunks) for name, mod in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (name, _), result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error("「%s」执行失败：%s", name, result)
                else:
                    logger.info("「%s」执行完成：%d 条记录", name, result)

        logger.info("二批模块全部执行完毕：文档=%s，总批次数=%d", ctx.document_id, len(batches))

    async def _run_one(self, name: str, module: PostProcModule, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> int:
        """执行单个模块并返回处理记录数。"""
        logger.info("「%s」开始执行：文档=%s", name, ctx.document_id)
        return await module.process(ctx, chunks)


def enabled_module_names(dispatcher: PostProcDispatcher | None = None) -> list[str]:
    """返回当前可重跑的二批模块名（已注册 + 非首批 + 门控开启）。

    门控判定与 _run_postproc_subtasks 的 settings_dict 计算完全同式：
    settings.postproc_<name> AND PostProcConfig().enable_<name>，
    防止"校验通过但 worker 实际不跑"。
    """
    from aion_knowledge.common.config import settings
    from aion_knowledge.infrastructure.models import PostProcConfig

    d = dispatcher or PostProcDispatcher({})
    cfg = PostProcConfig()
    return [
        name
        for name, mod in d._modules.items()
        if not mod.always_on
        and getattr(settings, f"postproc_{name}", False)
        and getattr(cfg, f"enable_{name}", False)
    ]

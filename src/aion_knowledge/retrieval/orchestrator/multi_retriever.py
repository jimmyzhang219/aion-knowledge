"""MultiRetriever — 多路并发检索调度器。"""
from __future__ import annotations

import asyncio
import logging

from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext

logger = logging.getLogger(__name__)


class MultiRetriever:
    """多路检索调度器，并发执行所有已注册的检索器。

    用法：
        mr = MultiRetriever(timeout=30.0)
        mr.register(BM25Retriever())
        mr.register(VectorRetriever())
        results = await mr.search(ctx)
    """

    def __init__(self, timeout: float = 30.0):
        self._retrievers: dict[str, BaseRetriever] = {}
        self._timeout = timeout

    def register(self, retriever: BaseRetriever) -> None:
        """注册一个检索器。"""
        if not retriever.name:
            raise ValueError("Retriever must have a non-empty name")
        self._retrievers[retriever.name] = retriever

    def register_all(self, retrievers: list[BaseRetriever]) -> None:
        """批量注册检索器。"""
        for r in retrievers:
            self.register(r)

    async def search(self, ctx: RetrieverContext) -> dict[str, list[ChunkResult]]:
        """并发执行所有已启用的检索器。

        返回 {路径名: [ChunkResult, ...]}，每个列表已按 score 降序。
        """
        enabled = ctx.enabled_paths or set(self._retrievers.keys())

        async def run_with_timeout(name: str, retriever: BaseRetriever) -> tuple[str, list[ChunkResult]]:
            try:
                results = await asyncio.wait_for(
                    retriever.retrieve(ctx),
                    timeout=self._timeout,
                )
                # 确保结果已排序
                results.sort(key=lambda r: r.score, reverse=True)
                return name, results
            except asyncio.TimeoutError:
                logger.warning("Retriever %r timed out after %ss", name, self._timeout)
                return name, []
            except Exception as exc:
                logger.error("Retriever %r failed: %s", name, exc, exc_info=True)
                return name, []

        # 并发启动所有已启用的检索器
        coros = []
        for name, retriever in self._retrievers.items():
            if name in enabled:
                coros.append(run_with_timeout(name, retriever))

        if not coros:
            return {}

        completed = await asyncio.gather(*coros, return_exceptions=False)

        result_dict: dict[str, list[ChunkResult]] = {}
        for name, results in completed:
            if results:
                result_dict[name] = results

        logger.info("MultiRetriever: %d/%d paths returned results", len(result_dict), len(coros))
        return result_dict

    @property
    def retriever_names(self) -> list[str]:
        return list(self._retrievers.keys())

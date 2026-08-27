"""RRFFusionStage — RRF 融合 → 阈值过滤 → 候选池。"""
from __future__ import annotations

from aion_knowledge.retrieval.orchestrator.rrf_fuser import RRFFuser
from aion_knowledge.retrieval.pipeline.base import Stage
from aion_knowledge.retrieval.pipeline.context import PipelineContext


class RRFFusionStage(Stage):
    """RRF 融合多路结果 → 阈值过滤 → 写入 ctx.results（保留 pool_size，不做最终截断）。"""

    def __init__(
        self,
        rrf_fuser: RRFFuser,
        pool_multiplier: int = 3,
        threshold_ratio: float | None = 0.3,
    ) -> None:
        self._rrf_fuser = rrf_fuser
        self._pool_multiplier = pool_multiplier
        self._threshold_ratio = threshold_ratio

    async def run(self, ctx: PipelineContext) -> None:
        if not ctx.path_results:
            ctx.results = []
            return

        pool_size = max(ctx.top_k * self._pool_multiplier, 50)
        ctx.results = self._rrf_fuser.fuse(
            ctx.path_results,
            top_k=pool_size,
            threshold_ratio=self._threshold_ratio,
        )

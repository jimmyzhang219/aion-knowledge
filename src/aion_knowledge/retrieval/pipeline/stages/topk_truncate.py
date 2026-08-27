"""TopKTruncateStage — 最终截断到 top_k。"""
from __future__ import annotations

from aion_knowledge.retrieval.pipeline.base import Stage
from aion_knowledge.retrieval.pipeline.context import PipelineContext


class TopKTruncateStage(Stage):
    """最终截断：ctx.results = ctx.results[:ctx.top_k]。"""

    async def run(self, ctx: PipelineContext) -> None:
        ctx.results = ctx.results[:ctx.top_k]

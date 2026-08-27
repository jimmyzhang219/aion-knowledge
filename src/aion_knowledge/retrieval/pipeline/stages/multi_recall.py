"""MultiRecallStage — 11 路并发检索。"""
from __future__ import annotations

from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.orchestrator.multi_retriever import MultiRetriever
from aion_knowledge.retrieval.pipeline.base import Stage
from aion_knowledge.retrieval.pipeline.context import PipelineContext


class MultiRecallStage(Stage):
    """10 路并发检索，结果写入 ctx.path_results。"""

    def __init__(self, multi_retriever: MultiRetriever) -> None:
        self._multi_retriever = multi_retriever

    async def run(self, ctx: PipelineContext) -> None:
        rctx = RetrieverContext(
            query=ctx.query,
            kb_id=ctx.kb_id,
            original_query=ctx.original_query,
            top_k=ctx.path_top_k,
            enabled_paths=ctx.enabled_paths,
            query_embedding=ctx.query_embedding,
            expansion_keywords=ctx.expansion_keywords,
            entities=ctx.entities,
        )
        rctx._llm = ctx._llm   # type: ignore[attr-defined]
        rctx._embedder = ctx._embedder  # type: ignore[attr-defined]
        ctx.path_results = await self._multi_retriever.search(rctx)

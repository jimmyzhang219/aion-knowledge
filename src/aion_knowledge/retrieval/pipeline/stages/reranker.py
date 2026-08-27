"""RerankerStage — bge-reranker 交叉编码器重新打分。"""
from __future__ import annotations

import logging

from aion_knowledge.common.config import settings
from aion_knowledge.retrieval.pipeline.base import Stage
from aion_knowledge.retrieval.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class RerankerStage(Stage):
    """bge-reranker 重新打分，不做截断。降级：服务不可用时保留原始分数。"""

    async def run(self, ctx: PipelineContext) -> None:
        if not ctx.results or not settings.reranker_enabled or not ctx.query:
            return

        from aion_knowledge.infrastructure.reranker.client import rerank as reranker_client

        try:
            contents = [r.content for r in ctx.results]
            scores = await reranker_client(ctx.query, contents)
            for r, score in zip(ctx.results, scores, strict=True):
                r.score = score
            ctx.results.sort(key=lambda r: r.score, reverse=True)
        except Exception as exc:
            # 降级：保留原始分数，不影响主线检索；记录异常便于排查
            logger.warning("RerankerStage: rerank 调用失败，降级保留原始分数：%s", exc)

"""向量检索器 — chunk_vector.embedding 的余弦相似度匹配。"""
from __future__ import annotations

import logging

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext
from aion_knowledge.storage.relational.vector_repo import VectorRepository

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    name = "vector"

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        if not ctx.query_embedding:
            logger.warning("VectorRetriever: query_embedding is None, skipping")
            return []
        results: list[ChunkResult] = []
        async with get_session() as session:
            repo = VectorRepository(session)
            vec_results = await repo.similarity_search(
                kb_id=ctx.kb_id, embedding=ctx.query_embedding, top_k=ctx.top_k,
            )
            results = [
                ChunkResult(
                    chunk_id=r.chunk_id, kb_id=r.kb_id,
                    document_id=r.document_id, content=r.content,
                    score=r.score, source_paths=[self.name],
                )
                for r in vec_results
            ]
        logger.debug("VectorRetriever: %d results", len(results))
        return results

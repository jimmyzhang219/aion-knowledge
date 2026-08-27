"""FAQ 检索器 — 三层匹配（精确 → 模糊 → 向量）+ boost + 负问过滤。

FAQ 数据存储在 chunk_text（chunk_type='faq'）和 chunk_vector 中，
元数据以 JSONB 形式保存在 chunk_text.metadata 列。
"""
from __future__ import annotations

import logging

from aion_knowledge.common.config import settings
from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.models.enums import ChunkType
from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext
from aion_knowledge.storage.relational.chunk_repo import ChunkRepository
from aion_knowledge.storage.relational.vector_repo import VectorRepository

logger = logging.getLogger(__name__)


class FAQRetriever(BaseRetriever):
    name = "faq"

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        # FAQ 的 standard/similar question 是自然语言短语，应优先用原始 query 匹配；
        # rewrite 后的 ctx.query 是关键词展开串，不适合整串匹配
        match_query = (ctx.original_query or ctx.query).strip()
        if not match_query:
            return []
        results: list[ChunkResult] = []
        query_lower = match_query.lower()
        async with get_session() as session:
            chunk_repo = ChunkRepository(session)
            faq_rows = await chunk_repo.search_faq(kb_id=ctx.kb_id, query=match_query)
            for faq_row in faq_rows:
                # 负问题过滤
                if faq_row.negative_questions:
                    neg_hit = any(
                        query_lower in nq.lower() or nq.lower() in query_lower
                        for nq in faq_row.negative_questions
                    )
                    if neg_hit:
                        continue
                # 计算匹配分
                if faq_row.standard_question and faq_row.standard_question.lower() == query_lower:
                    score = 1.0
                elif faq_row.similar_questions and any(query_lower in sq.lower() for sq in faq_row.similar_questions):
                    score = 0.8
                else:
                    score = 0.6
                score *= settings.faq_score_boost
                results.append(ChunkResult(
                    chunk_id=faq_row.id, kb_id=faq_row.kb_id,
                    document_id=faq_row.document_id, content=faq_row.content,
                    score=score, source_paths=[self.name],
                    chunk_type=ChunkType.faq.value,
                    metadata={"standard_question": faq_row.standard_question or "", "category": faq_row.category or ""},
                ))

        # 向量匹配层（如 FAQ 有嵌入且 query_embedding 可用）
        if ctx.query_embedding:
            async with get_session() as session:
                vec_repo = VectorRepository(session)
                vec_results = await vec_repo.similarity_search(
                    kb_id=ctx.kb_id, embedding=ctx.query_embedding,
                    top_k=ctx.top_k, chunk_type="faq",
                )
                for vr in vec_results:
                    # 负问题过滤
                    raw_nq = vr.metadata.get("negative_questions", [])
                    if isinstance(raw_nq, str):
                        import json
                        try:
                            raw_nq = json.loads(raw_nq) if raw_nq else []
                        except (json.JSONDecodeError, TypeError):
                            raw_nq = []
                    negative_qs = list(raw_nq) if isinstance(raw_nq, list) else []
                    if negative_qs:
                        query_lower = ctx.query.lower().strip()
                        neg_hit = any(
                            query_lower in nq.lower() or nq.lower() in query_lower
                            for nq in negative_qs
                        )
                        if neg_hit:
                            continue
                    results.append(ChunkResult(
                        chunk_id=vr.chunk_id, kb_id=ctx.kb_id,
                        document_id=vr.document_id, content=vr.content,
                        score=float(vr.score) * settings.faq_score_boost,
                        source_paths=[self.name], chunk_type=ChunkType.faq.value,
                    ))

        logger.debug("FAQRetriever: %d results", len(results))
        return results[: ctx.top_k]

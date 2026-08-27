"""摘要检索器 — summary_text BM25 + embedding_summary 向量混合。"""
from __future__ import annotations

import logging

from sqlalchemy import text as sql_text

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext

logger = logging.getLogger(__name__)

# 摘要内部融合权重
SUMMARY_BM25_WEIGHT = 0.3
SUMMARY_VECTOR_WEIGHT = 0.7


class SummaryRetriever(BaseRetriever):
    name = "summary"

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        if not ctx.query.strip():
            return []
        results: list[ChunkResult] = []
        async with get_session() as session:
            # 1. BM25 on summary_text
            bm25_rows = await session.execute(
                sql_text("""
                    SELECT c.id, c.content, c.document_id, c.kb_id,
                           ts_rank(to_tsvector('zh_cfg', COALESCE(c.summary_text, '')),
                                   plainto_tsquery('zh_cfg', :query)) AS score
                    FROM chunk_text c
                    JOIN doc_knowledge_documents d ON d.id = c.document_id AND NOT d.deleted
                    JOIN kb_knowledge_bases k ON k.id = c.kb_id AND NOT k.deleted
                    WHERE c.kb_id = :kb_id
                      AND c.summary_text IS NOT NULL
                      AND to_tsvector('zh_cfg', c.summary_text) @@ plainto_tsquery('zh_cfg', :query)
                    ORDER BY score DESC
                    LIMIT :limit
                """),
                {"query": ctx.query, "kb_id": ctx.kb_id, "limit": ctx.top_k},
            )
            bm25_scores: dict[str, float] = {}
            for row in bm25_rows:
                bm25_scores[str(row.id)] = float(row.score)

            # 2. Vector on embedding_summary（如有 query_embedding）
            vector_scores: dict[str, float] = {}
            if ctx.query_embedding:
                embedding_str = "[" + ",".join(str(v) for v in ctx.query_embedding) + "]"
                vec_rows = await session.execute(
                    sql_text("""
                        SELECT v.chunk_id,
                               (1 - (v.embedding_summary <=> CAST(:query_embedding AS vector))) AS score
                        FROM chunk_vector v
                        JOIN chunk_text c ON c.id = v.chunk_id::uuid
                        JOIN doc_knowledge_documents d ON d.id = c.document_id AND NOT d.deleted
                        JOIN kb_knowledge_bases k ON k.id = v.kb_id AND NOT k.deleted
                        WHERE v.kb_id = :kb_id
                          AND v.embedding_summary IS NOT NULL
                        ORDER BY score DESC
                        LIMIT :limit
                    """),
                    {"query_embedding": embedding_str, "kb_id": ctx.kb_id, "limit": ctx.top_k},
                )
                for row in vec_rows:
                    vector_scores[str(row.chunk_id)] = float(row.score)

            # 3. 混合得分
            all_ids = set(bm25_scores) | set(vector_scores)
            for chunk_id_str in all_ids:
                b = bm25_scores.get(chunk_id_str, 0.0) * SUMMARY_BM25_WEIGHT
                v = vector_scores.get(chunk_id_str, 0.0) * SUMMARY_VECTOR_WEIGHT
                results.append(ChunkResult(
                    chunk_id=chunk_id_str,
                    kb_id=ctx.kb_id,
                    document_id="",
                    content="",
                    score=b + v,
                    source_paths=[self.name],
                ))

            # 4. 回填 content（取 bm25 命中行，或单独查询）
            if results:
                ids = [r.chunk_id for r in results]
                placeholders = ",".join(f"'{i}'" for i in ids)
                fill = await session.execute(
                    sql_text(f"""
                        SELECT id, content, document_id
                        FROM chunk_text
                        WHERE id IN ({placeholders})
                          AND NOT EXISTS (SELECT 1 FROM doc_knowledge_documents d
                                          WHERE d.id = chunk_text.document_id AND d.deleted)
                    """),
                )
                content_map: dict[str, tuple[str, str]] = {}
                for row in fill:
                    content_map[str(row.id)] = (row.content, str(row.document_id))
                for r in results:
                    if r.chunk_id in content_map:
                        r.content, r.document_id = content_map[r.chunk_id]

        results.sort(key=lambda r: r.score, reverse=True)
        logger.debug("SummaryRetriever: %d results", len(results))
        return results[: ctx.top_k]

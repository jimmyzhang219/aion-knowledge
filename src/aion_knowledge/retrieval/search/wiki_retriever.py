"""Wiki 页面检索器 — 按 wiki 百科内容 ts_rank 全文检索匹配，命中后按 chunk_refs 回捞源 chunk 原文。"""
from __future__ import annotations

import logging

from sqlalchemy import text as sql_text

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext

logger = logging.getLogger(__name__)


class WikiRetriever(BaseRetriever):
    name = "wiki"

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        query_base = ctx.query.strip()
        if not query_base:
            return []
        if ctx.expansion_keywords:
            query_base += " " + " ".join(ctx.expansion_keywords)
        results: list[ChunkResult] = []
        async with get_session() as session:
            rows = await session.execute(
                sql_text("""
                    SELECT ct.id, ct.content, ct.document_id,
                           w.page_title, w.page_slug, w.id AS wiki_page_id,
                           GREATEST(
                               ts_rank(to_tsvector('zh_cfg', COALESCE(w.page_title, '')),
                                       plainto_tsquery('zh_cfg', :query)),
                               ts_rank(to_tsvector('zh_cfg', COALESCE(w.content, '')),
                                       plainto_tsquery('zh_cfg', :query))
                           ) AS score
                    FROM chunk_wiki w
                    JOIN kb_knowledge_bases k ON k.id = w.kb_id AND NOT k.deleted
                    INNER JOIN chunk_text ct
                            ON ct.kb_id = w.kb_id AND ct.id::text = ANY(w.chunk_refs)
                           AND NOT EXISTS (SELECT 1 FROM doc_knowledge_documents d
                                           WHERE d.id = ct.document_id AND d.deleted)
                    WHERE w.kb_id = :kb_id
                      AND (to_tsvector('zh_cfg', COALESCE(w.page_title, '')) @@ plainto_tsquery('zh_cfg', :query)
                           OR to_tsvector('zh_cfg', COALESCE(w.content, '')) @@ plainto_tsquery('zh_cfg', :query))
                    ORDER BY score DESC
                    LIMIT :limit
                """),
                {"query": query_base, "kb_id": ctx.kb_id, "limit": ctx.top_k},
            )
            for row in rows:
                results.append(ChunkResult(
                    chunk_id=str(row.id),
                    kb_id=ctx.kb_id,
                    document_id=str(row.document_id),
                    content=row.content,
                    score=float(row.score),
                    source_paths=[self.name],
                    metadata={
                        "page_title": row.page_title,
                        "page_slug": row.page_slug,
                        "wiki_page_id": str(row.wiki_page_id),
                    },
                ))
        logger.debug("WikiRetriever: %d results", len(results))
        return results

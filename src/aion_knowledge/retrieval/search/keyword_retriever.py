"""关键词检索器 — chunk_text.keywords 数组重叠匹配。"""
from __future__ import annotations

import logging

from sqlalchemy import text

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext

logger = logging.getLogger(__name__)

# 常见无意义停用词（检索用，不传数据库）
_STOP_WORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "之", "与", "及", "或", "但", "而", "以", "及", "等", "对", "把",
    "被", "让", "给", "为", "所", "从", "将", "并", "中", "其", "该",
})


class KeywordRetriever(BaseRetriever):
    name = "keyword"

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        query_base = ctx.query.strip()
        if not query_base:
            return []
        if ctx.expansion_keywords:
            query_base += " " + " ".join(ctx.expansion_keywords)

        results: list[ChunkResult] = []
        async with get_session() as session:
            # PG zhparser 分词（通过 tsvector 提取）
            row = await session.execute(
                text("SELECT unnest(tsvector_to_array(to_tsvector('zh_cfg', COALESCE(:q, ''))))"),
                {"q": query_base},
            )
            all_tokens = [r[0].lower() for r in row]
            keywords = [t for t in all_tokens if t not in _STOP_WORDS][:10]
            if not keywords:
                return []

            # 单次 SQL（ANY 替代逐词循环）
            patterns = [f"%{kw}%" for kw in keywords]
            rows = await session.execute(
                text("""
                    SELECT DISTINCT c.id, c.content, c.document_id, c.kb_id
                    FROM chunk_text c
                    JOIN doc_knowledge_documents d ON d.id = c.document_id AND NOT d.deleted
                    JOIN kb_knowledge_bases k ON k.id = c.kb_id AND NOT k.deleted
                    WHERE c.kb_id = :kb_id
                      AND EXISTS (
                          SELECT 1 FROM unnest(c.keywords) ck
                          WHERE ck ILIKE ANY(:patterns)
                      )
                    LIMIT :limit
                """),
                {"kb_id": ctx.kb_id, "patterns": patterns, "limit": ctx.top_k},
            )
            for r in rows:
                results.append(ChunkResult(
                    chunk_id=str(r.id), kb_id=str(r.kb_id),
                    document_id=str(r.document_id), content=r.content,
                    score=1.0, source_paths=[self.name],
                ))

        logger.debug("KeywordRetriever: %d results", len(results))
        return results

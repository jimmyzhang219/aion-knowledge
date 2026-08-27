"""BM25 检索器 — PostgreSQL pg_textsearch BM25 索引搜索。

使用 pg_textsearch 扩展的 bm25 访问方法，在 content_tokens 列上执行 BM25
全文检索。分词由索引的 text_config='zh_cfg' 配置的 zhparser 自动处理。

pg_textsearch 的 <@> 操作符返回的评分是负值（0 = 不匹配，越负越相关），
与常规相似度评分（越高越好）方向相反。RRF 融合使用排名位置而非分值大小，
因此原始负值评分可直接使用。排序方向：ORDER BY score ASC。

关于 asyncpg 兼容性：
  CAST(:query AS bm25query) 和 to_bm25query(:query) 在 asyncpg 参数化
  查询中均失效，原因是 asyncpg 的参数绑定改变了 PostgreSQL 的运算符类型
  推断路径。解决方案是将查询字符串直接内联到 SQL 文本中（作为字符串字面
  量），仅 kb_id / limit 等标准类型走参数化绑定。查询字符串经 replace 转义
  后不含 SQL 特殊字符，注入风险可控。
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    name = "bm25"

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        # 1. 预处理
        query_base = ctx.query.strip()
        if not query_base:
            return []
        if ctx.expansion_keywords:
            query_base += " " + " ".join(ctx.expansion_keywords)

        rows_data: list[dict[str, Any]] = []
        async with get_session() as session:
            # 2. pg_textsearch BM25 索引搜索
            # text_config='zh_cfg' 的 BM25 索引直接处理原始查询，由 zhparser 分词
            query_literal = query_base.replace("'", "''")
            sql = f"""
                SELECT c.id, c.content, c.document_id, c.kb_id, c.metadata,
                       c.content_tokens <@> to_bm25query('{query_literal}') AS score
                FROM chunk_text c
                JOIN doc_knowledge_documents d ON d.id = c.document_id AND NOT d.deleted
                JOIN kb_knowledge_bases k ON k.id = c.kb_id AND NOT k.deleted
                WHERE c.kb_id = CAST(:kb_id AS uuid)
                  AND c.content_tokens <@> to_bm25query('{query_literal}') IS NOT NULL
                ORDER BY score ASC
                LIMIT :limit
            """
            rows = await session.execute(
                text(sql),
                {"kb_id": ctx.kb_id, "limit": ctx.top_k},
            )
            rows_data = [
                {
                    "id": str(r.id),
                    "content": r.content,
                    "document_id": str(r.document_id),
                    "kb_id": str(r.kb_id),
                    "score": float(r.score),
                    "metadata": dict(r.metadata) if r.metadata else {},
                }
                for r in rows
            ]

        results = [
            ChunkResult(
                chunk_id=d["id"],
                kb_id=d["kb_id"],
                document_id=d["document_id"],
                content=d["content"],
                score=d["score"],
                source_paths=[self.name],
                metadata=d["metadata"],
            )
            for d in rows_data
        ]

        logger.debug(
            "BM25Retriever(pg_textsearch): %d results for query=%r", len(results), ctx.query[:50]
        )
        return results

"""社区报告检索器 — chunk_community.embedding 的余弦相似度匹配。"""
from __future__ import annotations

import logging

from sqlalchemy import text as sql_text

from aion_knowledge.common.community_text import build_community_text
from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext

logger = logging.getLogger(__name__)


class CommunityRetriever(BaseRetriever):
    name = "community"

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        if not ctx.query_embedding:
            logger.warning("CommunityRetriever: query_embedding is None, skipping")
            return []
        embedding_str = "[" + ",".join(str(v) for v in ctx.query_embedding) + "]"
        async with get_session() as session:
            rows = await session.execute(
                sql_text("""
                    SELECT chunk_community.id, community_id, community_level, summary, findings,
                           COALESCE(payload->>'title', '') AS title,
                           COALESCE(payload->'members', '[]') AS members,
                           (1 - (embedding <=> CAST(:query_embedding AS vector))) AS score
                    FROM chunk_community
                    JOIN kb_knowledge_bases k ON k.id = chunk_community.kb_id AND NOT k.deleted
                    WHERE chunk_community.kb_id = CAST(:kb_id AS uuid) AND embedding IS NOT NULL
                    ORDER BY score DESC
                    LIMIT :limit
                """),
                {
                    "query_embedding": embedding_str,
                    "kb_id": ctx.kb_id,
                    "limit": ctx.top_k,
                },
            )
            results = [
                ChunkResult(
                    chunk_id=str(r.id),
                    kb_id=ctx.kb_id,
                    document_id="",  # 社区为 KB 级，无文档归属
                    content=build_community_text(r.title, r.summary, r.findings or []),
                    score=float(r.score),
                    source_paths=[self.name],
                    metadata={
                        "community_id": r.community_id,
                        "community_level": r.community_level,
                        "members": r.members if r.members else [],  # payload->'members' 为 jsonb，asyncpg 已解码为 list
                    },
                )
                for r in rows
            ]
        logger.debug("CommunityRetriever: %d results", len(results))
        return results

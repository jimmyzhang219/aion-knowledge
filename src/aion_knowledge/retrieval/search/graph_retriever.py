"""知识图谱检索器 — 命中实体 → 回捞关联 chunk 原文。"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text as sql_text

from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext

logger = logging.getLogger(__name__)


class GraphRetriever(BaseRetriever):
    name = "graph"

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        # 降级链：entities → expansion_keywords → 原始查询
        if ctx.entities:
            entity_names = ctx.entities
        elif ctx.expansion_keywords:
            entity_names = ctx.expansion_keywords
        else:
            entity_names = [ctx.query]

        try:
            # 前置检查：KB 是否被删（Neo4j 无法 SQL join，单独查）
            from aion_knowledge.infrastructure.db import get_session

            async with get_session() as session:
                kb_deleted = await session.scalar(
                    sql_text("SELECT deleted FROM kb_knowledge_bases WHERE id = :kb_id"),
                    {"kb_id": ctx.kb_id},
                )
                if kb_deleted:
                    return []

            from aion_knowledge.infrastructure.graph import search_entities

            matched_entities = await search_entities(
                kb_id=ctx.kb_id,
                names=entity_names,
                top_k=ctx.top_k,
            )
            chunk_scores = self._aggregate_chunks(matched_entities, ctx.top_k)
            if not chunk_scores:
                return []

            # chunk_ids 已按 score 降序（_aggregate_chunks 排序），保持该顺序以维护 RRF rank
            score_map = dict(chunk_scores)
            chunk_ids = list(score_map.keys())

            from aion_knowledge.storage.relational.chunk_repo import ChunkRepository

            async with get_session() as session:
                repo = ChunkRepository(session)
                rows = await repo.get_by_ids(chunk_ids)

            # 按 chunk_ids（score 降序）重排——DB 对 IN 查询不保证返回顺序
            row_map = {row.id: row for row in rows}
            return [
                ChunkResult(
                    chunk_id=cid,
                    kb_id=ctx.kb_id,
                    document_id=row_map[cid].document_id,
                    content=row_map[cid].content,
                    score=score_map[cid],
                    source_paths=[self.name],
                    chunk_type=row_map[cid].chunk_type,
                    metadata=row_map[cid].chunk_metadata,
                )
                for cid in chunk_ids if cid in row_map
            ]
        except Exception as exc:
            logger.error("GraphRetriever search failed: %s", exc)
        return []

    @staticmethod
    def _aggregate_chunks(
        matched_entities: list[dict[str, Any]], top_k: int,
    ) -> list[tuple[str, float]]:
        """同 chunk 被多实体关联时取最高 similarity，按 score 降序截断 top_k。"""
        chunk_score: dict[str, float] = {}
        for ent in matched_entities:
            for cid in ent.get("chunk_ids") or []:
                sim = float(ent.get("similarity") or 0.0)
                if cid not in chunk_score or sim > chunk_score[cid]:
                    chunk_score[cid] = sim
        return sorted(chunk_score.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

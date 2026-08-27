"""KBGraphMerger — 跨文档合并：写 Neo4j + 更新 PG metadata 统计。"""
from __future__ import annotations

import logging
from typing import Any

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.infrastructure.graph import add_graph as neo4j_add_graph
from aion_knowledge.infrastructure.graph import get_stats as neo4j_get_stats
from aion_knowledge.storage.relational.graph_repo import GraphMetadataRepository

logger = logging.getLogger(__name__)


class KBGraphMerger:
    """将文档子图合并到 Neo4j 知识图谱，并更新 PG metadata 统计。"""

    async def merge_document(
        self,
        kb_id: str,
        doc_id: str,
        entities: dict[str, dict[str, Any]],
        relations: dict[tuple[str, str, str], dict[str, Any]],
    ) -> None:
        """合并一个文档的实体/关系到 KB 全局图。"""
        if not entities and not relations:
            return
        await neo4j_add_graph(kb_id, doc_id, entities, relations)
        await self._update_metadata_v2(kb_id)

    async def _update_metadata_v2(self, kb_id: str) -> None:
        """从 Neo4j 获取统计，更新 PG 缓存表。"""
        await update_kb_graph_stats(kb_id)


async def update_kb_graph_stats(kb_id: str) -> None:
    """刷新 KB 图谱统计：entity/relation/doc 来自 Neo4j，community 来自 PG 社区表。

    供 graph_extract（写图后）与 community（写社区后）共用，保证
    graph_metadata 的 doc_count / community_count 与真实数据一致。
    """
    stats = await neo4j_get_stats(kb_id)
    async with get_session() as session:
        repo = GraphMetadataRepository(session)
        community_count = await repo.count_communities(kb_id)
        await repo.upsert_stats(
            kb_id=kb_id,
            entity_count=stats["entity_count"],
            relation_count=stats["relation_count"],
            doc_count=stats["doc_count"],
            community_count=community_count,
        )
        await session.commit()

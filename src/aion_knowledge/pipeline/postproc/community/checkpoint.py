"""CommunityCheckpointManager — 基于 graph_metadata.checkpoints 的 hash 检查点。"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.storage.relational.graph_repo import GraphMetadataRepository

logger = logging.getLogger(__name__)


def compute_graph_hash(
    entity_names: list[str],
    relation_keys: list[tuple[str, str, str]],
) -> str:
    """计算 KB 图的 hash（entity+relation 排序后 SHA256）。

    对 relation 三元组内的前两个元素（源/目标）也排序，
    确保无向图中 A→B 与 B→A 视为同一条边。
    """
    normalized_relations = sorted(
        (min(s, t), max(s, t), r) for s, t, r in relation_keys
    )
    raw = json.dumps(
        {"entities": sorted(entity_names), "relations": normalized_relations},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CommunityCheckpointManager:
    """社区检查点管理器（基于 graph_metadata.checkpoints JSONB）。"""

    async def save(self, kb_id: str, graph_hash: str) -> None:
        """保存社区检测完成检查点（graph_hash + 完成时间）。"""
        async with get_session() as session:
            repo = GraphMetadataRepository(session)
            await repo.save_checkpoint(kb_id, "community", {
                "graph_hash": graph_hash,
                "version": 1,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            await session.commit()

    async def load(self, kb_id: str) -> str | None:
        """读取社区检查点中的 graph_hash，无记录返回 None。"""
        async with get_session() as session:
            repo = GraphMetadataRepository(session)
            return await repo.load_checkpoint(kb_id, key="community")

    async def should_skip(self, kb_id: str, current_hash: str) -> bool:
        """判断是否可跳过社区检测：hash 一致且已有报告产出时返回 True。"""
        saved = await self.load(kb_id)
        if saved != current_hash:
            return False
        count = await self._count_community_reports(kb_id)
        return count > 0

    async def _count_community_reports(self, kb_id: str) -> int:
        """统计该 KB 已生成的社区报告数量。"""
        async with get_session() as session:
            repo = GraphMetadataRepository(session)
            return await repo.count_communities(kb_id)

    async def clear(self, kb_id: str) -> None:
        """清除该 KB 的社区检查点。"""
        async with get_session() as session:
            repo = GraphMetadataRepository(session)
            await repo.clear_checkpoint(kb_id, key="community")
            await session.commit()

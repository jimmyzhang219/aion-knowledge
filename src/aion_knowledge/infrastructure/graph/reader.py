"""Neo4j 读取操作 — 图谱检索。"""
from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncManagedTransaction

from aion_knowledge.infrastructure.graph.client import Neo4jConnection

logger = logging.getLogger(__name__)


async def search_entities(
    kb_id: str,
    names: list[str],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """精确 + 别名反查 + 模糊匹配实体。"""
    conn = Neo4jConnection()

    async def _search(
        tx: AsyncManagedTransaction, kb: str, entity_names: list[str], limit: int
    ) -> list[dict[str, Any]]:
        # 精确匹配（含别名反查：名称或其别名命中任一搜索词）
        query_result = await tx.run(
            """
            MATCH (e:Entity {kb_id: $kb})
            WHERE e.name IN $names OR any(a IN e.aliases WHERE a IN $names)
            OPTIONAL MATCH (e)<-[:INSTANCE_OF]-(inst:EntityInstance {kb_id: $kb})
            RETURN e.name AS name, e.type AS type,
                   inst.description AS description, inst.weight AS weight,
                   inst.chunk_ids AS chunk_ids
            LIMIT $limit
            """,
            kb=kb, names=entity_names, limit=limit,
        )
        records = await query_result.fetch(limit)
        records = list(records)

        # 如果不足，模糊匹配补充
        existing_names = {r["name"] for r in records}
        if len(records) < limit:
            for name in entity_names:
                fuzzy_result = await tx.run(
                    """
                    MATCH (e:Entity {kb_id: $kb})
                    WHERE e.name CONTAINS $pattern
                      AND NOT e.name IN $names
                    OPTIONAL MATCH (e)<-[:INSTANCE_OF]-(inst:EntityInstance {kb_id: $kb})
                    RETURN e.name AS name, e.type AS type,
                           inst.description AS description, inst.weight AS weight,
                           inst.chunk_ids AS chunk_ids
                    LIMIT $fuzzy_limit
                    """,
                    kb=kb, pattern=name, names=entity_names,
                    fuzzy_limit=limit - len(records),
                )
                fuzzy_records = await fuzzy_result.fetch(limit - len(records))
                for r in fuzzy_records:
                    if r["name"] not in existing_names:
                        records.append(r)
                        existing_names.add(r["name"])

        entity_map: dict[str, dict[str, Any]] = {}
        for r in records:
            name = r["name"]
            if name not in entity_map:
                entity_map[name] = {
                    "entity_name": name,
                    "entity_type": r["type"] or "",
                    "descriptions": [],
                    "weights": [],
                    "chunk_ids": [],
                    "similarity": 1.0 if name in entity_names else 0.5,
                }
            if r["description"]:
                entity_map[name]["descriptions"].append(r["description"])
            if r["weight"] is not None:
                entity_map[name]["weights"].append(r["weight"])
            # 累加 instance 的 chunk_ids 并去重
            for cid in r["chunk_ids"] or []:
                if cid not in entity_map[name]["chunk_ids"]:
                    entity_map[name]["chunk_ids"].append(cid)

        result = []
        for name, data in entity_map.items():
            description = "<SEP>".join(data["descriptions"]) if data["descriptions"] else ""
            weight = sum(data["weights"]) / len(data["weights"]) if data["weights"] else 1.0
            result.append({
                "entity_name": name,
                "entity_type": data["entity_type"],
                "description": description,
                "weight": weight,
                "similarity": data["similarity"],
                "chunk_ids": data["chunk_ids"],
            })
        return result

    return await conn.execute_read(_search, kb_id, names, top_k) or []


async def expand_neighbors(
    kb_id: str,
    entity_names: list[str],
    hop: int = 1,
) -> list[dict[str, Any]]:
    """展开实体的邻居关系（当前仅支持 1-hop）。"""
    conn = Neo4jConnection()

    async def _expand(
        tx: AsyncManagedTransaction, kb: str, names: list[str]
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (e:Entity {kb_id: $kb})
            WHERE e.name IN $names OR any(a IN e.aliases WHERE a IN $names)
            MATCH (e)<-[:INSTANCE_OF]-(:EntityInstance {kb_id: $kb})
                  -[r:RELATION_INSTANCE {kb_id: $kb}]-
                  (:EntityInstance {kb_id: $kb})-[:INSTANCE_OF]->(neighbor:Entity)
            RETURN r.type AS type, r.weight AS weight,
                   r.description AS description,
                   e.name AS source, neighbor.name AS target
            LIMIT 200
            """,
            kb=kb, names=names,
        )
        records = await result.fetch(200)
        return [
            {
                "source": r["source"],
                "target": r["target"],
                "type": r["type"] or "",
                "description": r["description"] or "",
                "weight": r["weight"] if r["weight"] is not None else 1.0,
            }
            for r in records
        ]

    return await conn.execute_read(_expand, kb_id, entity_names) or []


async def load_kb_graph(kb_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """加载 KB 全量图谱（实体 + 聚合关系），供社区发现等批处理使用。"""
    conn = Neo4jConnection()

    async def _load(
        tx: AsyncManagedTransaction, kb: str
    ) -> tuple[list[Any], list[Any]]:
        # 读取所有实体
        ent_result = await tx.run(
            "MATCH (e:Entity {kb_id: $kb}) RETURN e.name AS name, e.type AS type",
            kb=kb,
        )
        entities = await ent_result.fetch(10000)
        # 读取所有关系（两阶段模型：EntityInstance 间的 RELATION_INSTANCE 边，按实体对聚合）
        rel_result = await tx.run(
            "MATCH (s:Entity {kb_id: $kb})<-[:INSTANCE_OF]-(:EntityInstance {kb_id: $kb})"
            "-[r:RELATION_INSTANCE {kb_id: $kb}]->(:EntityInstance {kb_id: $kb})"
            "-[:INSTANCE_OF]->(t:Entity {kb_id: $kb}) "
            "RETURN s.name AS source, t.name AS target, r.type AS type, "
            "SUM(r.weight) AS weight",
            kb=kb,
        )
        relations = await rel_result.fetch(10000)
        return entities, relations

    result = await conn.execute_read(_load, kb_id)
    if result is None:
        return [], []
    entities, relations = result
    return (
        [{"entity_name": r["name"], "entity_type": r["type"] or ""}
         for r in entities],
        [{"source_entity": r["source"], "target_entity": r["target"],
          "relation_type": r["type"], "weight": r["weight"] or 1.0}
         for r in relations],
    )

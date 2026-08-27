"""Neo4j 写入操作。"""
from __future__ import annotations

import logging
import uuid
from typing import Any, cast

from neo4j import AsyncManagedTransaction

from aion_knowledge.infrastructure.graph.client import Neo4jConnection

logger = logging.getLogger(__name__)


async def add_graph(
    kb_id: str,
    doc_id: str,
    entities: dict[str, dict[str, Any]],
    relations: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    """文档级实体/关系合并到 KB 图（EntityInstance 两阶段写入模式）。

    Stages:
        1. MERGE Entity 抽象节点（去重，仅保留 type 信息）
        2. MERGE EntityInstance 节点 + INSTANCE_OF 关系
        3. MERGE RELATION_INSTANCE 边（挂在 EntityInstance 之间）
    """
    conn = Neo4jConnection()

    if not entities and not relations:
        return

    # Stage 1: MERGE Entity + EntityInstance + INSTANCE_OF（原子事务）
    if entities:
        async def _merge_entity_layer(tx: AsyncManagedTransaction) -> None:
            await tx.run(
                """
                UNWIND $data AS row
                // 别名解析：名称命中某 canonical 的 aliases 时归一化为 canonical
                OPTIONAL MATCH (canon:Entity {kb_id: row.kb_id})
                WHERE any(a IN canon.aliases WHERE a = row.entity_name)
                WITH row, collect(canon) AS canons
                WITH row, CASE WHEN size(canons) > 0
                        THEN canons[0].name ELSE row.entity_name END AS resolved
                MERGE (e:Entity {name: resolved, kb_id: row.kb_id})
                SET e.type = row.type
                MERGE (inst:EntityInstance {
                    doc_id: row.doc_id, entity_name: resolved, kb_id: row.kb_id
                })
                ON CREATE SET inst.id = row.id
                SET inst.description = CASE
                        WHEN inst.description IS NULL OR inst.description = '' THEN row.description
                        WHEN row.description IS NULL OR row.description = '' THEN inst.description
                        ELSE inst.description + '<SEP>' + row.description
                    END,
                    inst.weight = CASE
                        WHEN inst.weight IS NULL THEN row.weight
                        ELSE inst.weight + row.weight
                    END,
                    inst.chunk_ids = CASE
                        WHEN inst.chunk_ids IS NULL THEN row.chunk_ids
                        ELSE [x IN inst.chunk_ids WHERE NOT x IN row.chunk_ids] + row.chunk_ids
                    END
                MERGE (inst)-[:INSTANCE_OF]->(e)
                """,
                data=[
                    {
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "entity_name": name,
                        "kb_id": kb_id,
                        "type": ent.get("type", ""),
                        "description": "<SEP>".join(ent.get("descriptions", [])),
                        "weight": ent.get("weight", 1.0),
                        "chunk_ids": list(ent.get("source_chunks", [])),
                    }
                    for name, ent in entities.items()
                ],
            )

        await conn.execute_write(_merge_entity_layer)
        logger.debug("Neo4j: merged Entity+EntityInstance for kb=%s, doc=%s", kb_id, doc_id)

    # Stage 2: MERGE RELATION_INSTANCE 边（挂在 EntityInstance 之间）
    if not relations:
        return

    async def _merge_relation_instances(tx: AsyncManagedTransaction) -> int:
        result = await tx.run(
            """
            UNWIND $data AS row
            // 别名解析：source/target 命中别名时归一化为 canonical（与 Stage 1 一致）
            OPTIONAL MATCH (cs:Entity {kb_id: row.kb_id})
            WHERE any(a IN cs.aliases WHERE a = row.source)
            WITH row, collect(cs) AS cs_list
            OPTIONAL MATCH (ct:Entity {kb_id: row.kb_id})
            WHERE any(a IN ct.aliases WHERE a = row.target)
            WITH row, cs_list, collect(ct) AS ct_list
            WITH row,
                 CASE WHEN size(cs_list) > 0 THEN cs_list[0].name ELSE row.source END AS src,
                 CASE WHEN size(ct_list) > 0 THEN ct_list[0].name ELSE row.target END AS tgt
            MATCH (s_inst:EntityInstance {doc_id: row.doc_id, entity_name: src, kb_id: row.kb_id})
            MATCH (t_inst:EntityInstance {doc_id: row.doc_id, entity_name: tgt, kb_id: row.kb_id})
            MERGE (s_inst)-[r:RELATION_INSTANCE {
                type: row.type, kb_id: row.kb_id, doc_id: row.doc_id
            }]->(t_inst)
            SET r.description = CASE
                    WHEN r.description IS NULL OR r.description = '' THEN row.description
                    WHEN row.description IS NULL OR row.description = '' THEN r.description
                    ELSE r.description + '<SEP>' + row.description
                END,
                r.weight = CASE
                    WHEN r.weight IS NULL THEN row.weight
                    ELSE r.weight + row.weight
                END
            RETURN count(r) AS count
            """,
            data=[
                {
                    "doc_id": doc_id,
                    "source": source,
                    "target": target,
                    "kb_id": kb_id,
                    "type": rel_type,
                    "description": "<SEP>".join(rel.get("descriptions", [])),
                    "weight": rel.get("weight", 1.0),
                }
                for (source, target, rel_type), rel in relations.items()
            ],
        )
        record = await result.single()
        # record["count"] 来自 Neo4j 记录，类型为 Any，此处收敛为 int
        return cast(int, record["count"]) if record else 0

    count = await conn.execute_write(_merge_relation_instances)
    logger.debug(
        "Neo4j: merged %d RELATION_INSTANCE edges for kb=%s, doc=%s",
        count or 0,
        kb_id,
        doc_id,
    )


async def _reattach_instances(
    tx: AsyncManagedTransaction, keep: str, remove: str, kb_id: str
) -> None:
    """合并 remove 到 keep：别名入库、实例归一化重挂、删除 remove 节点。

    归一化不变量：图中 EntityInstance.entity_name 一律为 canonical 名。
      1. remove 名称追加进 keep.aliases（去重），别名保留在图内
      2. remove 名下实例重挂 INSTANCE_OF 到 keep，entity_name 改为 keep
      3. 同文档已有 keep 实例（冲突）时：关系边与属性并入该实例后删除原实例
    """
    # 1) 别名注册 + 收集实例 + 删除别名节点（DETACH DELETE 级联删 INSTANCE_OF 边）
    result = await tx.run(
        """
        MERGE (keep:Entity {name: $keep, kb_id: $kb})
        WITH keep
        MATCH (d:Entity {name: $remove, kb_id: $kb})
        OPTIONAL MATCH (inst:EntityInstance {kb_id: $kb})-[:INSTANCE_OF]->(d)
        SET keep.aliases = [a IN coalesce(keep.aliases, []) + [$remove] WHERE a <> $keep]
        WITH keep, d, [i IN collect(inst) WHERE i IS NOT NULL] AS insts
        DETACH DELETE d
        RETURN [i IN insts | {id: elementId(i), doc_id: i.doc_id}] AS insts
        """,
        keep=keep, remove=remove, kb=kb_id,
    )
    record = await result.single()
    insts = record["insts"] if record else []

    for inst in insts:
        # 2) 同文档是否已有 keep 实例（冲突检测）
        dup_result = await tx.run(
            """
            MATCH (dup:EntityInstance {doc_id: $doc_id, entity_name: $keep, kb_id: $kb})
            RETURN dup
            """,
            doc_id=inst["doc_id"], keep=keep, kb=kb_id,
        )
        if await dup_result.single() is None:
            # 3a) 无冲突：改名 + 重挂 INSTANCE_OF
            await tx.run(
                """
                MATCH (inst:EntityInstance {id: $inst_id})
                SET inst.entity_name = $keep
                MERGE (inst)-[:INSTANCE_OF]->(:Entity {name: $keep, kb_id: $kb})
                """,
                inst_id=inst["id"], keep=keep, kb=kb_id,
            )
        else:
            # 3b) 有冲突：出/入边重挂到 dup + 属性并入 dup + 删除 inst
            await tx.run(
                """
                MATCH (inst:EntityInstance {id: $inst_id})
                MATCH (dup:EntityInstance {doc_id: $doc_id, entity_name: $keep, kb_id: $kb})
                // 出边重挂
                MATCH (inst)-[r:RELATION_INSTANCE]->(t:EntityInstance)
                MERGE (dup)-[r2:RELATION_INSTANCE {
                    type: r.type, kb_id: r.kb_id, doc_id: r.doc_id
                }]->(t)
                SET r2.description = CASE
                        WHEN coalesce(r2.description, '') = '' THEN coalesce(r.description, '')
                        WHEN r.description IS NULL OR r.description = '' THEN r2.description
                        ELSE r2.description + '<SEP>' + r.description
                    END,
                    r2.weight = coalesce(r2.weight, 0.0) + coalesce(r.weight, 0.0)
                DELETE r
                // 入边重挂
                WITH inst, dup
                MATCH (s:EntityInstance)-[r:RELATION_INSTANCE]->(inst)
                MERGE (s)-[r2:RELATION_INSTANCE {
                    type: r.type, kb_id: r.kb_id, doc_id: r.doc_id
                }]->(dup)
                SET r2.description = CASE
                        WHEN coalesce(r2.description, '') = '' THEN coalesce(r.description, '')
                        WHEN r.description IS NULL OR r.description = '' THEN r2.description
                        ELSE r2.description + '<SEP>' + r.description
                    END,
                    r2.weight = coalesce(r2.weight, 0.0) + coalesce(r.weight, 0.0)
                DELETE r
                // 实例属性并入 dup（chunk_ids 去重追加）
                WITH inst, dup
                SET dup.weight = coalesce(dup.weight, 0.0) + coalesce(inst.weight, 0.0),
                    dup.chunk_ids = [x IN coalesce(dup.chunk_ids, []) + coalesce(inst.chunk_ids, [])
                                     WHERE NOT x IN coalesce(dup.chunk_ids, [])],
                    dup.description = CASE
                        WHEN coalesce(dup.description, '') = '' THEN coalesce(inst.description, '')
                        WHEN inst.description IS NULL OR inst.description = '' THEN dup.description
                        ELSE dup.description + '<SEP>' + inst.description
                    END
                DETACH DELETE inst
                """,
                inst_id=inst["id"], doc_id=inst["doc_id"],
                keep=keep, kb=kb_id,
            )


async def merge_entities(kb_id: str, keep: str, remove: str) -> None:
    """实体消歧：将 remove 名下实例重挂到 keep，删除 remove 节点。"""
    conn = Neo4jConnection()
    await conn.execute_write(_reattach_instances, keep, remove, kb_id)
    logger.info("Neo4j: merged entity '%s' -> '%s'", remove, keep)


async def merge_aliases(kb_id: str, canonical: str, aliases: list[str]) -> None:
    """同义实体合并：将别名名下实例重挂到 canonical，删除别名节点。"""
    conn = Neo4jConnection()
    for alias in aliases:
        await conn.execute_write(_reattach_instances, canonical, alias, kb_id)
    logger.info("Neo4j: merged %d aliases into '%s'", len(aliases), canonical)


async def delete_document_graph(kb_id: str, doc_id: str) -> None:
    """删除文档在 Neo4j 图谱中的所有贡献（幂等）。

    流程：
    1. 删除该文档所有 RELATION_INSTANCE 边
    2. 删除所有 EntityInstance 节点（级联 INSTANCE_OF 关系）
    3. 清理没有 EntityInstance 指向的孤立 Entity
    """
    conn = Neo4jConnection()

    async def _delete(tx: AsyncManagedTransaction) -> None:
        # 删除该文档 EntityInstance 之间的 RELATION_INSTANCE 边
        await tx.run(
            """
            MATCH (a:EntityInstance {doc_id: $doc_id, kb_id: $kb_id})
            MATCH (a)-[r:RELATION_INSTANCE]->()
            DELETE r
            """,
            doc_id=doc_id, kb_id=kb_id,
        )
        # 删除 EntityInstance（级联 INSTANCE_OF 关系）
        await tx.run(
            """
            MATCH (n:EntityInstance {doc_id: $doc_id, kb_id: $kb_id})
            DETACH DELETE n
            """,
            doc_id=doc_id, kb_id=kb_id,
        )
        # 清理孤立 Entity
        await tx.run(
            """
            MATCH (e:Entity {kb_id: $kb_id})
            WHERE NOT EXISTS {
                MATCH (e)<-[:INSTANCE_OF]-(:EntityInstance)
            }
            DETACH DELETE e
            """,
            kb_id=kb_id,
        )

    await conn.execute_write(_delete)
    logger.info("Neo4j: 删除文档 %s 的图谱贡献（kb=%s）", doc_id, kb_id)


async def delete_graph(kb_id: str) -> None:
    """删除整个 KB 的图谱数据（EntityInstance + RELATION_INSTANCE + Entity）。"""
    conn = Neo4jConnection()

    async def _delete(tx: AsyncManagedTransaction, kb: str) -> None:
        # 删除 EntityInstance 节点（级联 INSTANCE_OF 和 RELATION_INSTANCE 关系）
        await tx.run(
            """
            MATCH (n:EntityInstance {kb_id: $kb})
            DETACH DELETE n
            """,
            kb=kb,
        )
        # 删除 Entity 节点
        await tx.run(
            """
            MATCH (n:Entity {kb_id: $kb})
            DETACH DELETE n
            """,
            kb=kb,
        )

    await conn.execute_write(_delete, kb_id)
    logger.info("Neo4j: deleted graph for kb=%s", kb_id)


async def get_stats(kb_id: str) -> dict[str, Any]:
    """获取 KB 的图谱统计（EntityInstance + RELATION_INSTANCE + 关联文档数）。"""
    conn = Neo4jConnection()

    async def _stats(tx: AsyncManagedTransaction, kb: str) -> dict[str, Any]:
        result = await tx.run(
            """
            MATCH (n:EntityInstance {kb_id: $kb})
            RETURN count(n) AS entity_count
            """,
            kb=kb,
        )
        record = await result.single()
        entity_count = record["entity_count"] if record else 0

        result2 = await tx.run(
            """
            MATCH ()-[r:RELATION_INSTANCE {kb_id: $kb}]->()
            RETURN count(r) AS relation_count
            """,
            kb=kb,
        )
        record2 = await result2.single()
        relation_count = record2["relation_count"] if record2 else 0

        result3 = await tx.run(
            """
            MATCH (i:EntityInstance {kb_id: $kb})
            RETURN count(DISTINCT i.doc_id) AS doc_count
            """,
            kb=kb,
        )
        record3 = await result3.single()
        doc_count = record3["doc_count"] if record3 else 0

        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "doc_count": doc_count,
        }

    return await conn.execute_read(_stats, kb_id) or {
        "entity_count": 0, "relation_count": 0, "doc_count": 0,
    }

"""RAPTOR 树遍历检索 — 选树 → children_ids 逐层剪枝 → 叶子 top-N → 路径拼装。

摘要链沿"遍历父"传播（下钻时经 children_ids JOIN 带出 frontier 归属，DISTINCT ON
取最相似父）：软聚类共享子节点（存储 parent_id 指向首父、首父不在 beam 内）继承
下钻到它的父的链，不截断。
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text as sql_text

from aion_knowledge.common.config import settings
from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext

logger = logging.getLogger(__name__)


def _row_map(row: Any) -> dict[str, Any]:
    """SQLAlchemy Row 或测试 SimpleNamespace 统一转 dict。"""
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    return dict(vars(row))


class RAPTORRetriever(BaseRetriever):
    name = "raptor"

    async def retrieve(self, ctx: RetrieverContext) -> list[ChunkResult]:
        if not ctx.query_embedding:
            logger.warning("RAPTORRetriever: query_embedding is None, skipping")
            return []
        embedding_str = "[" + ",".join(str(v) for v in ctx.query_embedding) + "]"
        async with get_session() as session:
            # Stage 1：选树 — root（parent_id IS NULL）按向量相似度取 top_trees
            root_rows = await session.execute(
                sql_text("""
                    SELECT r.id, r.doc_id, r.layer, r.summary, r.source_chunk_ids,
                           r.children_ids
                    FROM chunk_raptor r
                    WHERE r.kb_id = :kb_id
                      AND r.embedding IS NOT NULL
                      AND r.parent_id IS NULL
                      AND (r.doc_id IS NULL OR NOT EXISTS (
                          SELECT 1 FROM doc_knowledge_documents d
                          WHERE d.id = r.doc_id AND d.deleted))
                      AND NOT EXISTS (SELECT 1 FROM kb_knowledge_bases k
                                      WHERE k.id = r.kb_id AND k.deleted)
                    ORDER BY r.embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :top_trees
                """),
                {"query_embedding": embedding_str, "kb_id": ctx.kb_id,
                 "top_trees": settings.raptor_traverse_top_trees},
            )

            # 遍历父链：节点 id -> 下钻路径上的完整摘要链（root → 该节点）
            traversal_chain: dict[str, list[str]] = {}
            # 节点 id -> 路径上最近有 doc_id 的祖先的 doc_id（同树 doc_id 一致）
            doc_chain: dict[str, str] = {}
            node_source_ids: dict[str, list[str]] = {}
            paths: list[dict[str, Any]] = []  # {chain, source_ids, doc_id, bottom_id}
            seen_bottoms: set[str] = set()
            for row in root_rows:
                row_map = _row_map(row)
                rid = str(row_map["id"])
                traversal_chain[rid] = [row_map.get("summary") or ""]
                if row_map.get("doc_id"):
                    doc_chain[rid] = str(row_map["doc_id"])
                node_source_ids[rid] = [
                    str(s) for s in (row_map.get("source_chunk_ids") or [])
                ]
                # 退化树（root 即叶子，children_ids 为空）：直接发射 [root] 路径；
                # children_ids 缺失（mock 行）视为非退化，保持原下钻行为
                if row_map.get("children_ids") == []:
                    seen_bottoms.add(rid)
                    paths.append({
                        "bottom_id": rid,
                        "source_ids": node_source_ids[rid],
                        "chain": traversal_chain[rid],
                        "doc_id": doc_chain.get(rid, ""),
                    })

            # Stage 2：逐层剪枝 — children_ids 下钻（软聚类 DAG 不漏），每层取 beam 个。
            # JOIN 带出遍历父（frontier 归属）：共享子节点继承"下钻到它的父"的链，
            # 不依赖存储 parent_id（首父可能不在 beam 内，沿它回溯会截断）；
            # DISTINCT ON (r.id) 先按节点去重（保留 dist 最小 = 最相似父的行）再取 beam，
            # 重复行不消耗 beam 槽位
            frontier_ids = [str(r.id) for r in root_rows]
            while frontier_ids:
                level_rows = await session.execute(
                    sql_text("""
                        SELECT * FROM (
                            SELECT DISTINCT ON (r.id)
                                   r.id, r.doc_id, r.layer, r.summary, r.source_chunk_ids,
                                   r.parent_id, e.parent_id AS via_parent,
                                   (r.embedding <=> CAST(:query_embedding AS vector)) AS dist
                            FROM chunk_raptor r
                            JOIN (
                                SELECT p.id AS parent_id, unnest(p.children_ids) AS child_id
                                FROM chunk_raptor p
                                WHERE p.id = ANY(:parent_ids)
                            ) e ON e.child_id = r.id
                            WHERE r.kb_id = :kb_id
                              AND r.embedding IS NOT NULL
                              AND (r.doc_id IS NULL OR NOT EXISTS (
                                  SELECT 1 FROM doc_knowledge_documents d
                                  WHERE d.id = r.doc_id AND d.deleted))
                              AND NOT EXISTS (SELECT 1 FROM kb_knowledge_bases k
                                              WHERE k.id = r.kb_id AND k.deleted)
                            ORDER BY r.id, dist
                        ) t
                        ORDER BY dist
                        LIMIT :beam
                    """),
                    {"query_embedding": embedding_str, "parent_ids": frontier_ids,
                     "kb_id": ctx.kb_id, "beam": settings.raptor_traverse_beam},
                )
                chosen = list(level_rows)
                if not chosen:
                    # 断链回退：frontier 中尚未发射的节点发射为路径（children_ids
                    # 悬空/无子节点时节点本身仍可用，避免静默丢失；退化 root 已在
                    # Stage 1 发射并记入 seen_bottoms，不会重复）
                    for rid in frontier_ids:
                        if rid in seen_bottoms:
                            continue
                        seen_bottoms.add(rid)
                        paths.append({
                            "bottom_id": rid,
                            "source_ids": node_source_ids.get(rid, []),
                            "chain": traversal_chain.get(rid, [""]),
                            "doc_id": doc_chain.get(rid, ""),
                        })
                    break
                next_ids: list[str] = []
                for row in chosen:
                    row_map = _row_map(row)
                    rid = str(row_map["id"])
                    if rid in seen_bottoms:
                        continue  # 软聚类下同一节点可被多个父下钻到，按 id 去重（先到先得）
                    seen_bottoms.add(rid)
                    via_parent = str(row_map["via_parent"])
                    traversal_chain[rid] = traversal_chain.get(via_parent, []) + [
                        row_map.get("summary") or ""
                    ]
                    doc_chain[rid] = (
                        str(row_map["doc_id"]) if row_map.get("doc_id")
                        else doc_chain.get(via_parent, "")
                    )
                    node_source_ids[rid] = [
                        str(s) for s in (row_map.get("source_chunk_ids") or [])
                    ]
                    if int(row_map.get("layer") or 1) <= 1:
                        paths.append({  # 叶子层：路径终止
                            "bottom_id": rid,
                            "source_ids": node_source_ids[rid],
                            "chain": traversal_chain[rid],
                            "doc_id": doc_chain[rid],
                        })
                    else:
                        next_ids.append(rid)
                frontier_ids = next_ids

            # Stage 3：叶子 top-N + 路径拼装
            results: list[ChunkResult] = []
            for path in paths:
                source_ids = path["source_ids"]
                leaf_contents: list[str] = []
                if source_ids:
                    leaf_rows = await session.execute(
                        sql_text("""
                            SELECT ct.content
                            FROM chunk_text ct
                            JOIN chunk_vector cv ON cv.chunk_id = ct.id
                            JOIN doc_knowledge_documents d ON d.id = ct.document_id AND NOT d.deleted
                            WHERE ct.kb_id = :kb_id
                              AND ct.id = ANY(:source_ids)
                            ORDER BY cv.embedding <=> CAST(:query_embedding AS vector)
                            LIMIT :leaf_topk
                        """),
                        {"query_embedding": embedding_str, "source_ids": source_ids,
                         "kb_id": ctx.kb_id,
                         "leaf_topk": settings.raptor_traverse_leaf_topk},
                    )
                    leaf_contents = [
                        (_row_map(r).get("content") or "")[
                            : settings.raptor_traverse_max_leaf_chars
                        ]
                        for r in leaf_rows
                    ]
                    if not leaf_contents:
                        # 回退：无 chunk_vector 行时按 source_chunk_ids 顺序取前 N（有胜于无）
                        leaf_contents = source_ids[: settings.raptor_traverse_leaf_topk]

                content = "\n\n".join(
                    [c for c in path["chain"] if c] + leaf_contents
                )
                results.append(ChunkResult(
                    chunk_id=path["bottom_id"],
                    kb_id=ctx.kb_id,
                    document_id=path["doc_id"],
                    content=content,
                    score=1.0,  # RRF 只用 rank，reranker 会重排
                    source_paths=[self.name],
                    metadata={
                        "source_chunk_ids": source_ids[: settings.raptor_traverse_leaf_topk],
                    },
                ))
        logger.debug("RAPTORRetriever: %d results", len(results))
        return results

"""GraphExtractModule — 从 chunks 并发提取实体/关系到 Neo4j。

作用：
  通过 LLM 从每个 chunk 中提取实体（name + type + description）和
  关系（source → target + type + weight），经跨文档合并写入 Neo4j 知识图谱。

核心机制：
  - 并发控制：通过 Semaphore 限制同时提取的 chunk 数
    （max_concurrent 可配）
  - Gleaning 多轮补充：首次提取后，对已有实体列表发起多轮补充提取，
    直到无新实体或达到 max_gleanings 上限
  - 文档内去重：同一文档内的同名实体合并 descriptions 和 source_chunks
  - 关系权重累加：同一 (source, target, type) 关系的 weight 累加
  - 跨文档合并：通过 KBGraphMerger 将本次提取结果聚合到 Neo4j 知识图谱
  - 可选启用，依赖 text 模块先行

设计约束：
  图谱与知识库 1:1：同一 KB 下所有文档的实体/关系合并进同一个图
  （Neo4j 按 kb_id 标识），不按文档粒度建独立图。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from aion_knowledge.infrastructure.llm import LLMClient, get_llm_client_for_module
from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule
from aion_knowledge.pipeline.postproc.graph_extract.config import graph_config

logger = logging.getLogger(__name__)


class GraphExtractModule(PostProcModule):
    """实体/关系提取模块。二批执行，依赖 text，结果合并写入 Neo4j。"""

    always_on = False
    depends_on = ["text"]

    async def process(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> int:
        """并发提取全部 chunks 的实体关系并触发跨文档合并，返回实体数。"""
        if not chunks:
            return 0
        llm = get_llm_client_for_module("graph_extract")
        max_concurrent = graph_config.max_concurrent
        max_gleanings = graph_config.max_gleanings
        entity_types = graph_config.entity_types or ["person", "organization", "location", "event", "concept", "product"]

        sem = asyncio.Semaphore(max_concurrent)

        async def process_chunk(
            chunk: dict[str, Any],
        ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
            """单 chunk 提取：内容守卫后调用 _extract_with_gleaning。"""
            async with sem:
                content = chunk.get("content", "").strip()
                chunk_uuid = chunk.get("chunk_uuid", "")
                if not content or not chunk_uuid:
                    return chunk_uuid, [], []
                if not self._check_content(content):
                    logger.warning("【图谱提取】内容过少，跳过 chunk=%s", chunk_uuid)
                    return chunk_uuid, [], []
                entities, relations = await self._extract_with_gleaning(
                    llm, content, entity_types, max_gleanings
                )
                return chunk_uuid, entities, relations

        results = await asyncio.gather(*[process_chunk(c) for c in chunks])

        # 文档内去重
        all_entities: dict[str, dict[str, Any]] = {}
        all_relations: dict[tuple[str, str, str], dict[str, Any]] = {}
        for chunk_uuid, entities, relations in results:
            for ent in entities:
                name = ent["name"]
                if name in all_entities:
                    all_entities[name]["descriptions"].append(ent.get("description", ""))
                    if chunk_uuid not in all_entities[name]["source_chunks"]:
                        all_entities[name]["source_chunks"].append(chunk_uuid)
                else:
                    all_entities[name] = {
                        "type": ent.get("type", ""),
                        "descriptions": [ent.get("description", "")],
                        "source_chunks": [chunk_uuid],
                    }
            for rel in relations:
                key = (rel["source"], rel["target"], rel["type"])
                if key in all_relations:
                    all_relations[key]["weight"] += rel.get("weight", 1)
                    all_relations[key]["descriptions"].append(rel.get("description", ""))
                    if chunk_uuid not in all_relations[key]["source_chunks"]:
                        all_relations[key]["source_chunks"].append(chunk_uuid)
                else:
                    all_relations[key] = {
                        "source": rel["source"],
                        "target": rel["target"],
                        "type": rel["type"],
                        "weight": rel.get("weight", 1),
                        "descriptions": [rel.get("description", "")],
                        "source_chunks": [chunk_uuid],
                    }

        if not all_entities:
            return 0

        # 触发跨文档合并（写入 Neo4j）
        await self._trigger_merge(ctx.kb_id, ctx.document_id, all_entities, all_relations)

        logger.info("【图谱提取】处理完成：%d 实体，%d 关系，文档=%s",
                     len(all_entities), len(all_relations), ctx.doc_name)
        return len(all_entities)

    async def _extract_with_gleaning(
        self, llm: LLMClient, content: str, entity_types: list[str], max_gleanings: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """对单个 chunk 执行含 gleaning 补充的实体关系提取。"""
        from aion_knowledge.pipeline.postproc.graph_extract.extractor import (
            extract_entities_with_gleaning,
        )
        return await extract_entities_with_gleaning(llm, content, entity_types, max_gleanings)

    async def _trigger_merge(
        self, kb_id: str, doc_id: str,
        entities: dict[str, dict[str, Any]],
        relations: dict[tuple[str, str, str], dict[str, Any]],
    ) -> None:
        """触发跨文档合并。"""
        try:
            from aion_knowledge.pipeline.postproc.graph_extract.merger import KBGraphMerger
            merger = KBGraphMerger()
            await merger.merge_document(kb_id, doc_id, entities, relations)
        except Exception as exc:
            logger.error("【图谱提取】知识图谱合并失败：文档=%s，错误=%s", doc_id, exc)


def module() -> GraphExtractModule:
    """模块工厂函数，供调度器自动发现。"""
    return GraphExtractModule()

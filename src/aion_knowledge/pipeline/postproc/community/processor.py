"""CommunityModule — 知识图谱社区发现与报告生成。

作用：
  从已消歧的 Neo4j 知识图谱中读取实体和关系，构建 NetworkX 图，
  通过层次化社区检测发现紧密关联的实体群组，生成 LLM 驱动的
  社区摘要报告，写入 chunk_community 表。

核心机制：
  - 数据源：优先读取 Neo4j（已消歧的全局知识图谱）
  - 社区检测：层次化 Leiden 算法（hierarchical），支持 max_cluster_size
    限制，控制社区粒度；降级到基础 Leiden
  - 检查点（Checkpoint）：图 hash 未变时跳过重复检测，避免无效 LLM 调用
  - 报告生成：每个社区按层级（level 0/1）使用不同的 Prompt 模板，
    包含 title、summary、findings（含 explanation）和 rating 评分
  - 回退策略：无 Neo4j 图时，回退到 chunk 级 LLM 提取实体→构图→社区检测

依赖关系：
  - 依赖 text 模块先行（chunks 数据源）
  - 依赖 disambiguation 模块先行（消歧后的知识图谱）
  - 两个回退场景：有图（从 Neo4j 读）和无图（chunk 级提取）
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import networkx as nx  # type: ignore[import-untyped]  # networkx 无 stub，模块整体按 Any 处理
from sqlalchemy import text

from aion_knowledge.common.community_text import build_community_text
from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.infrastructure.embedder import create_embedder
from aion_knowledge.infrastructure.llm import LLMClient, get_llm_client_for_module
from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule
from aion_knowledge.pipeline.postproc.community.config import community_config
from aion_knowledge.pipeline.postproc.community.leiden import (
    detect_communities,
)

logger = logging.getLogger(__name__)

COMMUNITY_REPORT_PROMPT = {
    0: "以下是一个大型实体社区（紧密关联的实体群组）的摘要。请描述该社区的整体主题、核心实体和主要关系。"
       "以 JSON 格式输出："
       '{{"title": "社区标题", "summary": "高层摘要", '
       '"findings": [{{"summary": "发现项", "explanation": "详细解释"}}], '
       '"rating": 1-10}}\n\n实体：{entities}\n关系：{relations}',
    1: "以下是一组密切相关的实体。请详细描述它们之间的关系和交互方式。"
       "以 JSON 格式输出："
       '{{"title": "社区标题", "summary": "概要", '
       '"findings": [{{"summary": "发现项", "explanation": "详细解释"}}], '
       '"rating": 1-10}}\n\n实体：{entities}\n关系：{relations}',
}


class CommunityModule(PostProcModule):
    """社区发现与报告生成模块。二批执行，依赖 text + disambiguation。"""

    always_on = False
    depends_on = ["text", "disambiguation"]

    async def process(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> int:
        """对 KB 图执行社区发现并生成报告，返回写入的社区记录数。"""
        if not chunks:
            return 0

        kb_id = ctx.kb_id
        from aion_knowledge.infrastructure.graph import load_kb_graph
        entities, relations = await load_kb_graph(kb_id)

        # 检查点：图 hash 未变则跳过社区检测
        cpm = None
        current_hash: str | None = None
        if entities:
            from aion_knowledge.pipeline.postproc.community.checkpoint import (
                CommunityCheckpointManager,
                compute_graph_hash,
            )
            current_hash = compute_graph_hash(
                [e["entity_name"] for e in entities],
                [(r["source_entity"], r["target_entity"], r["relation_type"])
                 for r in relations],
            )
            if community_config.enable_checkpoint:
                cpm = CommunityCheckpointManager()
                if await cpm.should_skip(kb_id, current_hash):
                    logger.info("【社区发现】检查点命中，跳过知识库 %s", kb_id)
                    return await self._backfill_missing_embeddings(kb_id)

        if not entities:
            logger.info("【社区发现】未找到知识图谱，回退到 chunk 级提取")
            return await self._fallback_process(ctx, chunks)

        graph = nx.Graph()
        for ent in entities:
            graph.add_node(ent["entity_name"], type=ent.get("entity_type", ""))
        for rel in relations:
            s, t = rel["source_entity"], rel["target_entity"]
            if s in graph and t in graph:
                graph.add_edge(s, t, type=rel.get("relation_type", ""),
                           weight=rel.get("weight", 1.0))

        if graph.number_of_nodes() == 0:
            return 0

        try:
            from aion_knowledge.pipeline.postproc.community.leiden import (
                detect_communities_hierarchical,
            )
            max_cluster_size = community_config.max_cluster_size
            communities = detect_communities_hierarchical(
                graph, max_cluster_size=max_cluster_size
            )
        except ImportError:
            communities = detect_communities(graph)

        if not communities:
            return 0

        llm = get_llm_client_for_module("community")
        inserted = await self._generate_and_save_reports(llm, kb_id, communities, relations)
        # 刷新 KB 图谱统计（社区数随本次写入变化）；失败不阻断主流程
        try:
            from aion_knowledge.pipeline.postproc.graph_extract.merger import update_kb_graph_stats
            await update_kb_graph_stats(kb_id)
        except Exception as exc:
            logger.warning("【社区发现】统计刷新失败：%s", exc)
        # 保存检查点
        if cpm is not None and current_hash is not None:
            try:
                await cpm.save(kb_id, current_hash)
            except Exception as exc:
                logger.warning("【社区发现】检查点保存失败：%s", exc)
        logger.info("【社区发现】处理完成：%d 个社区，知识库=%s", inserted, kb_id)
        return inserted

    async def _generate_and_save_reports(
        self, llm: LLMClient, kb_id: str,
        communities: list[dict[str, Any]], all_relations: list[dict[str, Any]],
    ) -> int:
        """为所有社区生成报告并批量写入 chunk_community，返回写入数。"""
        from aion_knowledge.pipeline.postproc.community.orm import ChunkCommunity

        # 阶段 1：先完成全部 LLM 调用（不持有 DB 连接），收集报告数据
        reports: list[dict[str, Any]] = []
        for comm in communities:
            level = comm["level"]
            prompt = COMMUNITY_REPORT_PROMPT.get(level, COMMUNITY_REPORT_PROMPT[0])
            members = comm["members"]
            related_relations = [
                r for r in all_relations
                if r["source_entity"] in members and r["target_entity"] in members
            ]
            report = await self._generate_report(llm, prompt, members, related_relations)
            if not report:
                continue
            reports.append({
                "community_id": comm["id"],
                "level": level,
                "members": members,
                "report": report,
            })

        # 阶段 1.5：批量生成摘要向量（不持有 DB 连接；失败不阻断入库，向量留 NULL）
        await self._embed_reports(reports)

        # 阶段 2：获取会话，一次批量写入
        inserted = 0
        async with get_session() as session:
            session.add_all([
                ChunkCommunity(
                    chunk_id=uuid.UUID(int=0),  # KB 级社区，零值 UUID 表示非 chunk 级
                    kb_id=uuid.UUID(kb_id),
                    community_id=r["community_id"],
                    community_level=r["level"],
                    summary=r["report"].get("summary", ""),
                    findings=r["report"].get("findings", []),
                    embedding=r.get("embedding"),
                    payload={
                        "title": r["report"].get("title", ""),
                        "rating": r["report"].get("rating", 0),
                        "members": r["members"],
                        "source": "community_module",
                    },
                )
                for r in reports
            ])
            await session.flush()
            inserted = len(reports)
        return inserted

    async def _generate_report(self, llm: LLMClient, prompt: str, members: list[str],
                                relations: list[dict[str, Any]]) -> dict[str, Any] | None:
        """调用 LLM 为单个社区生成结构化报告（title/summary/findings/rating）。"""
        try:
            return await llm.generate_structured(
                prompt.format(
                    entities=", ".join(members),
                    relations=json.dumps(relations, ensure_ascii=False),
                ),
                output_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "summary": {"type": "string"},
                                    "explanation": {"type": "string"},
                                },
                                "required": ["summary", "explanation"],
                            },
                        },
                        "rating": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["title", "summary", "findings", "rating"],
                },
            )
        except Exception as exc:
            logger.warning("【社区发现】报告生成失败：%s", exc)
            return None

    async def _embed_reports(self, reports: list[dict[str, Any]]) -> None:
        """批量生成摘要向量并回填 reports 的 embedding 字段；失败全部留 None。"""
        try:
            provider = create_embedder()
            texts = [
                build_community_text(
                    r["report"].get("title", ""),
                    r["report"].get("summary", ""),
                    r["report"].get("findings", []),
                )
                for r in reports
            ]
            embeddings = await provider.embed_documents(texts) if texts else []
        except Exception as exc:
            logger.warning("【社区发现】摘要向量生成失败：%s", exc)
            embeddings = []
        if embeddings and len(embeddings) != len(reports):
            logger.warning("【社区发现】向量数量不匹配：%d 个向量 vs %d 个报告", len(embeddings), len(reports))
        for r, emb in zip(reports, embeddings):
            r["embedding"] = emb if emb else None

    async def _backfill_missing_embeddings(self, kb_id: str) -> int:
        """检查点命中后，为 embedding IS NULL 的存量社区补向量（不重跑 LLM）。"""
        async with get_session() as session:
            rows = (await session.execute(
                text("""SELECT id, COALESCE(payload->>'title', '') AS title,
                           COALESCE(summary, '') AS summary, COALESCE(findings, '[]') AS findings
                    FROM chunk_community
                    WHERE kb_id = CAST(:kb_id AS uuid) AND embedding IS NULL"""),
                {"kb_id": kb_id},
            )).fetchall()
        if not rows:
            return 0

        try:
            provider = create_embedder()
            texts = [build_community_text(r.title, r.summary, r.findings) for r in rows]
            embeddings = await provider.embed_documents(texts)
        except Exception as exc:
            logger.warning("【社区发现】补向量失败：%s", exc)
            return 0

        if embeddings and len(embeddings) != len(rows):
            logger.warning("【社区发现】补向量数量不匹配：%d 个向量 vs %d 条记录", len(embeddings), len(rows))

        updated = 0
        async with get_session() as session:
            for r, emb in zip(rows, embeddings):
                if not emb:
                    continue
                await session.execute(
                    text("UPDATE chunk_community SET embedding = CAST(:emb AS vector) WHERE id = CAST(:id AS uuid)"),
                    {"emb": str(emb), "id": r.id},
                )
                updated += 1
            await session.commit()
        return updated

    async def _fallback_process(self, ctx: PostProcContext,
                                 chunks: list[dict[str, Any]]) -> int:
        """无知识图谱时的回退路径：chunk 级提取实体 → 构图 → 社区发现。"""
        from aion_knowledge.pipeline.postproc.community.orm import ChunkCommunity
        from aion_knowledge.pipeline.postproc.graph_extract.extractor import (
            extract_entities_with_gleaning,
        )

        llm = get_llm_client_for_module("community")
        all_entities: list[tuple[str, str, str]] = []
        all_relations: list[dict[str, Any]] = []

        for chunk in chunks:
            content = chunk.get("content", "").strip()
            chunk_uuid = chunk.get("chunk_uuid", "")
            if not content or not chunk_uuid:
                continue
            if not self._check_content(content):
                logger.warning("【社区发现】内容过少，跳过 chunk=%s", chunk_uuid)
                continue
            # 改用共享提取（取代 _fallback_extract）
            extracted_entities, extracted_relations = await extract_entities_with_gleaning(
                llm, content,
                entity_types=["person", "organization", "location", "event", "concept", "product"],
                max_gleanings=1,
            )
            for ent in extracted_entities:
                all_entities.append((ent["name"], ent.get("type", ""), chunk_uuid))
            all_relations.extend(extracted_relations)

        if not all_entities:
            return 0

        graph = nx.Graph()
        for name, etype, _ in all_entities:
            graph.add_node(name, type=etype)
        for rel in all_relations:
            s, t = rel["source"], rel["target"]
            if s in graph and t in graph:
                graph.add_edge(s, t, type=rel["type"], weight=rel.get("weight", 1.0))

        communities = detect_communities(graph)
        if not communities:
            return 0

        entity_name_to_chunk = {e[0]: e[2] for e in all_entities}

        # 阶段 1：先完成全部 LLM 调用（不持有 DB 连接），收集报告数据
        reports: list[dict[str, Any]] = []
        for comm in communities:
            level = comm.get("level", 0)
            prompt = COMMUNITY_REPORT_PROMPT.get(level, COMMUNITY_REPORT_PROMPT[0])
            members = comm["members"]
            related = [r for r in all_relations
                       if r["source"] in members and r["target"] in members]
            report = await self._generate_report(llm, prompt, members, related)
            if not report:
                continue

            member_chunk = ""
            for member in members:
                if member in entity_name_to_chunk:
                    member_chunk = entity_name_to_chunk[member]
                    break
            cid = member_chunk or chunks[0].get("chunk_uuid", "")
            reports.append({
                "community_id": comm["id"],
                "level": level,
                "members": members,
                "report": report,
                "chunk_uuid": cid,
            })

        # 阶段 1.5：批量生成摘要向量（不持有 DB 连接；失败不阻断入库，向量留 NULL）
        await self._embed_reports(reports)

        # 阶段 2：获取会话，一次批量写入
        inserted = 0
        async with get_session() as session:
            session.add_all([
                ChunkCommunity(
                    chunk_id=uuid.UUID(r["chunk_uuid"]) if r["chunk_uuid"] else uuid.UUID(int=0),
                    kb_id=uuid.UUID(ctx.kb_id),
                    community_id=r["community_id"],
                    community_level=r["level"],
                    summary=r["report"].get("summary", ""),
                    findings=r["report"].get("findings", []),
                    embedding=r.get("embedding"),
                    payload={
                        "title": r["report"].get("title", ""),
                        "rating": r["report"].get("rating", 0),
                        "members": r["members"],
                        "source": "community_fallback",
                    },
                )
                for r in reports
            ])
            await session.flush()
            inserted = len(reports)
        # 刷新 KB 图谱统计（fallback 路径同样产生社区）；失败不阻断主流程
        try:
            from aion_knowledge.pipeline.postproc.graph_extract.merger import update_kb_graph_stats
            await update_kb_graph_stats(ctx.kb_id)
        except Exception as exc:
            logger.warning("【社区发现】统计刷新失败：%s", exc)
        logger.info("【社区发现】回退完成：%d 个社区，文档=%s", inserted, ctx.doc_name)
        return inserted

def module() -> CommunityModule:
    """模块工厂函数，供调度器自动发现。"""
    return CommunityModule()

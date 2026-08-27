"""WikiModule — MAP→REDUCE→PLAN→REFINE 四阶段 Wiki 构建。

作用：
  从文档 chunks 中自动提取候选百科页面，生成 Wiki 风格的 Markdown
  内容，写入 chunk_wiki 表。

四阶段流程：
  Phase 1 — MAP（提取候选）：LLM 从整篇文档（chunks 组合原文）一次
    提取候选概念/实体，包含 term、type（concept|entity）、reason。
  Phase 2 — REDUCE（跨文档合并）：与 KB 已有页面合并——slug 精确命中
    直接 merge，未命中交 LLM 判同。
  Phase 3 — PLAN（暂未独立）：REDUCE 后直接进入 REFINE。
  Phase 4 — REFINE（页面生成）：LLM 为每个候选生成 Wiki 页面，
    包含 slug、概述、详细说明、相关概念等章节；正文以 [[slug]]
    wikilink 互链（new 分支生成含链接正文，merge 分支仅追加引用）。

输出：
  - chunk_wiki 表：page_slug、page_title、content（Markdown，含
    [[slug]] wikilink）、chunk_refs/source_refs（引用 chunk 与贡献
    文档）、out_links/in_links（wikilink 互链）、taxonomy_path、
    status=draft
  - 可选启用，依赖 text 模块先行
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from aion_knowledge.common.config import settings
from aion_knowledge.common.model_registry import get_model_max_input_tokens, truncate_by_tokens
from aion_knowledge.infrastructure.llm import LLMClient, get_llm_client_for_module
from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule

logger = logging.getLogger(__name__)

WIKI_MAP_PROMPT = (
    "从以下文本中提取可以作为 Wiki 页面的候选概念或实体。\n\n"
    "文本：{content}\n\n"
    "请以 JSON 格式输出，结构如下：\n"
    "{{\"candidates\": [{{\"term\": \"概念或实体名称\", \"type\": \"concept 或 entity\", \"reason\": \"为何适合作为 Wiki 页面\"}}]}}"
)

WIKI_MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "type": {"type": "string", "enum": ["concept", "entity"]},
                    "reason": {"type": "string"},
                },
                "required": ["term", "type", "reason"],
            },
        },
    },
    "required": ["candidates"],
}

WIKI_DEDUP_PROMPT = (
    "判断候选概念是否与已有 Wiki 页面指代同一概念。\n\n"
    "候选：{term}\n已有页面：{titles}\n\n"
    "若同一概念输出 {{\"is_same\": true, \"existing_slug\": \"...\"}}，"
    "否则输出 {{\"is_same\": false}}。"
)

WIKI_DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "is_same": {"type": "boolean"},
        "existing_slug": {"type": "string"},
    },
    "required": ["is_same"],
}

WIKI_REFINE_PROMPT = (
    "为以下 Wiki 页面生成 Markdown 内容。"
    "包含概述、详细说明、相关概念等章节。\n\n"
    "标题：{title}\n类型：{type}\n来源内容：{source_content}\n\n"
    "可用页面 slug 列表：{available_slugs}\n"
    "规则：正文中提及可用列表中的概念时，必须写作 [[slug]] 或 [[slug|显示名]] 形式的链接；"
    "slug 只能来自上述列表，不得发明。\n\n"
    "以 JSON 格式输出，包含 slug 和 content 两个字段。"
)

WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


class WikiModule(PostProcModule):
    """Wiki 百科构建模块。二批执行，依赖 text。"""

    always_on = False
    depends_on = ["text"]

    async def process(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> int:
        """提取候选 → 跨文档合并 → 生成页面，写入 chunk_wiki，返回新建+合并页数。"""
        if not chunks:
            await self._save_checkpoint(ctx, "no_candidates")
            return 0
        llm = get_llm_client_for_module("wiki")
        candidates = await self._extract_candidates(llm, chunks)
        if not candidates:
            await self._save_checkpoint(ctx, "no_candidates")
            return 0

        # 阶段 1：查询 KB 已有页面（短会话，关后不再持有连接）
        from sqlalchemy import text  # 延迟 import（processor.py 顶部无 sqlalchemy）

        from aion_knowledge.infrastructure.db import get_session
        existing: list[dict[str, str]] = []
        async with get_session() as session:
            rows = await session.execute(
                text("SELECT page_slug, page_title FROM chunk_wiki WHERE kb_id = :kb"),
                {"kb": uuid.UUID(ctx.kb_id)},
            )
            existing = [{"slug": r.page_slug, "title": r.page_title} for r in rows]

        # 阶段 2：跨文档合并（LLM，无 DB 连接）
        merged, new = await self._merge_with_existing(llm, candidates, existing)
        # 批内去重：同文档同 slug 的候选合并（防 uq_chunk_wiki_kb_slug 冲突），source_terms 累积
        seen: dict[str, dict[str, Any]] = {}
        for candidate in new:
            slug = self._to_slug(candidate["term"])
            if slug in seen:
                prev = seen[slug]
                prev["source_terms"] = [
                    *prev.get("source_terms", [prev["term"]]), candidate["term"],
                ]
            else:
                candidate["source_terms"] = [candidate["term"]]
                seen[slug] = candidate
        new = list(seen.values())
        # 页面 chunk 关联：文档级粗粒度（本文档全部 chunk）
        doc_chunks = sorted(
            {str(c.get("chunk_uuid", "")) for c in chunks if c.get("chunk_uuid")}
        )

        # 阶段 3：REFINE — new 生成页面；merge 仅追加引用不改 content
        existing_slugs = ([e["slug"] for e in existing] + [
            self._to_slug(c["term"]) for c in new
        ])[:20]  # 与 _judge_same 的标题列表上限一致
        valid_slugs = set(existing_slugs)
        pages: list[dict[str, Any]] = []
        for candidate in new:
            page = await self._generate_page(
                llm, candidate,
                source_content=_candidate_source(candidate, chunks),
                available_slugs=existing_slugs,
            )
            # 页面 slug 恒由 term 派生：LLM 返回的 slug 不落库（防撞唯一约束、链接落空）
            pages.append({
                "kb_id": ctx.kb_id,
                "slug": self._to_slug(candidate["term"]),
                "title": candidate["term"], "content": page.get("content", ""),
                "type": candidate.get("type", "concept"),
                # 批内去重已为候选写入累积的 source_terms，勿覆盖
                "source_terms": candidate.get("source_terms", [candidate["term"]]),
                "chunk_refs": [c for c in doc_chunks if c],
                # out_links 白名单过滤：仅可用列表内 slug，且排除自链接
                "out_links": [s for s in self._extract_out_links(page.get("content", ""))
                              if s in valid_slugs and s != self._to_slug(candidate["term"])],
            })
        merged_pages: list[dict[str, Any]] = []
        for candidate in merged:
            merged_pages.append({
                "existing_slug": candidate["existing_slug"],
                "chunk_refs": [c for c in doc_chunks if c],
                "source_ref": ctx.document_id,
                "source_terms": [candidate["term"]],
            })

        # 阶段 4：短会话批量写入（先 LLM 后写库，遵守基类纪律）
        from aion_knowledge.pipeline.postproc.wiki.orm import ChunkWiki  # 保留现有延迟 import
        inserted = len(pages)
        affected = 0
        async with get_session() as session:
            for p in pages:
                session.add(ChunkWiki(
                    kb_id=uuid.UUID(p["kb_id"]),
                    page_slug=p["slug"], page_title=p["title"],
                    content=p["content"],
                    chunk_refs=p["chunk_refs"], source_refs=[ctx.document_id],
                    out_links=p["out_links"],
                    taxonomy_path=f"/{p['type']}", status="draft",
                    payload={
                        "wiki_type": p["type"], "source_terms": p["source_terms"],
                        "source": "wiki_module",
                    },
                ))
            for m in merged_pages:
                row = await session.execute(
                    text("""
                        SELECT source_refs, chunk_refs, payload
                        FROM chunk_wiki WHERE kb_id = :kb AND page_slug = :slug
                    """),
                    {"kb": uuid.UUID(ctx.kb_id), "slug": m["existing_slug"]},
                )
                cur = row.first()
                if cur is None:
                    continue  # 目标页面已不存在（被删/未落库），跳过
                source_refs = list(dict.fromkeys([*cur.source_refs, m["source_ref"]]))
                chunk_refs = list(dict.fromkeys([*cur.chunk_refs, *m["chunk_refs"]]))
                payload = dict(cur.payload or {})
                terms = list(dict.fromkeys([*(payload.get("source_terms") or []), *m["source_terms"]]))
                payload["source_terms"] = terms
                await session.execute(
                    text("""
                        UPDATE chunk_wiki
                        SET source_refs = :refs, chunk_refs = :chunks, payload = :payload,
                            updated_at = now()
                        WHERE id = (
                            SELECT id FROM chunk_wiki
                            WHERE kb_id = :kb AND page_slug = :slug LIMIT 1
                        )
                    """),
                    # payload 需 json 序列化：text SQL 无类型标注，asyncpg 无法直接编码 dict
                    {"refs": source_refs, "chunks": chunk_refs,
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "kb": uuid.UUID(ctx.kb_id), "slug": m["existing_slug"]},
                )
                affected += 1
            # 先 flush 再维护 in_links：确保新页面已落库，UPDATE 目标必然命中
            await session.flush()
            for p in pages:
                for target in p["out_links"]:
                    await session.execute(
                        text("""
                            UPDATE chunk_wiki
                            SET in_links = array_append(in_links, :link_slug), updated_at = now()
                            WHERE kb_id = :kb AND page_slug = :target
                              AND NOT (:link_slug = ANY(in_links))
                        """),
                        {"kb": uuid.UUID(p["kb_id"]), "target": target, "link_slug": p["slug"]},
                    )
        await self._save_checkpoint(
            ctx, "completed", page_count=inserted + affected,
            candidate_count=len(candidates),
        )
        logger.info(
            "【百科】处理完成：新建 %d / 合并 %d，文档=%s", inserted, affected, ctx.doc_name,
        )
        return inserted + affected

    async def _save_checkpoint(
        self,
        ctx: PostProcContext,
        status: str,
        page_count: int = 0,
        candidate_count: int = 0,
    ) -> None:
        """记录该文档的 wiki 执行检查点；失败仅告警，不阻断主流程。"""
        try:
            from aion_knowledge.pipeline.postproc.wiki.checkpoint import WikiCheckpointManager
            await WikiCheckpointManager().save(
                ctx.kb_id, ctx.document_id,
                status=status, page_count=page_count, candidate_count=candidate_count,
            )
        except Exception as exc:
            logger.warning("【百科】检查点保存失败：%s", exc)

    @staticmethod
    def _compose_full_text(chunks: list[dict[str, Any]]) -> str:
        """按 seq_num 组合 chunks 为文档原文（图片/表格内容已回写 content，无需查库）。"""
        ordered = sorted(chunks, key=lambda c: c.get("seq_num", 0))
        return "\n\n".join(c.get("content", "") for c in ordered if c.get("content"))

    async def _extract_candidates(
        self, llm: LLMClient, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Phase 1 MAP：整篇文档（chunks 组合原文）一次 LLM 提取候选。"""
        full_text = self._compose_full_text(chunks)
        if not self._check_content(full_text):
            return []
        try:
            max_tokens = get_model_max_input_tokens(settings.llm_model, ratio=0.05)
            content = truncate_by_tokens(full_text, max_tokens)
            if len(content) < len(full_text):
                logger.warning(
                    "【百科】文档过长已截断：%d → %d 字符，尾部候选可能丢失",
                    len(full_text), len(content),
                )
            data = await llm.generate_structured(
                WIKI_MAP_PROMPT.format(content=content),
                output_schema=WIKI_MAP_SCHEMA,
            )
            return self._normalize_candidates(data)
        except Exception as exc:
            logger.warning("【百科】LLM 调用失败：%s", exc)
            return []

    async def _merge_with_existing(
        self,
        llm: LLMClient,
        candidates: list[dict[str, Any]],
        existing: list[dict[str, str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """跨文档去重合并。返回 (merged, new)。

        merged 项带 existing_slug；existing 为 [{"slug", "title"}]（KB 内已有页面）。
        规则预筛：slug 归一化精确命中；未命中交 LLM 判同。
        """
        by_slug = {e["slug"]: e for e in existing}
        merged: list[dict[str, Any]] = []
        new: list[dict[str, Any]] = []
        for c in candidates:
            slug = self._to_slug(c["term"])
            if slug in by_slug:
                merged.append({**c, "existing_slug": slug})
                continue
            hit = await self._judge_same(llm, c["term"], existing)
            if hit:
                merged.append({**c, "existing_slug": hit})
            else:
                new.append(c)
        return merged, new

    async def _judge_same(
        self, llm: LLMClient, term: str, existing: list[dict[str, str]]
    ) -> str | None:
        """LLM 判同：候选 term 与已有页面是否同一概念，返回命中 slug 或 None。

        返回前校验 existing_slug 真实存在于 existing，防 LLM 幻觉 slug
        导致后续 merge UPDATE 静默无匹配（产生重复页面）。
        """
        if not existing:
            return None
        valid_slugs = {e["slug"] for e in existing}
        titles = "、".join(f"{e['title']}({e['slug']})" for e in existing[:20])
        try:
            data = await llm.generate_structured(
                WIKI_DEDUP_PROMPT.format(term=term, titles=titles),
                output_schema=WIKI_DEDUP_SCHEMA,
            )
            if isinstance(data, dict) and data.get("is_same") and data.get("existing_slug"):
                slug = str(data["existing_slug"])
                if slug in valid_slugs:
                    return slug
        except Exception as exc:
            logger.warning("【百科】判同失败：%s", exc)
        return None

    @staticmethod
    def _normalize_candidates(data: Any) -> list[dict[str, Any]]:
        """归一化 LLM 返回的候选结构，兼容 candidates/wiki_candidates/terms/entities 键与 dict/字符串元素。"""
        if not isinstance(data, dict):
            return []
        raw: list[Any] | None = None
        for key in ("candidates", "wiki_candidates", "terms", "entities"):
            value = data.get(key)
            if isinstance(value, list) and value:
                raw = value
                break
        if raw is None:
            return []
        normalized: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict) and item.get("term"):
                normalized.append({
                    "term": str(item["term"]).strip(),
                    "type": str(item.get("type") or "concept"),
                    "reason": str(item.get("reason") or ""),
                })
            elif isinstance(item, str) and item.strip():
                normalized.append({
                    "term": item.strip(),
                    "type": "concept",
                    "reason": "",
                })
        return [c for c in normalized if c["term"]]

    @staticmethod
    def _extract_out_links(content: str) -> list[str]:
        """从 content 抽取 [[slug]] / [[slug|显示名]] 中的 slug，去重保序。"""
        slugs: list[str] = []
        for m in WIKI_LINK_RE.finditer(content or ""):
            slug = m.group(1).strip()
            if slug and slug not in slugs:
                slugs.append(slug)
        return slugs

    async def _generate_page(
        self, llm: LLMClient, candidate: dict[str, Any],
        source_content: str, available_slugs: list[str],
    ) -> dict[str, Any]:
        """Phase 4 REFINE：为单个候选生成 Wiki 页面内容（含 wikilink 约束）。"""
        try:
            result = await llm.generate_structured(
                WIKI_REFINE_PROMPT.format(
                    title=candidate["term"],
                    type=candidate["type"],
                    source_content=source_content,
                    available_slugs="、".join(available_slugs) or "（无）",
                ),
                output_schema={
                    "type": "object",
                    "properties": {"slug": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["slug", "content"],
                },
            )
            return result
        except Exception as exc:
            logger.warning("【百科】REFINE 精炼失败：「%s」：%s", candidate["term"], exc)
            return {"slug": self._to_slug(candidate["term"]), "content": ""}

    @staticmethod
    def _to_slug(title: str) -> str:
        """将页面标题转换为 URL slug。"""
        slug = title.lower().strip()
        slug = re.sub(r"[^a-z0-9一-鿿]+", "_", slug)
        slug = slug.strip("_")
        return slug or "untitled"


def _candidate_source(candidate: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
    """REFINE 来源内容：候选 term 所在 chunk 的 content（截断）。"""
    content = next(
        (c.get("content", "") for c in chunks
         if candidate["term"] and candidate["term"] in c.get("content", "")),
        "",
    ) or (chunks[0].get("content", "") if chunks else "")
    return content[:500]


def module() -> WikiModule:
    """模块工厂函数，供调度器自动发现。"""
    return WikiModule()

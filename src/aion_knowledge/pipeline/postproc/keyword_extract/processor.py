"""KeywordExtractModule — 三层关键词提取与去重合并。

作用：
  为每个 chunk 提取关键词，写入 chunk_text.keywords 字段。
  可选启用（always_on = False），依赖 text 模块。

三层递进机制：
  Tier 1（开放集）：调用 LLM 从 chunk 内容自由生成 topN 个关键词。
    提取结果用逗号/空格分隔，适用于没有预设标签的场景。

  Tier 2（封闭集精确匹配）：将 chunk 内容小写化后，与 KB 预设的 tags
    列表做精确子串匹配。无需 LLM 调用，成本低、准确率高。
    匹配结果作为 Tier 3 的 few-shot 示例。

  Tier 3（封闭集约束选取）：当 Tier 2 匹配不足 3 个时，调用 LLM
    从剩余 tags 中按内容 relevance 选取 topN 个。携带 Tier 2 的
    成功匹配样例作为 few-shot 引导。

最终合并三个 tier 的结果，去重后写入 chunk_text。
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

from aion_knowledge.common.config import settings
from aion_knowledge.common.model_registry import get_model_max_input_tokens, truncate_by_tokens
from aion_knowledge.infrastructure.llm import LLMClient, get_llm_client_for_module
from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule

logger = logging.getLogger(__name__)

TIER1_PROMPT = "从以下文本中提取{topn}个关键词，用中文逗号分隔。只返回关键词，不要序号。\n\n{content}"

TIER3_PROMPT = """从以下标签列表中，选择最匹配文本内容的{topn}个标签。
只返回标签名，用逗号分隔。标签必须来自可用标签列表。

可用标签：{tags}

{few_shot}
文本内容：{content}"""

TIER3_TOPN = 3


class KeywordExtractModule(PostProcModule):
    """三层关键词提取模块。二批执行，依赖 text，结果写回 chunk_text.keywords。"""

    always_on = False
    depends_on = ["text"]

    async def process(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> int:
        """为每个 chunk 执行三层关键词提取并合并写入，返回处理 chunk 数。"""
        if not chunks:
            return 0

        # 加载 KB 预设 tags
        kb_tags = await self._load_kb_tags(ctx.kb_id)
        if not kb_tags:
            logger.info("【关键词】知识库 %s 无预设标签，跳过 Tier 2/3", ctx.kb_id)

        llm = get_llm_client_for_module("keyword_extract")
        from aion_knowledge.infrastructure.db import get_session

        examples: list[dict[str, Any]] = []  # Tier 2 成功匹配的 chunk，作为 Tier 3 few-shot
        results: list[dict[str, Any]] = []

        # ── Tier 1: LLM 自由生成 ──
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            chunk_uuid = chunk.get("chunk_uuid", "")
            if not content or not chunk_uuid:
                continue
            if not self._check_content(content):
                logger.warning("【关键词】内容过少，跳过 chunk=%s", chunk_uuid)
                continue

            tier1_keywords = await self._extract_keywords_llm(llm, content)

            # ── Tier 2: 精确子串匹配 ──
            _max_tokens_tier2 = get_model_max_input_tokens(settings.llm_model, ratio=0.005)
            tier2_matched: list[str] = []
            if kb_tags:
                tier2_matched = self._match_tags_exact(content, kb_tags)
                if tier2_matched:
                    examples.append({
                        "content": (
                            truncate_by_tokens(content, _max_tokens_tier2)
                            if content else content[:500]
                        ),
                        "tags": tier2_matched,
                    })

            # ── Tier 3: LLM 约束选取（仅当 Tier 2 匹配 < 3 个且有 tags）──
            tier3_selected: list[str] = []
            if kb_tags and len(tier2_matched) < TIER3_TOPN:
                remaining = [t for t in kb_tags if t not in tier2_matched]
                if remaining:
                    tier3_selected = await self._select_tags_llm(
                        llm, content, remaining, examples,
                    )

            # 合并去重
            seen: set[str] = set()
            all_keywords: list[str] = []
            for kw in tier1_keywords + tier2_matched + tier3_selected:
                if kw not in seen:
                    seen.add(kw)
                    all_keywords.append(kw)

            if not all_keywords:
                continue

            results.append({
                "chunk_uuid": chunk_uuid,
                "keywords": all_keywords,
            })

        # ── 批量写入 chunk_text.keywords ──
        if not results:
            return 0

        from aion_knowledge.storage.relational.chunk_repo import ChunkRepository

        async with get_session() as session:
            repo = ChunkRepository(session)
            for r in results:
                await repo.update_keywords(r["chunk_uuid"], r["keywords"])

        logger.info("【关键词】处理完成：%d 个 chunk，文档=%s", len(results), ctx.doc_name)
        return len(results)

    # ── Tier 1 ──

    async def _extract_keywords_llm(self, llm: LLMClient, content: str, topn: int = 5) -> list[str]:
        """LLM 自由生成关键词（开放集）。"""
        _max_tokens = get_model_max_input_tokens(settings.llm_model, ratio=0.03)
        try:
            raw = await llm.generate(TIER1_PROMPT.format(
                topn=topn, content=truncate_by_tokens(content, _max_tokens) if content else "",
            ))
            parts = re.split(r"[,，、;；\s]+", raw.strip())
            return [p.strip() for p in parts if len(p.strip()) > 1][:topn]
        except Exception as exc:
            logger.warning("【关键词】Tier 1 提取失败：%s", exc)
            return []

    # ── Tier 2 ──

    @staticmethod
    def _match_tags_exact(content: str, kb_tags: list[str]) -> list[str]:
        """精确子串匹配（封闭集，无 LLM）。"""
        content_lower = content.lower()
        return [tag for tag in kb_tags if tag.lower() in content_lower]

    # ── Tier 3 ──

    async def _select_tags_llm(
        self,
        llm: LLMClient,
        content: str,
        remaining_tags: list[str],
        examples: list[dict[str, Any]],
        topn: int = TIER3_TOPN,
    ) -> list[str]:
        """LLM 从预设 tags 中约束选取（封闭集）。"""
        # 构造 few-shot 示例（最多 2 个）
        few_shot = ""
        if examples:
            selected = random.sample(examples, min(2, len(examples)))
            lines = []
            for ex in selected:
                lines.append(f"文本：{ex['content'][:200]}")
                lines.append(f"标签：{'，'.join(ex['tags'])}")
            few_shot = "\n".join(lines) + "\n\n"

        _max_tokens = get_model_max_input_tokens(settings.llm_model, ratio=0.02)
        prompt = TIER3_PROMPT.format(
            topn=topn,
            tags="，".join(remaining_tags),
            few_shot=few_shot,
            content=truncate_by_tokens(content, _max_tokens) if content else "",
        )

        try:
            raw = await llm.generate(prompt)
            parts = re.split(r"[,，、;；\s]+", raw.strip())
            # 只保留在标签列表中的结果
            validated = [
                p.strip() for p in parts
                if p.strip() in remaining_tags
            ]
            return validated[:topn]
        except Exception as exc:
            logger.warning("【关键词】Tier 3 标签选择失败：%s", exc)
            return []

    # ── 工具方法 ──

    @staticmethod
    async def _load_kb_tags(kb_id: str) -> list[str]:
        """从 kb_knowledge_bases 读取预设 tags。"""
        from sqlalchemy import text

        from aion_knowledge.infrastructure.db import get_session

        async with get_session() as session:
            row = await session.execute(
                text("SELECT tags FROM kb_knowledge_bases WHERE id = :kid"),
                {"kid": kb_id},
            )
            result = row.first()
            return result[0] if result else []


def module() -> KeywordExtractModule:
    """模块工厂函数，供调度器自动发现。"""
    return KeywordExtractModule()

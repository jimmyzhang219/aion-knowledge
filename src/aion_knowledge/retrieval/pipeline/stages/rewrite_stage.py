"""RewriteStage — LLM 查询改写，提升多路召回覆盖率。"""
from __future__ import annotations

import logging
from typing import Any

from aion_knowledge.infrastructure.llm.client import LLMClient
from aion_knowledge.retrieval.pipeline.base import Stage
from aion_knowledge.retrieval.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

# 改写输出 JSON schema，用于 generate_structured
_REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rewritten_query": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["rewritten_query", "keywords", "entities"],
}

_SYSTEM_PROMPT = """你是一个搜索查询改写助手。将用户输入改写为适合知识库检索的形式。

要求：
1. 去除口语化表达和冗余词，保留核心搜索意图
2. 改写后的查询不要求保留问句形式，可以是关键词组合（如"多路召回 配置 说明"）
3. 将缩写/简写补充为完整术语（如"LLM"同时保留原文+全称 → 在关键词中展开）
4. 提取 3-10 个关键词，用于精确匹配检索
5. 提取 1-5 个实体（人名、产品名、组织名、术语等），无实体则返回空列表
6. 改写后的查询保持简洁，30 词以内
7. 如果原始查询已经很简洁清晰，保持原样

输出 JSON 格式（不要额外的 markdown 或说明）：
{
  "rewritten_query": "改写后的查询字符串",
  "keywords": ["关键词1", "关键词2"],
  "entities": ["实体1", "实体2"]
}"""


class RewriteStage(Stage):
    """LLM 查询改写阶段。

    前置条件：
        - enabled=True 且 llm_client 不为 None
    执行后：
        - ctx.query 被 rewritten_query 覆盖
        - ctx.expansion_keywords 被 keywords 填充
        - ctx.entities 被 entities 填充
    跳过条件：
        - enabled=False 或 llm_client 为 None
        - LLM 调用异常（日志 warning，不中断 pipeline）
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        enabled: bool = False,
    ) -> None:
        self._llm_client = llm_client
        self._enabled = enabled

    async def run(self, ctx: PipelineContext) -> None:
        if not self._enabled or self._llm_client is None:
            logger.info("【查询改写】跳过 (enabled=%s, llm=%s)", self._enabled, self._llm_client is not None)
            return

        query = ctx.query.strip()
        if not query:
            return

        try:
            result = await self._llm_client.generate_structured(
                prompt=query,
                output_schema=_REWRITE_SCHEMA,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=200,
            )
        except Exception:
            logger.warning("RewriteStage: LLM call failed, keeping original query", exc_info=True)
            return

        if not isinstance(result, dict):
            logger.warning("RewriteStage: unexpected LLM output type %s, skipping", type(result).__name__)
            return

        # 保留原始 query，供依赖自然语言原文的检索器（如 FAQ）使用
        if ctx.original_query is None:
            ctx.original_query = query

        # 覆盖 ctx.query
        rewritten = result.get("rewritten_query", "")
        if rewritten and isinstance(rewritten, str):
            ctx.query = rewritten.strip()

        # 填充 expansion_keywords
        keywords = result.get("keywords")
        if keywords and isinstance(keywords, list):
            cleaned = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]
            if cleaned:
                ctx.expansion_keywords = cleaned

        # 填充 entities
        entities = result.get("entities")
        if entities and isinstance(entities, list):
            cleaned_entities = [e.strip() for e in entities if isinstance(e, str) and e.strip()]
            if cleaned_entities:
                ctx.entities = cleaned_entities

        logger.info(
            "【查询改写】原始: %s | 改写: %s | 关键词: %s | 实体: %s",
            query,
            ctx.query,
            ctx.expansion_keywords or [],
            ctx.entities or [],
        )

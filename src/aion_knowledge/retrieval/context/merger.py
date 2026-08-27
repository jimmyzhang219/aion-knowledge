"""上下文合并与组装 — 合并搜索结果 + FAQ 直接答案通道。"""

from __future__ import annotations

from typing import Any

from aion_knowledge.common.config import settings
from aion_knowledge.models.enums import ChunkType

Query = str


def merge_context(
    results: list[dict[str, Any]],
    query: Query,
) -> list[dict[str, Any]]:
    """合并检索结果到上下文。

    对 FAQ 结果：若分数 >= faq_direct_answer_threshold，以 faq_direct 类型直接注入。
    其他结果：以 chunk 类型传递。
    """
    context: list[dict[str, Any]] = []

    for r in results:
        is_faq = r.get("chunk_type") == ChunkType.faq.value
        if is_faq and r.get("score", 0.0) >= settings.faq_direct_answer_threshold:
            context.append({
                "type": "faq_direct",
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {}),
            })
        else:
            context.append({
                "type": "chunk",
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {}),
            })

    return context

"""上下文截断 — 按 token 精确计数丢弃低分结果。"""
from __future__ import annotations

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# 缓存编码器（进程内只初始化一次）
_ENCODING = None


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def _estimate_tokens(text: str) -> int:
    """用 tiktoken 精确估算 token 数。"""
    return len(_get_encoding().encode(text))


def truncate_context(
    context: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """按分数保留结果，直到 token 数不超限。

    Args:
        context: 检索结果上下文列表（需包含 content, score）。
        max_tokens: 最大 token 数。为 None 时根据 settings.llm_model
                    的 context_window 动态计算（取其 75%）。

    Returns:
        截断后的上下文列表（按 score 降序）。
    """
    if not context:
        return []

    if max_tokens is None:
        from aion_knowledge.common.config import settings
        from aion_knowledge.common.model_registry import get_registry
        context_window = get_registry().context_window(settings.llm_model)
        max_tokens = int(context_window * 0.75)
        max_tokens = max(max_tokens, 1024)

    # 按 score 降序排列
    sorted_ctx = sorted(context, key=lambda x: x.get("score", 0), reverse=True)

    # 从高分到低分累积 token，至少保留一条
    result: list[dict[str, Any]] = []
    total_tokens = 0
    for item in sorted_ctx:
        content = item.get("content", "")
        item_tokens = _estimate_tokens(content)
        if result and total_tokens + item_tokens > max_tokens:
            break
        result.append(item)
        total_tokens += item_tokens

    return result

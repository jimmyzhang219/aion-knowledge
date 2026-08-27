"""LLM Answer 生成 — 组装 prompt + 调用 create_llm。"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from aion_knowledge.infrastructure.llm import create_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个知识库问答助手。请根据以下检索结果回答用户问题。

检索结果：
{context}

用户问题：{query}

请基于检索结果作答。如果结果不足以回答问题，请直接说明缺少什么信息。
不要编造答案。"""


def build_prompt(query: str, context: list[dict[str, Any]]) -> str:
    """组装 QA prompt。

    Args:
        query: 用户问题。
        context: 截断后的检索结果上下文列表。

    Returns:
        完整的 prompt 字符串。
    """
    lines = []
    for i, item in enumerate(context, 1):
        content = item.get("content", "")
        lines.append(f"[{i}] {content}")

    context_str = "\n\n".join(lines) if lines else "（无检索结果）"
    return SYSTEM_PROMPT.format(context=context_str, query=query)


async def generate_answer(
    query: str,
    context: list[dict[str, Any]],
    stream: bool = False,
) -> str | AsyncIterator[str]:
    """生成 LLM 回答。

    Args:
        query: 用户问题。
        context: 截断后的检索结果上下文。
        stream: 是否流式输出。

    Returns:
        stream=False → 完整 answer 字符串。
        stream=True  → AsyncIterator[str]。
    """
    prompt = build_prompt(query, context)
    llm = create_llm()

    if stream:
        return llm.stream(prompt)
    else:
        return await llm.generate(prompt)

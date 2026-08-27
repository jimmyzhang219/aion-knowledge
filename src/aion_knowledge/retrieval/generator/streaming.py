"""SSE 流式输出 — 事件格式 + StreamingResponse 构造。"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse


async def _generate_sse(
    query: str,
    context: list[dict[str, Any]],
    llm: Any,
) -> AsyncIterator[dict[str, str]]:
    """生成 SSE 事件序列。

    事件类型：
        retrieval — 检索完成，携带结果数量
        token     — 逐 token 内容
        done      — 生成完成，携带完整 answer
    """
    # 1. 检索完成事件
    yield {"event": "retrieval", "data": json.dumps({"count": len(context)}, ensure_ascii=False)}

    # 2. 逐 token 生成
    from aion_knowledge.retrieval.generator.qa import build_prompt

    prompt = build_prompt(query, context)

    full_text = ""
    async for token in llm.stream(prompt):
        if token:
            full_text += token
            yield {"event": "token", "data": json.dumps(token, ensure_ascii=False)}

    # 3. 完成事件
    yield {"event": "done", "data": json.dumps({"answer": full_text}, ensure_ascii=False)}


def build_sse_response(
    query: str,
    context: list[dict[str, Any]],
    llm: Any,
) -> StreamingResponse:
    """构造 SSE StreamingResponse。

    Args:
        query: 用户问题。
        context: 截断后的检索结果上下文。
        llm: LangChain BaseChatModel 实例（支持 astream）。

    Returns:
        text/event-stream 的 StreamingResponse。
    """

    async def event_stream() -> AsyncIterator[bytes]:
        async for event in _generate_sse(query, context, llm):
            yield f"event: {event['event']}\ndata: {event['data']}\n\n".encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )

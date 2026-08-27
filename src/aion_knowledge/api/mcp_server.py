"""MCP 服务器 — 提供 search_knowledge 工具。

使用 FastMCP（mcp SDK）注册工具，通过 Streamable HTTP 传输暴露 MCP 端点。
挂载到主 FastAPI 应用的 `/mcp` 路径下（见 api/__init__.py）。
"""
from __future__ import annotations

import logging
import time
from contextvars import Token
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette

from aion_knowledge.common.trace import get_trace_id, reset_trace_id, set_trace_id
from aion_knowledge.retrieval.orchestrator.router import RetrievalRouter
from aion_knowledge.storage.relational.kb_repo import KnowledgeBaseRepo

logger = logging.getLogger(__name__)

# 全局 FastMCP 实例 (使用 json_response 方便集成测试解析)
mcp = FastMCP(
    "Aion Knowledge",
    instructions="""知识库语义检索服务（Aion Knowledge）。

支持多路召回（向量 / BM25 / 关键词 / FAQ / 图谱等 11 路）→ RRF 融合排序。
只返回原始检索片段（chunk），不含 LLM 生成回答。

使用场景：
- 当需要从知识库中检索与用户问题相关的文档片段时使用
- 搜索参数：query（查询文本）、kb_id（知识库 ID）、top_k（返回条数，默认 10）

无需关注内部检索路径，多路融合是自动完成的。""",
    stateless_http=True,
    streamable_http_path="/",
    json_response=True,
)
_router = RetrievalRouter()


def _enter_trace(trace_id: str | None) -> tuple[str, Token[str]]:
    """工具入口：确定 trace_id（未传则生成）并注入 contextvar，返回 (trace_id, reset_token)。"""
    tid = trace_id or get_trace_id()
    return tid, set_trace_id(tid)


def _log_tool_response(tool: str, tid: str, start: float, result: dict[str, Any]) -> dict[str, Any]:
    """记 MCP 工具响应日志（status=ok/error + 耗时 + 结果条数），返回结果供直接 return。"""
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    if "error" in result:
        logger.info(
            "MCP RESPONSE %s | trace_id=%s | status=error | %dms | error=%s",
            tool,
            tid,
            elapsed_ms,
            result["error"],
        )
    else:
        items = len(result.get("results") or result.get("kbs") or [])
        logger.info(
            "MCP RESPONSE %s | trace_id=%s | status=ok | %dms | items=%d",
            tool,
            tid,
            elapsed_ms,
            items,
        )
    return result


@mcp.tool()
async def list_knowledge_bases(trace_id: str | None = None) -> dict[str, Any]:
    """列出所有可用的知识库及其 ID 和名称，供后续 search_knowledge 使用。

    Agent 应优先调用此工具获取目标 kb_id，再传给 search_knowledge。

    Args:
        trace_id: 请求链路追踪 ID（可选，未传则服务端生成）
    """
    tid, token = _enter_trace(trace_id)
    start = time.perf_counter()
    logger.info("MCP REQUEST list_knowledge_bases | trace_id=%s", tid)
    try:
        repo = KnowledgeBaseRepo()
        kbs = await repo.list_all()
        return _log_tool_response(
            "list_knowledge_bases",
            tid,
            start,
            {
                "trace_id": tid,
                "kbs": [
                    {"id": str(kb.id), "name": kb.name, "description": kb.description}
                    for kb in kbs
                ],
            },
        )
    except Exception as exc:
        # 与 search_knowledge 一致：工具内捕获返回 error dict，保证 request/response 日志成对
        logger.exception("MCP list_knowledge_bases failed")
        return _log_tool_response(
            "list_knowledge_bases", tid, start, {"trace_id": tid, "error": str(exc)}
        )
    finally:
        reset_trace_id(token)


@mcp.tool()
async def search_knowledge(
    query: str,
    kb_id: str,
    top_k: int = 10,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """在知识库中进行多路检索，返回 RRF 融合结果（不含 LLM 生成）。

    kb_id 参数应通过 list_knowledge_bases 工具获取。

    Args:
        query: 搜索查询
        kb_id: 知识库 ID（先调用 list_knowledge_bases 获取）
        top_k: 返回结果数（默认 10）
        trace_id: 请求链路追踪 ID（可选，未传则服务端生成）
    """
    tid, token = _enter_trace(trace_id)
    start = time.perf_counter()
    logger.info(
        "MCP REQUEST search_knowledge | trace_id=%s | query=%r | kb_id=%r | top_k=%r",
        tid,
        query,
        kb_id,
        top_k,
    )
    try:
        if not query or not query.strip():
            return _log_tool_response(
                "search_knowledge",
                tid,
                start,
                {"trace_id": tid, "error": "query must not be empty"},
            )

        try:
            fused, source_breakdown, path_stats = await _router.search(
                query=query,
                kb_id=kb_id,
                top_k=top_k,
            )
        except Exception as exc:
            logger.exception("MCP search_knowledge failed")
            return _log_tool_response(
                "search_knowledge", tid, start, {"trace_id": tid, "error": str(exc)}
            )
        return _log_tool_response(
            "search_knowledge",
            tid,
            start,
            {
                "trace_id": tid,
                "results": [
                    {
                        "chunk_id": r.chunk_id,
                        "content": r.content,
                        "score": r.score,
                        "source_paths": r.source_paths,
                        "chunk_type": r.chunk_type,
                    }
                    for r in fused
                ],
                "source_breakdown": source_breakdown,
            },
        )
    finally:
        reset_trace_id(token)


def get_mcp_streamable_http_app() -> "Starlette":
    """返回 MCP Streamable HTTP 传输的 Starlette ASGI 应用，供 FastAPI mount。"""
    return mcp.streamable_http_app()



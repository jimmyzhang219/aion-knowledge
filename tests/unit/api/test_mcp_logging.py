"""MCP 工具 request/response 日志测试（直接调用工具函数，绕过 FastMCP 传输层）。"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from aion_knowledge.api import mcp_server
from aion_knowledge.storage.relational.kb_repo import KnowledgeBaseRepo

_LOG_NAME = "aion_knowledge.api.mcp_server"


class _FakeRouter:
    """最小 RetrievalRouter 替身：search 返回固定融合结果。"""

    async def search(self, query: str, kb_id: str, top_k: int):
        fused = [
            SimpleNamespace(
                chunk_id="c1",
                content="测试内容",
                score=0.9,
                source_paths=["doc.md"],
                chunk_type="text",
            )
        ]
        return fused, {"vector": 1}, {"elapsed_ms": 12}


@pytest.fixture(autouse=True)
def _patch_router(monkeypatch) -> None:
    """替换模块级 _router，避免真实检索链路。"""
    monkeypatch.setattr(mcp_server, "_router", _FakeRouter())


def _lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _LOG_NAME]


async def test_search_knowledge_logs_request_and_response(caplog: pytest.LogCaptureFixture) -> None:
    """search_knowledge：REQUEST（工具名+参数+trace_id）与 RESPONSE（状态+耗时+条数）成对。"""
    caplog.set_level(logging.INFO, logger=_LOG_NAME)
    result = await mcp_server.search_knowledge(
        query="测试问题", kb_id="kb-1", top_k=3, trace_id="t-1"
    )
    assert result["trace_id"] == "t-1"
    assert len(result["results"]) == 1

    lines = _lines(caplog)
    req = [line for line in lines if "MCP REQUEST search_knowledge" in line]
    resp = [line for line in lines if "MCP RESPONSE search_knowledge" in line]
    assert len(req) == 1, lines
    assert len(resp) == 1, lines
    assert "trace_id=t-1" in req[0]
    assert "query='测试问题'" in req[0]
    assert "kb_id='kb-1'" in req[0]
    assert "top_k=3" in req[0]
    assert "trace_id=t-1" in resp[0]
    assert "status=ok" in resp[0]
    assert "items=1" in resp[0]
    assert "ms" in resp[0]


async def test_search_knowledge_empty_query_logs_error_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """空 query：REQUEST 与 status=error 的 RESPONSE 成对。"""
    caplog.set_level(logging.INFO, logger=_LOG_NAME)
    result = await mcp_server.search_knowledge(query="", kb_id="kb-1", trace_id="t-2")
    assert "error" in result

    lines = _lines(caplog)
    req = [line for line in lines if "MCP REQUEST search_knowledge" in line]
    resp = [line for line in lines if "MCP RESPONSE search_knowledge" in line]
    assert len(req) == 1, lines
    assert len(resp) == 1, lines
    assert "status=error" in resp[0]
    assert "query must not be empty" in resp[0]


async def test_list_knowledge_bases_logs_request_and_response(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """list_knowledge_bases：REQUEST/RESPONSE 成对，RESPONSE 带 kbs 条数。"""
    caplog.set_level(logging.INFO, logger=_LOG_NAME)

    async def fake_list_all(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(id="kb-1", name="KB1", description="测试库")]

    monkeypatch.setattr(KnowledgeBaseRepo, "list_all", fake_list_all)

    result = await mcp_server.list_knowledge_bases(trace_id="t-3")
    assert len(result["kbs"]) == 1

    lines = _lines(caplog)
    req = [line for line in lines if "MCP REQUEST list_knowledge_bases" in line]
    resp = [line for line in lines if "MCP RESPONSE list_knowledge_bases" in line]
    assert len(req) == 1, lines
    assert len(resp) == 1, lines
    assert "trace_id=t-3" in req[0]
    assert "status=ok" in resp[0]
    assert "items=1" in resp[0]
    assert "ms" in resp[0]

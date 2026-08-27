"""MCP Streamable HTTP 搜索工具集成测试。"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aion_knowledge.api import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(
        create_app(),
        base_url="http://localhost:19531",
        headers={"Accept": "application/json, text/event-stream"},
    ) as c:
        yield c


def _mcp_request(client, method: str, params: dict | None = None):
    """发送 MCP Streamable HTTP JSON-RPC 请求。"""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
    }
    if params is not None:
        body["params"] = params
    return client.post("/mcp", json=body)


def test_mcp_search_knowledge(client):
    """测试 MCP search_knowledge 工具能正确返回检索结果。"""
    resp = _mcp_request(client, "tools/call", {
        "name": "search_knowledge",
        "arguments": {
            "query": "测试",
            "kb_id": "00000000-0000-0000-0000-000000000001",
            "top_k": 5,
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    content = data["result"]["content"]
    assert len(content) == 1
    text_content = content[0]["text"]
    result = json.loads(text_content)
    assert "results" in result
    assert isinstance(result["results"], list)
    assert "source_breakdown" in result


def test_mcp_search_empty_query_fails(client):
    """空查询应返回错误。"""
    resp = _mcp_request(client, "tools/call", {
        "name": "search_knowledge",
        "arguments": {
            "query": "",
            "kb_id": "00000000-0000-0000-0000-000000000001",
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    content = data["result"]["content"][0]["text"]
    result = json.loads(content)
    assert "error" in result
    assert "trace_id" in result


def test_mcp_search_knowledge_with_trace_id(client):
    """传入 trace_id：返回结果携带同一 trace_id。"""
    resp = _mcp_request(client, "tools/call", {
        "name": "search_knowledge",
        "arguments": {
            "query": "测试",
            "kb_id": "00000000-0000-0000-0000-000000000001",
            "top_k": 5,
            "trace_id": "mcp-trace-1",
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    result = json.loads(data["result"]["content"][0]["text"])
    assert result["trace_id"] == "mcp-trace-1"


def test_mcp_search_knowledge_generates_trace_id(client):
    """未传 trace_id：返回生成的 uuid7 trace_id。"""
    resp = _mcp_request(client, "tools/call", {
        "name": "search_knowledge",
        "arguments": {
            "query": "测试",
            "kb_id": "00000000-0000-0000-0000-000000000001",
            "top_k": 5,
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    result = json.loads(data["result"]["content"][0]["text"])
    assert len(result["trace_id"]) == 36



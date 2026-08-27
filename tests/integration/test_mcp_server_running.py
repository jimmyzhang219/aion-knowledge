"""对运行中的 MCP Server 进行真实 HTTP 集成测试。

与 test_mcp_search.py 不同，本测试不 mock 也不 in-process 启动应用，
而是连接一个已在运行中的 MCP Server 实例，发送真实 JSON-RPC 请求。

用法：
    # 终端 1：先启动完整 FastAPI 应用
    python -m aion_knowledge                          # 完整 API（端口 19531，MCP 在 /mcp）

    # 终端 2：运行测试
    pytest tests/integration/test_mcp_server_running.py -v

环境变量：
    MCP_BASE_URL  可自定义端点（默认 http://localhost:19531/mcp）
"""
from __future__ import annotations

import json
import os
from urllib.parse import urlparse

import httpx
import pytest

pytestmark = pytest.mark.integration

# 默认连接配置
#   python -m aion_knowledge  → http://localhost:19531/mcp
DEFAULT_MCP_URL = "http://localhost:19531/mcp"
ENV_KEY = "MCP_BASE_URL"


def _resolve_url() -> str:
    """返回 MCP 端点 URL，优先使用环境变量。"""
    return os.environ.get(ENV_KEY, DEFAULT_MCP_URL)


def _split_url(url: str) -> tuple[str, str]:
    """将 URL 拆分为 (base, path)，避免 httpx 自动加尾缀斜杠。"""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path or "/"
    return base, path


@pytest.fixture(scope="module")
def mcp_url() -> str:
    return _resolve_url()


@pytest.fixture(scope="module")
def client(mcp_url: str) -> httpx.Client:  # 同步 client，module scope 复用
    """连接运行中的 MCP Server，不可达则 skip。"""
    try:
        base, path = _split_url(mcp_url)
        c = httpx.Client(
            base_url=base,
            timeout=5.0,
            headers={"Accept": "application/json"},
        )
        c._mcp_path = path  # 保存请求路径，供 helper 使用
        # 发一个 lightweight 请求验证连通性
        resp = c.post(
            path,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        resp.raise_for_status()
        return c
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        pytest.skip(f"MCP Server 未运行（{mcp_url}）: {e}")
        raise  # never reached, but satisfies type checker


# ── 辅助函数 ──────────────────────────────────────────────


def _mcp_request(
    client: httpx.Client,
    method: str,
    params: dict | None = None,
) -> dict:
    """发送 JSON-RPC 请求并返回 result 字段内容。"""
    body = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    path = getattr(client, "_mcp_path", "/")
    resp = client.post(path, json=body)
    resp.raise_for_status()
    data = resp.json()

    # JSON-RPC 错误
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"JSON-RPC error {err.get('code')}: {err.get('message')}")

    return data["result"]


def _call_tool(client: httpx.Client, name: str, arguments: dict | None = None) -> object:
    """调用 MCP 工具并解析返回内容。

    优先使用 structuredContent（保留原始类型），
    兜底回退到 content[0].text（JSON 字符串）。
    """
    result = _mcp_request(client, "tools/call", {"name": name, "arguments": arguments or {}})

    # structuredContent 保留原始返回值类型（list / dict）
    sc = result.get("structuredContent")
    if sc is not None:
        return sc.get("result", {})

    # 兜底：从 text 解析
    items = result.get("content", [])
    if not items:
        return {}
    text = items[0].get("text", "{}")
    return json.loads(text)


# ── 测试用例 ──────────────────────────────────────────────


def test_mcp_server_info(client: httpx.Client):
    """打印 MCP Server 初始化信息（serverInfo + instructions），供 Agent Prompt 使用。"""
    path = getattr(client, "_mcp_path", "/")
    resp = client.post(
        path,
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {
                  "protocolVersion": "2024-11-05",
                  "capabilities": {},
                  "clientInfo": {"name": "test", "version": "1.0"},
              }},
    )
    resp.raise_for_status()
    data = resp.json()
    result = data.get("result", {})
    print("\n=== serverInfo ===")
    print(json.dumps(result.get("serverInfo", {}), ensure_ascii=False, indent=2))
    print("\n=== instructions（Agent Prompt 用） ===")
    print(result.get("instructions", "(无)"))


def test_list_knowledge_bases(client: httpx.Client):
    """列出知识库，Agent 用来获取可用 kb_id（返回 {trace_id, kbs} 包装）。"""
    data = _call_tool(client, "list_knowledge_bases")
    print("\n=== list_knowledge_bases 响应 ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    assert isinstance(data, dict), f"应返回 dict 包装: {data}"
    assert "trace_id" in data, f"缺少 trace_id: {data}"
    kbs = data["kbs"]
    assert isinstance(kbs, list), f"kbs 应为列表: {kbs}"
    if kbs:
        item = kbs[0]
        assert "id" in item, f"缺少 id: {item}"
        assert "name" in item, f"缺少 name: {item}"


class TestSearchKnowledge:
    """search_knowledge 工具测试。"""

    def test_search_returns_results(self, client: httpx.Client):
        """正常搜索应返回 result + source_breakdown。"""
        data = _call_tool(client, "search_knowledge", {
            "query": "第一节",
            "kb_id": "eaf9b1d0-0c53-4644-9f30-772259498832",
            "top_k": 5,
        })
        print("\n=== search_knowledge('第一节') 响应 ===")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        assert "results" in data, f"响应缺少 results 字段: {data}"
        assert "source_breakdown" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["source_breakdown"], dict)

    def test_search_result_structure(self, client: httpx.Client):
        """每个结果应包含 chunk_id / content / score / source_paths / chunk_type。"""
        data = _call_tool(client, "search_knowledge", {
            "query": "test",
            "kb_id": "00000000-0000-0000-0000-000000000001",
            "top_k": 3,
        })
        print("\n=== search_knowledge('test') 响应 ===")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        for item in data["results"]:
            assert "chunk_id" in item, f"缺少 chunk_id: {item}"
            assert "content" in item
            assert "score" in item
            assert "source_paths" in item
            assert "chunk_type" in item

    def test_empty_query_returns_error(self, client: httpx.Client):
        """空查询应返回 error。"""
        data = _call_tool(client, "search_knowledge", {
            "query": "",
            "kb_id": "00000000-0000-0000-0000-000000000001",
        })
        print("\n=== search_knowledge('') 错误响应 ===")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        assert "error" in data, f"空查询未返回 error: {data}"

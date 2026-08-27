"""HTTP 搜索 API 集成测试。"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from aion_knowledge.api import create_app


@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_search_endpoint_returns_200(client: AsyncClient):
    """POST /api/v1/search 应返回 200。"""
    response = await client.post("/api/v1/search", json={
        "query": "test query",
        "kb_id": "00000000-0000-0000-0000-000000000001",
        "top_k": 5,
        "generate_answer": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "source_breakdown" in data


@pytest.mark.asyncio
async def test_search_empty_query_returns_422(client: AsyncClient):
    """空查询应返回 422 验证错误。"""
    response = await client.post("/api/v1/search", json={
        "query": "",
        "kb_id": "00000000-0000-0000-0000-000000000001",
    })
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_with_enabled_paths(client: AsyncClient):
    """指定 enabled_paths 时应仅调用指定路径。"""
    response = await client.post("/api/v1/search", json={
        "query": "test",
        "kb_id": "00000000-0000-0000-0000-000000000001",
        "enabled_paths": ["bm25", "vector"],
        "generate_answer": False,
    })
    assert response.status_code == 200
    data = response.json()
    # 仅 bm25 和 vector 出现在 source_breakdown 中
    for path in data["source_breakdown"]:
        assert path in ("bm25", "vector")


@pytest.mark.asyncio
async def test_generate_answer_default_false(client: AsyncClient):
    """generate_answer=False 时返回 results 且 answer=None。"""
    response = await client.post("/api/v1/search", json={
        "query": "test",
        "kb_id": "00000000-0000-0000-0000-000000000001",
        "generate_answer": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["answer"] is None

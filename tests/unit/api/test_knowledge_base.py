"""KnowledgeBase API 路由测试。"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from aion_knowledge.api import create_app

# 测试创建 KB 使用的固定名字，fixture 按名清理，避免污染真实库
_CREATED_NAMES = ["项目 Alpha", "KB 1", "Get Me"]


@pytest.fixture(autouse=True)
async def cleanup_created_kbs():
    """清理测试创建的 KB，避免污染真实库。"""
    from sqlalchemy import delete

    from aion_knowledge.infrastructure.db import get_session
    from aion_knowledge.models.orm import KnowledgeBase

    yield

    async with get_session() as session:
        await session.execute(
            delete(KnowledgeBase).where(KnowledgeBase.name.in_(_CREATED_NAMES))
        )


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """AsyncClient 直接跑在测试事件循环上，避免 TestClient 每请求新建 loop。"""
    async with AsyncClient(
        transport=ASGITransport(app=create_app()),
        base_url="http://test",
    ) as c:
        yield c


async def test_create_knowledge_base(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/knowledge",
        json={"name": "项目 Alpha", "tags": ["研发"], "description": "测试"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "项目 Alpha"
    assert data["tags"] == ["研发"]
    assert data["description"] == "测试"
    assert "id" in data


async def test_create_knowledge_base_missing_name(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/knowledge", json={})
    assert resp.status_code == 422


async def test_list_knowledge_bases(client: AsyncClient) -> None:
    # 先创建一条
    await client.post(
        "/api/v1/knowledge",
        json={"name": "KB 1"},
    )
    resp = await client.get("/api/v1/knowledge")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert isinstance(data["items"], list)


async def test_create_knowledge_base_persists_trace_id(client: AsyncClient) -> None:
    """X-Trace-ID header 值落库到 kb_knowledge_bases.trace_id。"""
    from sqlalchemy import select

    from aion_knowledge.infrastructure.db import get_session
    from aion_knowledge.models.orm import KnowledgeBase

    resp = await client.post(
        "/api/v1/knowledge",
        headers={"X-Trace-ID": "kb-trace-99"},
        json={"name": "KB 1"},
    )
    assert resp.status_code == 201
    assert resp.headers.get("X-Trace-ID") == "kb-trace-99"

    async with get_session() as session:
        row = (
            await session.execute(select(KnowledgeBase).where(KnowledgeBase.name == "KB 1"))
        ).scalar_one()
        assert row.trace_id == "kb-trace-99"


async def test_get_knowledge_base_not_found(client: AsyncClient) -> None:
    resp = await client.get(f"/api/v1/knowledge/{uuid.uuid4()}")
    assert resp.status_code == 404


async def test_get_knowledge_base_found(client: AsyncClient) -> None:
    created = await client.post(
        "/api/v1/knowledge",
        json={"name": "Get Me"},
    )
    kb_id = created.json()["id"]
    resp = await client.get(f"/api/v1/knowledge/{kb_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Get Me"

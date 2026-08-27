"""删除 API 路由测试（真实 DB）。"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from aion_knowledge.api import create_app

_CREATED_NAMES = ["删除测试库"]


@pytest.fixture(autouse=True)
async def cleanup_deleted_kbs():
    """清理测试创建的 KB 及其文档行，避免污染真实库。"""
    from sqlalchemy import delete, select

    from aion_knowledge.infrastructure.db import get_session
    from aion_knowledge.models.orm import KnowledgeBase, KnowledgeDocument

    yield

    async with get_session() as session:
        await session.execute(
            delete(KnowledgeDocument).where(
                KnowledgeDocument.kb_id.in_(
                    select(KnowledgeBase.id).where(
                        KnowledgeBase.name.in_(_CREATED_NAMES)
                    )
                )
            )
        )
        await session.execute(
            delete(KnowledgeBase).where(KnowledgeBase.name.in_(_CREATED_NAMES))
        )


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=create_app()),
        base_url="http://test",
    ) as c:
        yield c


async def _create_kb(client: AsyncClient) -> str:
    resp = await client.post("/api/v1/knowledge", json={"name": "删除测试库"})
    assert resp.status_code == 201
    return str(resp.json()["id"])


@pytest.mark.asyncio
async def test_delete_kb_endpoint(client: AsyncClient):
    """DELETE /knowledge/{kb_id}：200 + deleted=true。"""
    kb_id = await _create_kb(client)

    resp = await client.delete(f"/api/v1/knowledge/{kb_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["kb_id"] == kb_id
    assert body["deleted"] is True


@pytest.mark.asyncio
async def test_delete_kb_not_found(client: AsyncClient):
    resp = await client.delete(f"/api/v1/knowledge/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_kb_idempotent(client: AsyncClient):
    """重复删除返回 200（幂等）。"""
    kb_id = await _create_kb(client)
    assert (await client.delete(f"/api/v1/knowledge/{kb_id}")).status_code == 200
    assert (await client.delete(f"/api/v1/knowledge/{kb_id}")).status_code == 200


@pytest.mark.asyncio
async def test_delete_document_endpoint(client: AsyncClient):
    """DELETE /knowledge/{kb_id}/documents/{document_id}：200。"""
    kb_id = await _create_kb(client)
    # 直接插一条文档行（不走 ingestion）
    from aion_knowledge.infrastructure.db import get_session
    from aion_knowledge.ingestion.document_repo import create_document

    async with get_session() as session:
        doc = await create_document(session, kb_id, "t.md", "md", "h1", 10)
        await session.commit()
        doc_id = str(doc.id)

    resp = await client.delete(f"/api/v1/knowledge/{kb_id}/documents/{doc_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == doc_id
    assert body["deleted"] is True

    # 锁住「逻辑删除提交点」：deleted=true 已落库
    from sqlalchemy import select

    from aion_knowledge.models.orm import KnowledgeDocument

    async with get_session() as session:
        row = (
            await session.execute(
                select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
            )
        ).scalar_one()
        assert row.deleted is True


@pytest.mark.asyncio
async def test_delete_document_not_found(client: AsyncClient):
    kb_id = await _create_kb(client)
    resp = await client.delete(f"/api/v1/knowledge/{kb_id}/documents/{uuid.uuid4()}")
    assert resp.status_code == 404

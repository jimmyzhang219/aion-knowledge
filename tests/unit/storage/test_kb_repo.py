"""KnowledgeBaseRepo 测试。"""

from __future__ import annotations

import uuid

import pytest

from aion_knowledge.storage.relational.kb_repo import KnowledgeBaseRepo

# 测试创建 KB 使用的固定名字，fixture 按名清理，避免污染真实库
_CREATED_NAMES = [
    "Test KB", "Test KB with trace", "Find Me",
    "Find Me Deleted", "List Filter Deleted", "List Filter Alive",
]


@pytest.fixture(scope="module", autouse=True)
async def ensure_schema_synced():
    """确保 DB schema 与 ORM 同步（新增列需要 process 内 diff 加列）。"""
    from aion_knowledge.infrastructure.db import init_db
    await init_db()


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


@pytest.mark.asyncio
async def test_create_knowledge_base() -> None:
    repo = KnowledgeBaseRepo()
    result = await repo.create(name="Test KB", tags=["dev"], description="desc")
    assert result.name == "Test KB"
    assert result.tags == ["dev"]
    assert result.description == "desc"
    assert result.id is not None


@pytest.mark.asyncio
async def test_list_knowledge_bases() -> None:
    repo = KnowledgeBaseRepo()
    result = await repo.list_all()
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_get_by_id_found() -> None:
    repo = KnowledgeBaseRepo()
    created = await repo.create(name="Find Me")
    found = await repo.get_by_id(created.id)
    assert found is not None
    assert found.name == "Find Me"


@pytest.mark.asyncio
async def test_get_by_id_not_found() -> None:
    repo = KnowledgeBaseRepo()
    found = await repo.get_by_id(uuid.uuid4())
    assert found is None


@pytest.mark.asyncio
async def test_create_knowledge_base_with_trace_id() -> None:
    """create 落库 trace_id。"""
    repo = KnowledgeBaseRepo()
    result = await repo.create(name="Test KB with trace", tags=["dev"], description="desc",
                               trace_id="kb-trace-77")
    assert result.trace_id == "kb-trace-77"


@pytest.mark.asyncio
async def test_get_by_id_excludes_deleted() -> None:
    """已逻辑删除的 KB 查询视为不存在（get_by_id 返回 None）。"""
    from sqlalchemy import update

    from aion_knowledge.infrastructure.db import get_session
    from aion_knowledge.models.orm import KnowledgeBase

    repo = KnowledgeBaseRepo()
    created = await repo.create(name="Find Me Deleted")
    async with get_session() as session:
        await session.execute(
            update(KnowledgeBase).where(KnowledgeBase.id == created.id).values(deleted=True)
        )
        await session.commit()

    found = await repo.get_by_id(created.id)
    assert found is None


@pytest.mark.asyncio
async def test_list_all_excludes_deleted() -> None:
    """已逻辑删除的 KB 不出现在列表。"""
    from sqlalchemy import update

    from aion_knowledge.infrastructure.db import get_session
    from aion_knowledge.models.orm import KnowledgeBase

    repo = KnowledgeBaseRepo()
    deleted_kb = await repo.create(name="List Filter Deleted")
    alive_kb = await repo.create(name="List Filter Alive")
    async with get_session() as session:
        await session.execute(
            update(KnowledgeBase).where(KnowledgeBase.id == deleted_kb.id).values(deleted=True)
        )
        await session.commit()

    items = await repo.list_all()
    ids = [kb.id for kb in items]
    assert alive_kb.id in ids
    assert deleted_kb.id not in ids

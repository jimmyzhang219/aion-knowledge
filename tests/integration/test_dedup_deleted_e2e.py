"""查重规则 + deleted 全局防护端到端测试（直插库，不跑 ingestion）。

覆盖：改名重传判重、跨 KB 不判重、逻辑删除后重传放行、已删 KB 上传拦截、
已删文档不可见（postproc 重跑 404 的根因；rerun 404 本身由单测覆盖——
test_postproc_rerun.test_document_not_found + Task 4 过滤）、
同 hash 双非删除行回归保护（scalars().first() 不抛 MultipleResultsFound）。
运行: cd <repo> && pytest tests/integration/test_dedup_deleted_e2e.py -v
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text as sql_text

from aion_knowledge.infrastructure.db import _engine, dispose_engine, get_session
from aion_knowledge.ingestion.document_repo import get_document_by_id
from aion_knowledge.ingestion.kb_guard import KnowledgeBaseNotFoundError, ensure_kb_exists
from aion_knowledge.ingestion.strategy.regular.strategy import RegularIngestionStrategy

pytestmark = pytest.mark.asyncio(loop_scope="session")

_KB1 = uuid.uuid4()
_KB2 = uuid.uuid4()
_CONTENT = b"same content for dedup test"


async def _cleanup() -> None:
    """清理两个测试 KB 的全部关联数据（测试开头/结尾各调一次，幂等）。"""
    await dispose_engine()
    async with _engine.begin() as conn:
        for kb in (_KB1, _KB2):
            await conn.execute(sql_text("DELETE FROM chunk_text WHERE kb_id = :kb"), {"kb": kb})
            await conn.execute(
                sql_text("DELETE FROM doc_knowledge_documents WHERE kb_id = :kb"), {"kb": kb}
            )
            await conn.execute(
                sql_text("DELETE FROM kb_knowledge_bases WHERE id = :kb"), {"kb": kb}
            )


async def _insert_kb(kb_id: uuid.UUID, deleted: bool = False) -> None:
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO kb_knowledge_bases (id, name, tags, description, created_at, updated_at, deleted) "
                "VALUES (:id, :name, '{}', '', now(), now(), :deleted)"
            ),
            {"id": kb_id, "name": f"Dedup-E2E-{kb_id}", "deleted": deleted},
        )


async def _insert_doc(
    kb_id: uuid.UUID, doc_id: uuid.UUID, file_hash: str, doc_name: str, deleted: bool = False
) -> None:
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO doc_knowledge_documents "
                "(id, kb_id, doc_name, suffix, hash, size, status, tags, source_label, "
                " creator, file_path, chunk_strategy, created_at, updated_at, deleted) "
                "VALUES (:id, :kb, :name, '', :hash, :size, 'completed', '{}', '', 'test', '', "
                " 'auto', now(), now(), :deleted)"
            ),
            {
                "id": doc_id,
                "kb": kb_id,
                "name": doc_name,
                "hash": file_hash,
                "size": len(_CONTENT),
                "deleted": deleted,
            },
        )


async def test_rename_upload_same_content_is_duplicate() -> None:
    """改名重传同内容 → 判重复（新规则：不看 doc_name）。"""
    await _cleanup()
    await _insert_kb(_KB1)
    doc_id = uuid.uuid4()
    await _insert_doc(
        _KB1, doc_id, RegularIngestionStrategy("md")._compute_hash(_CONTENT), "orig.md"
    )

    strategy = RegularIngestionStrategy(suffix="md")
    result = await strategy._pre_process(kb_id=str(_KB1), content=_CONTENT, file_name="renamed.md")

    assert result is not None
    assert result["status"] == "duplicate"
    assert result["document_id"] == str(doc_id)
    await _cleanup()


async def test_cross_kb_same_content_not_duplicate() -> None:
    """跨 KB 同内容 → 不判重（查重范围限定单 KB）。"""
    await _cleanup()
    await _insert_kb(_KB1)
    await _insert_kb(_KB2)
    await _insert_doc(
        _KB1, uuid.uuid4(), RegularIngestionStrategy("md")._compute_hash(_CONTENT), "orig.md"
    )

    strategy = RegularIngestionStrategy(suffix="md")
    result = await strategy._pre_process(kb_id=str(_KB2), content=_CONTENT, file_name="orig.md")

    assert result is None
    await _cleanup()


async def test_deleted_doc_can_be_reuploaded() -> None:
    """逻辑删除后重传同内容 → 不判重（可重新入库）。"""
    await _cleanup()
    await _insert_kb(_KB1)
    await _insert_doc(
        _KB1,
        uuid.uuid4(),
        RegularIngestionStrategy("md")._compute_hash(_CONTENT),
        "orig.md",
        deleted=True,
    )

    strategy = RegularIngestionStrategy(suffix="md")
    result = await strategy._pre_process(kb_id=str(_KB1), content=_CONTENT, file_name="orig.md")

    assert result is None
    await _cleanup()


async def test_duplicate_hash_two_live_rows_no_crash() -> None:
    """同 KB 两条未删除同 hash 行（旧规则改名逃重遗留）→ 返回 duplicate 而非抛异常。

    任务 1 修复回归保护：get_document_by_hash 用 scalars().first()，
    scalar_one_or_none 会抛 MultipleResultsFound。
    """
    await _cleanup()
    await _insert_kb(_KB1)
    file_hash = RegularIngestionStrategy("md")._compute_hash(_CONTENT)
    await _insert_doc(_KB1, uuid.uuid4(), file_hash, "orig.md")
    await _insert_doc(_KB1, uuid.uuid4(), file_hash, "orig-copy.md")

    strategy = RegularIngestionStrategy(suffix="md")
    result = await strategy._pre_process(kb_id=str(_KB1), content=_CONTENT, file_name="renamed.md")

    assert result is not None
    assert result["status"] == "duplicate"
    await _cleanup()


async def test_deleted_kb_rejects_upload() -> None:
    """已删 KB：ensure_kb_exists 抛异常（上传入口 404）。"""
    await _cleanup()
    await _insert_kb(_KB1, deleted=True)

    with pytest.raises(KnowledgeBaseNotFoundError):
        await ensure_kb_exists(str(_KB1))
    await _cleanup()


async def test_deleted_doc_invisible_to_repo() -> None:
    """已删文档经 get_document_by_id 不可见（postproc 重跑 404 的根因）。"""
    await _cleanup()
    await _insert_kb(_KB1)
    live_doc_id = uuid.uuid4()
    deleted_doc_id = uuid.uuid4()
    await _insert_doc(_KB1, live_doc_id, "deadbeef" * 8, "orig.md")
    await _insert_doc(_KB1, deleted_doc_id, "deadbeef" * 8, "orig-deleted.md", deleted=True)

    async with get_session() as session:
        # 正对照：活行必须可见，否则查询整体坏掉（恒 None）时本用例会假绿
        assert await get_document_by_id(session, live_doc_id) is not None
        assert await get_document_by_id(session, deleted_doc_id) is None
    await _cleanup()

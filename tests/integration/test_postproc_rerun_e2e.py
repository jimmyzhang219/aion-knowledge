"""后处理重跑 E2E：真实 DB 验证 helper → 队列 → worker only 传递全链路。

运行: cd /Users/jimmy/VSCodeProjects/aion-knowledge && python -m pytest tests/integration/test_postproc_rerun_e2e.py -v -s
"""
from __future__ import annotations

import pytest

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.infrastructure.db import _engine
from aion_knowledge.infrastructure.queues import postproc_queue
from aion_knowledge.ingestion.postproc_rerun import enqueue_postproc_rerun

_TEST_KB_ID = "00000000-0000-0000-0000-00000000e2e3"

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _ensure_test_kb() -> None:
    from sqlalchemy import text as sql_text

    async with _engine.begin() as conn:
        row = (await conn.execute(
            sql_text("SELECT id FROM kb_knowledge_bases WHERE id = :id"),
            {"id": _TEST_KB_ID},
        )).one_or_none()
        if not row:
            await conn.execute(
                sql_text(
                    "INSERT INTO kb_knowledge_bases (id, name, tags, description, created_at, updated_at) "
                    "VALUES (:id, :name, :tags, :desc, now(), now())"
                ),
                {
                    "id": _TEST_KB_ID,
                    "name": "RERUN-E2E-Test-KB",
                    "tags": ["测试"],
                    "desc": "后处理重跑 E2E 测试用知识库",
                },
            )


async def _insert_doc_with_chunk(doc_id, doc_name: str) -> None:
    """ORM 插入文档 + chunk。

    注意：chunk_text 的 context_header/keywords/token_count 等字段 nullable=False
    且只有 Python 层默认值（无 server_default），裸 SQL INSERT 会违反 NOT NULL，
    必须走 ORM（default 在 ORM 层生效）。
    """
    import uuid as uuid_mod

    from aion_knowledge.infrastructure.db import get_session
    from aion_knowledge.models.enums import ChunkStrategy, DocumentStatus
    from aion_knowledge.models.orm import KnowledgeDocument
    from aion_knowledge.pipeline.postproc.text.orm import ChunkText

    async with get_session() as session:
        session.add(KnowledgeDocument(
            id=doc_id,
            kb_id=uuid_mod.UUID(_TEST_KB_ID),
            doc_name=doc_name,
            suffix="md",
            hash="e2e-hash-rerun-" + doc_name,
            size=10,
            status=DocumentStatus.completed,
            creator="test",
            chunk_strategy=ChunkStrategy.auto,
        ))
        session.add(ChunkText(
            id=uuid7(),
            document_id=doc_id,
            kb_id=uuid_mod.UUID(_TEST_KB_ID),
            content="重跑测试内容",
            seq_num=1,
            chunk_type="text",
        ))
        await session.commit()


async def _cleanup() -> None:
    from sqlalchemy import text as sql_text

    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "DELETE FROM chunk_text WHERE document_id IN "
                "(SELECT id FROM doc_knowledge_documents WHERE kb_id = :kb_id)"
            ),
            {"kb_id": _TEST_KB_ID},
        )
        await conn.execute(
            sql_text(
                "DELETE FROM doc_knowledge_documents WHERE kb_id = :kb_id"
            ),
            {"kb_id": _TEST_KB_ID},
        )


async def test_rerun_full_chain(monkeypatch):
    """helper 真实执行 → 队列收到任务 → worker only 白名单传递。"""
    from aion_knowledge.common.config import settings
    from aion_knowledge.infrastructure.workers import _run_postproc_subtasks

    # 先清残留再建数据（断言中途失败时，下次运行先清掉上次残留）
    await _cleanup()
    await _ensure_test_kb()
    doc_id = uuid7()
    doc_name = "rerun-e2e.md"
    await _insert_doc_with_chunk(doc_id, doc_name)

    # 确定性：确保 raptor 处于启用状态
    monkeypatch.setattr(settings, "postproc_raptor", True)

    # 1. helper 真实执行
    task = await enqueue_postproc_rerun(str(_TEST_KB_ID), str(doc_id), ["raptor"])
    assert str(task.document_id) == str(doc_id)
    assert task.doc_name == doc_name
    assert task.suffix == "md"
    assert task.chunk_count == 1
    assert task.modules == ["raptor"]

    # 2. 队列真实收到任务
    qtask = await postproc_queue.get()
    assert qtask.document_id == str(doc_id)
    assert qtask.modules == ["raptor"]
    postproc_queue.task_done()

    # 3. worker 消费：only 白名单传递（spy dispatcher，真实 chunks 加载与 settings_dict 计算）
    captured: dict = {}

    class _FakeDispatcher:
        def __init__(self, settings_dict, only=None):
            captured["settings"] = settings_dict
            captured["only"] = only

        async def run_second_batch(self, ctx, chunks):
            captured["chunks"] = chunks  # 记录真实加载的 chunks

    monkeypatch.setattr(
        "aion_knowledge.pipeline.postproc.dispatcher.PostProcDispatcher",
        _FakeDispatcher,
    )
    await _run_postproc_subtasks(qtask)
    assert captured["only"] == ["raptor"]
    assert captured["settings"]["raptor"] is True
    # worker 真实从 DB 加载到 chunk（chunk 加载异常会被 workers.py 静默吞掉，必须显式断言）
    assert len(captured["chunks"]) == 1
    assert captured["chunks"][0]["content"] == "重跑测试内容"

"""真实端到端集成测试：零 Mock，全链路访问 OSS + PostgreSQL。

重要：使用 loop_scope="session" 确保 asyncpg 连接在同一个事件循环中复用。
"""

from __future__ import annotations

import logging

import pytest

from aion_knowledge.indexing.executor import IndexingExecutor
from aion_knowledge.infrastructure.models import PostProcTask, UnifiedContext
from aion_knowledge.infrastructure.queues import ctx_queue, postproc_queue
from aion_knowledge.ingestion.strategy.registry import get_strategy
from aion_knowledge.models.enums import StrategyName

pytestmark = pytest.mark.asyncio(loop_scope="session")
logger = logging.getLogger(__name__)

_TEST_KB_ID = "00000000-0000-0000-0000-0000000000e2"
_TEST_DOC_NAME = "e2e_real_test.md"


async def _cleanup_db() -> None:
    """清理测试 KB 所有记录。"""
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import _engine, dispose_engine

    await dispose_engine()  # 释放旧连接，避免 "another operation is in progress"
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
                "DELETE FROM task_ingestion_tasks WHERE document_id IN "
                "(SELECT id FROM doc_knowledge_documents WHERE kb_id = :kb_id)"
            ),
            {"kb_id": _TEST_KB_ID},
        )
        await conn.execute(
            sql_text("DELETE FROM doc_knowledge_documents WHERE kb_id = :kb_id"),
            {"kb_id": _TEST_KB_ID},
        )
    await _ensure_test_kb()  # 清理后恢复测试 KB，供 execute() 校验通过


async def _ensure_test_kb() -> None:
    """确保测试知识库存在（kb 关系由程序维护，集成测试需自建 KB）。"""
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import _engine

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
                    "name": "E2E-Real-Test-KB",
                    "tags": [],
                    "desc": "E2E real 测试用知识库",
                },
            )
            logger.info("Test KB created: %s", _TEST_KB_ID)


async def delete_document_cascade(doc_id: str) -> None:
    """根据 doc_knowledge_documents.id 级联删除文档及其全部关联数据。

    删除范围：
      - OSS（S3/MinIO）或 LocalFileStore 中的原始上传文件
      - LocalFileStore 中的 ``converted.md``（执行器转换产物）
      - chunk_text（切片）
      - task_ingestion_tasks（入库任务）
      - chunk_vector（向量嵌入，待真实 Provider 接入后生效）
      - doc_knowledge_documents（文档元数据）
    """
    from sqlalchemy import text as sql_text

    from aion_knowledge.common.config import settings
    from aion_knowledge.infrastructure.db import _engine, dispose_engine

    await dispose_engine()

    # 1) 查询文档信息（需要 kb_id/hash 定位文件目录）
    async with _engine.begin() as conn:
        row = (await conn.execute(
            sql_text("SELECT kb_id, doc_name, hash FROM doc_knowledge_documents WHERE id = :id"),
            {"id": doc_id},
        )).one_or_none()

    if not row:
        logger.warning("delete_document_cascade: document %s not found", doc_id)
        return

    kb_id, doc_name, doc_hash = row

    # 2) 删除对象存储文件
    # 目录布局：{s3_prefix}/{kb_id}/{hash}/（原始文件 + converted.md + images/）
    file_dir = f"{kb_id}/{doc_hash}"
    if settings.s3_access_key:
        try:
            import boto3
            from botocore.config import Config as BotoConfig

            boto_config = BotoConfig(
                region_name=settings.s3_region,
                signature_version="s3v4",
                s3={"addressing_style": settings.s3_addressing_style},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            )
            client = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint,
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=settings.s3_secret_key,
                config=boto_config,
            )
            prefix = f"{settings.s3_prefix}/{file_dir}/"
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix)
            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    client.delete_object(Bucket=settings.s3_bucket, Key=key)
                    logger.info("OSS deleted: %s", key)
        except Exception as exc:
            logger.error("OSS cleanup failed for doc %s: %s", doc_id, exc)

    # LocalFileStore 清理（S3 未配置时原始文件、converted.md 等产物落在本地存储）
    try:
        from pathlib import Path

        local_dir = Path("/tmp/aion_storage") / file_dir
        if local_dir.is_dir():
            for f in local_dir.rglob("*"):
                if f.is_file():
                    f.unlink()
                    logger.info("Local file deleted: %s", f)
            # 从下往上删除空目录
            for d in sorted(local_dir.rglob("*"), reverse=True):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            local_dir.rmdir()
            logger.info("Local dir deleted: %s", local_dir)
    except Exception as exc:
        logger.error("Local file cleanup failed for doc %s: %s", doc_id, exc)

    # 3) 删除数据库关联记录
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text("DELETE FROM chunk_text WHERE document_id = :id"),
            {"id": doc_id},
        )
        await conn.execute(
            sql_text("DELETE FROM task_ingestion_tasks WHERE document_id = :id"),
            {"id": doc_id},
        )
        await conn.execute(
            sql_text("DELETE FROM doc_knowledge_documents WHERE id = :id"),
            {"id": doc_id},
        )
    logger.info("Document cascade deleted: id=%s name=%s hash=%s", doc_id, doc_name, doc_hash[:16])


# ── 测试 1：enqueue_upload 真实 OSS + DB ────────────────────


@pytest.mark.asyncio
async def test_enqueue_upload_real() -> None:
    """enqueue_upload → 真实 OSS 上传 + metadata DB 写入。"""
    await _cleanup_db()

    md_content = b"# E2E Real\n\nHello.\n\n## A\n\nContent A.\n\n## B\n\nContent B.\n"

    strategy = get_strategy(StrategyName.regular, suffix="md")
    result = await strategy.execute(
        kb_id=_TEST_KB_ID,
        content=md_content,
        file_name=_TEST_DOC_NAME,
    )

    assert result["status"] == "queued"
    doc_id = result.get("document_id")
    assert doc_id

    ctx: UnifiedContext = await ctx_queue.get()
    assert ctx.doc_name == _TEST_DOC_NAME
    assert ctx.original_file_ref.startswith("s3://"), f"original_file_ref={ctx.original_file_ref}"

    # 验证 DB（使用同一 async 引擎）
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import _engine

    async with _engine.begin() as conn:
        row = (await conn.execute(
            sql_text("SELECT status, hash FROM doc_knowledge_documents WHERE id = :id"),
            {"id": doc_id},
        )).one()
        assert row.status == "pending"
        assert row.hash

        row = (await conn.execute(
            sql_text("SELECT status FROM task_ingestion_tasks WHERE document_id = :id"),
            {"id": doc_id},
        )).one()
        assert row.status == "pending"

    ctx_queue.task_done()
    await _cleanup_db()


# ── 测试 2：IndexingExecutor 全流程 + chunk_text 写入 ──


@pytest.mark.asyncio
async def test_pipeline_executor_real(tmp_path) -> None:
    """IndexingExecutor → OSS 下载 → 解析 → 分块 → chunk_text 真实写入。"""
    await _cleanup_db()

    md_content = (
        b"# Full Pipeline\n\n"
        b"Testing real pipeline flow.\n\n"
        b"## Section A\n\n"
        b"Content of A.\n\n"
        b"## Section B\n\n"
        b"Content of B.\n"
    )

    strategy = get_strategy(StrategyName.regular, suffix="md")
    result = await strategy.execute(
        kb_id=_TEST_KB_ID,
        content=md_content,
        file_name=_TEST_DOC_NAME,
    )
    assert result["status"] == "queued"
    doc_id = result["document_id"]

    ctx: UnifiedContext = await ctx_queue.get()
    executor = IndexingExecutor()
    postproc_task = await executor.run(ctx)

    assert isinstance(postproc_task, PostProcTask)
    assert postproc_task.chunk_count > 0

    # 验证 chunk_text 表真实写入（同一 async 引擎）
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import _engine

    async with _engine.begin() as conn:
        row = (await conn.execute(
            sql_text("SELECT status FROM doc_knowledge_documents WHERE id = :id"),
            {"id": doc_id},
        )).one()
        assert row.status == "completed", f"doc status: {row.status}"

        rows = (await conn.execute(
            sql_text(
                "SELECT content, seq_num FROM chunk_text "
                "WHERE document_id = :id ORDER BY seq_num"
            ),
            {"id": doc_id},
        )).all()
        assert len(rows) == postproc_task.chunk_count
        for r in rows:
            assert r.content

    ctx_queue.task_done()
    await _cleanup_db()


# ── 测试 3：完整 Worker 链路 ────────────────────────────────


@pytest.mark.asyncio
async def test_full_pipeline_with_workers() -> None:
    """启动真实 Worker1/Worker2，全链路自动执行完毕。"""
    import asyncio

    await _cleanup_db()

    md_content = (
        b"# Worker E2E\n\n"
        b"Workers processing this doc.\n\n"
        b"## Chapter X\n\n"
        b"Content X.\n\n"
        b"## Chapter Y\n\n"
        b"Content Y.\n"
    )

    # 启动 Worker
    from aion_knowledge.infrastructure.workers import pipeline_worker, postproc_worker

    w1 = asyncio.create_task(pipeline_worker(), name="pipeline_worker")
    w2 = asyncio.create_task(postproc_worker(), name="postproc_worker")
    await asyncio.sleep(0.05)

    # 上传
    strategy = get_strategy(StrategyName.regular, suffix="md")
    result = await strategy.execute(
        kb_id=_TEST_KB_ID,
        content=md_content,
        file_name=_TEST_DOC_NAME,
    )
    assert result["status"] == "queued"
    doc_id = result["document_id"]

    # 等待 Worker 消费
    await asyncio.wait_for(ctx_queue.join(), timeout=30)
    await asyncio.wait_for(postproc_queue.join(), timeout=30)

    w1.cancel()
    w2.cancel()
    await asyncio.gather(w1, w2, return_exceptions=True)

    # 验证最终状态（同一 async 引擎）
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import _engine

    async with _engine.begin() as conn:
        row = (await conn.execute(
            sql_text("SELECT status FROM doc_knowledge_documents WHERE id = :id"),
            {"id": doc_id},
        )).one()
        assert row.status == "completed", f"doc status: {row.status}"

        row = (await conn.execute(
            sql_text("SELECT status FROM task_ingestion_tasks WHERE document_id = :id"),
            {"id": doc_id},
        )).one()
        assert row.status == "completed"

        count = (await conn.execute(
            sql_text("SELECT COUNT(*) FROM chunk_text WHERE document_id = :id"),
            {"id": doc_id},
        )).scalar()
        assert count > 0, "Expected chunks in chunk_text"

    await _cleanup_db()


# _run_worker 已由 pipeline_worker / postproc_worker 直接启动

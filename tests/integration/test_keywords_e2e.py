"""Keywords 迁移 E2E 验证：全链路测试 keywords 写入 chunk_text。

运行: uv run --active pytest tests/integration/test_keywords_e2e.py -v -s
"""
from __future__ import annotations

import logging

import pytest

from aion_knowledge.infrastructure.db import _engine, dispose_engine, get_session

logger = logging.getLogger(__name__)

_TEST_KB_ID = "00000000-0000-0000-0000-00000000e2e1"

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _ensure_test_kb() -> None:
    """确保测试知识库存在（含预设 tags），同时确保 chunk_text 有 keywords 列。"""
    from sqlalchemy import text as sql_text

    async with _engine.begin() as conn:
        # 迁移：确保 chunk_text 表有 keywords 列（开发环境尚未跑 migration）
        await conn.execute(
            sql_text(
                "ALTER TABLE chunk_text ADD COLUMN IF NOT EXISTS "
                "keywords VARCHAR[] NOT NULL DEFAULT '{}'::VARCHAR[]"
            )
        )

        # 确保测试 KB 存在
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
                    "name": "Keywords-E2E-Test-KB",
                    "tags": ["人工智能", "医疗", "药物研发", "深度学习"],
                    "desc": "Keywords 迁移测试用知识库",
                },
            )
            logger.info("Test KB created: %s", _TEST_KB_ID)


async def _cleanup() -> None:
    """清理本次测试产生的数据。"""
    from sqlalchemy import text as sql_text

    await dispose_engine()
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


@pytest.mark.asyncio
async def test_keywords_written_to_chunk_text_e2e() -> None:
    """全链路测试：上传 → 解析 → 分块 → TextModule → KeywordExtract → 验证 chunk_text.keywords。"""
    await _cleanup()
    await _ensure_test_kb()

    # ── 1. 准备测试内容（包含 KB tags 中的词汇，触发 Tier 2 匹配）──
    md_content = (
        "# AI in Healthcare\n\n"
        "## 诊断\n\n"
        "人工智能在医疗诊断中发挥重要作用。深度学习模型可以分析医学影像。\n\n"
        "## 药物\n\n"
        "药物研发借助机器学习加速，AI 技术降低研发成本。\n\n"
        "## 结尾\n\n"
        "这是最后一段，没有关键词匹配。\n"
    ).encode("utf-8")

    # ── 2. 上传文件，触发 Pipeline ──
    from aion_knowledge.ingestion.strategy.registry import get_strategy
    from aion_knowledge.models.enums import StrategyName

    strategy = get_strategy(StrategyName.regular, suffix="md")
    result = await strategy.execute(
        kb_id=_TEST_KB_ID,
        content=md_content,
        file_name="keywords_e2e_test.md",
    )
    assert result["status"] == "queued", f"enqueue failed: {result}"
    doc_id = result["document_id"]
    logger.info("Document created: %s", doc_id)

    # ── 3. 从 Queue1 取消息，执行 IndexingExecutor ──
    from aion_knowledge.infrastructure.models import PostProcTask
    from aion_knowledge.infrastructure.queues import ctx_queue

    ctx = await ctx_queue.get()
    assert ctx.doc_name == "keywords_e2e_test.md"

    from aion_knowledge.indexing.executor import IndexingExecutor

    executor = IndexingExecutor()
    postproc_task = await executor.run(ctx)
    assert isinstance(postproc_task, PostProcTask)
    assert postproc_task.chunk_count > 0
    logger.info("Pipeline done: chunk_count=%d", postproc_task.chunk_count)

    ctx_queue.task_done()

    # ── 4. 执行后处理（启用 keyword_extract）──
    from aion_knowledge.pipeline.postproc.base import PostProcContext
    from aion_knowledge.pipeline.postproc.dispatcher import PostProcDispatcher

    settings_dict = {
        "keyword_extract":  True,
        "question_gen":     False,
        "summarizer":       False,
        "raptor":           False,
        "graph_extract":    False,
        "community":        False,
        "disambiguation":   False,
        "wiki":             False,
    }

    # 从 chunk_text 加载 chunks
    from sqlalchemy import text as sql_text

    async with get_session() as session:
        rows = await session.execute(
            sql_text(
                "SELECT id, content, seq_num, metadata FROM chunk_text "
                "WHERE document_id = :doc_id AND chunk_type = 'text' ORDER BY seq_num"
            ),
            {"doc_id": doc_id},
        )
        chunks = []
        for row in rows:
            row_dict = dict(row._mapping)
            chunks.append({
                "chunk_uuid": str(row_dict["id"]),
                "content": row_dict["content"],
                "seq_num": row_dict["seq_num"],
                "chunk_metadata": row_dict.get("metadata", {}),
            })

    logger.info("Loaded %d chunks from chunk_text", len(chunks))

    dispatcher = PostProcDispatcher(settings_dict)
    proc_ctx = PostProcContext(
        document_id=doc_id,
        kb_id=_TEST_KB_ID,
        doc_name="keywords_e2e_test.md",
    )
    await dispatcher.run_second_batch(proc_ctx, chunks)

    # ── 5. 验证 chunk_text.keywords ──
    async with get_session() as session:
        rows = await session.execute(
            sql_text(
                "SELECT id, content, keywords, seq_num FROM chunk_text "
                "WHERE document_id = :doc_id AND chunk_type = 'text' ORDER BY seq_num"
            ),
            {"doc_id": doc_id},
        )
        results = list(rows)

    assert len(results) > 0, "No chunk_text rows found!"
    logger.info("=== Keywords 验证结果 ===")
    for row in results:
        d = dict(row._mapping)
        kw = d.get("keywords", [])
        logger.info(
            "  seq=%s keywords=%s content_preview=%s...",
            d["seq_num"], kw, d["content"][:40],
        )

    # ── 6. 验证至少有一个 chunk 包含 KB tags 中的关键词 ──
    all_keywords_flat = []
    for row in results:
        d = dict(row._mapping)
        all_keywords_flat.extend(d.get("keywords", []))

    logger.info("All extracted keywords: %s", all_keywords_flat)
    expected_tags = {"人工智能", "医疗", "药物研发", "深度学习"}
    found_tags = expected_tags & set(all_keywords_flat)
    logger.info("KB tags found in keywords: %s", found_tags)
    assert len(found_tags) >= 1, (
        f"Expected at least 1 KB tag in keywords, got {found_tags}. "
        f"All keywords: {all_keywords_flat}"
    )

    # ── 7. 验证文档状态为 completed ──
    async with _engine.begin() as conn:
        row = (await conn.execute(
            sql_text("SELECT status FROM doc_knowledge_documents WHERE id = :id"),
            {"id": doc_id},
        )).one()
        assert row.status == "completed", f"doc status: {row.status}"

    logger.info("=== E2E 测试全部通过 ===")

    # 清理
    await _cleanup()

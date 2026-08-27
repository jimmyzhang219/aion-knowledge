"""
真实端到端测试：FAQ + 普通文档上传，检验数据正确性。

直接调用 IngestionStrategy → Queue → IndexingExecutor → DB 验证。
不 Mock 任何组件，使用真实 PostgreSQL + LocalFileStore。
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# 强制使用 local file storage（避免依赖 S3）
os.environ["AION_S3_ACCESS_KEY"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_e2e")

_TEST_KB_ID = "00000000-0000-0000-0000-00000000feed"

# ── 辅助函数 ──────────────────────────────────────────────────────


async def cleanup_kb(kb_id: str) -> None:
    """清理测试 KB 的全部数据。"""
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import _engine

    async with _engine.begin() as conn:
        # 先查所有 document ID
        doc_ids = (
            await conn.execute(
                sql_text("SELECT id FROM doc_knowledge_documents WHERE kb_id = :kb_id"),
                {"kb_id": kb_id},
            )
        ).scalars().all()

        for doc_id in doc_ids:
            # 清理 chunk 数据（chunk_vector 关联 chunk_text.id，需先清理）
            chunk_text_ids = (
                await conn.execute(
                    sql_text("SELECT id FROM chunk_text WHERE document_id = :id"),
                    {"id": doc_id},
                )
            ).scalars().all()
            for ctid in chunk_text_ids:
                await conn.execute(
                    sql_text("DELETE FROM chunk_vector WHERE chunk_id = :id"),
                    {"id": ctid},
                )
            await conn.execute(
                sql_text("DELETE FROM chunk_text WHERE document_id = :id"),
                {"id": doc_id},
            )
            await conn.execute(
                sql_text("DELETE FROM task_ingestion_tasks WHERE document_id = :id"),
                {"id": doc_id},
            )

        # 清理文档记录
        await conn.execute(
            sql_text("DELETE FROM doc_knowledge_documents WHERE kb_id = :kb_id"),
            {"kb_id": kb_id},
        )

    # 清理本地文件
    local_base = Path("/tmp/aion_storage")
    if local_base.is_dir():
        for f in local_base.iterdir():
            if f.is_file():
                f.unlink()
        docs_dir = local_base / "docs"
        if docs_dir.is_dir():
            for sub in docs_dir.iterdir():
                if sub.is_dir():
                    for f in sub.iterdir():
                        f.unlink()
                    sub.rmdir()
            # docs_dir.rmdir()  # 可能还有别的

    logger.info("已清理 KB=%s 的数据", kb_id)


def create_faq_csv() -> tuple[bytes, str]:
    """创建 FAQ CSV 测试文件。"""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["分类", "问题", "相似问题", "负问题", "答案", "答案策略"])
    writer.writerow(["技术", "如何重置密码？", "密码忘了", "如何查看密码", "进入设置页面修改密码", "all"])
    writer.writerow(["技术", "如何开启双因素认证？", "2FA怎么开", "", "在安全设置中开启双因素认证", "all"])
    writer.writerow(["业务", "退货流程是什么？", "如何退货", "换货流程", "在订单页面申请退货退款", "all"])
    writer.writerow(["业务", "如何联系客服？", "客服热线", "", "拨打 400-xxx-xxxx", "all"])

    content = buf.getvalue().encode("utf-8")
    return content, "faq_test.csv"


def create_markdown_doc() -> tuple[bytes, str]:
    """创建 Markdown 测试文档。"""
    content = (
        "# Aion Knowledge 用户手册\n\n"
        "## 快速开始\n\n"
        "欢迎使用 Aion Knowledge。本系统支持多格式文档的知识库构建。\n\n"
        "### 安装\n\n"
        "1. 确保 Python 3.12+ 已安装\n"
        "2. 克隆仓库并安装依赖\n"
        "3. 配置 PostgreSQL 数据库\n\n"
        "## 功能特性\n\n"
        "### 多源数据接入\n\n"
        "系统支持多种数据源的接入，包括文件上传、URL 导入、API 直接接入等。\n\n"
        "### 智能分块\n\n"
        "系统内置了多种分块策略，包括递归字符分割、语义分割等。\n\n"
        "### 向量检索\n\n"
        "支持多种 embedding 模型，包括 BGE、OpenAI 等。\n\n"
        "## 架构设计\n\n"
        "Aion Knowledge 采用经典 RAG 三阶段架构：\n\n"
        "- Ingestion: 数据接入层\n"
        "- Indexing: 索引构建层\n"
        "- Retrieval: 检索引擎层\n\n"
        "## API 参考\n\n"
        "### POST /api/v1/knowledge\n\n"
        "创建知识库。\n\n"
        "### POST /api/v1/knowledge/{kb_id}/documents/upload\n\n"
        "上传文档。\n\n"
        "### POST /api/v1/search\n\n"
        "检索。\n\n"
    ).encode("utf-8")
    return content, "user_manual.md"


async def test_ingest_and_index(
    content: bytes,
    file_name: str,
    strategy_name: str,
    suffix: str,
    source: str,
    expected_verify_fn,
) -> dict:
    """上传 + 索引 + 验证全流程。"""
    from aion_knowledge.indexing.executor import IndexingExecutor
    from aion_knowledge.infrastructure.models import UnifiedContext
    from aion_knowledge.infrastructure.queues import ctx_queue
    from aion_knowledge.ingestion.strategy.registry import get_strategy as get_ingestion_strategy

    # Step 1: 通过 IngestionStrategy 上传
    kwargs = {}
    if strategy_name == "regular":
        kwargs["suffix"] = suffix
    strategy = get_ingestion_strategy(strategy_name, **kwargs)
    result = await strategy.execute(
        kb_id=_TEST_KB_ID,
        content=content,
        file_name=file_name,
        creator="test",
    )
    assert result["status"] in ("queued", "duplicate"), f"入队失败: {result}"
    document_id = result.get("document_id")
    logger.info("✅ 入队成功: file=%s status=%s doc_id=%s", file_name, result["status"], document_id)

    # Step 2: 从队列中取出 UnifiedContext
    ctx: UnifiedContext = await ctx_queue.get()
    assert ctx.doc_name == file_name
    assert ctx.kb_id == _TEST_KB_ID
    logger.info("✅ 出队成功: ctx=%s source=%s", ctx.context_id, ctx.source)

    # Step 3: IndexingExecutor 执行
    executor = IndexingExecutor()
    postproc_task = await executor.run(ctx)
    assert postproc_task.chunk_count > 0
    logger.info(
        "✅ 索引完成: doc=%s chunks=%d",
        file_name, postproc_task.chunk_count,
    )

    ctx_queue.task_done()

    return {
        "ctx": ctx,
        "task": postproc_task,
        "document_id": document_id or ctx.ext_metadata.get("document_id"),
    }


async def verify_faq_data(doc_id: str, kb_id: str) -> None:
    """验证 FAQ 数据是否正确入库。"""
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import _engine

    async with _engine.begin() as conn:
        # 验证 document 状态
        doc = (
            await conn.execute(
                sql_text(
                    "SELECT id, doc_name, suffix, status FROM doc_knowledge_documents WHERE id = :id"
                ),
                {"id": doc_id},
            )
        ).one_or_none()
        assert doc is not None, f"文档 {doc_id} 不存在"
        assert doc.status == "completed", f"状态应为 completed，实际为 {doc.status}"
        logger.info("  文档状态: %s", doc.status)

        # 验证 FAQ chunk 数据
        chunks = (
            await conn.execute(
                sql_text(
                    "SELECT seq_num, content, metadata, chunk_type "
                    "FROM chunk_text WHERE document_id = :id ORDER BY seq_num"
                ),
                {"id": doc_id},
            )
        ).all()

        assert len(chunks) == 4, f"预期 4 个 FAQ chunk，实际 {len(chunks)}"
        for i, c in enumerate(chunks):
            meta = json.loads(c.metadata) if isinstance(c.metadata, str) else c.metadata
            assert c.chunk_type == "faq", f"chunk_type 应为 faq，实际为 {c.chunk_type}"
            logger.info("  FAQ[%d]: %s", i, meta.get("standard_question", str(meta)[:60]))

        logger.info("✅ FAQ 数据验证通过: %d chunks", len(chunks))


async def verify_markdown_data(doc_id: str, kb_id: str) -> None:
    """验证 Markdown 文档数据是否正确入库。"""
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import _engine

    async with _engine.begin() as conn:
        # 验证 document 状态
        doc = (
            await conn.execute(
                sql_text(
                    "SELECT id, doc_name, suffix, status FROM doc_knowledge_documents WHERE id = :id"
                ),
                {"id": doc_id},
            )
        ).one_or_none()
        assert doc is not None, f"文档 {doc_id} 不存在"
        assert doc.status == "completed", f"状态应为 completed，实际为 {doc.status}"
        logger.info("  文档状态: %s", doc.status)

        # 验证 chunk 数据
        chunks = (
            await conn.execute(
                sql_text(
                    "SELECT seq_num, content, chunk_type FROM chunk_text "
                    "WHERE document_id = :id ORDER BY seq_num"
                ),
                {"id": doc_id},
            )
        ).all()

        assert len(chunks) > 0, "应有 chunk 数据"
        logger.info("  Markdown chunks: %d", len(chunks))

        # 验证内容包含原文关键段落
        all_text = " ".join(c.content for c in chunks)
        assert "Aion Knowledge" in all_text, "缺少标题内容"
        assert "多源数据接入" in all_text, "缺少中文内容"
        assert "Ingestion" in all_text, "缺少 Ingestion 引用"
        logger.info("✅ Markdown 数据验证通过: %d chunks", len(chunks))


async def main():
    logger.info("=" * 60)
    logger.info("开始端到端真实测试")
    logger.info("=" * 60)

    # 清理遗留数据
    await cleanup_kb(_TEST_KB_ID)

    # ── 测试 1: FAQ 文档 ─────────────────────────────────────
    logger.info("\n" + "─" * 40)
    logger.info("测试 1: FAQ 文档")
    logger.info("─" * 40)

    faq_content, faq_name = create_faq_csv()
    faq_result = await test_ingest_and_index(
        content=faq_content,
        file_name=faq_name,
        strategy_name="faq",
        suffix="csv",
        source="faq",
        expected_verify_fn=None,
    )
    await verify_faq_data(faq_result["document_id"], _TEST_KB_ID)

    # ── 测试 2: Markdown 文档 ────────────────────────────────
    logger.info("\n" + "─" * 40)
    logger.info("测试 2: Markdown 文档")
    logger.info("─" * 40)

    md_content, md_name = create_markdown_doc()
    md_result = await test_ingest_and_index(
        content=md_content,
        file_name=md_name,
        strategy_name="regular",
        suffix="md",
        source="regular",
        expected_verify_fn=None,
    )
    await verify_markdown_data(md_result["document_id"], _TEST_KB_ID)

    # ── 汇总 ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("🎉 全部测试通过！")
    logger.info("=" * 60)

    # 清理测试数据
    await cleanup_kb(_TEST_KB_ID)
    logger.info("测试数据已清理")


if __name__ == "__main__":
    asyncio.run(main())

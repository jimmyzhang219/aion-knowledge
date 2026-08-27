#!/usr/bin/env python
"""FAQ 模块端到端集成测试 — 真实数据库 + 真实嵌入。

用法：
  cd /Users/jimmy/VSCodeProjects/aion-knowledge
  python scripts/test_faq_e2e.py

前提：
  - PostgreSQL 运行中，aion_knowledge 库存在
  - 所有表已创建
  - Ollama 运行中（bge-m3 模型可用）
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("faq_e2e")

TEST_KB_ID = "00000000-0000-0000-0000-00000000e2e2"


async def ensure_test_kb() -> str:
    """创建测试知识库，返回 kb_id。"""
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import _engine

    async with _engine.begin() as conn:
        row = (await conn.execute(
            sql_text("SELECT id FROM kb_knowledge_bases WHERE id = :id"),
            {"id": TEST_KB_ID},
        )).one_or_none()
        if not row:
            await conn.execute(
                sql_text(
                    "INSERT INTO kb_knowledge_bases (id, name, tags, description) "
                    "VALUES (:id, :name, :tags, :desc)"
                ),
                {
                    "id": TEST_KB_ID,
                    "name": "FAQ-E2E-Test-KB",
                    "tags": ["测试"],
                    "desc": "FAQ 集成测试用知识库",
                },
            )
            logger.info("Test KB created: %s", TEST_KB_ID)
        else:
            logger.info("Test KB already exists: %s", TEST_KB_ID)
    return TEST_KB_ID


async def cleanup_kb() -> None:
    """清理该 KB 下所有与 FAQ 相关的数据。"""
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import _engine

    async with _engine.begin() as conn:
        await conn.execute(
            sql_text("DELETE FROM chunk_vector WHERE kb_id = :kb_id"),
            {"kb_id": TEST_KB_ID},
        )
        await conn.execute(
            sql_text("DELETE FROM chunk_text WHERE kb_id = :kb_id"),
            {"kb_id": TEST_KB_ID},
        )
        await conn.execute(
            sql_text(
                "DELETE FROM task_ingestion_tasks WHERE document_id IN "
                "(SELECT id FROM doc_knowledge_documents WHERE kb_id = :kb_id)"
            ),
            {"kb_id": TEST_KB_ID},
        )
        await conn.execute(
            sql_text("DELETE FROM doc_knowledge_documents WHERE kb_id = :kb_id"),
            {"kb_id": TEST_KB_ID},
        )
    logger.info("Cleaned up KB: %s", TEST_KB_ID)


async def test_faq_import_basic() -> bool:
    """测试 1：基础 FAQ 导入 — 写入 chunk_text + chunk_vector。"""
    logger.info("\n===== 测试 1：基础 FAQ 导入 =====")

    from aion_knowledge.ingestion.faq_import import import_faq_file

    # 准备测试数据：3 条 FAQ 条目
    entries = [
        {
            "standard_question": "如何重置密码？",
            "similar_questions": ["密码忘了怎么办", "忘记密码"],
            "answers": ["进入设置页面点击重置密码", "联系管理员重置"],
            "tags": ["账户安全"],
        },
        {
            "standard_question": "如何修改邮箱？",
            "similar_questions": ["变更邮箱地址"],
            "negative_questions": ["如何删除账号"],
            "answers": ["在个人设置中修改邮箱"],
            "tags": ["账户安全"],
        },
        {
            "standard_question": "如何导出数据？",
            "similar_questions": [],
            "answers": ["在设置页面选择导出"],
        },
    ]
    content = json.dumps(entries).encode("utf-8")

    result = await import_faq_file(
        content=content,
        file_ext="json",
        kb_id=TEST_KB_ID,
        mode="append",
        creator="e2e_test",
    )
    logger.info("Import result: %s", result.model_dump())

    assert result.mode == "append", f"Expected append, got {result.mode}"
    assert result.total == 3, f"Expected 3 total, got {result.total}"
    assert result.inserted == 3, f"Expected 3 inserted, got {result.inserted}"
    assert result.errors == 0, f"Expected 0 errors, got {result.errors}"
    assert result.error_details == [], f"Errors: {result.error_details}"

    logger.info("✅ 测试 1 通过：%d 条 FAQ 导入成功", result.inserted)
    return True


async def test_faq_chunks_written() -> bool:
    """测试 2：验证 FAQ chunks 已写入 chunk_text 表。"""
    logger.info("\n===== 测试 2：验证 chunk_text 写入 =====")

    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import get_session

    async with get_session() as session:
        rows = await session.execute(
            sql_text(
                "SELECT id, content, chunk_type, metadata, kb_id "
                "FROM chunk_text "
                "WHERE kb_id = :kb_id AND chunk_type = 'faq' "
                "ORDER BY id"
            ),
            {"kb_id": TEST_KB_ID},
        )
        chunks = list(rows)

    assert len(chunks) == 3, f"Expected 3 FAQ chunks, got {len(chunks)}"

    for i, row in enumerate(chunks):
        d = dict(row._mapping)
        metadata = d["metadata"]
        logger.info("  Chunk %d: type=%s kb=%s", i + 1, d["chunk_type"], d["kb_id"])
        logger.info("    content: %s", d["content"][:60])
        logger.info("    standard_question: %s", metadata.get("standard_question"))
        logger.info("    similar_questions: %s", metadata.get("similar_questions"))
        assert d["chunk_type"] == "faq"
        assert d["kb_id"] == TEST_KB_ID
        assert "Q:" in d["content"]
        assert "A:" in d["content"]

    logger.info("✅ 测试 2 通过：3 条 FAQ chunks 验证成功")
    return True


async def test_faq_vectors_written() -> bool:
    """测试 3：验证 FAQ 向量已写入 chunk_vector 表。"""
    logger.info("\n===== 测试 3：验证 chunk_vector 写入 =====")

    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import get_session

    async with get_session() as session:
        rows = await session.execute(
            sql_text(
                "SELECT cv.chunk_id, cv.kb_id, cv.embedding IS NOT NULL AS has_emb, "
                "       cv.payload->>'chunk_type' AS payload_type "
                "FROM chunk_vector cv "
                "WHERE cv.kb_id = :kb_id "
                "ORDER BY cv.id"
            ),
            {"kb_id": TEST_KB_ID},
        )
        vectors = list(rows)

    assert len(vectors) == 3, f"Expected 3 vectors, got {len(vectors)}"

    for i, row in enumerate(vectors):
        d = dict(row._mapping)
        logger.info("  Vector %d: chunk_id=%s has_emb=%s payload_type=%s",
                     i + 1, d["chunk_id"], d["has_emb"], d["payload_type"])
        assert d["has_emb"] is True, f"Vector {i + 1} has no embedding!"
        assert d["payload_type"] == "faq"

    logger.info("✅ 测试 3 通过：3 条 FAQ 向量验证成功（均有真实嵌入）")
    return True


async def test_faq_import_replace() -> bool:
    """测试 4：replace 模式 — 先清空再导入。"""
    logger.info("\n===== 测试 4：Replace 模式测试 =====")

    from aion_knowledge.ingestion.faq_import import import_faq_file

    entries = [
        {
            "standard_question": "替换后的新问题？",
            "answers": ["这是替换后的新答案"],
        },
    ]
    content = json.dumps(entries).encode("utf-8")

    result = await import_faq_file(
        content=content,
        file_ext="json",
        kb_id=TEST_KB_ID,
        mode="replace",
        creator="e2e_test",
    )
    logger.info("Replace result: %s", result.model_dump())

    assert result.mode == "replace"
    assert result.inserted == 1, f"Expected 1, got {result.inserted}"

    # 验证旧数据已被清除
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import get_session

    async with get_session() as session:
        count = await session.scalar(
            sql_text("SELECT COUNT(*) FROM chunk_text WHERE kb_id = :kb_id AND chunk_type = 'faq'"),
            {"kb_id": TEST_KB_ID},
        )
    assert count == 1, f"Expected 1 FAQ chunk after replace, got {count}"

    logger.info("✅ 测试 4 通过：replace 模式工作正常")
    return True


async def test_faq_concurrent_guard() -> bool:
    """测试 5：并发导入防护 — RuntimeError。"""
    logger.info("\n===== 测试 5：并发导入防护 =====")

    from aion_knowledge.ingestion.faq_import import _running_imports, import_faq_file

    # 模拟正在导入
    _running_imports.add(TEST_KB_ID)

    try:
        entries = [{"standard_question": "Q", "answers": ["A"]}]
        content = json.dumps(entries).encode("utf-8")

        import pytest
        with pytest.raises(RuntimeError) as exc_info:
            await import_faq_file(content, "json", TEST_KB_ID, "append")
        logger.info("  RuntimeError message: %s", str(exc_info.value))
        assert "正在导入" in str(exc_info.value)
    finally:
        _running_imports.remove(TEST_KB_ID)

    logger.info("✅ 测试 5 通过：并发导入正确拦截")
    return True


async def test_faq_search_postprocessing() -> bool:
    """测试 6：FAQ 检索后处理（分数提升 + 负问过滤）。"""
    logger.info("\n===== 测试 6：FAQ 检索后处理 =====")

    from aion_knowledge.retrieval.search.kb_search import apply_faq_postprocessing

    # 模拟搜索结果
    results = [
        {"chunk_type": "faq", "score": 0.8, "content": "Q: 如何重置密码？\nA: 进入设置页面",
         "metadata": {"negative_questions": ["如何删除账号"]}},
        {"chunk_type": "faq", "score": 0.8, "content": "Q: 如何删除账号？\nA: 联系客服",
         "metadata": {"negative_questions": []}},
        {"chunk_type": "text", "score": 0.75, "content": "普通文档内容", "metadata": {}},
    ]

    # 测试负问过滤
    filtered = apply_faq_postprocessing(results, "如何删除账号")
    assert len(filtered) == 2, f"Expected 2 after negative filter, got {len(filtered)}"
    # 第一条 FAQ 应该被过滤（因为负问匹配）
    assert filtered[0]["chunk_type"] == "faq"  # 第二条 FAQ
    assert filtered[1]["chunk_type"] == "text"

    # 测试分数提升
    boosted = apply_faq_postprocessing(results, "如何重置密码")
    faq_scores = [r["score"] for r in boosted if r["chunk_type"] == "faq"]
    text_scores = [r["score"] for r in boosted if r["chunk_type"] == "text"]
    for s in faq_scores:
        assert s > 0.8, f"FAQ score should be boosted: {s}"
    for s in text_scores:
        assert s == 0.75, f"Non-FAQ score unchanged: {s}"

    logger.info("✅ 测试 6 通过：FAQ 分数提升 + 负问过滤正确")
    return True


async def test_faq_direct_answer() -> bool:
    """测试 7：FAQ 直接答案通道。"""
    logger.info("\n===== 测试 7：FAQ 直接答案通道 =====")

    from aion_knowledge.retrieval.context.merger import merge_context

    # 高分数 → 直接答案
    results_high = [
        {"chunk_type": "faq", "score": 0.9,
         "content": "Q: 如何重置密码？\nA: 进入设置页面",
         "metadata": {"standard_question": "如何重置密码？"}},
    ]
    ctx = merge_context(results_high, "如何重置密码")
    faq_direct = [c for c in ctx if c.get("type") == "faq_direct"]
    assert len(faq_direct) == 1, f"Expected 1 faq_direct, got {len(faq_direct)}"

    # 低分数 → 普通 chunk
    results_low = [
        {"chunk_type": "faq", "score": 0.5,
         "content": "Q: test\nA: answer", "metadata": {}},
    ]
    ctx = merge_context(results_low, "test")
    faq_direct = [c for c in ctx if c.get("type") == "faq_direct"]
    assert len(faq_direct) == 0, "Low score FAQ should not be direct answer"

    # 非 FAQ → 不受影响
    results_normal = [
        {"chunk_type": "text", "score": 0.95, "content": "normal", "metadata": {}},
    ]
    ctx = merge_context(results_normal, "test")
    assert all(c.get("type") == "chunk" for c in ctx)

    logger.info("✅ 测试 7 通过：FAQ 直接答案通道正确")
    return True


async def test_csv_import() -> bool:
    """测试 8：CSV 导入。"""
    logger.info("\n===== 测试 8：CSV 格式导入 =====")

    from aion_knowledge.ingestion.faq_import import import_faq_file

    csv_content = (
        "分类,问题,相似问题,负问题,答案,答案策略\n"
        "网络,如何连接WiFi？,WiFi连不上##无法连接WiFi,有线网络问题,打开设置连接WiFi,all\n"
        "网络,如何设置VPN？,VPN配置方法,,在设置中添加VPN配置,all\n"
    )

    result = await import_faq_file(
        content=csv_content.encode("utf-8"),
        file_ext="csv",
        kb_id=TEST_KB_ID,
        mode="replace",
        creator="e2e_test",
    )
    logger.info("CSV import result: %s", result.model_dump())
    assert result.inserted == 2, f"Expected 2, got {result.inserted}"

    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import get_session
    async with get_session() as session:
        rows = await session.execute(
            sql_text("SELECT content FROM chunk_text WHERE kb_id = :kb_id AND chunk_type = 'faq' ORDER BY id"),
            {"kb_id": TEST_KB_ID},
        )
        contents = [r[0] for r in rows]
    logger.info("  CSV imported contents:")
    for c in contents:
        logger.info("    - %s", c[:60])

    assert any("如何连接WiFi" in c for c in contents), "WiFi question not found"
    assert any("如何设置VPN" in c for c in contents), "VPN question not found"

    logger.info("✅ 测试 8 通过：CSV 导入正确")
    return True


async def main() -> None:
    logger.info("=" * 60)
    logger.info("FAQ 模块端到端真实测试")
    logger.info("=" * 60)

    await cleanup_kb()
    await ensure_test_kb()

    tests = [
        test_faq_import_basic,
        test_faq_chunks_written,
        test_faq_vectors_written,
        test_faq_import_replace,
        test_faq_concurrent_guard,
        test_faq_search_postprocessing,
        test_faq_direct_answer,
        test_csv_import,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
            logger.info("")
        except Exception as e:
            failed += 1
            logger.error("❌ %s 失败: %s", test.__name__, e, exc_info=True)
            logger.info("")

    logger.info("=" * 60)
    logger.info("测试完成：%d passed, %d failed", passed, failed)
    logger.info("=" * 60)

    # 清理
    await cleanup_kb()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

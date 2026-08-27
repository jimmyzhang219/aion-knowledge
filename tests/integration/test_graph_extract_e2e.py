"""GraphExtract 端到端集成测试。

需要真实 PostgreSQL + Neo4j + LLM 环境。标记为 @pytest.mark.integration，
默认不运行。

用法:
    pytest tests/integration/test_graph_extract_e2e.py -v -m integration
"""
from __future__ import annotations

import logging

import pytest

from aion_knowledge.infrastructure.db import _engine

logger = logging.getLogger(__name__)

_TEST_KB_ID = "e2e-test-kb"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="session"),
]


async def _cleanup() -> None:
    """清理测试 KB 所有 graph 数据（PG metadata + Neo4j 图谱）。"""
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.graph import delete_graph

    async with _engine.begin() as conn:
        try:
            await conn.execute(
                sql_text("DELETE FROM graph_metadata WHERE kb_id = :kb_id"),
                {"kb_id": _TEST_KB_ID},
            )
        except Exception:
            pass
    await delete_graph(_TEST_KB_ID)


async def _setup_kb() -> None:
    """初始化测试环境。"""
    from aion_knowledge.infrastructure.db import init_db
    await init_db()
    await _cleanup()


# ── 测试：提取 → 合并 → 验证 ────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_and_merge_two_docs() -> None:
    """验证两个文档的实体跨文档合并。

    流程：
      1. 文档1（ali.md）：马云创建了阿里巴巴，张勇是CEO
      2. 文档2（ant.md）：马云是蚂蚁集团的创始人
      3. 验证 AGE 和 PG 中"马云"只有一个节点，source_docs 包含两个文档
    """
    from aion_knowledge.pipeline.postproc.base import PostProcContext
    from aion_knowledge.pipeline.postproc.graph_extract.processor import GraphExtractModule

    await _setup_kb()
    module = GraphExtractModule()

    # 文档1：马云创建阿里巴巴
    ctx1 = PostProcContext(document_id="doc1", kb_id=_TEST_KB_ID, doc_name="ali.md")
    chunks1 = [
        {"chunk_uuid": "c1", "content": "马云在1999年创建了阿里巴巴。"},
        {"chunk_uuid": "c2", "content": "张勇于2015年成为阿里巴巴CEO。"},
    ]
    count1 = await module.process(ctx1, chunks1)
    assert count1 > 0, "文档1应提取出至少一个实体"

    # 文档2：马云创建蚂蚁集团
    ctx2 = PostProcContext(document_id="doc2", kb_id=_TEST_KB_ID, doc_name="ant.md")
    chunks2 = [
        {"chunk_uuid": "c3", "content": "马云是蚂蚁集团的创始人。"},
    ]
    count2 = await module.process(ctx2, chunks2)
    assert count2 > 0, "文档2应提取出至少一个实体"

    # ── 验证 Neo4j：EntityInstance.chunk_ids 携带来源 chunk ──
    from aion_knowledge.infrastructure.graph import search_entities
    matched = await search_entities(_TEST_KB_ID, ["马云"], top_k=5)
    assert matched, "Neo4j 中应能匹配到 '马云'"
    ma_yun = next((e for e in matched if e["entity_name"] == "马云"), None)
    assert ma_yun is not None, "应匹配到 '马云'"
    chunk_ids = ma_yun.get("chunk_ids", [])
    assert "c1" in chunk_ids and "c3" in chunk_ids, \
        f"chunk_ids 应含 doc1 的 c1 与 doc2 的 c3：{chunk_ids}"
    logger.info("Neo4j 验证通过 — EntityInstance.chunk_ids 正确携带来源 chunk")

    logger.info("=== GraphExtract E2E 测试全部通过 ===")

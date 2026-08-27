"""社区向量化检索 E2E：真实 DB 验证 生成摘要向量 → 向量检索命中。

运行: cd /Users/jimmy/VSCodeProjects/aion-knowledge && python -m pytest tests/integration/test_community_e2e.py -v -s
依赖: embedder + LLM 配置可用（由 .env 指定 provider）；postproc 配置启用 community 模块。
"""
from __future__ import annotations

import uuid as uuid_mod

import pytest
from sqlalchemy import text as sql_text

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.infrastructure.db import _engine, get_session
from aion_knowledge.infrastructure.embedder import create_embedder
from aion_knowledge.models.enums import ChunkStrategy, DocumentStatus
from aion_knowledge.models.orm import KnowledgeDocument
from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.community.processor import CommunityModule
from aion_knowledge.pipeline.postproc.text.orm import ChunkText
from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.search.community_retriever import CommunityRetriever

_TEST_KB_ID = "00000000-0000-0000-0000-00000000e2e4"

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _ensure_test_kb() -> None:
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
                {"id": _TEST_KB_ID, "name": "COMMUNITY-E2E-Test-KB",
                 "tags": ["测试"], "desc": "社区向量化 e2e 测试知识库"},
            )


async def _insert_doc_with_chunk() -> str:
    doc_id = uuid7()
    async with get_session() as session:
        session.add(KnowledgeDocument(
            id=doc_id, kb_id=uuid_mod.UUID(_TEST_KB_ID),
            doc_name="community-e2e.md", suffix="md",
            hash="e2e-hash-community-" + str(doc_id)[:8],
            size=10, status=DocumentStatus.completed, creator="e2e",
            chunk_strategy=ChunkStrategy.auto,
        ))
        session.add(ChunkText(
            id=uuid7(), kb_id=uuid_mod.UUID(_TEST_KB_ID), document_id=doc_id,
            seq_num=1, chunk_type="text",
            content="沉浸式VR大空间技术栈包含头显定位与渲染引擎，由甲公司和乙公司联合开发。",
        ))
        await session.commit()
    return str(doc_id)


async def _cleanup() -> None:
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text("DELETE FROM chunk_community WHERE kb_id = :kb_id"),
            {"kb_id": _TEST_KB_ID},
        )
        await conn.execute(
            sql_text(
                "DELETE FROM chunk_text WHERE document_id IN "
                "(SELECT id FROM doc_knowledge_documents WHERE kb_id = :kb_id)"
            ),
            {"kb_id": _TEST_KB_ID},
        )
        await conn.execute(
            sql_text("DELETE FROM doc_knowledge_documents WHERE kb_id = :kb_id"),
            {"kb_id": _TEST_KB_ID},
        )


async def test_community_vector_e2e():
    """无图回退路径：社区模块生成摘要+向量 → 检索命中 KB 级社区摘要。"""
    # 先清残留再建数据（上次运行中途失败时，残留会污染 rows 断言）
    await _cleanup()
    await _ensure_test_kb()
    try:
        doc_id = await _insert_doc_with_chunk()

        # 1. 无图（回退路径）跑社区模块
        module = CommunityModule()
        ctx = PostProcContext(document_id=doc_id, kb_id=_TEST_KB_ID, doc_name="community-e2e.md")
        # 注意：chunk_uuid 须为字符串（workers.py 传 str(r.id)，processor 内 uuid.UUID(...) 再解析）
        chunks = [{"chunk_uuid": str(uuid7()), "content": "沉浸式VR大空间技术栈包含头显定位与渲染引擎。"}]
        count = await module.process(ctx, chunks)
        assert count >= 1, "应至少生成 1 个社区"

        # 2. 落库带向量
        async with get_session() as session:
            rows = (await session.execute(
                sql_text("SELECT id, community_id, embedding FROM chunk_community WHERE kb_id = :kb_id"),
                {"kb_id": _TEST_KB_ID},
            )).fetchall()
        assert rows, "chunk_community 应有记录"
        assert all(r.embedding is not None for r in rows), "社区记录应生成摘要向量"

        # 3. 向量检索命中
        provider = create_embedder()
        query_emb = (await provider.embed_documents(["沉浸式VR技术栈"]))[0]

        retriever = CommunityRetriever()
        results = await retriever.retrieve(RetrieverContext(
            query="沉浸式VR技术栈是什么", kb_id=_TEST_KB_ID, query_embedding=query_emb,
        ))
        assert results, "向量检索应命中社区"
        assert results[0].metadata["community_id"] == rows[0].community_id
        assert results[0].content, "content 应为社区摘要文本"
    finally:
        await _cleanup()

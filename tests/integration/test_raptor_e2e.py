"""RAPTOR E2E 验证：全链路测试 RAPTOR 写入 chunk_raptor。

运行: cd /Users/jimmy/VSCodeProjects/aion-knowledge && python -m pytest tests/integration/test_raptor_e2e.py -v -s
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from aion_knowledge.infrastructure.db import _engine, get_session

logger = logging.getLogger(__name__)

_TEST_KB_ID = "00000000-0000-0000-0000-00000000e2e2"

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
                    "name": "RAPTOR-E2E-Test-KB",
                    "tags": ["技术", "文档"],
                    "desc": "RAPTOR 迁移测试用知识库",
                },
            )
            logger.info("Test KB created: %s", _TEST_KB_ID)


async def _cleanup() -> None:
    from sqlalchemy import text as sql_text
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "DELETE FROM chunk_raptor WHERE kb_id = :kb_id"
            ),
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
            sql_text(
                "DELETE FROM chunk_vector WHERE kb_id = :kb_id"
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


def _make_long_topic(topic: str, size: int = 15) -> str:
    """生成多段、多角度的长文本，便于 RAPTOR 做有意义的聚类。"""
    aspects = {
        "人工智能": [
            "人工智能正在深刻改变各行各业的生产方式。",
            "机器学习算法通过大量数据训练，能够自动发现规律和模式。",
            "深度学习利用多层神经网络处理复杂任务如图像识别。",
            "自然语言处理技术让计算机理解人类语言成为可能。",
            "强化学习在游戏和机器人控制领域取得了突破性进展。",
            "大语言模型展示了令人惊叹的文本生成和理解能力。",
            "AI 伦理和可解释性是当前研究的热点方向。",
            "联邦学习能保护数据隐私的同时训练高质量模型。",
            "多模态 AI 融合视觉、语言和语音信息进行综合判断。",
            "边缘 AI 将计算能力带到终端设备，降低延迟。",
            "AI 辅助编程工具正在提升开发者的工作效率。",
            "计算机视觉在自动驾驶中用于识别道路和障碍物。",
            "知识图谱为 AI 系统提供结构化的事实知识。",
            "对抗训练增强了 AI 模型的鲁棒性和安全性。",
            "AI 芯片专门化是提升深度学习推理性能的关键。",
        ],
        "数据库技术": [
            "关系型数据库通过 ACID 事务保证数据一致性。",
            "NoSQL 数据库提供了灵活的数据模型和高可扩展性。",
            "分布式数据库将数据分散到多个节点以支持海量存储。",
            "向量数据库专门处理高维向量数据，支持相似度搜索。",
            "pgvector 扩展让 PostgreSQL 具备了向量存储和检索能力。",
            "索引是数据库查询性能优化最重要的手段之一。",
            "事务隔离级别需要根据应用场景做合理权衡。",
            "数据库分片策略决定了系统的横向扩展能力。",
            "基于日志的复制保证了数据库主从节点数据一致。",
            "OLAP 和分析型数据库针对复杂查询进行了优化。",
            "内存数据库大大缩短了数据访问的延迟。",
            "数据仓库是组织大型分析查询的核心基础设施。",
            "数据库备份和恢复策略是运维的基础保障。",
            "查询优化器通过代价模型选择最高效的执行计划。",
            "NewSQL 数据库试图结合传统数据库的 ACID 和 NoSQL 的可扩展性。",
        ],
        "软件工程方法论": [
            "敏捷开发强调迭代交付和快速响应需求变化。",
            "持续集成要求开发者频繁将代码合并到主干。",
            "持续交付确保软件随时可以发布到生产环境。",
            "代码审查可以有效发现缺陷并提升团队编码水平。",
            "测试驱动开发要求先写测试再写实现代码。",
            "领域驱动设计关注核心业务模型的建模和演进。",
            "微服务架构将应用拆分为独立部署的小型服务。",
            "设计模式提供了解决常见设计问题的可复用方案。",
            "重构是在不改变外部行为的前提下改善内部结构。",
            "DevOps 强调开发与运维的协作和自动化。",
            "监控和可观测性是保障系统稳定运行的基础。",
            "API 设计的一致性直接影响开发效率和用户体验。",
            "Git 分支策略决定了团队协作的工作流程。",
            "软件架构文档有助于新成员理解系统设计决策。",
            "性能压测可以发现系统在高负载下的瓶颈和风险。",
        ],
    }
    parts = []
    for i in range(size):
        parts.append(aspects.get(topic, ["内容段落"])[i % 15])
    return "\n\n".join(parts)


@pytest.mark.asyncio
async def test_raptor_e2e() -> None:
    """全链路测试：上传 → 解析 → 分块 → TextModule → RAPTOR → 验证 chunk_raptor。"""
    await _cleanup()
    await _ensure_test_kb()

    # ── 1. 生成多主题测试内容（确保 RAPTOR 能聚类）──
    md_content = (
        f"# 技术文档：AI 与数据库\n\n"
        f"## 第一部分\n\n{_make_long_topic('人工智能')}\n\n"
        f"## 第二部分\n\n{_make_long_topic('数据库技术')}\n\n"
        f"## 第三部分\n\n{_make_long_topic('软件工程方法论')}\n\n"
    ).encode("utf-8")

    # ── 2. 上传文件 ──
    from aion_knowledge.ingestion.strategy.registry import get_strategy
    from aion_knowledge.models.enums import StrategyName

    strategy = get_strategy(StrategyName.regular, suffix="md")
    result = await strategy.execute(
        kb_id=_TEST_KB_ID,
        content=md_content,
        file_name="raptor_e2e_test.md",
    )
    assert result["status"] == "queued", f"enqueue failed: {result}"
    doc_id = result["document_id"]
    logger.info("Document created: %s", doc_id)

    # ── 3. 执行 IndexingExecutor（解析 + 分块 + TextModule + VectorModule）──
    from aion_knowledge.infrastructure.models import PostProcTask
    from aion_knowledge.infrastructure.queues import ctx_queue

    ctx = await ctx_queue.get()
    assert ctx.doc_name == "raptor_e2e_test.md"

    from aion_knowledge.indexing.executor import IndexingExecutor

    executor = IndexingExecutor()
    postproc_task = await executor.run(ctx)
    assert isinstance(postproc_task, PostProcTask)
    assert postproc_task.chunk_count > 0
    logger.info("Pipeline done: chunk_count=%d", postproc_task.chunk_count)
    ctx_queue.task_done()

    # ── 4. 加载 chunks 并执行 RAPTOR ═══
    from sqlalchemy import text as sql_text

    from aion_knowledge.pipeline.postproc.base import PostProcContext
    from aion_knowledge.pipeline.postproc.dispatcher import PostProcDispatcher

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

    logger.info("Loaded %d chunks from chunk_text, running RAPTOR...", len(chunks))
    assert len(chunks) >= 3, f"Too few chunks: {len(chunks)}"

    # 启用 RAPTOR
    settings_dict = {
        "raptor": True,
    }
    dispatcher = PostProcDispatcher(settings_dict)
    proc_ctx = PostProcContext(
        document_id=doc_id,
        kb_id=_TEST_KB_ID,
        doc_name="raptor_e2e_test.md",
        file_type="md",
        parser_id="naive",
    )

    # 确保 RAPTOR 模块配置好
    from aion_knowledge.common.config import settings as global_settings
    global_settings.postproc_raptor = True

    await dispatcher.run_second_batch(proc_ctx, chunks)

    # ── 5. 验证 chunk_raptor 数据 ═══
    async with get_session() as session:
        rows = await session.execute(
            sql_text(
                "SELECT id, title, summary, layer, cluster_id, "
                "       source_chunk_ids, tree_builder, clustering_method, output_mode "
                "FROM chunk_raptor "
                "WHERE kb_id = :kb_id ORDER BY layer, id"
            ),
            {"kb_id": _TEST_KB_ID},
        )
        records = list(rows)

    logger.info("=== RAPTOR 验证结果 ===")
    logger.info("chunk_raptor 记录数: %d", len(records))
    assert len(records) >= 1, "No RAPTOR records found!"

    for row in records:
        d = dict(row._mapping)
        src_count = len(d.get("source_chunk_ids", []))

        # 检查 embedding 是否被赋值（chunk_raptor 行应该已有）
        emb_row = (await asyncio.get_event_loop().run_in_executor(None, lambda: None)) or None
        async with get_session() as s:
            emb_res = await s.execute(
                sql_text("SELECT embedding IS NOT NULL AS has_emb FROM chunk_raptor WHERE id = :rid"),
                {"rid": d["id"]},
            )
            emb_row = emb_res.one_or_none()

        has_emb = emb_row[0] if emb_row else False

        logger.info(
            "  id=%-3d layer=%-2d title=%-20s builder=%-8s method=%-5s "
            "source=%d emb=%s summary_preview=%-30s",
            d["id"],
            d["layer"],
            d["title"][:18],
            d["tree_builder"],
            d["clustering_method"],
            src_count,
            "✅" if has_emb else "❌",
            d.get("summary", "")[:28],
        )

    # ── 6. 验证数据质量 ──
    assert len(records) >= 1, "RAPTOR 没有生成任何摘要"

    # 验证至少有一条记录有 source_chunk_ids
    has_provenance = any(
        len(dict(r._mapping).get("source_chunk_ids", [])) > 0 for r in records
    )
    assert has_provenance, "没有记录有 source_chunk_ids（溯源性丢失）"

    # 验证 tree_builder 正确
    for r in records:
        d = dict(r._mapping)
        assert d["tree_builder"] == "raptor"
        assert d["clustering_method"] in ("gmm", "ahc")
        assert d["output_mode"] == "flat"

    # ── 7. 验证文档状态 ──
    async with _engine.begin() as conn:
        row = (await conn.execute(
            sql_text("SELECT status FROM doc_knowledge_documents WHERE id = :id"),
            {"id": doc_id},
        )).one()
        assert row.status == "completed", f"doc status: {row.status}"

    logger.info("=== RAPTOR E2E 测试全部通过 ===")

    # 展示数据
    from aion_knowledge.infrastructure.db import _engine as eng
    async with eng.begin() as conn:
        rows = await conn.execute(
            sql_text(
                "SELECT id, LEFT(title, 40) AS title, layer, cluster_id, "
                "       source_chunk_ids, tree_builder, clustering_method, "
                "       output_mode, LENGTH(summary) AS summary_len, "
                "       embedding IS NOT NULL AS has_emb "
                "FROM chunk_raptor WHERE kb_id = :kb_id ORDER BY layer, id"
            ),
            {"kb_id": _TEST_KB_ID},
        )
        for r in rows:
            logger.info("  id=%-3s title=%-40s layer=%-2s cluster=%-8s src=%s builder=%-6s method=%-5s mode=%-5s sum_len=%-4s emb=%s",
                        r[0], r[1][:38], r[2], r[3], len(r[4]), r[5], r[6], r[7], r[8], "✅" if r[9] else "❌")

    # 如需清理取消下行注释：
    # await _cleanup()

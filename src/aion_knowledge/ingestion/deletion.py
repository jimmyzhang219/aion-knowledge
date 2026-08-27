"""删除编排：逻辑删除（置 deleted 标记）+ 真实删除（异步 purge 数据）。

删除流程（文档级 / KB 级共用）：
1. 校验存在 → 不存在返回 False（API 层转 404）
2. 置 deleted=true 提交 —— 检索屏蔽的提交点
3. AION_DELETION_LOGICAL=true（默认）→ 到此为止，仅逻辑删除
4. false → asyncio.create_task 异步执行真实删除，请求立即返回
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from sqlalchemy import text as sql_text

from aion_knowledge.common.config import settings
from aion_knowledge.infrastructure.db import get_session

logger = logging.getLogger(__name__)


def _log_purge_result(task: asyncio.Future[Any]) -> None:
    """purge 后台任务兜底回调：成功/异常都记日志（防任务异常静默丢失）。"""
    try:
        task.result()
    except Exception:
        logger.exception("purge 后台任务异常")
    else:
        logger.info("purge 后台任务完成")


async def delete_document(kb_id: str, document_id: str) -> bool:
    """逻辑删除文档。返回 False 表示文档不存在（或不属于该 KB）。"""
    async with get_session() as session:
        row = await session.execute(
            sql_text("SELECT id FROM doc_knowledge_documents WHERE id = :doc AND kb_id = :kb"),
            {"doc": uuid.UUID(document_id), "kb": uuid.UUID(kb_id)},
        )
        if row.first() is None:
            return False
        await session.execute(
            sql_text("UPDATE doc_knowledge_documents SET deleted = true WHERE id = :doc"),
            {"doc": uuid.UUID(document_id)},
        )
        await session.commit()
    if not settings.deletion_logical:
        task = asyncio.create_task(_purge_document(kb_id, document_id))
        task.add_done_callback(_log_purge_result)
    logger.info(
        "文档删除（logical=%s）：doc=%s kb=%s", settings.deletion_logical, document_id, kb_id
    )
    return True


async def delete_kb(kb_id: str) -> bool:
    """逻辑删除整个 KB（KB + KB 下所有文档）。返回 False 表示 KB 不存在。"""
    async with get_session() as session:
        row = await session.execute(
            sql_text("SELECT id FROM kb_knowledge_bases WHERE id = :kb"),
            {"kb": uuid.UUID(kb_id)},
        )
        if row.first() is None:
            return False
        await session.execute(
            sql_text("UPDATE kb_knowledge_bases SET deleted = true WHERE id = :kb"),
            {"kb": uuid.UUID(kb_id)},
        )
        await session.execute(
            sql_text("UPDATE doc_knowledge_documents SET deleted = true WHERE kb_id = :kb"),
            {"kb": uuid.UUID(kb_id)},
        )
        await session.commit()
    if not settings.deletion_logical:
        task = asyncio.create_task(_purge_kb(kb_id))
        task.add_done_callback(_log_purge_result)
    logger.info("KB 删除（logical=%s）：kb=%s", settings.deletion_logical, kb_id)
    return True


async def _purge_document(kb_id: str, document_id: str) -> None:
    """真实删除文档全部数据（幂等，不碰 KB 级数据）。

    PG chunk 系单事务（1-5 步）→ Neo4j → 对象存储 → checkpoint → 主行物理删（收尾，单独事务）。
    主行最后删：外部步骤失败时主行仍在（deleted 已屏蔽检索），可经 API 重删重试；
    各步失败均记日志不抛异常（幂等，重复执行安全）。
    """
    doc_uuid = uuid.UUID(document_id)
    kb_uuid = uuid.UUID(kb_id)
    file_path = ""
    try:
        async with get_session() as session:
            # 0. 先取 file_path（主行删除后无法再查）
            row = await session.execute(
                sql_text("SELECT file_path FROM doc_knowledge_documents WHERE id = :doc"),
                {"doc": doc_uuid},
            )
            r = row.first()
            if r is not None:
                file_path = r[0]
            # 1. chunk_vector（先删，chunk_id 关联该文档的 chunk_text）
            await session.execute(
                sql_text("""
                    DELETE FROM chunk_vector WHERE chunk_id IN (
                        SELECT id FROM chunk_text WHERE document_id = :doc)
                """),
                {"doc": doc_uuid},
            )
            # 2. chunk_disambiguation 文档级条目（chunk_id 非 NULL；须在 chunk_text 删除前按子查询匹配）
            await session.execute(
                sql_text("""
                    DELETE FROM chunk_disambiguation WHERE chunk_id IN (
                        SELECT id FROM chunk_text WHERE document_id = :doc)
                """),
                {"doc": doc_uuid},
            )
            # 3. chunk_text（含 FAQ）
            await session.execute(
                sql_text("DELETE FROM chunk_text WHERE document_id = :doc AND kb_id = :kb"),
                {"doc": doc_uuid, "kb": kb_uuid},
            )
            # 4. chunk_raptor 文档级树（doc_id 为 NULL 的 KB 级树不碰）
            await session.execute(
                sql_text("DELETE FROM chunk_raptor WHERE kb_id = :kb AND doc_id = :doc"),
                {"doc": doc_uuid, "kb": kb_uuid},
            )
            # 5. task_ingestion_tasks
            await session.execute(
                sql_text("DELETE FROM task_ingestion_tasks WHERE document_id = :doc"),
                {"doc": doc_uuid},
            )
            await session.commit()
    except Exception:
        logger.exception(
            "文档 purge PG 事务失败（deleted 已屏蔽检索，可重删重试）：doc=%s", document_id
        )
        return

    # 6. Neo4j 文档图谱（独立事务，幂等）
    try:
        from aion_knowledge.infrastructure.graph import delete_document_graph

        await delete_document_graph(kb_id, document_id)
    except Exception:
        logger.exception("Neo4j 删除文档图谱失败（幂等可重试）：doc=%s", document_id)

    # 7. 对象存储原文件
    if file_path:
        try:
            from aion_knowledge.infrastructure.storage import resolve_storage

            await resolve_storage().delete(file_path)
        except Exception:
            logger.exception("对象存储删除失败：%s", file_path)

    # 8. checkpoint 文件
    try:
        from aion_knowledge.indexing.checkpoint import CheckpointManager

        await CheckpointManager().delete(document_id)
    except Exception:
        logger.exception("checkpoint 删除失败：doc=%s", document_id)

    # 9. 主行物理删（收尾，单独事务；失败记日志——deleted 已屏蔽检索，可重删重试）
    try:
        async with get_session() as session:
            await session.execute(
                sql_text("DELETE FROM doc_knowledge_documents WHERE id = :doc AND kb_id = :kb"),
                {"doc": doc_uuid, "kb": kb_uuid},
            )
            await session.commit()
    except Exception:
        logger.exception(
            "文档主行删除失败（deleted 已屏蔽检索，可重删重试）：doc=%s", document_id
        )


async def _purge_kb(kb_id: str) -> None:
    """真实删除整个 KB：全部文档级数据 + KB 级数据（community/wiki/raptor/disambiguation/graph_metadata/Neo4j/KB 行）。

    PG 失败（文档列表查询或 KB 级事务）记日志后直接返回：deleted 已屏蔽检索，可经 API 重删重试；
    单文档 purge 失败记日志继续，不影响其余文档与 KB 级清理；Neo4j 删除独立 try/except；
    KB 主行最后删（单独事务）：Neo4j 失败时主行仍在，可重删重试。
    """
    kb_uuid = uuid.UUID(kb_id)
    try:
        async with get_session() as session:
            rows = await session.execute(
                sql_text("SELECT id FROM doc_knowledge_documents WHERE kb_id = :kb"),
                {"kb": kb_uuid},
            )
            doc_ids = [str(r[0]) for r in rows.fetchall()]
        for doc_id in doc_ids:
            try:
                await _purge_document(kb_id, doc_id)
            except Exception:
                logger.exception("文档 purge 失败（继续其余文档与 KB 级清理）：doc=%s", doc_id)
                continue

        # KB 级数据（不含 KB 主行；主行收尾最后删）
        async with get_session() as session:
            await session.execute(sql_text("DELETE FROM chunk_community WHERE kb_id = :kb"), {"kb": kb_uuid})
            await session.execute(sql_text("DELETE FROM chunk_wiki WHERE kb_id = :kb"), {"kb": kb_uuid})
            await session.execute(sql_text("DELETE FROM chunk_raptor WHERE kb_id = :kb"), {"kb": kb_uuid})
            await session.execute(sql_text("DELETE FROM chunk_disambiguation WHERE kb_id = :kb"), {"kb": kb_uuid})
            await session.execute(sql_text("DELETE FROM graph_metadata WHERE kb_id = :kb"), {"kb": kb_uuid})
            await session.commit()
    except Exception:
        logger.exception(
            "KB purge PG 失败（deleted 已屏蔽检索，可重删重试）：kb=%s", kb_id
        )
        return
    try:
        from aion_knowledge.infrastructure.graph import delete_graph

        await delete_graph(kb_id)
    except Exception:
        logger.exception("Neo4j 删除 KB 图谱失败（幂等可重试）：kb=%s", kb_id)

    # KB 主行物理删（收尾，单独事务；失败记日志——deleted 已屏蔽检索，可重删重试）
    try:
        async with get_session() as session:
            await session.execute(
                sql_text("DELETE FROM kb_knowledge_bases WHERE id = :kb"),
                {"kb": kb_uuid},
            )
            await session.commit()
    except Exception:
        logger.exception(
            "KB 主行删除失败（deleted 已屏蔽检索，可重删重试）：kb=%s", kb_id
        )

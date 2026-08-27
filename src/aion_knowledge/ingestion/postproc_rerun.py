"""后处理重跑入口 — 对已入库文档单独执行指定二批模块。

API 路由薄壳调用本模块；校验与入队编排均在此完成，
处理执行完全复用 postproc_worker 的 _run_postproc_subtasks。
"""

from __future__ import annotations

import logging
import uuid

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.infrastructure.models import PostProcConfig, PostProcTask
from aion_knowledge.infrastructure.queues import postproc_queue
from aion_knowledge.ingestion.document_repo import get_document_by_id
from aion_knowledge.storage.relational.chunk_repo import ChunkRepository

logger = logging.getLogger(__name__)


class DocumentNotFoundError(Exception):
    """文档不存在、不属于指定知识库，或 id 格式非法。"""


class ModuleValidationError(Exception):
    """模块白名单校验失败。"""


def validate_modules(modules: list[str]) -> None:
    """校验模块白名单：非空、合法、非首批、已启用（.env 门控 × 出厂硬控）。

    校验逻辑与 _run_postproc_subtasks 的 settings_dict 计算同式，
    指定的模块任一门控未开即抛 ModuleValidationError，避免"queued 后静默不跑"。
    """
    from aion_knowledge.pipeline.postproc.dispatcher import (
        PostProcDispatcher,
        enabled_module_names,
    )

    if not modules:
        raise ModuleValidationError("modules 不能为空，至少指定一个二批模块")

    dispatcher = PostProcDispatcher({})
    known = set(dispatcher._modules)
    unknown = sorted(set(modules) - known)
    if unknown:
        raise ModuleValidationError(f"未知模块: {unknown}，合法模块: {sorted(known)}")

    always_on = {n for n, m in dispatcher._modules.items() if m.always_on}
    forbidden = sorted(set(modules) & always_on)
    if forbidden:
        raise ModuleValidationError(f"首批模块不可重跑（由上传流程执行）: {forbidden}")

    disabled = sorted(set(modules) - set(enabled_module_names(dispatcher)))
    if disabled:
        raise ModuleValidationError(f"模块未启用（.env 门控或出厂硬控未开）: {disabled}")


async def enqueue_postproc_rerun(kb_id: str | uuid.UUID, document_id: str, modules: list[str]) -> PostProcTask:
    """校验文档并构造重跑任务入队 postproc_queue，返回任务（调用方入队即返回）。"""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError as exc:
        raise DocumentNotFoundError(f"invalid document id: {document_id}") from exc

    # kb_id 兼容 str（任意大小写）与 uuid.UUID 对象，统一为 UUID 做值比较
    if isinstance(kb_id, uuid.UUID):
        kb_uuid = kb_id
        kb_str = str(kb_id)
    else:
        kb_str = kb_id
        try:
            kb_uuid = uuid.UUID(kb_id)
        except ValueError as exc:
            raise DocumentNotFoundError(f"invalid kb id: {kb_id}") from exc

    # 白名单先校验：模块名写错时快速失败，不打 DB 查询
    validate_modules(modules)

    async with get_session() as session:
        doc = await get_document_by_id(session, doc_uuid)
        if doc is None or doc.kb_id != kb_uuid:
            raise DocumentNotFoundError(
                f"Document {document_id} not found in kb {kb_id}"
            )
        chunk_count = await ChunkRepository(session).count_by_document(document_id)

    task = PostProcTask(
        document_id=document_id,
        kb_id=kb_str,
        doc_name=doc.doc_name,
        suffix=doc.suffix,
        chunk_count=chunk_count,
        modules=modules,
        # 出厂硬控由 validate_modules 已预检通过；任务携带出厂配置，
        # worker 侧 AND settings.postproc_* 与 _run_postproc_subtasks 同式
        postproc_config=PostProcConfig(),
    )
    await postproc_queue.put(task)
    logger.info("后处理重跑已入队：文档=%s，模块=%s", document_id, modules)
    return task

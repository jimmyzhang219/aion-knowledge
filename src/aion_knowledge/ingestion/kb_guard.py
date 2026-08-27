"""知识库存在性校验守卫 — 程序化维护 kb 关系（表无 FK 约束）。

所有写入入口（上传/FAQ/API 直投）通过 IngestionStrategy 基类调用本守卫，
确保只有存在的知识库才能写入文档。
"""

from __future__ import annotations

import uuid

from aion_knowledge.storage.relational.kb_repo import KnowledgeBaseRepo


class KnowledgeBaseNotFoundError(Exception):
    """知识库不存在或 kb_id 非法。"""


async def ensure_kb_exists(kb_id: str) -> None:
    """校验知识库存在，不存在/非法则抛 KnowledgeBaseNotFoundError。"""
    try:
        kb_uuid = uuid.UUID(kb_id)
    except ValueError:
        raise KnowledgeBaseNotFoundError(f"invalid kb id: {kb_id}") from None
    kb = await KnowledgeBaseRepo().get_by_id(kb_uuid)
    if kb is None:
        raise KnowledgeBaseNotFoundError(f"KnowledgeBase {kb_uuid} not found")

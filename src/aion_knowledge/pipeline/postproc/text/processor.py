"""TextModule — 文本 chunk 落库与 parent 结构生成。

作用：
  将 parser 产出的分块结果（chunks）持久化到 chunk_text 表，
  同时为后续模块（vector、summarizer 等）提供 chunk_uuid。
  始终启用（always_on = True）。

父块(parent)机制：
  - 每 2 个 text 类型子块合并为 1 个父块（seq_num 递增，排在子块之后）
  - 非 text 子块（table、image 等）不参与合并
  - 父块记录 parent_of 字段指向其包含的子块 ID
  - 父块同样写入 chunks 列表供 vector 等下游模块处理
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, ClassVar

from sqlalchemy import text

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.models.enums import ChunkType
from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule
from aion_knowledge.pipeline.postproc.text.orm import ChunkText

logger = logging.getLogger(__name__)


class TextModule(PostProcModule):
    """明文 chunk 模块。始终启用，固定生成 parent 结构。"""

    always_on = True
    depends_on: ClassVar[list[str]] = []

    async def process(
        self,
        ctx: PostProcContext,
        chunks: list[dict[str, Any]],
    ) -> int:
        """写入 chunk_text 表，返回总行数（子块 + 父块）。"""
        if not chunks:
            return 0

        from aion_knowledge.infrastructure.db import get_session
        from aion_knowledge.pipeline.cleaner import clean_passage

        doc_uuid = uuid.UUID(ctx.document_id) if ctx.document_id else None

        child_rows = []
        for c in chunks:
            chunk_type = c.get("chunk_type", ChunkType.text.value)
            chunk_metadata: dict[str, Any] = {
                "token_count": c.get("token_count", 0),
                "chunk_id": c.get("chunk_id", ""),
                "heading_path": c.get("heading_path", ""),
            }
            # 合并自定义 metadata（如 FAQ 的 standard_question、answers 等）
            # 先写入自定义字段，再写入默认字段确保固定字段不会被覆盖
            if c.get("metadata"):
                chunk_metadata = {**c["metadata"], **chunk_metadata}
            # 表格 caption 存入 metadata
            if c.get("table_caption"):
                chunk_metadata["table_caption"] = c["table_caption"]
            if c.get("context_above"):
                chunk_metadata["context_above"] = c["context_above"]
            if c.get("context_below"):
                chunk_metadata["context_below"] = c["context_below"]

            # 将 image_url 写入 image_refs + chunk_metadata，避免静默丢失
            image_refs: list[str] = list(c.get("image_refs", []))
            if c.get("image_url"):
                image_refs.append(c["image_url"])
                chunk_metadata["image_url"] = c["image_url"]

            child_rows.append(ChunkText(
                document_id=doc_uuid,
                kb_id=uuid.UUID(ctx.kb_id),
                content=clean_passage(c.get("content", "")),
                context_header=c.get("heading_path", ""),
                seq_num=c.get("seq_num", 0),
                chunk_type=chunk_type,
                token_count=c.get("token_count", 0),
                chunk_metadata=chunk_metadata,
                image_refs=image_refs,
            ))

        async with get_session() as session:
            session.add_all(child_rows)
            await session.flush()

            # 固定 parent 结构：每 2 个 text 子块合并为一个父块（非 text 块跳过）
            parent_rows: list[ChunkText] = []
            scale = 2
            text_indices = [i for i, r in enumerate(child_rows) if r.chunk_type == ChunkType.text.value]
            for idx, i in enumerate(range(0, len(text_indices), scale)):
                idx_group = text_indices[i:i + scale]
                group = [child_rows[j] for j in idx_group]
                parent_content = "".join(c.content for c in group)
                parent_tokens = sum(c.token_count for c in group)
                parent_id = uuid7()

                for child in group:
                    child.parent_chunk_id = parent_id

                parent_rows.append(ChunkText(
                    id=parent_id,
                    document_id=doc_uuid,
                    kb_id=uuid.UUID(ctx.kb_id),
                    content=parent_content,
                    context_header=chunks[idx_group[0]].get("heading_path", ""),
                    seq_num=len(chunks) + idx,
                    chunk_type=ChunkType.parent.value,
                    token_count=parent_tokens,
                    chunk_metadata={
                        "parent_of": [str(r.id) for r in group],
                        "token_count": parent_tokens,
                    },
                ))

            session.add_all(parent_rows)
            await session.flush()

            # 批量更新 content_tokens（由 PG zhparser 从 content 计算）
            chunk_ids = [str(row.id) for row in child_rows] + [str(row.id) for row in parent_rows]
            if chunk_ids:
                await session.execute(
                    text("""
                        UPDATE chunk_text
                        SET content_tokens = tsvector_to_array(
                                to_tsvector('zh_cfg', COALESCE(content, '')))
                        WHERE id = ANY(:ids)
                    """),
                    {"ids": chunk_ids},
                )

            # 回写 chunk_uuid 到 chunks 列表，供后续模块（如 vector）使用
            for orig, row in zip(chunks, child_rows):
                orig["chunk_uuid"] = str(row.id)

            # 追加 parent 到 chunks 列表，供后续模块（如 vector）使用
            for parent_row in parent_rows:
                chunks.append({
                    "content": parent_row.content,
                    "chunk_uuid": str(parent_row.id),
                    "seq_num": parent_row.seq_num,
                    "chunk_type": ChunkType.parent.value,
                    "token_count": parent_row.token_count,
                    "heading_path": parent_row.context_header or "",
                })

        total = len(child_rows) + len(parent_rows)
        logger.info("【Text】写入完成：%d 子块 + %d 父块 = %d，文档=%s",
                     len(child_rows), len(parent_rows), total, ctx.doc_name)
        return total


def module() -> TextModule:
    """模块工厂函数，供调度器自动发现。"""
    return TextModule()

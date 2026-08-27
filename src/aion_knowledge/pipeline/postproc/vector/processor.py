"""VectorModule — 向量嵌入生成与落库。

作用：
  对已经 text 模块处理后的 chunks 计算向量嵌入，写入 chunk_vector 表。
  始终启用（always_on = True），依赖 text 模块先执行。

机制：
  - 跳过空内容 chunk（避免 embedding provider 对空字符串挂死）
  - 从 settings.embedding_provider（ollama / openai）选择嵌入服务
  - 每行向量记录携带 chunk_type、seq_num、doc_name 等 payload
  - 单模块失败不影响文档处理的整体流程
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.infrastructure.embedder import create_embedder
from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule

logger = logging.getLogger(__name__)



class VectorModule(PostProcModule):
    """向量嵌入模块。始终启用，依赖 text 模块先执行拿到 chunk_uuid。"""

    always_on = True
    depends_on: ClassVar[list[str]] = ["text", "vlm_caption"]

    async def process(
        self,
        ctx: PostProcContext,
        chunks: list[dict[str, Any]],
    ) -> int:
        """计算嵌入并写入 chunk_vector 表，返回写入记录数。"""
        if not chunks:
            return 0

        # 过滤空内容 chunk（如 VLM 尚未处理的图片），避免 Ollama 对空字符串挂死
        valid: list[tuple[int, dict[str, Any]]] = [
            (i, c) for i, c in enumerate(chunks) if c.get("content", "").strip()
        ]
        if not valid:
            logger.info("【Vector】无可处理的有内容 chunk，跳过")
            return 0

        texts = [c["content"] for _, c in valid]
        indices = [i for i, _ in valid]

        try:
            provider = create_embedder()
            embeddings = await provider.embed_documents(texts)
        except Exception as exc:
            logger.error("【Vector】向量化失败：文档=%s，错误=%s", ctx.doc_name, exc)
            return 0

        if len(embeddings) != len(texts):
            logger.error("【Vector】数量不匹配：%d 个向量 vs %d 个内容",
                         len(embeddings), len(texts))
            return 0

        from aion_knowledge.infrastructure.db import get_session
        from aion_knowledge.storage.relational.vector_repo import VectorRepository

        async with get_session() as session:
            repo = VectorRepository(session)
            for idx, emb in zip(indices, embeddings):
                chunk = chunks[idx]
                await repo.insert(
                    id=uuid7(),
                    chunk_id=chunk.get("chunk_uuid", chunk.get("chunk_id", "")),
                    kb_id=ctx.kb_id,
                    embedding=emb,
                    payload={
                        "chunk_type": chunk.get("chunk_type", "text"),
                        "seq_num": chunk.get("seq_num", 0),
                        "doc_name": ctx.doc_name,
                    },
                )

        logger.info("【Vector】向量写入完成：%d 条，文档=%s", len(embeddings), ctx.doc_name)
        return len(embeddings)


def module() -> VectorModule:
    """模块工厂函数，供调度器自动发现。"""
    return VectorModule()

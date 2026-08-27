"""SummarizerModule — 逐 chunk 摘要生成与落库。

作用：
  为每个 chunk 生成 50-200 字的简洁摘要，同时写入三个存储层：
  - chunk_text.summary_text 字段（文本）
  - chunk_text.summary_tokens 字段（zhparser 分词，供 BM25 检索）
  - chunk_vector 表（摘要向量，用于语义检索）

机制：
  - 依赖 text 模块先执行拿到 chunk_uuid 和 content
  - 通过 LLM 逐 chunk 生成摘要（content 截断 4000 字符）
  - 摘要向量化逻辑与 VectorModule 共享同一 provider 配置
  - 向量化失败不回滚，只记录 warning
  - 写入 summary_text 后同步更新 summary_tokens（供 BM25 索引使用）
  - 可选启用，通过配置控制
"""
from __future__ import annotations

import logging
from typing import Any

from aion_knowledge.common.config import settings
from aion_knowledge.common.model_registry import get_model_max_input_tokens, truncate_by_tokens
from aion_knowledge.infrastructure.embedder import create_embedder
from aion_knowledge.infrastructure.llm import LLMClient, get_llm_client_for_module
from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = "为以下文本生成一段简洁的摘要（50-200字）：\n\n{content}"


class SummarizerModule(PostProcModule):
    """逐 chunk 摘要生成模块。二批执行，依赖 text + vector。"""

    always_on = False
    depends_on = ["text", "vector"]

    async def process(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> int:
        """为每个 chunk 生成摘要并写入三个存储层，返回生成数。"""
        if not chunks:
            return 0

        llm = get_llm_client_for_module("summarizer")
        embedder = create_embedder()
        from aion_knowledge.infrastructure.db import get_session
        from aion_knowledge.storage.relational.chunk_repo import ChunkRepository
        from aion_knowledge.storage.relational.vector_repo import VectorRepository

        updated = 0
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            chunk_uuid = chunk.get("chunk_uuid", "")
            if not content or not chunk_uuid:
                continue
            if not self._check_content(content):
                logger.warning("【摘要】内容过少，跳过 chunk=%s", chunk_uuid)
                continue

            summary = await self._summarize_chunk(llm, content)
            if not summary:
                continue

            # 计算摘要向量
            embedding_summary = None
            try:
                emb_list = await embedder.embed_documents([summary])
                if emb_list:
                    embedding_summary = emb_list[0]
            except Exception as exc:
                logger.warning("【摘要】向量化失败：%s", exc)

            async with get_session() as session:
                chunk_repo = ChunkRepository(session)
                vec_repo = VectorRepository(session)
                await chunk_repo.update_summary(chunk_id=chunk_uuid, summary_text=summary)
                # 同步更新 summary_tokens（由 PG zhparser 从 summary_text 计算）
                await chunk_repo.update_summary_tokens(chunk_id=chunk_uuid)
                if embedding_summary:
                    await vec_repo.update_summary_embedding(
                        chunk_id=chunk_uuid, embedding_summary=embedding_summary
                    )
            updated += 1

        logger.info("【摘要】处理完成：%d 个摘要，文档=%s", updated, ctx.doc_name)
        return updated

    async def _summarize_chunk(self, llm: LLMClient, content: str) -> str:
        """调用 LLM 生成单 chunk 摘要，失败返回空串。"""
        try:
            _max_tokens = get_model_max_input_tokens(settings.llm_model, ratio=0.05)
            return await llm.generate(SUMMARY_PROMPT.format(
                content=truncate_by_tokens(content, _max_tokens) if content else "",
            ))
        except Exception as exc:
            logger.warning("【摘要】生成失败：%s", exc)
            return ""


def module() -> SummarizerModule:
    """模块工厂函数，供调度器自动发现。"""
    return SummarizerModule()

"""QuestionGenModule — 逐 chunk 问题生成与向量化。

作用：
  基于 chunk 内容生成 topN 个用户可能提出的问题，同时写入：
  - chunk_vector 表的 questions 字段（逗号拼接的文本）
  - 问题的向量嵌入（用于语义检索匹配）

机制：
  - 调用 LLM 生成问题，每行一条，自动去除序号前缀
  - 生成的问题通过逗号拼接为单字符串后向量化
  - 向量化失败不阻塞主流程（warning 记录）
  - 可选启用，依赖 text 模块先行
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

QUESTION_PROMPT = "基于以下文本内容，生成{topn}个用户可能会问的问题。每个问题一行，不要序号。\n\n{content}"


class QuestionGenModule(PostProcModule):
    """逐 chunk 问题生成模块。二批执行，依赖 text + vector。"""

    always_on = False
    depends_on = ["text", "vector"]

    async def process(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> int:
        """为每个 chunk 生成问题并写入 chunk_vector，返回更新的 chunk 数。"""
        if not chunks:
            return 0

        llm = get_llm_client_for_module("question_gen")
        embedder = create_embedder()
        from aion_knowledge.infrastructure.db import get_session
        from aion_knowledge.storage.relational.vector_repo import VectorRepository

        updated = 0
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            chunk_uuid = chunk.get("chunk_uuid", "")
            if not content or not chunk_uuid:
                continue
            if not self._check_content(content):
                logger.warning("【问题生成】内容过少，跳过 chunk=%s", chunk_uuid)
                continue

            questions = await self._generate_questions(llm, content)
            if not questions:
                continue

            # 逗号拼接为单字符串
            questions_text = "，".join(questions)

            # 计算问题向量
            embedding_questions = None
            try:
                emb_list = await embedder.embed_documents([questions_text])
                if emb_list:
                    embedding_questions = emb_list[0]
            except Exception as exc:
                logger.warning("【问题生成】向量化失败：%s", exc)

            async with get_session() as session:
                repo = VectorRepository(session)
                count = await repo.update_questions(
                    chunk_id=chunk_uuid, questions=questions_text, embedding_questions=embedding_questions,
                )
                if count:
                    updated += 1

        logger.info("【问题生成】处理完成：%d 个 chunk，文档=%s", updated, ctx.doc_name)
        return updated

    async def _generate_questions(self, llm: LLMClient, content: str, topn: int = 5) -> list[str]:
        """调用 LLM 生成 topn 个问题并清洗行格式。"""
        try:
            _max_tokens = get_model_max_input_tokens(settings.llm_model, ratio=0.03)
            raw = await llm.generate(QUESTION_PROMPT.format(
                topn=topn, content=truncate_by_tokens(content, _max_tokens) if content else "",
            ))
            lines = [ln.strip().lstrip("1234567890.、. ") for ln in raw.strip().split("\n") if ln.strip()]
            return [ln for ln in lines if len(ln) > 5][:topn]
        except Exception as exc:
            logger.warning("【问题生成】生成失败：%s", exc)
            return []


def module() -> QuestionGenModule:
    """模块工厂函数，供调度器自动发现。"""
    return QuestionGenModule()

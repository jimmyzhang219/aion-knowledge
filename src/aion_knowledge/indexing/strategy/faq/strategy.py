"""FAQChunkingStrategy — FAQ 文档的索引构建策略。

完全重写基类 execute()，不走默认的 解析→清洗→切分 流水线：
FAQ 数据源本身已是结构化 Q/A 条目，无需 Markdown 转换或语义切分。
简化流水线：下载原始文件（CSV/JSON）→ 解析 → 每条目 = 1 chunk。
"""

from __future__ import annotations

import logging
from typing import Any

from aion_knowledge.common.model_registry import count_tokens
from aion_knowledge.indexing.strategy.base import ChunkingStrategy
from aion_knowledge.indexing.strategy.faq.parser import ParseError, parse_faq_file
from aion_knowledge.indexing.strategy.faq.schemas import FAQChunkMetadata, FAQEntry
from aion_knowledge.indexing.strategy.registry import register_strategy
from aion_knowledge.infrastructure.models import UnifiedContext
from aion_knowledge.models.enums import ChunkType, StrategyName

logger = logging.getLogger(__name__)


@register_strategy(StrategyName.faq)
class FAQChunkingStrategy(ChunkingStrategy):
    """FAQ 文档的索引构建策略。

    完全重写 execute()，绕过基类默认流水线：FAQ 数据源本身结构化，
    直接按条目生成 chunk，无需 Markdown 转换、清洗或章节切分。
    """

    strategy_key = "faq"

    async def execute(
        self,
        ctx: UnifiedContext,
    ) -> list[dict[str, Any]]:
        """下载原始 FAQ 文件 → 解析为条目 → 每条目生成 1 个 chunk。

        Args:
            ctx: 统一上下文，suffix 决定按 CSV 还是 JSON 解析。

        Returns:
            chunk dict 列表；解析失败或无有效条目时返回空列表（不抛异常）。
        """
        from aion_knowledge.pipeline.downloader import Downloader

        # 1. 从存储下载原始文件
        local_path = await Downloader().download(ctx.original_file_ref, ctx.doc_name)

        # 2. 读取并解析
        with open(local_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        try:
            # 从 ctx.suffix 获取实际文件后缀
            file_ext = ctx.suffix
            entries = parse_faq_file(raw_content.encode("utf-8"), file_ext)
        except ParseError:
            logger.warning("FAQChunkingStrategy：文档 %s 解析失败", ctx.doc_name)
            return []

        if not entries:
            logger.warning("FAQChunkingStrategy：文档 %s 无有效条目", ctx.doc_name)
            return []

        # 3. 每条目 = 1 chunk
        return _build_chunk_dicts(entries)


def _build_faq_content(metadata: FAQChunkMetadata) -> str:
    """构建人工可读的 FAQ content 字符串。"""
    lines = [f"Q: {metadata.standard_question}"]
    for a in metadata.answers:
        lines.append(f"A: {a}")
    return "\n".join(lines)


def _build_chunk_dicts(entries: list[FAQEntry]) -> list[dict[str, Any]]:
    """将 FAQ 条目转换为统一的 chunk dict 列表。"""
    chunks: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        chunk_metadata = FAQChunkMetadata(
            standard_question=entry.standard_question,
            similar_questions=entry.similar_questions,
            negative_questions=entry.negative_questions,
            answers=entry.answers,
            answer_strategy=entry.answer_strategy,
        )
        content = _build_faq_content(chunk_metadata)

        chunks.append({
            "content": content,
            "chunk_type": ChunkType.faq.value,
            "token_count": count_tokens(content),
            "seq_num": idx,
            "metadata": chunk_metadata.model_dump(),
        })
    return chunks

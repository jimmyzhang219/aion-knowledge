"""文档索引构建策略包。"""
from aion_knowledge.indexing.strategy.base import ChunkingStrategy
from aion_knowledge.indexing.strategy.faq.strategy import FAQChunkingStrategy
from aion_knowledge.indexing.strategy.registry import get_strategy, register_strategy
from aion_knowledge.indexing.strategy.regular.strategy import RegularChunkingStrategy

__all__ = [
    "ChunkingStrategy",
    "RegularChunkingStrategy",
    "FAQChunkingStrategy",
    "register_strategy",
    "get_strategy",
]

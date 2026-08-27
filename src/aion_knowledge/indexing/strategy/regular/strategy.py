"""RegularChunkingStrategy — 常规文档的索引构建策略。

使用基类 ChunkingStrategy 的默认 5 步流水线：
_download → _parse → _clean → _prepare_chunks → _assemble

如有个性化需求，覆写对应步骤方法即可。
"""

from __future__ import annotations

from aion_knowledge.indexing.strategy.base import ChunkingStrategy
from aion_knowledge.indexing.strategy.registry import register_strategy
from aion_knowledge.models.enums import StrategyName


@register_strategy(StrategyName.regular)
@register_strategy(StrategyName.url_import)
@register_strategy(StrategyName.manual_entry)
class RegularChunkingStrategy(ChunkingStrategy):
    """常规文档的索引构建策略，使用基类默认流水线。"""

    strategy_key = "regular"

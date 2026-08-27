"""文档接入策略包。

每种文档类型实现一个 IngestionStrategy 子类，
管理从原始输入到 UnifiedContext 入队的完整流程。
"""

from aion_knowledge.ingestion.strategy.api_direct.strategy import ApiDirectIngestionStrategy
from aion_knowledge.ingestion.strategy.base import IngestionStrategy
from aion_knowledge.ingestion.strategy.faq.strategy import FAQIngestionStrategy
from aion_knowledge.ingestion.strategy.registry import get_strategy
from aion_knowledge.ingestion.strategy.regular.strategy import RegularIngestionStrategy

__all__ = [
    "ApiDirectIngestionStrategy",
    "IngestionStrategy",
    "RegularIngestionStrategy",
    "FAQIngestionStrategy",
    "get_strategy",
]

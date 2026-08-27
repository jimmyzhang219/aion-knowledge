"""策略注册表 — 按 source 自动发现和实例化 IngestionStrategy。"""

from __future__ import annotations

from typing import Any, Callable

from aion_knowledge.ingestion.strategy.base import IngestionStrategy
from aion_knowledge.models.enums import StrategyName

_registry: dict[str, type[IngestionStrategy]] = {}


def _resolve(name: str | StrategyName) -> str:
    return name.value if isinstance(name, StrategyName) else name


def register_source(
    source: str | StrategyName,
) -> Callable[[type[IngestionStrategy]], type[IngestionStrategy]]:
    """装饰器：将策略类注册到 source。

    Usage:
        @register_source(StrategyName.faq)
        class FAQIngestionStrategy(IngestionStrategy):
            source = "faq"
    """
    key = _resolve(source)

    def _wrap(cls: type[IngestionStrategy]) -> type[IngestionStrategy]:
        _registry[key] = cls
        return cls
    return _wrap


def get_strategy(source: str | StrategyName, **kwargs: Any) -> IngestionStrategy:
    """按 source 获取策略实例。"""
    key = _resolve(source)
    cls = _registry.get(key)
    if cls is None:
        raise ValueError(
            f"未知策略 source: {key}，可用: {list(_registry)}"
        )
    return cls(**kwargs)

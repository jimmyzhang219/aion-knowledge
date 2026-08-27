"""策略注册表 — 按 source 自动发现和实例化 ChunkingStrategy。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aion_knowledge.indexing.strategy.base import ChunkingStrategy
from aion_knowledge.models.enums import StrategyName

_StrategyDecorator = Callable[[type[ChunkingStrategy]], type[ChunkingStrategy]]

_registry: dict[str, type[ChunkingStrategy]] = {}


def _resolve(name: str | StrategyName) -> str:
    return name.value if isinstance(name, StrategyName) else name


def register_strategy(source: str | StrategyName) -> _StrategyDecorator:
    """装饰器：将策略类注册到 source。

    Usage:
        @register_strategy(StrategyName.faq)
        class FAQChunkingStrategy(ChunkingStrategy):
            strategy_key = "faq"
    """
    key = _resolve(source)

    def _wrap(cls: type[ChunkingStrategy]) -> type[ChunkingStrategy]:
        _registry[key] = cls
        return cls

    return _wrap


def get_strategy(source: str | StrategyName, **kwargs: Any) -> ChunkingStrategy:
    """按 source 获取策略实例。"""
    key = _resolve(source)
    cls = _registry.get(key)
    if cls is None:
        raise ValueError(
            f"未知索引策略 source: {key}，可用: {list(_registry)}"
        )
    return cls(**kwargs)

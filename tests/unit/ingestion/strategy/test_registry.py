"""StrategyRegistry 测试。"""

from __future__ import annotations

import pytest

from aion_knowledge.ingestion.strategy.base import IngestionStrategy


class _TestStrategy(IngestionStrategy):
    source = "test_source"
    suffix = "test_doc"

    async def _build_context(self, *args, **kwargs):
        return None


class TestRegisterSource:
    def test_register_and_get(self):
        from aion_knowledge.ingestion.strategy.registry import (
            _registry,
            get_strategy,
            register_source,
        )

        _registry.clear()
        register_source("test_source")(_TestStrategy)
        strategy = get_strategy("test_source")
        assert isinstance(strategy, _TestStrategy)
        assert strategy.source == "test_source"

    def test_get_strategy_passes_kwargs(self):
        from aion_knowledge.ingestion.strategy.registry import (
            _registry,
            get_strategy,
            register_source,
        )

        _registry.clear()
        register_source("test_source")(_TestStrategy)
        strategy = get_strategy("test_source")
        assert isinstance(strategy, _TestStrategy)

    def test_get_strategy_unknown(self):
        from aion_knowledge.ingestion.strategy.registry import _registry, get_strategy

        _registry.clear()
        with pytest.raises(ValueError, match="未知策略 source"):
            get_strategy("nonexistent")

    def test_multiple_sources_same_class(self):
        from aion_knowledge.ingestion.strategy.registry import (
            _registry,
            get_strategy,
            register_source,
        )

        _registry.clear()

        @register_source("a")
        @register_source("b")
        class MultiStrategy(IngestionStrategy):
            source = "multi"
            suffix = "multi"
            async def _build_context(self, *args, **kwargs):
                return None

        assert isinstance(get_strategy("a"), MultiStrategy)
        assert isinstance(get_strategy("b"), MultiStrategy)

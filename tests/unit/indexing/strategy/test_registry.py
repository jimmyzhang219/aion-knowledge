"""register_strategy / get_strategy 注册表测试。"""

import pytest

from aion_knowledge.indexing.strategy.base import ChunkingStrategy
from aion_knowledge.indexing.strategy.registry import (
    _registry,
    get_strategy,
    register_strategy,
)


class TestRegistry:
    _saved_registry: dict = {}

    def setup_method(self):
        self._saved_registry = _registry.copy()
        _registry.clear()

    def teardown_method(self):
        _registry.clear()
        _registry.update(self._saved_registry)

    def test_register_and_get(self):
        @register_strategy("test_type")
        class TestStrategy(ChunkingStrategy):
            strategy_key = "test_type"

        instance = get_strategy("test_type")
        assert isinstance(instance, TestStrategy)

    def test_get_unknown_source(self):
        with pytest.raises(ValueError, match="未知索引策略"):
            get_strategy("nonexistent")

    def test_multiple_sources(self):
        @register_strategy("type_a")
        class StrategyA(ChunkingStrategy):
            strategy_key = "type_a"

        @register_strategy("type_b")
        class StrategyB(ChunkingStrategy):
            strategy_key = "type_b"

        assert isinstance(get_strategy("type_a"), StrategyA)
        assert isinstance(get_strategy("type_b"), StrategyB)

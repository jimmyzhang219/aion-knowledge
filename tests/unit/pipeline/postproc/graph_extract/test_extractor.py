"""extractor.py — 共享实体/关系提取测试。

真实环境 bug 回归：LLM 输出 shape 不可信（generate_structured 走 json_mode，
不强制 schema），gleaning 轮可能把 relations 返回为字符串列表，导致
'r' 取 .get() 时崩溃（'str' object has no attribute 'get'），
整个 graph_extract 模块失败。
"""
from __future__ import annotations

import pytest

from aion_knowledge.pipeline.postproc.graph_extract.extractor import (
    extract_entities_with_gleaning,
)


class FakeLLM:
    """模拟 generate_structured：按序返回预设响应。"""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)

    async def generate_structured(self, prompt: str, output_schema: dict, **kwargs) -> dict:
        return self._responses.pop(0) if self._responses else {}


@pytest.mark.asyncio
async def test_glean_relations_as_str_should_not_crash():
    """回归：gleaning 轮 relations 返回字符串列表时不得崩溃，合法数据应保留。"""
    llm = FakeLLM([
        {
            "entities": [{"name": "A", "type": "X", "description": ""}],
            "relations": [{"source": "A", "target": "B", "type": "rel", "weight": 8}],
        },
        {
            "entities": [],
            "relations": ["B -> C"],  # LLM 非确定性：字符串而非对象
        },
    ])
    entities, relations = await extract_entities_with_gleaning(llm, "文本", ["X"], max_gleanings=2)
    assert entities == [{"name": "A", "type": "X", "description": ""}]
    assert relations == [{"source": "A", "target": "B", "type": "rel", "weight": 8}]


@pytest.mark.asyncio
async def test_top_level_list_should_not_crash():
    """回归：LLM 顶层输出为 JSON 数组（generate_structured 透传 list）时不得崩溃。

    真实环境错误：'list' object has no attribute 'get'（result.get 在 list 上调用）。
    """
    llm = FakeLLM([
        [{"name": "A", "type": "X", "description": ""}],  # 顶层数组而非对象
    ])
    entities, relations = await extract_entities_with_gleaning(llm, "文本", ["X"], max_gleanings=0)
    assert entities == []
    assert relations == []


@pytest.mark.asyncio
async def test_glean_top_level_list_should_not_crash():
    """回归：gleaning 轮 LLM 顶层输出为数组时不得崩溃，返回首轮已提取的合法数据。"""
    llm = FakeLLM([
        {
            "entities": [{"name": "A", "type": "X", "description": ""}],
            "relations": [{"source": "A", "target": "B", "type": "rel", "weight": 8}],
        },
        [{"name": "B", "type": "Y", "description": ""}],  # glean 轮顶层数组
    ])
    entities, relations = await extract_entities_with_gleaning(llm, "文本", ["X"], max_gleanings=2)
    assert entities == [{"name": "A", "type": "X", "description": ""}]
    assert relations == [{"source": "A", "target": "B", "type": "rel", "weight": 8}]


@pytest.mark.asyncio
async def test_initial_round_malformed_should_not_crash():
    """回归：首轮 entities/relations 混入字符串时不得崩溃，合法 dict 条目应保留。"""
    llm = FakeLLM([
        {
            "entities": ["实体A", {"name": "B", "type": "X", "description": ""}],
            "relations": ["A -> B", {"source": "B", "target": "C", "type": "rel"}],
        },
    ])
    entities, relations = await extract_entities_with_gleaning(llm, "文本", ["X"], max_gleanings=0)
    assert entities == [{"name": "B", "type": "X", "description": ""}]
    assert relations == [{"source": "B", "target": "C", "type": "rel"}]

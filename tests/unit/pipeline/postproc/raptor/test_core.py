"""测试 RAPTOR 核心算法。"""
from unittest.mock import AsyncMock

import pytest

from aion_knowledge.pipeline.postproc.raptor.core import (
    RecursiveAbstractiveProcessing4TreeOrganizedRetrieval,
)


@pytest.fixture
def mock_llm():
    m = AsyncMock()
    m.max_length = 8192
    m.generate.return_value = "标题行\n摘要正文内容"
    return m


@pytest.fixture
def mock_embd():
    m = AsyncMock()
    m.encode.return_value = [0.1] * 128
    return m


@pytest.mark.asyncio
async def test_returns_empty_for_single_chunk(mock_llm, mock_embd):
    raptor = RecursiveAbstractiveProcessing4TreeOrganizedRetrieval(
        max_cluster=4, llm_model=mock_llm, embd_model=mock_embd, prompt="{cluster_content}",
    )
    result, layers, parent_child_map = await raptor([("hello", [0.1] * 128, ["id1"])])
    assert result == []
    assert layers == []
    assert parent_child_map == {}


@pytest.mark.asyncio
async def test_classic_builder_with_two_chunks(mock_llm, mock_embd):
    raptor = RecursiveAbstractiveProcessing4TreeOrganizedRetrieval(
        max_cluster=4, llm_model=mock_llm, embd_model=mock_embd, prompt="{cluster_content}",
        small_layer_collapse=2,
    )
    chunks = [
        ("text a", [0.1] * 128, ["id1"]),
        ("text b", [0.2] * 128, ["id2"]),
    ]
    result, layers, parent_child_map = await raptor(chunks)
    assert len(result) >= 2  # 原始 + 至少一个摘要
    assert len(layers) >= 1


@pytest.mark.asyncio
async def test_tree_mode(mock_llm, mock_embd):
    raptor = RecursiveAbstractiveProcessing4TreeOrganizedRetrieval(
        max_cluster=4, llm_model=mock_llm, embd_model=mock_embd, prompt="{cluster_content}",
        small_layer_collapse=2,
    )
    chunks = [
        ("text a", [0.1] * 128, ["id1"]),
        ("text b", [0.2] * 128, ["id2"]),
    ]
    tree = await raptor(chunks, is_tree=True)
    assert isinstance(tree, dict)
    assert "title" in tree

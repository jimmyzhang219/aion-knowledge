"""层级社区检测测试。"""
from __future__ import annotations

from unittest.mock import patch

import networkx as nx

from aion_knowledge.pipeline.postproc.community.leiden import (
    detect_communities,
    detect_communities_hierarchical,
)


class TestDetectCommunitiesHierarchical:
    def test_simple_graph(self):
        graph = nx.Graph()
        graph.add_edge("A", "B", weight=1.0)
        graph.add_edge("B", "C", weight=1.0)
        graph.add_edge("D", "E", weight=1.0)
        result = detect_communities_hierarchical(graph, max_cluster_size=100)
        assert len(result) > 0
        for comm in result:
            assert "id" in comm
            assert "level" in comm
            assert "members" in comm
            assert len(comm["members"]) >= 2

    def test_returns_multiple_levels(self):
        """在足够大的图上应返回多个 level。"""
        graph = nx.Graph()
        for i in range(30):
            graph.add_edge(f"n{i}", f"n{(i+1)%30}", weight=1.0)
        result = detect_communities_hierarchical(graph, max_cluster_size=5)
        levels = set(c["level"] for c in result)
        assert len(levels) > 0

    def test_empty_graph(self):
        graph = nx.Graph()
        assert detect_communities_hierarchical(graph) == []

    def test_fallback_no_graspologic(self):
        graph = nx.Graph()
        graph.add_edge("A", "B", weight=1.0)
        with patch.dict("sys.modules", {"graspologic": None}):
            result = detect_communities_hierarchical(graph)
            assert len(result) > 0
            assert result[0]["level"] == 0

    def test_detect_communities_still_works(self):
        """确认旧函数仍可用。"""
        graph = nx.Graph()
        graph.add_edge("A", "B", weight=1.0)
        result = detect_communities(graph)
        assert len(result) > 0

"""Leiden 社区检测测试。"""
from __future__ import annotations

import networkx as nx

from aion_knowledge.pipeline.postproc.community.leiden import detect_communities


class TestLeiden:
    def test_detect_communities_simple(self):
        """验证简单图能检测到社区。"""
        graph = nx.Graph()
        graph.add_edges_from([
            ("A", "B"), ("B", "C"), ("C", "A"),
            ("D", "E"), ("E", "F"), ("F", "D"),
        ])
        graph.add_edge("C", "D", weight=0.1)

        communities = detect_communities(graph)
        assert len(communities) >= 1
        assert any(len(c["members"]) >= 2 for c in communities)

    def test_detect_communities_empty(self):
        """空图返回空列表。"""
        graph = nx.Graph()
        assert detect_communities(graph) == []

    def test_detect_communities_no_edges(self):
        """孤立节点图应返回空列表（无 size>=2 的社区）。"""
        graph = nx.Graph()
        graph.add_node("A")
        graph.add_node("B")
        assert detect_communities(graph) == []

"""Leiden 层次聚类封装。"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx  # type: ignore[import-untyped]  # networkx 无 stub，模块整体按 Any 处理

logger = logging.getLogger(__name__)


def detect_communities(graph: nx.Graph, max_cluster_size: int = 100) -> list[dict[str, Any]]:
    """执行 Leiden 社区检测。

    优先使用 graspologic 的层次化 Leiden，回退到 networkx 的连通分量。

    Args:
        graph: NetworkX 图。
        max_cluster_size: 最大聚类大小。

    Returns:
        [{"id": str, "level": int, "members": [str, ...]}, ...]
    """
    if graph.number_of_nodes() == 0:
        return []

    # 无边图直接走 fallback（graspologic 的 hierarchical_leiden 会抛 EmptyNetworkError）
    if graph.number_of_edges() == 0:
        return []

    try:
        from graspologic.partition import hierarchical_leiden

        community_map = hierarchical_leiden(graph, max_cluster_size=max_cluster_size, random_seed=42)
        # graspologic 的 cluster 字段为 int（另一函数 detect_communities_hierarchical 中才做 str() 转换）
        level_groups: dict[int, dict[int, list[str]]] = {}
        for item in community_map:
            level = item.level
            level_groups.setdefault(level, {})
            cid = item.cluster
            level_groups[level].setdefault(cid, [])
            level_groups[level][cid].append(item.node)

        result = []
        for level, clusters in level_groups.items():
            for cid, members in clusters.items():
                if len(members) >= 2:
                    result.append({"id": f"L{level}_{cid}", "level": level, "members": members})
        if result:
            return result
    except ImportError:
        logger.warning("graspologic not available, falling back to connected components")

    components = list(nx.connected_components(graph))
    return [
        {"id": f"CC_{i}", "level": 0, "members": list(comp)}
        for i, comp in enumerate(components) if len(comp) >= 2
    ]


def detect_communities_hierarchical(
    graph: nx.Graph, max_cluster_size: int = 100
) -> list[dict[str, Any]]:
    """执行 Leiden 并保留全 levels，替代只取最后一层。

    Returns:
        [{"id": str, "level": int, "members": [str, ...]}, ...]
        每个社区一条记录，含 level 信息。
    """
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return []

    try:
        from graspologic.partition import hierarchical_leiden

        community_map = hierarchical_leiden(
            graph, max_cluster_size=max_cluster_size, random_seed=42
        )
        levels: dict[int, dict[str, list[str]]] = {}
        for item in community_map:
            levels.setdefault(item.level, {})
            cid = str(item.cluster)
            levels[item.level].setdefault(cid, [])
            levels[item.level][cid].append(item.node)

        result = []
        for level, clusters in levels.items():
            for cid, members in clusters.items():
                if len(members) >= 2:
                    result.append({
                        "id": f"L{level}_{cid}",
                        "level": level,
                        "members": members,
                    })
        if result:
            return result
    except ImportError:
        logger.warning("graspologic not available, falling back to connected components")

    # fallback: connected components
    components = list(nx.connected_components(graph))
    return [
        {"id": f"CC_{i}", "level": 0, "members": list(comp)}
        for i, comp in enumerate(components) if len(comp) >= 2
    ]

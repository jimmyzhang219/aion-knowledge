"""测试 RAPTOR 聚类模块。"""
import numpy as np

from aion_knowledge.pipeline.postproc.raptor.clustering import (
    adjust_tree_nodes,
    cluster_ahc,
    cluster_gmm,
    get_optimal_clusters_gmm,
)


def test_gmm_returns_one_for_single_point():
    n, clusters = cluster_gmm(np.random.randn(1, 10), max_cluster=5, threshold=0.1)
    assert n == 1
    assert clusters == [[0]]


def test_gmm_soft_clustering_overlaps():
    """两个分离簇 + 桥接点：桥接点应同时属于两个簇（软聚类）。

    注：簇心取 ±2、尺度 1.6（而非大幅分离），否则 BIC 会给孤立的
    桥接点单独建组件（k=3），任何 threshold > 0 都不产生重叠。
    """
    rng = np.random.RandomState(42)
    left = rng.randn(20, 10) * 1.6 - 2.0  # 簇 A：-2 附近
    right = rng.randn(20, 10) * 1.6 + 2.0  # 簇 B：+2 附近
    bridge = np.zeros((1, 10))            # 桥接点：两簇中间
    emb = np.vstack([left, bridge, right])
    n, clusters = cluster_gmm(emb, max_cluster=5, threshold=0.05)
    # 桥接点在数据里的索引是 20
    containing = [c for c in clusters if 20 in c]
    assert len(containing) >= 2, f"桥接点应属于多个簇，实际 {len(containing)} 个"
    assert all(len(c) > 1 for c in clusters), "不应出现空簇"


def test_ahc_returns_one_for_single_point():
    n, labels = cluster_ahc(np.random.randn(1, 10), max_cluster=5)
    assert n == 1
    assert labels == [0]


def test_gmm_bic_returns_reasonable_k():
    """两个明显分离的簇，BIC 应选择 k≈2。"""
    rng = np.random.RandomState(42)
    centers = [np.ones(10) * -5, np.ones(10) * 5]
    points = np.vstack([c + rng.randn(50, 10) * 0.5 for c in centers])
    k = get_optimal_clusters_gmm(points, max_cluster=10, random_state=42)
    assert 1 <= k <= 5, f"BIC 选的 k={k} 不合理"


def test_adjust_tree_nodes_converges():
    rng = np.random.RandomState(42)
    emb = rng.randn(20, 5)
    labels = rng.randint(0, 3, size=20)
    result = adjust_tree_nodes(emb, labels)
    assert len(result) == 20
    assert np.unique(result).size <= 3

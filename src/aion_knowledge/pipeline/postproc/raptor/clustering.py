"""RAPTOR 聚类方法：GMM 软聚类（BIC，一个点可同时属多簇）+ AHC 硬聚类（Ward 最大 gap）。"""

import logging
from typing import Any, cast

import numpy as np
from sklearn.cluster import (  # type: ignore[import-untyped]  # sklearn 无 stub
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]  # sklearn 无 stub

logger = logging.getLogger(__name__)

_GMM_REG_COVAR = 1e-4  # 防止 GMM 协方差矩阵奇异


def get_optimal_clusters_gmm(embeddings: np.ndarray[Any, Any], max_cluster: int,
                             random_state: int = 0) -> int:
    """用 BIC 选择最优 GMM cluster 数量。"""
    n = len(embeddings)
    max_clusters = min(max_cluster, n)
    if max_clusters <= 1:
        return 1

    n_clusters = np.arange(1, max_clusters + 1)
    bics = []
    for k in n_clusters:
        gm = GaussianMixture(
            n_components=k, random_state=random_state,
            covariance_type="diag", reg_covar=_GMM_REG_COVAR,
        )
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return int(n_clusters[np.argmin(bics)])


def get_clusters_ahc(embeddings: np.ndarray[Any, Any], max_cluster: int) -> np.ndarray[Any, Any]:
    """用 Ward 链接 AHC + 树状图最大 gap 切分。"""
    n = len(embeddings)
    if n <= 1:
        return np.zeros(n, dtype=int)
    if n == 2:
        return np.arange(n)

    full = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0,
        compute_distances=True, linkage="ward",
    )
    full.fit(embeddings)

    distances = full.distances_
    if len(distances) > 1:
        gaps = np.diff(distances)
        max_gap_idx = int(np.argmax(gaps))
        n_clusters = max(1, min(n - max_gap_idx - 1, max_cluster))
    else:
        n_clusters = max(1, min(n, max_cluster))

    if n_clusters <= 1:
        return np.zeros(n, dtype=int)

    clust = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    # sklearn 整体按 Any 处理，fit_predict 返回 Any，cast 到 ndarray 保证返回类型精确
    return cast(np.ndarray[Any, Any], clust.fit_predict(embeddings))


def adjust_tree_nodes(embeddings: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
                      max_iter: int = 5) -> np.ndarray[Any, Any]:
    """Centroid 精调：将 AHC 结果按最近 centroid 重新分配。"""
    labels = labels.copy()
    for _ in range(max_iter):
        unique = np.unique(labels)
        if len(unique) <= 1:
            return labels
        centroids = np.stack([embeddings[labels == lbl].mean(axis=0) for lbl in unique])
        diffs = embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        sq_dists = (diffs ** 2).sum(axis=2)
        new_indices = np.argmin(sq_dists, axis=1)
        new_labels = unique[new_indices]
        if np.array_equal(new_labels, labels):
            break
        remap = {int(old): idx for idx, old in enumerate(np.unique(new_labels))}
        labels = np.array([remap[int(label)] for label in new_labels])
    return labels


def cluster_gmm(embeddings: np.ndarray[Any, Any], max_cluster: int, threshold: float,
                random_state: int = 0) -> tuple[int, list[list[int]]]:
    """GMM 软聚类：一个点可同时属于多个簇（prob > threshold 全部候选）。

    Returns:
        (n_clusters, clusters)；clusters[i] 是第 i 个簇的索引列表，
        索引可出现在多个簇中。无候选时回退 argmax（硬分配）。
    """
    n_clusters = get_optimal_clusters_gmm(embeddings, max_cluster, random_state)
    if n_clusters <= 1:
        return 1, [list(range(len(embeddings)))]

    gm = GaussianMixture(
        n_components=n_clusters, random_state=random_state,
        covariance_type="diag", reg_covar=_GMM_REG_COVAR,
    )
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    clusters: list[list[int]] = [[] for _ in range(n_clusters)]
    for i, prob in enumerate(probs):
        candidates = np.where(prob > threshold)[0]
        if len(candidates) == 0:
            candidates = np.array([int(np.argmax(prob))])
        for c in candidates:
            clusters[int(c)].append(i)
    return n_clusters, clusters


def cluster_ahc(embeddings: np.ndarray[Any, Any], max_cluster: int) -> tuple[int, list[int]]:
    """AHC 聚类 + centroid 精调。"""
    raw_labels = get_clusters_ahc(embeddings, max_cluster)
    raw_count = np.unique(raw_labels).size

    if raw_count > 1:
        labels = adjust_tree_nodes(embeddings, raw_labels)
    else:
        labels = raw_labels

    unique = np.unique(labels)
    label_map = {int(old): idx for idx, old in enumerate(unique)}
    normalized = [label_map[int(label)] for label in labels]
    return len(unique), normalized

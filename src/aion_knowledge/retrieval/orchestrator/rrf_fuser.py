"""RRF（Reciprocal Rank Fusion）融合器。"""
from __future__ import annotations

import logging
from dataclasses import replace

from aion_knowledge.retrieval.base import ChunkResult

logger = logging.getLogger(__name__)


class RRFFuser:
    """倒数排序融合器。

    RRF_score(chunk) = Σ weight_i / (k + rank_i(chunk))

    weights 由调用方传入（如 vector=0.7, keyword=0.3），
    无显式 weights 时为空字典（所有路径 RRF 贡献为 0）。
    """

    def __init__(self, k: int = 60, weights: dict[str, float] | None = None):
        self.k = k
        self.weights = weights or {}

    def fuse(
        self,
        path_results: dict[str, list[ChunkResult]],
        top_k: int = 10,
        threshold_ratio: float | None = None,
    ) -> list[ChunkResult]:
        """执行 RRF 融合。

        参数：
            path_results: {路径名: [ChunkResult, ...]}，每个列表已按 score 降序
            top_k: 最终保留的最大结果数
            threshold_ratio: 若指定，则过滤掉分数 < max_score * threshold_ratio 的结果

        返回：[ChunkResult, ...] 按 RRF 分数降序（不会变异输入对象）
        """
        if not path_results:
            return []

        # 1. 构建 rank 映射：路径名 → {chunk_id: rank(1-indexed)}
        ranks: dict[str, dict[str, int]] = {}
        for path_name, results in path_results.items():
            path_ranks: dict[str, int] = {}
            for i, r in enumerate(results):
                if r.chunk_id not in path_ranks:
                    path_ranks[r.chunk_id] = i + 1  # 1-indexed
            ranks[path_name] = path_ranks

        # 2. 收集所有唯一 chunk，保留最高原始分的元数据
        chunk_map: dict[str, ChunkResult] = {}
        for results in path_results.values():
            for r in results:
                if r.chunk_id not in chunk_map or r.score > chunk_map[r.chunk_id].score:
                    chunk_map[r.chunk_id] = r

        # 3. 计算 RRF 分数（使用 replace 避免变异输入）
        fused: list[ChunkResult] = []
        for chunk_id, result in chunk_map.items():
            rrf_score = 0.0
            hit_paths: list[str] = []
            for path_name, path_ranks in ranks.items():
                rank = path_ranks.get(chunk_id)
                if rank is not None:
                    weight = self.weights.get(path_name, 0.0)
                    rrf_score += weight / (self.k + rank)
                    hit_paths.append(path_name)

            fused.append(replace(result, score=rrf_score, source_paths=hit_paths))

        # 4. 降序排列
        fused.sort(key=lambda r: r.score, reverse=True)

        # 5. 阈值过滤
        if threshold_ratio is not None and fused:
            max_score = fused[0].score
            if max_score > 0:
                fused = [r for r in fused if r.score >= max_score * threshold_ratio]

        # 6. top_k 截断
        return fused[:top_k]

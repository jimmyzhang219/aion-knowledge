"""RAPTOR 核心算法编排：经典 RAPTOR。"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable

import numpy as np
import umap  # type: ignore[import-untyped]  # umap-learn 无 py.typed/stub，整体按 Any 处理

from aion_knowledge.pipeline.postproc.raptor.clustering import cluster_ahc, cluster_gmm

logger = logging.getLogger(__name__)


class RecursiveAbstractiveProcessing4TreeOrganizedRetrieval:
    """构建 RAPTOR 摘要层：经典 RAPTOR 树策略。"""

    def __init__(
        self,
        max_cluster: int = 64,
        llm_model: Any = None,
        embd_model: Any = None,
        prompt: str = "",
        max_token: int = 256,
        threshold: float = 0.1,
        small_layer_collapse: int = 8,
        max_errors: int = 3,
        clustering_method: str = "gmm",
        context_window: int | None = None,
    ):
        """初始化构建器参数（聚类/摘要/容错相关配置）。"""
        self._max_cluster = max_cluster
        self._small_layer_collapse = small_layer_collapse
        self._llm_model = llm_model
        self._context_window = context_window or 8192
        self._embd_model = embd_model
        self._threshold = threshold
        self._prompt = prompt
        self._max_token = max_token
        self._max_errors = max(1, max_errors)
        self._error_count = 0
        self._clustering_method = clustering_method

    async def _summarize_texts(
        self, texts: list[str], callback: Callable[..., Any] | None = None,
    ) -> tuple[str, str, list[float]] | None:
        """用 LLM 总结一组文本，返回 (title, summary, embedding)。"""
        llm_max_len = self._context_window
        len_per_chunk = max(1, int((llm_max_len - self._max_token) / max(len(texts), 1)))
        cluster_content = "\n".join(t[:len_per_chunk] for t in texts)

        try:
            prompt = self._prompt.format(cluster_content=cluster_content)
            response = await self._llm_model.generate(prompt)
            response = re.sub(r"^.*</think>", "", response, flags=re.DOTALL)

            lines = response.strip().split("\n", 1)
            title = lines[0].strip()[:500]
            body = lines[1].strip() if len(lines) > 1 else ""

            emb = await self._embd_model.encode(body)
            return title, body, emb
        except Exception as exc:
            self._error_count += 1
            logger.warning("[RAPTOR] 摘要生成失败: %s", exc)
            if callback:
                callback(msg=f"[RAPTOR] 跳过 cluster（{len(texts)} 个chunk）: {exc}")
            if self._error_count >= self._max_errors:
                raise RuntimeError(f"RAPTOR 超过最大错误数 ({self._max_errors})") from exc
            return None

    def _cluster_layer(
        self, embeddings: list[list[float]], random_state: int,
    ) -> tuple[int, list[list[int]]]:
        """对一层 embedding 做 UMAP 降维后聚类。"""
        if not embeddings:
            return 0, []
        # 防御：过滤维度不一致的 embedding，防止 numpy inhomogeneous error
        dims = {len(e) for e in embeddings if hasattr(e, '__len__')}
        if len(dims) > 1:
            target_dim = max(dims)
            filtered = [e for e in embeddings if len(e) == target_dim]
            logger.warning("RAPTOR _cluster_layer: inhomogeneous dims %s, filtering %d→%d",
                           dims, len(embeddings), len(filtered))
            if len(filtered) < 2:
                return 0, []
            embeddings = filtered
        arr = np.asarray(embeddings, dtype=np.float64)
        n = len(arr)
        if n <= 1:
            return 0, []

        n_neighbors = max(2, min(int((n - 1) ** 0.8), 100))
        n_components = min(12, max(1, n - 2))
        reduced = umap.UMAP(
            n_neighbors=n_neighbors, n_components=n_components,
            metric="cosine",
        ).fit_transform(arr)

        if self._clustering_method == "ahc":
            n_clusters, ahc_labels = cluster_ahc(reduced, self._max_cluster)
            clusters = [[i for i, lbl in enumerate(ahc_labels) if lbl == c]
                        for c in range(n_clusters)]
        else:
            n_clusters, clusters = cluster_gmm(reduced, self._max_cluster,
                                                self._threshold, random_state)
        return n_clusters, clusters

    async def __call__(
        self,
        chunks: list[tuple[str, list[float], list[str]]],
        random_state: int = 0,
        callback: Callable[..., Any] | None = None,
        is_tree: bool = False,
    ) -> tuple[list[Any], list[Any], dict[int, list[int]]] | dict[str, Any] | None:
        """执行 RAPTOR 构建。

        Args:
            chunks: [(content, embedding, [source_chunk_id, ...]), ...]
            random_state: 随机种子
            callback: 进度回调
            is_tree: True 返回序列化树 dict，False 返回 (summaries, layers, parent_child_map)

        Returns:
            is_tree=False: (all_items, [(layer_start, layer_end), ...], parent_child_map)
            is_tree=True: 树的 dict 或 None（输入不足时）
        """
        normalized = [(t, v, s) for t, v, s in chunks if t and v is not None and len(v) > 0]
        if len(normalized) <= 1:
            return None if is_tree else ([], [], {})

        return await self._build_classic_layers(normalized, random_state, callback, is_tree)

    async def _build_classic_layers(
        self, chunks: list[Any], random_state: int,
        callback: Callable[..., Any] | None, is_tree: bool,
    ) -> tuple[list[Any], list[Any], dict[int, list[int]]] | dict[str, Any] | None:
        """经典 RAPTOR 层次聚类树。"""
        parent_child_map: dict[int, list[int]] = {}
        n_originals = len(chunks)
        layers = [(0, len(chunks))]
        start, end = 0, len(chunks)

        def _merge_source_ids(indices: list[int]) -> list[str]:
            """合并节点索引的 source_chunk_ids 并去重。"""
            merged: list[str] = []
            seen: set[str] = set()
            for i in indices:
                for src in chunks[i][2]:
                    if src and src not in seen:
                        seen.add(src)
                        merged.append(src)
            return merged

        async def _summarize_cluster(indices: list[int]) -> None:
            """汇总一个簇的文本并追加摘要节点到 chunks。"""
            nonlocal chunks
            texts = [chunks[i][0] for i in indices]
            result = await self._summarize_texts(texts, callback)
            if result is not None:
                title, summary, emb = result
                merged_ids = _merge_source_ids(indices)
                chunks.append((summary, emb, merged_ids, title))
                parent_child_map[len(chunks) - 1] = list(indices)

        while end - start > 1:
            embs = [chunks[i][1] for i in range(start, end)]

            if end - start <= self._small_layer_collapse:
                await _summarize_cluster(list(range(start, end)))
                produced = len(chunks) - end
                if produced == 0:
                    break
                layers.append((end, len(chunks)))
                if callback:
                    callback(msg=f"[RAPTOR] 小层 collapse: {end-start} → {produced}")
                break

            n_clusters, clusters = self._cluster_layer(embs, random_state)

            # 软聚类退化为整层一个簇时防死循环（簇数 ≥ 节点数时强制单簇）
            if n_clusters >= len(embs):
                clusters = [list(range(len(embs)))]

            tasks = []
            for cluster in clusters:
                if cluster:
                    # 簇索引相对本层（embs），偏移到 chunks 的绝对索引再消费
                    indices = [i + start for i in cluster]
                    tasks.append(asyncio.create_task(_summarize_cluster(indices)))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            produced = len(chunks) - end
            if produced == 0:
                break
            layers.append((end, len(chunks)))
            if callback:
                callback(msg=f"[RAPTOR] 聚类一层: {end-start} → {produced}")
            start = end
            end = len(chunks)

        if is_tree:
            return self._materialize_tree(chunks, layers, parent_child_map, n_originals)
        return chunks, layers, parent_child_map

    @staticmethod
    def _materialize_tree(
        chunks: list[Any], layers: list[tuple[int, int]],
        parent_child_map: dict[int, list[int]], n_originals: int,
    ) -> dict[str, Any] | None:
        """将 flat summaries 转为树 dict。"""
        if not layers or len(chunks) == 0:
            return None
        top_start, top_end = layers[-1]
        if top_end <= top_start:
            return None

        def _title_at(idx: int) -> str:
            """取节点标题（chunk 元组第 4 位）。"""
            return chunks[idx][3] if len(chunks[idx]) >= 4 else ""

        def _desc_at(idx: int) -> str:
            """取节点描述（chunk 元组首位）。"""
            return chunks[idx][0] if chunks[idx] else ""

        def _build_node(idx: int) -> dict[str, Any]:
            """递归构建树节点：叶子节点带 source_chunk_ids，内部节点带 children。"""
            children_idx = parent_child_map.get(idx, [])
            if children_idx and all(c < n_originals for c in children_idx):
                ids: list[str] = []
                seen: set[str] = set()
                for c in children_idx:
                    for s in chunks[c][2]:
                        if s and s not in seen:
                            seen.add(s)
                            ids.append(s)
                return {"title": _title_at(idx), "source_chunk_ids": ids, "description": _desc_at(idx)}
            return {
                "children": [_build_node(c) for c in children_idx],
                "title": _title_at(idx),
                "description": _desc_at(idx),
            }

        top = [_build_node(i) for i in range(top_start, top_end)]
        if len(top) == 1:
            return top[0]
        return {"title": "(root)", "children": top}

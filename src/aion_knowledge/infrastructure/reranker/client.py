"""Reranker HTTP 客户端 —— 调独立部署的 reranker 服务（``POST /rerank``）。

候选按 ``reranker_max_batch`` 分批发送（受 ``reranker_max_concurrency`` 并发限流），所有候选都打分，不截断丢弃。
任一批失败则抛错，由上层 RerankerStage 兜底降级（保留 RRF 分）。
"""
from __future__ import annotations

import asyncio

import httpx

from aion_knowledge.common.config import settings

DEFAULT_TIMEOUT = 30.0


def _parse_response(data: object, expected_len: int) -> list[float]:
    """把单批响应解析为与该批等长、按批内顺序对齐的分数列表。

    支持三种返回格式：
      [0.99, 0.13, ...]                  → 纯分数列表
      [{"index":0,"score":0.99}, ...]    → 对象列表（index 为批内局部下标，可能乱序）
      {"results": [...]}                 → 包装在 results 键下
    """
    if isinstance(data, dict):
        if "results" in data:
            data = data["results"]
        elif data:
            data = list(data.values())[0]
        else:
            data = []

    if not isinstance(data, list) or not data:
        return [0.0] * expected_len

    if isinstance(data[0], dict):
        # 对象列表（TEI 格式）：index 为批内局部下标，按其还原顺序
        scores: list[float] = [0.0] * expected_len
        for entry in data:
            idx = entry.get("index")
            if isinstance(idx, int) and 0 <= idx < expected_len and "score" in entry:
                scores[idx] = entry["score"]
        return scores

    return [float(s) for s in data[:expected_len]]


async def rerank(query: str, texts: list[str]) -> list[float]:
    """调 reranker 服务对 ``(query, text)`` 批量打分。

    候选按 ``reranker_max_batch`` 分批**并发**发送（受 ``reranker_max_concurrency`` 限流），所有候选都拿到真实分数，
    不截断丢弃。任一批请求失败则抛错（由上层 RerankerStage 兜底降级）。

    Args:
        query: 用户查询。
        texts: 候选文本列表。

    Returns:
        每个文本的相关性分数（0~1），与 ``texts`` 顺序一致。
    """
    if not query or not texts:
        return [0.0] * len(texts)

    # 字符 payload 防护（非 token 预算）：中文 ~1 字/token，×1.5 留标点/英文混排余量。
    # 真 token 截断由 TEI 服务端按模型 max_seq_length 完成。
    max_chars = int(settings.reranker_max_tokens * 1.5)
    truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]

    scores: list[float] = [0.0] * len(texts)
    batch_size = settings.reranker_max_batch
    sem = asyncio.Semaphore(max(1, settings.reranker_max_concurrency))

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, trust_env=False) as client:
        async def _send_batch(start: int, batch: list[str]) -> tuple[int, list[float]]:
            async with sem:
                resp = await client.post(
                    settings.reranker_endpoint,
                    json={"query": query, "texts": batch},
                )
                resp.raise_for_status()
                return start, _parse_response(resp.json(), len(batch))

        batches = [
            (start, truncated[start:start + batch_size])
            for start in range(0, len(truncated), batch_size)
        ]
        for start, batch_scores in await asyncio.gather(*(_send_batch(s, b) for s, b in batches)):
            for j, score in enumerate(batch_scores):
                scores[start + j] = score

    return scores

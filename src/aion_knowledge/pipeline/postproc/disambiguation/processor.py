"""DisambiguationModule — 实体消歧（批量 LLM 裁决 + 预过滤）。

作用：
  从 Neo4j 读取同一知识库下的所有实体，发现指向同一
  现实世界对象的不同名称并合并。消歧结果触发 Neo4j 知识图谱合并。

机制：
  1. 按 entity_type 分组，避免跨类型消歧（如同一名称"苹果"作为
     fruit 和 company 类型不应合并）
  2. 文本相似度预过滤：英文用 Levenshtein 编辑距离（≤3），
     CJK 用 Jaccard 字符重叠率（≥0.7）+ 子串包含检测
  3. LLM 批量裁决：按 batch_size 分批送入 LLM，逐对判断 is_same
     并给出 canonical 标准名
  4. 结果输出为 (canonical, [alias1, alias2, ...]) 列表
  5. 通过 DisambiguationMerger 写入 Neo4j 知识图谱

跳过策略：
  KB 无图谱数据时跳过（消歧为 KB 级操作，需先启用 graph_extract）。
"""
from __future__ import annotations

import itertools
import logging
import re
from typing import Any

from aion_knowledge.infrastructure.llm import get_llm_client_for_module
from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule
from aion_knowledge.pipeline.postproc.disambiguation.config import disambiguation_config
from aion_knowledge.pipeline.postproc.disambiguation.merger import DisambiguationMerger

logger = logging.getLogger(__name__)

DISAMBIGUATE_BATCH_PROMPT = (
    "判断以下每对实体是否指向现实世界中的同一对象。"
    "以 JSON 数组格式输出：\n"
    '[{{"entity_a": "...", "entity_b": "...", "is_same": true/false, "canonical": "标准名"}}]\n\n'
    "候选对：\n{pairs}"
)


def _is_cjk(s: str) -> bool:
    """检测字符串是否含有 CJK 字符。"""
    return bool(re.search(r'[一-鿿㐀-䶿]', s))


def _is_similar(a: str, b: str, edit_threshold: int = 3,
                 jaccard_threshold: float = 0.7) -> bool:
    """预过滤：相同 → True；英文编辑距离 ≤ threshold；CJK Jaccard 字符重叠 ≥ threshold。"""
    if a == b:
        return True
    if not a or not b:
        return False
    if _is_cjk(a) or _is_cjk(b):
        # 子串包含直接判定相似（如"苹果公司"包含"苹果"）
        if a in b or b in a:
            return True
        set_a, set_b = set(a), set(b)
        overlap = len(set_a & set_b) / max(len(set_a | set_b), 1)
        return overlap >= jaccard_threshold
    return _levenshtein(a.lower(), b.lower()) <= edit_threshold


def _levenshtein(a: str, b: str) -> int:
    """计算编辑距离（不使用外部库）。"""
    n, m = len(a), len(b)
    if n > m:
        a, b, n, m = b, a, m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add = previous[j] + 1
            delete = current[j - 1] + 1
            change = previous[j - 1] + (0 if a[j - 1] == b[i - 1] else 1)
            current[j] = min(add, delete, change)
    return current[n]


def _generate_candidates(
    entities: list[dict[str, str]],
    edit_threshold: int = 3,
    jaccard_threshold: float = 0.7,
) -> list[tuple[str, str]]:
    """按 entity_type 分组，组内生成候选对（无序，a < b 保证）。"""
    by_type: dict[str, list[str]] = {}
    for ent in entities:
        by_type.setdefault(ent.get("entity_type", ""), []).append(ent["entity_name"])
    candidates: list[tuple[str, str]] = []
    for etype, names in by_type.items():
        seen: set[tuple[str, str]] = set()
        for a, b in itertools.combinations(sorted(set(names)), 2):
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            if _is_similar(a, b, edit_threshold, jaccard_threshold):
                candidates.append((a, b))
    return candidates


class DisambiguationModule(PostProcModule):
    """实体消歧后处理模块（二批执行，依赖 text + graph_extract）。"""

    always_on = False
    depends_on = ["text", "graph_extract"]

    async def process(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> int:
        """执行 KB 级实体消歧，返回合并的实体组数量。

        流程：加载 KB 图谱 → hash 检查点跳过（图未变）→ 相似度预过滤
        生成候选对 → LLM 批量裁决 → Neo4j 合并 → 保存检查点。
        """
        if not chunks:
            return 0

        kb_id = ctx.kb_id
        llm = get_llm_client_for_module("disambiguation")
        entities, relations = await self._load_kb_entities(kb_id)

        if not entities:
            logger.warning("【消歧】KB 无图谱数据，跳过（消歧为 KB 级操作，需先启用 graph_extract）")
            return 0

        # KB 图 hash 检查点：hash 未变说明无新实体，跳过重复消歧（参照 community 检查点模式）
        from aion_knowledge.infrastructure.db import get_session
        from aion_knowledge.pipeline.postproc.community.checkpoint import compute_graph_hash
        from aion_knowledge.storage.relational.graph_repo import GraphMetadataRepository

        graph_hash = compute_graph_hash(
            [e["entity_name"] for e in entities],
            [(r["source_entity"], r["target_entity"], r["relation_type"]) for r in relations],
        )
        async with get_session() as session:
            repo = GraphMetadataRepository(session)
            saved = await repo.load_checkpoint(kb_id, "disambiguation")
            if saved == graph_hash:
                logger.info("【消歧】KB 图谱 hash 未变，跳过")
                return 0

        candidates = _generate_candidates(
            entities,
            edit_threshold=disambiguation_config.edit_distance_threshold,
            jaccard_threshold=disambiguation_config.jaccard_threshold,
        )

        if not candidates:
            logger.info("【消歧】预过滤后无候选对")
            # 无候选也保存检查点（当前图 hash，图未被修改）——下次 hash 未变时命中跳过
            await self._save_checkpoint(kb_id, graph_hash)
            return 0

        groups = await self._resolve_batches(llm, candidates,
                                              disambiguation_config.batch_size)

        merger = DisambiguationMerger()
        await merger.batch_merge(groups, kb_id)

        await self._save_checkpoint(kb_id, graph_hash)

        logger.info("【消歧】处理完成：合并 %d 个实体组，知识库=%s",
                     len(groups), kb_id)
        return len(groups)

    async def _load_kb_entities(self, kb_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """从 Neo4j 加载 KB 的全部实体与关系。"""
        from aion_knowledge.infrastructure.graph import load_kb_graph

        return await load_kb_graph(kb_id)

    async def _save_checkpoint(self, kb_id: str, graph_hash: str) -> None:
        """保存消歧检查点（失败容错：warning 不影响主流程）。"""
        from aion_knowledge.infrastructure.db import get_session
        from aion_knowledge.storage.relational.graph_repo import GraphMetadataRepository

        try:
            async with get_session() as session:
                repo = GraphMetadataRepository(session)
                await repo.save_checkpoint(kb_id, "disambiguation", {"graph_hash": graph_hash})
        except Exception as exc:
            logger.warning("【消歧】保存检查点失败：%s", exc)

    async def _resolve_batches(
        self, llm: Any, candidates: list[tuple[str, str]], batch_size: int
    ) -> list[tuple[str, list[str]]]:
        """分批调用 LLM 裁决候选对，归并为 (canonical, [别名...]) 分组。

        单批失败仅告警跳过，不影响其他批次；非数组/非 dict 返回视为空。
        """
        canonical_map: dict[str, set[str]] = {}

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            pairs_text = "\n".join(
                f"{j+1}. {a} | {b}" for j, (a, b) in enumerate(batch)
            )
            try:
                result = await llm.generate_structured(
                    DISAMBIGUATE_BATCH_PROMPT.format(pairs=pairs_text),
                    output_schema={
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity_a": {"type": "string"},
                                "entity_b": {"type": "string"},
                                "is_same": {"type": "boolean"},
                                "canonical": {"type": "string"},
                            },
                            "required": ["entity_a", "entity_b", "is_same", "canonical"],
                        },
                    },
                )
                # 兼容 LLM 返回形态：纯数组 / thinking 包装（{"reasoning":..., "answer":[...]}）/
                # 单个裁决对象，统一归一化为裁决对象列表。
                if isinstance(result, list):
                    items = result
                elif isinstance(result, dict):
                    answer = result.get("answer")
                    if isinstance(answer, list):
                        items = answer
                    elif isinstance(answer, dict):
                        items = [answer]
                    else:
                        items = [result]
                else:
                    items = []
                for item in items:
                    if item.get("is_same"):
                        c = item["canonical"]
                        for e in (item["entity_a"], item["entity_b"]):
                            if e != c:
                                canonical_map.setdefault(c, set()).add(e)
            except Exception as exc:
                logger.warning("【消歧】批量处理第 %d 批失败：%s", i // batch_size, exc)
                continue

        return [(c, list(aliases)) for c, aliases in canonical_map.items()]


def module() -> DisambiguationModule:
    """模块工厂函数，供调度器自动发现。"""
    return DisambiguationModule()

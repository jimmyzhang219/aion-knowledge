"""RaptorModule — 递归摘要树（RAPTOR）。

作用：
  对文档 chunks 执行递归聚类 + 摘要生成，构建层次化的树状结构，
  写入 chunk_raptor 表。适用于长文档的语义压缩和多粒度检索。

机制：
  1. 从 chunk_vector 表加载每段 chunk 的 embedding 向量
  2. 将 (text, embedding, [chunk_id]) 组装为 RAPTOR 输入
  3. 通过 RecursiveAbstractiveProcessing4TreeOrganizedRetrieval 执行：
     a. 聚类：默认 GMM 软聚类（prob > threshold 的全候选取，
        一个节点可同时属于多个簇），亦支持 AHC 硬聚类
     b. 每类生成摘要（LLM summarization），同一文本可参与多个簇摘要
     c. 递归：将摘要视为新的"chunk"继续聚类，直到剩余节点
        不超过 small_layer_collapse 时折叠为根
  4. 输出模式：
     - tree 模式：输出完整的树结构（tree_json），写入单条记录
       （仅供展示，不参与检索）
     - flat 模式：每层摘要单独记录，包含 embedding 回写、
       source_chunk_ids 溯源；并落 children_ids（全部摘要子节点，
       软聚类多父结构）与 parent_id（首个父节点），
       检索侧据此做自顶向下树遍历召回
  5. 幂等：写入前删除同 (kb_id, doc_id) 的旧树——仅当本次确认有新摘要
     时执行，与新增同一事务，LLM 全失败时保留旧树

自动跳过检查：
  - 扫描 PDF（parser_id=scanned_pdf）跳过
  - 图片类文档跳过
  - 文档 chunks 少于 2 个跳过
  - 根据 file_type / parser_id / parser_config 综合判断

可选启用，依赖 text + vector 模块先行（需要 embedding 数据）。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, cast

from aion_knowledge.common.config import settings
from aion_knowledge.common.model_registry import get_registry
from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.infrastructure.embedder import create_embedder
from aion_knowledge.infrastructure.llm import get_llm_client_for_module
from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule
from aion_knowledge.pipeline.postproc.raptor.config import raptor_config
from aion_knowledge.pipeline.postproc.raptor.core import (
    RecursiveAbstractiveProcessing4TreeOrganizedRetrieval,
)
from aion_knowledge.pipeline.postproc.raptor.utils import (
    get_skip_reason,
    should_skip_raptor,
)

logger = logging.getLogger(__name__)


class _EmbeddingAdapter:
    """简单 embedding 适配器，通过统一工厂获取嵌入服务。"""

    async def encode(self, text: str) -> list[float]:
        """对文本做查询向量化。"""
        if not text:
            return []
        embedder = create_embedder()
        return await embedder.embed_query(text)


def _parse_vector(raw: object) -> list[float]:
    """将 pgvector 返回的值解析为 float 列表。

    asyncpg 未注册 pgvector 类型适配器时返回的是字符串
    ``[-0.0366,-0.0163,...]``，需要手动解析为 ``list[float]``。
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip().strip("[]")
        if not stripped:
            return []
        return [float(x) for x in stripped.split(",")]
    return []


async def _load_vectors(document_id: str) -> dict[str, list[float]]:
    """从 chunk_vector 表加载该文档所有 chunk 的 embedding。"""
    from sqlalchemy import text as sql_text

    from aion_knowledge.infrastructure.db import get_session

    vectors: dict[str, list[float]] = {}
    try:
        async with get_session() as session:
            rows = await session.execute(
                sql_text(
                    "SELECT cv.chunk_id, cv.embedding FROM chunk_vector cv "
                    "WHERE cv.chunk_id::text IN "
                    "(SELECT ct.id::text FROM chunk_text ct WHERE ct.document_id = :doc_id)"
                ),
                {"doc_id": document_id},
            )
            for row in rows:
                row_dict = dict(row._mapping)
                chunk_id = str(row_dict["chunk_id"])
                vectors[chunk_id] = _parse_vector(row_dict["embedding"])
    except Exception as exc:
        logger.warning("Failed to load vectors for doc=%s: %s", document_id, exc)
    return vectors


class RaptorModule(PostProcModule):
    """RAPTOR 递归摘要模块。二批执行，依赖 text + vector。"""

    always_on = False
    depends_on = ["text", "vector"]

    async def process(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> int:
        """对文档 chunks 构建摘要树并写入 chunk_raptor，返回写入记录数。"""
        if not chunks or len(chunks) < 2:
            return 0

        # 自动禁用检查
        if should_skip_raptor(ctx.suffix, ctx.parser_id, ctx.parser_config):
            reason = get_skip_reason(ctx.suffix, ctx.parser_id, ctx.parser_config)
            logger.info("RAPTOR 跳过 doc=%s: %s", ctx.doc_name, reason)
            return 0

        clustering_method = raptor_config.clustering_method
        output_mode = raptor_config.output_mode
        max_cluster = raptor_config.max_cluster
        max_token = raptor_config.max_token
        threshold = raptor_config.threshold
        prompt = raptor_config.prompt
        random_seed = raptor_config.random_seed
        small_layer_collapse = raptor_config.small_layer_collapse
        max_errors = raptor_config.max_errors

        llm = get_llm_client_for_module("raptor")
        embd_adapter = _EmbeddingAdapter()

        # 加载 embedding 向量
        vectors = await _load_vectors(ctx.document_id)

        # 组装 RAPTOR 输入：(text, embedding, [chunk_id])
        raptor_input = []
        for c in chunks:
            chunk_id = str(c.get("chunk_uuid", ""))
            content = c.get("content", "")
            vec = vectors.get(chunk_id, [])
            if content and vec:
                raptor_input.append((content, vec, [chunk_id]))

        if len(raptor_input) < 2:
            return 0

        raptor = RecursiveAbstractiveProcessing4TreeOrganizedRetrieval(
            max_cluster=max_cluster,
            llm_model=llm,
            embd_model=embd_adapter,
            prompt=prompt,
            max_token=max_token,
            threshold=threshold,
            small_layer_collapse=small_layer_collapse,
            max_errors=max_errors,
            clustering_method=clustering_method,
            context_window=get_registry().context_window(settings.llm_model),
        )

        is_tree_mode = output_mode == "tree"
        result = await raptor(
            raptor_input,
            random_state=random_seed,
            callback=None,
            is_tree=is_tree_mode,
        )

        if result is None:
            return 0

        from sqlalchemy import text as sql_text

        from aion_knowledge.infrastructure.db import get_session
        from aion_knowledge.pipeline.postproc.raptor.orm import ChunkRaptor

        inserted = 0
        async with get_session() as session:
            if is_tree_mode:
                # is_tree=True 时 __call__ 保证返回树 dict，
                # cast 仅消除 tuple|dict 联合类型的窄化歧义，不改运行时行为
                tree_result = cast(dict[str, Any], result)
                # 幂等：重跑前删除同文档旧树（tree 分支必然插入 1 行，同一事务）
                await session.execute(
                    sql_text(
                        "DELETE FROM chunk_raptor "
                        "WHERE kb_id = :kb_id AND doc_id IS NOT DISTINCT FROM :doc_id"
                    ),
                    {"kb_id": ctx.kb_id, "doc_id": ctx.document_id},
                )
                row = ChunkRaptor(
                    kb_id=uuid.UUID(ctx.kb_id),
                    doc_id=uuid.UUID(ctx.document_id) if ctx.document_id else None,
                    title=tree_result.get("title", ""),
                    summary=tree_result.get("description", ""),
                    layer=0,
                    output_mode="tree",
                    tree_builder="raptor",
                    clustering_method=clustering_method,
                    tree_json=tree_result,
                )
                session.add(row)
                inserted = 1
            else:
                # is_tree=False 时 __call__ 保证返回 3 元组，
                # cast 仅消除 tuple|dict 联合类型的窄化歧义，不改运行时行为
                summaries, layers, parent_child_map = cast(
                    tuple[list[Any], list[Any], dict[int, list[int]]], result
                )
                original_count = len(raptor_input)
                # 幂等：确认有新摘要才删旧树并插入（LLM 全失败不抛异常时零插入，
                # 不删旧树避免检索窗口空洞），与新增同一事务
                if len(summaries) > original_count:
                    await session.execute(
                        sql_text(
                            "DELETE FROM chunk_raptor "
                            "WHERE kb_id = :kb_id AND doc_id IS NOT DISTINCT FROM :doc_id"
                        ),
                        {"kb_id": ctx.kb_id, "doc_id": ctx.document_id},
                    )
                # 反查表：子节点索引 → 首个父节点索引（软聚类下子可属多父，取第一个）
                child_to_first_parent: dict[int, int] = {}
                for p_idx, children in parent_child_map.items():
                    for c_idx in children:
                        child_to_first_parent.setdefault(c_idx, p_idx)
                # 预生成全部摘要节点 id：children_ids/parent_id 引用其他节点，需先于插入确定
                id_map: dict[int, uuid.UUID] = {
                    idx: uuid7() for idx in range(original_count, len(summaries))
                }
                chunk_layer = {}
                for li, (ls, le) in enumerate(layers):
                    if li == 0:
                        continue
                    for ci in range(ls, le):
                        chunk_layer[ci] = li

                for idx in range(original_count, len(summaries)):
                    entry = summaries[idx]
                    content, emb, source_ids = entry[0], entry[1], entry[2]
                    title = entry[3] if len(entry) >= 4 else ""

                    row = ChunkRaptor(
                        kb_id=uuid.UUID(ctx.kb_id),
                        doc_id=uuid.UUID(ctx.document_id) if ctx.document_id else None,
                        title=title,
                        summary=content,
                        layer=chunk_layer.get(idx, 1),
                        cluster_id=f"L{chunk_layer.get(idx, 1)}_C{idx}",
                        source_chunk_ids=[uuid.UUID(s) for s in source_ids if s],
                        parent_id=id_map[child_to_first_parent[idx]]
                        if idx in child_to_first_parent
                        else None,
                        children_ids=[
                            id_map[c] for c in parent_child_map.get(idx, []) if c >= original_count
                        ],
                        tree_builder="raptor",
                        clustering_method=clustering_method,
                        output_mode="flat",
                    )
                    session.add(row)
                    inserted += 1

                    # flush 立即拿到 row.id，再通过 raw SQL 回写 embedding
                    await session.flush()
                    if emb and len(emb) > 0:
                        update_emb_sql = (
                            "UPDATE chunk_raptor SET embedding = CAST(:emb AS vector) "
                            "WHERE id = :id"
                        )
                        await session.execute(
                            sql_text(update_emb_sql),
                            {"emb": str(emb), "id": row.id},
                        )

            await session.flush()
            logger.info("RAPTOR: 写入 %d 条记录到 chunk_raptor（doc=%s）", inserted, ctx.doc_name)

        return inserted


def module() -> RaptorModule:
    """模块工厂函数，供调度器自动发现。"""
    return RaptorModule()

"""Retrieval —— 多路径检索 & 生成模块。

完整的 RAG（检索增强生成）pipeline，流程为：

    查询理解 → 多路搜索 → 结果融合 → 重排序 → 上下文组装 → 生成

路由策略：
    - 根据查询特征（关键词、语义、FAQ、知识图谱等）自动路由到最优检索策略
    - 支持多检索器并行调用、结果融合（RRF）
    - 支持 RAPTOR 摘要树检索

主要组件：
    - search/：      各种检索器（向量/BERT/关键词/BM25/FAQ/知识图谱/RAPTOR/社区等）
    - orchestrator/：检索路由、多检索器并发调用、RRF 结果融合
    - context/：     上下文组装（父子块合并、截断器）
    - pipeline/：    完整检索 pipeline（多阶段编排）
    - generator/：   答案生成（QA、流式输出、Agent 模式）

用法：
    from aion_knowledge.retrieval import BaseRetriever, ChunkResult, RetrieverContext

    # 子模块的高层入口在对应 pipeline 中：
    from aion_knowledge.retrieval.pipeline import RetrievalPipeline
"""

from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult, RetrieverContext

__all__ = [
    "BaseRetriever",
    "ChunkResult",
    "RetrieverContext",
]

"""知识图谱实体/关系提取模块。

二批后处理模块（``always_on=False``），依赖 ``text``：从文档 chunks 中
通过 LLM 并发提取实体与关系，经文档内去重 + 跨文档合并写入 Neo4j
知识图谱，并刷新 PG ``graph_metadata`` 统计。

设计约束：图谱与知识库 1:1——同一 KB 下所有文档的实体/关系合并进
同一张 Neo4j 图（按 ``kb_id`` 标识），不按文档粒度建独立图。

实现逻辑：
  - ``processor.GraphExtractModule``：业务编排——
    1. 并发控制：``asyncio.Semaphore(max_concurrent)`` 限制同时提取的
       chunk 数，``asyncio.gather`` 并发处理全文档 chunks；
    2. 内容守卫：空内容 / 过短 chunk 跳过；
    3. 文档内去重：同名实体合并 ``descriptions`` + ``source_chunks``，
       同一 ``(source, target, type)`` 关系累加 ``weight``、合并描述；
    4. 触发跨文档合并（``merger.KBGraphMerger.merge_document``）。
  - ``extractor.extract_entities_with_gleaning``：LLM 提取核心，被
    graph_extract / community fallback 共享——
    1. 首轮结构化提取（entities + relations，JSON Schema 约束）；
    2. Gleaning 多轮补充：把已有实体列表回灌 prompt，让 LLM 补遗漏，
       直到无新实体或达 ``max_gleanings`` 上限；
    3. token 截断防御（``truncate_by_tokens``）防止超模型上下文。
  - ``merger.update_kb_graph_stats``：文档子图写入 Neo4j（``add_graph``）
    后，从 Neo4j 取实体/关系/文档数、从 PG 取社区数，upsert 到
    ``graph_metadata``（供 graph_extract / community 写后共用刷新）。

涉及技术 / 算法：
  - LLM 结构化提取 + Gleaning（多轮渐进式补全，源自 GraphRAG 思路）。
  - asyncio.Semaphore 并发节流，规避 API rate limit。
  - 文档内基于字典的实体/关系去重与权重累加。

输出：Neo4j 知识图谱（实体/关系/属性）+ PG ``graph_metadata`` 统计缓存
（图 hash 检查点亦存于此，供下游 disambiguation / community 复用）。

配置：``config.graph_config``（``AION_GRAPH_*``）——
``max_concurrent``、``max_gleanings``、``entity_types``。
"""

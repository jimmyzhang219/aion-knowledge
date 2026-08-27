"""知识图谱社区发现与社区报告生成模块。

二批后处理模块（``always_on=False``），依赖 ``text`` + ``disambiguation``：
从已消歧的全局知识图谱中读取实体/关系，通过层次化 Leiden 算法发现
紧密关联的实体社区，再为每个社区生成 LLM 摘要报告，写入
``chunk_community``，供社区级检索与全局摘要使用。

实现逻辑（``processor.CommunityModule``）：
  1. 数据源：``infrastructure.graph.load_kb_graph`` 从 Neo4j 读取 KB 级
     已消歧的实体（节点）与关系（边），构造 NetworkX 无向图。
  2. 检查点：``checkpoint.compute_graph_hash`` 对实体名 + 关系三元组算
     hash，与上次记录比对——未变则跳过检测与 LLM，仅回填补缺失向量
     （``_backfill_missing_embeddings``），避免无效开销。
  3. 社区检测（``leiden.py``）：优先 graspologic 的
     ``hierarchical_leiden``（层次化，保留全部 level 而非只取末层），
     ``max_cluster_size`` 控制社区粒度；graspologic 缺失或无边图时
     回退到 NetworkX 连通分量。
  4. 报告生成：每个社区按 level 0/1 选不同 prompt 模板（高层主题 vs
     细节关系），LLM 结构化输出 title/summary/findings/rating；再批量
     生成摘要向量（``common.build_community_text`` 拼装文本）。
  5. 两阶段入库：先全部 LLM 调用（不持有 DB 连接）→ 批量 embedding →
     单 session ``session.add_all`` 写 ``chunk_community``，KB 级社区
     ``chunk_id`` 置零值 UUID。
  6. 收尾：刷新 ``graph_metadata`` 统计（社区数随本次写入变化）。

涉及技术 / 算法：
  - graspologic ``hierarchical_leiden``：Leiden 算法在 Louvain 局部
    移动之上增加 refinement 阶段，保证社区连通性、缓解分辨率极限。
  - NetworkX：图建模与连通分量回退。
  - 图 hash 检查点：增量式跳过重复检测。
  - LLM ``generate_structured`` + JSON Schema 约束输出。

回退路径：KB 无图谱时（``_fallback_process``），退化为 chunk 级实体提取
（复用 graph_extract 的 ``extract_entities_with_gleaning``）→ 构图 →
社区检测，``chunk_id`` 取成员实体所在 chunk。

输出：``chunk_community`` 表（community_id / level / summary /
findings / embedding）。

配置：``config.community_config``（``AION_COMMUNITY_*``）——
``max_cluster_size``、``enable_checkpoint``。
"""

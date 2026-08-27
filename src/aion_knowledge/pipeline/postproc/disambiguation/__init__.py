"""实体消歧模块。

二批后处理模块（``always_on=False``），依赖 ``text`` + ``graph_extract``：
从已建好的 KB 知识图谱中读取全部实体，发现指向同一现实世界对象的不同
名称并合并到 canonical 标准名，合并结果回写 Neo4j 图谱并记录到
``chunk_disambiguation``。

实现逻辑（``processor.DisambiguationModule`` + ``merger.DisambiguationMerger``）：
  1. 数据源：``load_kb_graph`` 从 Neo4j 读取 KB 全部实体与关系；无图谱
     数据直接跳过（KB 级操作，需先启用 graph_extract）。
  2. 检查点：与 community 共用 ``compute_graph_hash``，hash 未变则跳过
     整轮消歧。
  3. 候选对生成（``_generate_candidates``）：按 ``entity_type`` 分组，
     组内配对——避免跨类型误并（如"苹果"作为 fruit 与 company）。
  4. 文本相似度预过滤（``_is_similar``）：先做廉价的字面过滤再送 LLM，
     显著减少裁决对数——
     - 相同字符串直接通过；
     - 英文（非 CJK）：手写 Levenshtein 编辑距离 DP，≤阈值（默认 3）通过；
     - CJK：字符级 Jaccard 重叠率 ≥阈值（默认 0.7），子串包含直接通过
       （如"苹果公司"含"苹果"）。
  5. LLM 批量裁决（``_resolve_batches``）：按 ``batch_size`` 分批，每批
     逐对判断 ``is_same`` 并给出 ``canonical``；用结构化输出（JSON Schema）
     约束，并兼容 thinking 包装（``{reasoning, answer}``）与单对象返回，
     统一归一化为裁决列表；聚合为 ``canonical_map``（canonical → 别名集）。
  6. 合并落地（``DisambiguationMerger``）：``neo4j_merge_aliases`` 在
     Neo4j 中把别名实体合并到 canonical，再逐条写合并历史到
     ``chunk_disambiguation``（KB 级决策，``chunk_id=None``）。
  7. 单批失败仅告警跳过，不影响其他批次。

涉及技术 / 算法：
  - 手写 Levenshtein 距离（动态规划，无外部依赖）+ 字符级 Jaccard
    作为 LLM 前的廉价预过滤。
  - CJK 检测正则 ``[一-鿿㐀-䶿]`` 区分中英文走不同相似度策略。
  - 图 hash 检查点（与 community 同源）。
  - LLM 批量结构化裁决，按 entity_type 隔离避免语义混淆。

输出：Neo4j 别名合并 + ``chunk_disambiguation`` 合并历史。

配置：``config.disambiguation_config``（``AION_DISAMBIGUATION_*``）——
``edit_distance_threshold``、``jaccard_threshold``、``batch_size``。
"""

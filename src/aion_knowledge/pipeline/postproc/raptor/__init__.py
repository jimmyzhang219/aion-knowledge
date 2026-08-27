"""RAPTOR 递归摘要树模块（Recursive Abstractive Processing for
Tree-Organized Retrieval）。

二批后处理模块（``always_on=False``），依赖 ``text`` + ``vector``：对
文档 chunks 执行递归聚类 + 摘要生成，构建层次化摘要树，写入
``chunk_raptor``，供检索侧做多粒度树遍历召回。适用于长文档的语义压缩。

实现逻辑：
  - ``processor.RaptorModule``：编排——
    1. 从 ``chunk_vector`` 加载该文档每个 chunk 的 embedding；
    2. 组装 RAPTOR 输入 ``(text, embedding, [source_chunk_id])``，
       缺文本或缺向量的 chunk 丢弃；
    3. 调 ``core.RecursiveAbstractiveProcessing4TreeOrganizedRetrieval``
       构建摘要树，按 ``output_mode`` 落库。
  - ``core.py``：经典 RAPTOR 层次聚类树——
    1. 逐层聚类：每层先 UMAP 降维（cosine 度量，``n_neighbors``/
       ``n_components`` 随节点数自适应），再聚类；
    2. 聚类方法（``clustering.py``）：
       - GMM 软聚类（默认）：用 BIC 选最优簇数，``predict_proba`` 中
         prob > ``threshold`` 的点全部入选——一个节点可同属多个簇；
         无候选时回退 argmax 硬分配；
       - AHC 硬聚类：Ward 链接 + 树状图最大 gap 切分 + centroid 精调
         （k-means 风格重分配到最近中心）；
    3. 每簇 LLM 摘要 → 追加为新的"chunk"节点（含摘要 embedding + 合并
       后的 source_chunk_ids），记录 parent→children 映射；
    4. 递归直到剩余节点 ≤ ``small_layer_collapse`` 时折叠为根；
    5. 容错：累计 ``max_errors`` 次失败则中止；软聚类退化为整层单簇时
       防死循环。
  - 输出两种模式：
    - **tree**：序列化整棵树 dict，写单条记录（``tree_json``），仅供
      展示，不参与检索；
    - **flat**（默认）：每个摘要节点单独成行，回写 embedding、
      ``source_chunk_ids`` 溯源，并落 ``children_ids``（全部摘要子节点，
      软聚类下为多父 DAG）与 ``parent_id``（首个父节点），检索侧据此
      自顶向下树遍历召回。
  - 幂等：写入前按 ``(kb_id, doc_id)`` 删旧树——仅当本次确认产出了新
    摘要才删（LLM 全失败时不删，避免检索窗口空洞），删与插在同一事务。
  - 自动跳过：扫描 PDF（``parser_id=scanned_pdf``）、图片类文档、
    chunks 少于 2 个（``utils.should_skip_raptor``）。

涉及技术 / 算法：
  - UMAP（``umap-learn``）降维 + cosine 度量。
  - scikit-learn ``GaussianMixture``（GMM / BIC 软聚类）、
    ``AgglomerativeClustering``（Ward 层次聚类）。
  - 经典 RAPTOR 递归摘要：摘要节点复用为下一层聚类输入。
  - 软聚类 → 多父 DAG（区别于原始 RAPTOR 的硬聚类树）。

输出：``chunk_raptor`` 表（layer / summary / embedding /
source_chunk_ids / parent_id / children_ids）。

配置：``config.raptor_config``（``AION_RAPTOR_*``）——
``clustering_method``、``output_mode``、``max_cluster``、``threshold``、
``small_layer_collapse``、``max_errors``、``prompt`` 等。
"""

"""pipeline 后处理模块包。

文档解析完成后，对分块结果执行的系列后处理操作。
每种后处理是一个独立的子包，通过统一的 PostProcModule 接口注册
到 PostProcDispatcher，由调度器按 DAG 拓扑序分批执行。

已注册模块（11 个）：
  text           文本 chunk 落库 + parent 结构生成           [always_on]
  vector         向量嵌入生成与落库                           [always_on]
  vlm_caption    VLM 图片描述 + OCR 合并                     [always_on]
  summarizer     逐 chunk 摘要生成                           [可选]
  keyword_extract 三层关键词提取                              [可选]
  question_gen   逐 chunk 问题生成                            [可选]
  wiki           MAP→REFINE 四阶段 Wiki 页面生成              [可选]
  graph_extract  实体关系提取 → Neo4j                         [可选]
  disambiguation 实体消歧（预过滤 + LLM 裁决）                [可选]
  community      社区发现（Leiden） + 报告生成                 [可选]
  raptor         递归摘要树（RAPTOR）                         [可选]
"""

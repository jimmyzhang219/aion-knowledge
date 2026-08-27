# 10 路检索：设计详解

> 每路拆解为：建索引（后处理做了什么）→ 数据结构（数据怎么存的）→ 召回（查询时怎么匹配和算分）

---

## 基础双路（默认启用，可由配置开关关闭）

---

### 1. bm25 — 全文检索

**建索引：**

解析器从文档提取原始文本，经过清洗和按策略切块后，写入 `chunk_text` 表。分块策略支持 `auto`（根据文档画像自动选择）、`heading`（按标题层级）、`heuristic`（启发式）、`recursive`（递归）和 `no_split`（整篇不分）。每块记录 chunk_type 为 `text`（叶子块）或 `parent`（每 2 个子块合并为一个父块，用于上下文召回）。

Text 后处理模块（首批执行）将每个 chunk 的原始文本写入 `chunk_text.content` 列后，PostgreSQL 内部通过 `zhparser` 分词器将内容自动切词并存入专用的 `content_tokens`（`text[]` 列），同时在该列上建立了 **pg_textsearch 扩展的 BM25 索引**（`idx_bm25_content_tokens`）。

**数据结构：**

`chunk_text` 表是最核心的文本存储，记录了每个文本块的内容、所属文档、顺序编号和类型。每个块有一个 `metadata` JSON 列，用于存储 FAQ 等链路的结构化扩展信息。

BM25 检索依赖两列：
- `content_tokens` — zhparser 自动分词后的 token 数组（text[]），已建 BM25 索引
- `summary_tokens` — 摘要分词后的 token 数组，用于 summary 检索

**召回逻辑：**

收到查询后，BM25Retriever 使用 pg_textsearch 扩展的 `to_bm25query()` 函数将原始查询文本转为 BM25 查询表达式，通过 `<@>` 操作符直接在 `content_tokens` BM25 索引上完成全文检索：

```sql
SELECT c.id, c.content, c.document_id, c.kb_id,
       c.content_tokens <@> to_bm25query('查询文本') AS score
FROM chunk_text c
WHERE c.kb_id = '...'
  AND c.content_tokens <@> to_bm25query('查询文本') IS NOT NULL
ORDER BY score ASC
```

> **注意：** pg_textsearch 的 `<@>` 操作符返回的评分是**负值**（0 = 不匹配，越负越相关），与常规相似度评分（越高越好）方向相反。但 RRF 融合使用排名位置而非分值大小，因此原始负值评分可直接使用，排序方向为 `ORDER BY score ASC`（最负的排第一）。
>
> 查询字符串直接内联到 SQL 中（而非作为参数绑定），因为 asyncpg 的参数化查询会破坏 pg_textsearch 操作符的类型推断路径。

分词完全由后端 PostgreSQL 的 **zhparser** 扩展（text_config `zh_cfg`）自动完成，不依赖 Python 端的 jieba 库。整个计算发生在数据库内，不需要遍历所有 chunk。

最终按分数降序（负值最小 = 最相关）取 top_k 返回。排序结果直接参与 RRF 融合，排名越靠前贡献越大。RRF 权重：**0.10**。

---

### 2. vector — 向量检索

**建索引：**

vector 模块是首批执行的模块之一，在 text 模块完成 chunk 落库后立即开始。它读取所有 chunk 的文本内容，调用 embedding 接口（默认 OpenAI `text-embedding-3-small`，1536 维，可通过 `.env` 切换为 Ollama `bge-m3` 1024 维或其他兼容模型），批量计算嵌入向量。每个 chunk 生成一个向量后，写入 `chunk_vector` 表。索引采用 vectorscale 扩展的 **diskann** 访问方法（`idx_diskann_embedding`），支持大规模向量近似搜索。

**数据结构：**

`chunk_vector` 表对标 pgvector 的标准用法，每条记录对应一个 chunk 的向量。除了主 embedding 列（所有模块共用的基础向量），还有两个可选的向量列：

- `embedding` — 全文向量，被 vector 召回路径使用
- `embedding_questions` — 问题向量，被 question_gen 召回路径使用
- `embedding_summary` — 摘要向量，被 summary 召回路径使用

payload 字段用 JSONB 记录 chunk 的类型和序号，便于过滤。

**召回逻辑：**

这是整个系统的主检索路径，RRF 权重最高（**0.15**）。查询时，先用同样的 embedding 模型把用户问题转为查询向量，然后在 PostgreSQL 中用 `<=>` 运算符（余弦距离）做近似最近邻搜索：

- `1 - 余弦距离` 就是余弦相似度，结果在 0 到 1 之间
- 1 表示方向完全一致，0 表示正交，越低越不相关
- 按相似度降序取 top_k

向量检索不依赖具体关键词匹配，能捕捉语义相似——比如搜"VR 看展"也能匹配到描述"沉浸式虚拟参观"的 chunk。

---

## 语义增强路（9 路可配置）

---

### 3. keyword — 关键词匹配

**建索引：**

keyword_extract 模块通过三条递进式的 Tier 链为每个 chunk 提炼关键词：

- **Tier 1（LLM 自由生成，开放集）：** 把 chunk 文本发给 LLM，要求它提取 3 到 10 个能概括段落最核心概念的关键词，返回逗号分隔的列表。这一步不限制词汇范围，任何词都可能出现。

- **Tier 2（无 LLM 精确匹配，封闭集）：** 如果知识库预定义了标签（tags），用 Python 做子串匹配，看 chunk 文本里出现了哪些标签。这一步不需要 LLM 调用，纯粹本地计算，用于兜底。

- **Tier 3（LLM 约束选择，封闭集）：** 从知识库预设的标签中，用 LLM 选出当前 chunk 最相关的标签。这一步对比 Tier 1 的开放集和 Tier 2 的精确匹配，做一个补充：预设标签虽然有限，但经过 LLM 理解语义后可能匹配到 Tier 1 遗漏的概念。

三个阶段的结果合并去重后写入 chunk_text 的 keywords 列。

**数据结构：**

keywords 列是 PostgreSQL 的变长字符串数组类型（`varchar[]`）。每个元素是一个关键词，如 `{'虚拟现实', '6DOF', '沉浸式体验'}`。这个列跟 content 在同一张表里，不需要额外的 JOIN。

**召回逻辑：**

收到查询后，KeywordRetriever 通过 PostgreSQL 的 `zhparser` 分词器（`to_tsvector('zh_cfg', ...)`）提取查询中的有效 token，过滤停用词后取前 10 个。然后对每个 token，通过 SQL 的 `unnest(c.keywords)` 展开数组，用 `ILIKE ANY` 做子串匹配——查询词"沉浸"能匹配到 keywords 里的"沉浸式体验"。所有命中的 chunk 得分为 **1.0**（相同权重），不做分数递减。这是因为 keyword 路径本身就是一个高精度的二值信号。RRF 权重：**0.08**。

---

### 4. question_gen — 问题生成检索

**建索引：**

这个模块的核心思路是"把文档翻译成问答对"。对每一个 chunk，LLM 被要求站在用户角度生成 5 个可能提问的问题。生成的问题合并成一个字符串，经过 embedding 模型转换为向量，写入 `chunk_vector.embedding_questions` 列。做向量检索时，用户的问题与这些生成的问题向量做相似度对比。

**数据结构：**

复用了 `chunk_vector` 表的两列：

- `embedding_questions` — 问题向量（1536 维，或与所选 embedding 模型一致）
- `questions` — 原始问题文本，存盘用（当前仅用于展示和调试）

这两列都允许为空，只有运行了 question_gen 模块的记录才有值。

**召回逻辑：**

只查 `embedding_questions` 不为空的记录。用 pgvector 的 `<=>` 做余弦距离排序，用户问题向量与每个 chunk 的问题向量对比，返回相似度最高的 top_k。RRF 权重：**0.10**。

这条路径的优势在于它能桥接"用户怎么问"和"文档怎么写"之间的语义鸿沟。用户不一定会用文档中的专业术语提问，但 LLM 生成的问题已经覆盖了各种提问方式。

---

### 5. summary — 摘要混合检索

**建索引：**

summarizer 模块对每个 chunk 调用 LLM 生成 50 到 200 个汉字的摘要。prompt 要求摘要覆盖原文的核心观点，剔除细枝末节。生成后，摘要文本写入 `chunk_text.summary_text` 列，摘要向量写入 `chunk_vector.embedding_summary` 列。

这里的向量不是用原文做的，而是用摘要做的——这意味着它捕捉的是 chunk "讲了什么"，而不是 chunk "原文是什么"。

**数据结构：**

两条记录分别在两张表里：

- `chunk_text.summary_text` — 摘要文本，用于 BM25 全文匹配
- `chunk_vector.embedding_summary` — 摘要向量，用于语义相似度

返回时需要回查 `chunk_text` 表获取原始 content。

**召回逻辑：**

这条路径采用混合检索策略，融合两种互补的信号：

- **BM25 分量（权重 0.3）：** PostgreSQL 的 `ts_rank` 函数，在 `summary_text` 列上做全文搜索。使用 zhparser 配置 `zh_cfg` 做分词，将查询转为 `plainto_tsquery('zh_cfg', :query)`，摘要转为 `to_tsvector('zh_cfg', ...)`，`ts_rank` 计算向量间的匹配密度。这个分量擅长精确匹配——用户搜的词如果在摘要里出现了，得分就会高。

- **向量分量（权重 0.7）：** pgvector 余弦距离，在 `query_embedding` 和 `embedding_summary` 之间做语义匹配。这个分量擅长捕捉语义相似——即使摘要里没用用户的原词，只要意思接近就能匹配到。

最终分数是 0.3 × BM25 + 0.7 × 向量分。向量权重更高，因为摘要本身就是语义浓缩的产物。RRF 权重：**0.10**。

---

### 6. raptor — 递归摘要树检索

**建索引：**

RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）是一种通过层次化聚类 + LLM 总结来构建文档摘要树的算法。执行过程分三层：

**第一层：聚类。** 从 `chunk_vector` 表加载所有 chunk 的向量，先做 UMAP 降维（从原始维度降到 12 维左右，保留局部结构），再用高斯混合模型（GMM）或凝聚层次聚类（AHC）分组。这层的作用是把语义相近的 chunk 分到同一族。

**第二层：摘要。** 对每个聚类族，把族内所有 chunk 的文本拼起来，用 LLM 生成一段摘要，并计算摘要向量。

**第三层：递归。** 把生成的摘要当作新的"chunk"（带向量），回到第一步再次聚类、摘要。每一轮产生的摘要构成树的一个新层级，直到某一层只剩 1 到 2 个节点为止。最终形成的树结构，下层是原文 chunk，上层是逐步抽象的高层摘要。

结果支持两种输出模式：`flat` 把所有摘要写入 `chunk_raptor` 表（每层摘要一行记录），`tree` 序列化为 JSON 树存入 `tree_json` 列。

**数据结构：**

`chunk_raptor` 表存储每个摘要层的信息：

- `layer` — 层级编号（0 是最底层摘要，越大越抽象）
- `title` — 摘要标题
- `summary` — LLM 生成的摘要文本
- `embedding` — 摘要的向量（直接存本表，不依赖 chunk_vector）
- `source_chunk_ids` — 这个摘要涉及哪些原始 chunk（UUID 数组）
- `cluster_id` — 聚类标识，格式为 `L{层}_C{簇编号}`
- `parent_id` — 父节点 ID（NULL=根节点）
- `tree_json` — 整棵树序列化后的 JSON（output_mode=tree 时使用）
- `tree_builder` — 树构建算法（raptor）
- `clustering_method` — 聚类算法（gmm/ahc）
- `output_mode` — 输出模式（flat/tree）

**召回逻辑：**

RAPTORRetriever 用 pgvector 的 `<=>` 找与查询向量最相似的摘要向量，返回时不区分层级（底层或顶层摘要都一起参与排序）。所有返回结果的 score 统一设为 **1.0**，因为 RAPTOR 的价值在于提供高层视角，RRF 融合时靠排名区分就够了。RRF 权重：**0.05**。

---

### 7. graph — 知识图谱检索

**建索引：**

graph_extract 模块是整条链路中最复杂、最重的后处理模块。它用 LLM 从 chunk 中提取实体和关系，流程分为四步：

**第一步：实体提取。** 对每个 chunk，按照预定义的实体类型 schema 提取命名实体。LLM 被要求返回结构化的实体列表，每个实体包含名称、类型和简短描述。

**第二步：多轮 gleaning。** 一次 LLM 调用可能漏检实体，gleaning 机制把已找到的实体传回给 LLM，要求它"检查是否还有遗漏的"——最多重复 2 轮，直到新增实体数为 0 或达上限。

**第三步：关系提取。** 对每对在同个 chunk 中出现的实体，LLM 判断它们之间存在什么关系（如"属于"、"依赖"、"位于"），带有权重。

**第四步：跨文档合并。** 通过 Merger 写入 Neo4j 图数据库。Neo4j 中每个实体是一个节点，每种关系是一条边。如果两个文档提到了同一实体，Neo4j 会做实体融合。

**数据结构：**

PostgreSQL 中的关系型存储用于 chunk 级溯源：

- `chunk_graph_entity` — 实体记录（entity_name, entity_type, description）
- `chunk_graph_relation` — 关系记录（source_entity, target_entity, relation_type, weight）

Neo4j 中的图存储用于 KB 级去重和图遍历（两阶段模型）：

- `(:Entity {name, type, aliases, kb_id})` — 实体节点（canonical 名 + 消歧别名数组）
- `(:EntityInstance {doc_id, entity_name, kb_id})` — 实例节点，`INSTANCE_OF` 指向实体
- `[:RELATION_INSTANCE {type, weight, description, kb_id}]` — 关系边（挂在实例之间）

**实体消歧（disambiguation）：** 后处理模块按类型分组 + 文本相似度预过滤 + LLM 批量裁决，把指向同一对象的名称合并：canonical 名保留在 `Entity.name`，其余名称写入 `Entity.aliases`（别名节点删除、实例归一化到 canonical）。合并历史记录在 `chunk_disambiguation` 表（审计用途）。消歧结果通过本路（graph）的别名反查生效——原独立 disambiguation 检索路已删除。

**召回逻辑：**

GraphRetriever（原 KGRetriever）直接检索 Neo4j + PostgreSQL：

1. **实体匹配：** 在 Neo4j 中做精确名匹配（`e.name IN $names OR any(a IN e.aliases WHERE a IN $names)`，含别名反查）+ CONTAINS 模糊匹配，返回命中实体实例关联的 `chunk_ids`。
2. **chunk 聚合：** 同 chunk 被多实体关联时取最高 similarity，按 score 降序截断 top_k（`_aggregate_chunks`）。
3. **原文回捞：** 按 chunk_ids 从 `chunk_text` 回捞 chunk 原文，构造 ChunkResult 参与 RRF 融合。

RRF 权重：**0.10**。

---

### 8. community — 社区发现

**建索引：**

community 模块负责从实体关系图中检测"社区"（紧密关联的实体群组）：

**第一步：构建图。** 从 `chunk_graph_entity` 和 `chunk_graph_relation` 读取所有实体和关系，构建 NetworkX 图数据结构。节点是实体，边是关系。

**第二步：社区检测。** 用 Leiden 算法做分层社区发现。Leiden 是 Louvain 的改进版，能保证检测出的社区在内部连接上是稠密的。结果分层——顶层是大社区，底层是子社区。

**第三步：报告生成。** 对每个检测到的社区，LLM 生成结构化报告：

- **title：** 社区主题（如"沉浸式VR大空间技术栈"）
- **summary：** 一段概括性描述
- **findings：** 3 到 5 条关键发现
- **rating：** 1 到 10 分的评分

**回退路径：** 如果无图谱数据（如 graph_extract 未启用），直接用 LLM 从 chunk 文本提取实体基元，用简单的相似度聚类替代社区发现，然后生成报告。

**数据结构：**

`chunk_community` 表的每条记录对应一个社区：

- `community_id` — 社区编号（如 "L0_9"）
- `summary` — 社区总结
- `findings` — JSONB 格式的关键发现列表
- `payload` — 额外的元数据

**召回逻辑：**

**向量化改造（2026-08-04）：** 旧逻辑基于 `zhparser` 分词做 token 重叠匹配（`to_tsvector('zh_cfg', ...)` + `tsvector_to_array()`，`overlap_ratio = |query_tokens ∩ text_tokens| / |query_tokens|`），且排除 KB 级社区。现改为向量检索（GraphRAG Global Search 式）：query_embedding 由上游 RewriteStage 计算并经 RetrieverContext 传入，CommunityRetriever 对 `chunk_community.embedding` 做 pgvector 余弦相似度检索：

```
score = 1 - (embedding <=> CAST(:query_embedding AS vector))
```

过滤 `embedding IS NOT NULL` 的记录，按得分降序取 top_k。KB 级社区（全零 chunk_id）与回退路径社区均可被召回（不再排除）。返回社区摘要文本（title + summary + findings 拼接，与嵌入文本同源）作为 content，chunk_id = `chunk_community.id`（uuid7 空间，不参与 chunk_text 空间的 RRF 撞键）。RRF 权重：**0.08**。

---

### 9. wiki — 百科检索

**建索引：**

wiki 模块用 LLM 从文档中提炼百科知识，构建 KB 级页面池（一页引用多个 chunk + wikilink 互链），有四阶段流水线：

**MAP 阶段：** 对整篇文档（chunks 组合原文），LLM 一次提取可以作为百科词条的概念或实体，输出 JSON 数组，每个元素包含 term（术语名）、type（概念/实体）、reason（为什么值得收录）。

**REDUCE 阶段（跨文档合并）：** 与 KB 已有页面合并——slug 精确命中直接 merge（仅追加 chunk_refs/source_refs，不改正文），未命中交 LLM 判同（existing_slug 须真实存在）。

**REFINE 阶段：** 对每个新候选词条，LLM 生成完整的百科页面 Markdown，包含概述段落、详细说明章节；正文中提及可用页面时写作 `[[slug]]`（或 `[[slug|显示名]]`）wikilink，out_links 由正文链接白名单过滤提取。

**写入阶段：** 每个页面写入 `chunk_wiki` 表，slug 恒由 term 经去除标点、空格替换下划线派生；落库同时写入 `chunk_refs`（引用 chunk 列表）、`source_refs`（贡献文档列表）、`out_links`（链接出去的 slug 列表），目标页 `in_links` 同步追加（反向链接）。

**数据结构：**

`chunk_wiki` 表的每条记录对应一个百科页面：

- `page_title` — 百科标题
- `page_slug` — URL 友好的标识
- `content` — Markdown 格式的百科正文（含 `[[slug]]` wikilink）
- `chunk_refs` — 引用 chunk UUID 列表
- `source_refs` — 贡献文档 UUID 列表
- `out_links` — 本页链接出去的 slug 列表
- `in_links` — 反向链接（被哪些 slug 链向）
- `taxonomy_path` — 分类路径（如 "技术/渲染/HDRP"）
- `status` — 发布状态（published/draft）

**召回逻辑：**

在 `page_title` 和 `content` 两列上分别做 PostgreSQL 的 `ts_rank` 全文检索，取两个 `ts_rank` 值的最大值作为该条目的分数。命中页面后按 `chunk_refs` 关联回捞全部源 chunk（INNER JOIN `chunk_text`），一页对应多个源 chunk 时逐条返回源 chunk 原文，metadata 携带 `page_title` / `page_slug` / `wiki_page_id`。RRF 权重：**0.07**。

---

### 10. faq — 常见问题检索

**建索引：**

FAQ 不走文档解析管道，而是通过独立的 FAQ 导入接口写入。导入时以 `chunk_type='faq'` 写入 `chunk_text` 表，元数据里包含标准化的问答结构。每条 FAQ 记录的 metadata 字段包含：

- `standard_question` — 标准问法
- `similar_questions` — 相似问法数组
- `negative_questions` — 负问数组，用于排除误匹配
- `answers` — 答案数组
- `category` — 分类标签

**数据结构：**

FAQ 数据存储在 `chunk_text`（`chunk_type='faq'`）和 `chunk_vector` 两张表中，没有专有的 FAQ 表。

**召回逻辑：**

FAQRetriever 采用三层匹配 + 负问过滤 + 分数提升 + 向量补充的策略：

**第一层（精确匹配，score = 1.0）：** 查询与 `standard_question` 完全一致（不区分大小写）。

**第二层（模糊匹配，score = 0.8）：** 查询出现在 `similar_questions` 数组中任一问题的 ILIKE 匹配中。

**第三层（内容匹配，score = 0.6）：** `content` 包含查询词，最宽松的兜底匹配。

**负问过滤：** 每一层都检查 `negative_questions`——如果查询与任一问法互为子串，跳过该条 FAQ。

**分数提升：** 三层匹配后，score 统一乘以 `faq_score_boost`（默认 **×1.2**）。

**向量补充（可选）：** 如果 `query_embedding` 可用，还会在 `chunk_vector` 表上做一个额外的 `<=>` 余弦距离搜索，只搜索 `chunk_type='faq'` 的记录，同样经过负问过滤，得分以 `(1 - 距离) × 1.2` 计入结果集。

FAQ 因为提供的是高置信度的直接答案，在检索结果中享有 boost 加成。RRF 权重：**0.05**。

---

## 附录：RRF 融合原理

10 路检索各自返回自己的结果列表后，RRFFuser 将它们融合为一个最终列表。

做法：对每个 ChunkResult，遍历所有路径，如果它在某一路出现了，就根据它在那一路的排名位置给它加分：

```
RRF_score = Σ weight_i / (k + rank_i)
```

- `rank_i` 从 1 开始计数（第 1 名 rank=1，第 2 名 rank=2，...）
- `k = 60` 是一个平滑常数，避免排名靠后的结果分数骤降
- `weight_i` 是每路的预设权重（见下表）

| 路径 | 权重 |
|------|------|
| vector | 0.15 |
| bm25 | 0.10 |
| question_gen | 0.10 |
| summary | 0.10 |
| graph | 0.10 |
| keyword | 0.08 |
| community | 0.08 |
| wiki | 0.07 |
| faq | 0.05 |
| raptor | 0.05 |

一个 chunk 出现得越靠前、覆盖的路径越多，RRF 分数就越高。最终按总分降序取 top_k 返回给用户。

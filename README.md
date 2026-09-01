<p align="center">
  <h1 align="center">Aion Knowledge-Kernel</h1>
  <p align="center">知识库内核 — Ingestion · Indexing · Retrieval</p>
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=fff)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18%2B-4169E1?logo=postgresql&logoColor=fff)
[![pgvector](https://img.shields.io/badge/pgvector-✓-5865F2)](https://github.com/pgvector/pgvector)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-018bff?logo=neo4j&logoColor=fff)](https://neo4j.com)
![License](https://img.shields.io/badge/License-MIT-5865F2)

</div>

<p align="center"><em>by Jimmy Zhang</em></p>

专注于 RAG 三阶段核心流程：**数据接入 → 索引构建 → 检索引擎**。  
不含多租户、权限、账号、审批等定制类业务功能。

---

## 架构

```

                                 数据接入（Ingestion）  
              多源数据接入 ──► 对象存储 + 元数据入库 ──► UnifiedContext ──► ctx_queue  
    ───────────────────────────────────────────────────────────────────────────────────────
                                         │
                                         ▼
                                 索引构建（Indexing）
        IndexingExecutor — 编排器：状态管理 → 策略执行 → 首批后处理
        ├─ 状态管理：标记 processing / completed（含 document + ingestion_task）
        ├─ ChunkingStrategy — 模板方法：按文档类型差异适配
        │   └─ execute：download → parse → clean → upload_md → prepare_chunks → assemble
        │       ├─ FAQChunkingStrategy          FAQ 文档专用（每条目 = 1 chunk）
        │       ├─ (预留：未来需特殊处理的文档类型)
        │       └─ RegularChunkingStrategy      通用类型（复用基类默认流水线）
        └─ 首批后处理（所有类型共享）
            ├─ text         → chunk_text 落库 + content_tokens BM25 索引
            ├─ vlm_caption  → 图片描述（context_above/below + caption + OCR）
            └─ vector       → chunk_vector 向量索引
                                         ▼
                                 postproc_queue 异步
                                         ▼
                        异步后处理（8 种可选模块，配置启用，所有 chunk 类型统一处理）
                        ├─ keyword_extract    关键词提取
                        ├─ question_gen       问题生成向量
                        ├─ summarizer         摘要文本+向量
                        ├─ raptor             递归摘要树
                        ├─ graph_extract      知识图谱（实体+关系→Neo4j）
                        ├─ disambiguation     实体消歧
                        ├─ community          社区检测+报告
                        └─ wiki               百科页面生成
    ───────────────────────────────────────────────────────────────────────────────────────
                                         │
                                         ▼
                                  检索（Retrieval）    
             检索管线（RetrievalPipeline） — 顺序编排 6 个 Stage
             ├── 查询改写 — LLM 改写 + 关键词/实体提取（可选，默认启用）
             ├── 多路召回 — 10 路并发检索（配置启用）
             │   ├── ① vector 向量 ANN（语义相似度）             权重 0.15    
             │   ├── ② text bm25 全文检索                      权重 0.10                     
             │   ├── ③ question_gen 问题嵌入向量搜索             权重 0.10                    
             │   ├── ④ summary 摘要混合检索                     权重 0.10                     
             │   ├── ⑤ graph 知识图谱（Neo4j 图检索）            权重 0.10                     
             │   ├── ⑥ keyword 关键词 ILIKE 匹配                权重 0.08                    
             │   ├── ⑦ community 社区报告 zhparser 分词         权重 0.08                     
             │   ├── ⑧ wiki 百科页面 BM25 检索                  权重 0.07                     
             │   ├── ⑨ faq 三层匹配                             权重 0.05                    
             │   └── ⑩ raptor RAPTOR 层级摘要向量检索            权重 0.05                    
             ├── RRF（融合+筛选）
             │   ├── 构建 rank 映射 — 各路径内 1-indexed 序号（rank 替代原始分数，消除跨路尺度差）
             │   ├── 去重合并 — 多路结果按 chunk_id 合并
             │   ├── RRF 计分 — Σ(路径权重 / (k + rank))，k=60
             │   ├── 排序 — 按 RRF 分数降序
             │   ├── 阈值过滤 — 去掉 < 30% 最高分的候选项
             │   └── 留候选池 — max(top_k × 3, 50)
             ├── Reranker 精排 — bge-reranker 重新打分
             ├── MMR 多样性重排 — embedding cosine / jieba 分词 Jaccard 兜底
             └── 最终截断 → [:top_k]
    ───────────────────────────────────────────────────────────────────────────────────────
```

---

## 环境要求

| 组件         | 版本                      |
|------------|-------------------------|
| Python     | 3.12+                   |
| PostgreSQL | 17+（zhparser + pg_textsearch + vector + vectorscale 扩展） |
| Neo4j      | 5.x+（可选，图谱支持）          |
| 对象存储       | S3 协议（原始文档、Markdown、图片） |

#### PostgreSQL 扩展

| 扩展 | 版本 | 说明 |
|------|------|------|
| pg_textsearch | 1.4.0-dev | 基于 BM25 排名的全文检索引擎 |
| vector | 0.8.6 | 向量数据类型及 ivfflat、hnsw 索引访问方法 |
| vectorscale | 0.9.0 | diskann 向量索引访问方法（vector 的补充插件，不替代 vector） |
| zhparser | 2.3 | 中文全文搜索解析器 |

### 依赖（AI 模型 / 推理服务）

| 服务         | 默认值 / 推荐模型            | 说明 |
|------------|---------------------------|------|
| Embedding  | `text-embedding-3-small`（可切换 Ollama `bge-m3`、阿里云、Jina、Azure、Gemini、智谱） | 通过 `AION_EMBEDDING_PROVIDER` 配置 |
| Reranker   | 独立部署的 `POST /rerank` 服务（如 Infinity、TEI） | 端点通过 `AION_RERANKER_ENDPOINT` 配置 |
| LLM        | `gpt-4o`（可切换阿里云 Qwen、DeepSeek、Anthropic Claude、Gemini、智谱 GLM、Ollama 及任何 OpenAI 兼容服务） | 通过 `AION_LLM_PROVIDER` 配置 |
| VLM        | `qwen-vl-plus`（图片描述专用，独立于文字 LLM） | 通过 `AION_VLM_PROVIDER` / `AION_VLM_MODEL` 配置 |
| OCR | 三级降级链路（扫描 PDF 页）：PaddleOCR → Tesseract → VLM OCR。主路径 PaddleOCR 需 `pip install -e ".[paddle-ocr]"`，模型部署方式见 [docs/ocr-deployment.md](docs/ocr-deployment.md) | Tesseract 为系统级二进制：macOS: `brew install tesseract`，Ubuntu: `apt-get install tesseract-ocr` |


---

## 启动

```bash
python -m aion_knowledge
```

默认监听 `http://localhost:19531`。如需自定义端口，通过环境变量 `AION_PORT` 或修改 `.env` 配置——启动入口当前使用固定端口，暂不支持 CLI `--port` 参数。

---

## 接口总览

API 默认地址：`http://localhost:19531`

### 数据接入

#### 支持的文件类型

| 接入类型 | 文件格式 | 扩展名 | 解析引擎 |
|---------|---------|--------|---------|
| 文档上传 | PDF | `.pdf` | builtin / MarkItDown / OpenDataLoader |
| | Word（现代/97-2003） | `.docx`、`.doc` | builtin / MarkItDown |
| | Excel | `.xlsx`、`.xls` | builtin / MarkItDown |
| | Markdown | `.md` | builtin / MarkItDown |
| | PowerPoint | `.pptx`、`.ppt`| MarkItDown |
| | CSV | `.csv` | MarkItDown |
| | EPUB | `.epub` | builtin |
| | MHTML 网页归档 | `.mhtml` | builtin |
| | 图片 | `.jpg`、`.jpeg`、`.png`、`.gif`、`.bmp`、`.tiff`、`.webp` | builtin |
| FAQ 导入 | CSV | `.csv` | FAQ 专用解析器（支持中英文表头别名） |
| | Excel | `.xlsx`、`.xls` | FAQ 专用解析器 |
| | JSON | `.json` | FAQ 专用解析器 |

> **说明：** 文档上传仅作扩展名推断，不做上传时硬性拦截；实际解析能力由后端注册的解析引擎决定。

#### POST `/api/v1/knowledge` — 创建知识库（隔离空间）

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | Body (JSON) | 知识库名称 |
| `tags` | Body (JSON), optional | 标签列表 |
| `description` | Body (JSON), optional | 描述 |

```bash
curl -X POST http://localhost:19531/api/v1/knowledge \
  -H "Content-Type: application/json" \
  -d '{"name": "我的知识库", "tags": ["技术文档", "运维"], "description": "团队技术文档汇总"}'
```

#### GET `/api/v1/knowledge` — 知识库列表

```bash
curl http://localhost:19531/api/v1/knowledge
```

#### GET `/api/v1/knowledge/{kb_id}` — 知识库详情

```bash
curl http://localhost:19531/api/v1/knowledge/<kb_id>
```

---

#### POST `/api/v1/knowledge/{kb_id}/documents/upload` — 文件上传接入

上传文档文件（PDF / DOCX / MD 等格式），触发完整的知识摄入管道：S3协议存储 → 元数据入库 → 解析 → 分块 → 后处理。

| 参数 | 类型 | 说明 |
|------|------|------|
| `kb_id` | Form | 目标知识库 ID |
| `file` | File | 上传的文件 |
| `chunk_strategy` | Form, default=`auto` | 切片策略 — `auto` 自适应（默认），`no_split` 整篇不分 |
| `creator` | Form, default=`system` | 创建者标识 |

**响应：**
```json
{
  "status": "queued",
  "context_id": "uuid",
  "document_id": "uuid"
}
```

状态 `duplicate` 表示文件已存在（按 hash 去重），`queued` 表示已入队等待处理。

```bash
curl -X POST "http://localhost:19531/api/v1/knowledge/<kb_id>/documents/upload" \
  -F "file=@文档.pdf" \
  -F "chunk_strategy=auto" \
  -F "creator=admin"
```

---

#### POST `/api/v1/knowledge/{kb_id}/documents/{document_id}/postproc/run` — 后处理重跑

对已入库文档单独执行指定的后处理二批模块（如新启用后需要补跑的模块），从 `chunk_text` 读取该文档原始 chunk，异步入队执行。常用于在 `.env` 新开启 `AION_POSTPROC_*` 后对存量文档补跑。

| 参数 | 类型 | 说明 |
|------|------|------|
| `kb_id` | Path | 目标知识库 ID |
| `document_id` | Path | 目标文档 ID |
| `modules` | Body (JSON), 必填 | 二批模块名列表（如 `raptor` / `graph_extract` / `wiki`），仅执行指定模块；首批模块（text/vector 等）不可重跑 |

**响应（201）：**
```json
{
  "status": "queued",
  "document_id": "uuid",
  "modules": ["raptor"]
}
```

**错误：**
- `404` — 文档不存在或不属于该知识库
- `400` — modules 为空 / 含未知模块 / 指定首批模块 / 模块未启用（`.env` 门控或出厂硬控未开）
- `422` — 缺少 modules 字段

```bash
# 重跑 raptor 模块
curl -X POST "http://localhost:19531/api/v1/knowledge/<kb_id>/documents/<document_id>/postproc/run" \
  -H "Content-Type: application/json" \
  -d '{"modules": ["raptor"]}'
```

---

#### POST `/api/v1/knowledge/{kb_id}/faq/import` — FAQ 批量导入

导入 FAQ 条目（CSV / XLSX / JSON 格式），直接写入 `chunk_text` 并计算向量写入 `chunk_vector`，不走完整解析管道。

| 参数 | 类型 | 说明 |
|------|------|------|
| `kb_id` | Path | 目标知识库 ID |
| `file` | File | FAQ 文件（CSV / XLSX / JSON） |
| `mode` | Form, default=`append` | 导入模式 — `append` 追加，`replace` 先清空再导入 |
| `creator` | Form, default=`system` | 创建者标识 |

**响应：**
```json
{
  "status": "queued",
  "context_id": "uuid",
  "document_id": "uuid"
}
```
状态 `duplicate` 表示文件已存在（按 hash 去重），`queued` 表示已入队等待处理。

```bash
# CSV 导入
curl -X POST "http://localhost:19531/api/v1/knowledge/<kb_id>/faq/import" \
  -F "file=@faq.csv" \
  -F "mode=append" \
  -F "creator=admin"

# JSON 导入
curl -X POST "http://localhost:19531/api/v1/knowledge/<kb_id>/faq/import" \
  -F "file=@faq.json" \
  -F "mode=replace" \
  -F "creator=admin"
```

---


### 系统

#### GET `/health` — 健康检查

```bash
curl http://localhost:19531/health
```

```json
{"status": "ok", "version": "0.1.0", "timestamp": "2026-07-18T..."}
```

---

## 检索 — 多路召回入口

RESTful `/api/v1/search` 和 MCP 工具是两种并存的检索入口，共享同一套底层引擎（10 路并发召回 → RRF 融合，详见 [docs/multi-recall.md](docs/multi-recall.md)）。

### REST API：`POST /api/v1/search`

适合上层应用使用，返回多路召回后由 LLM 生成的回答，可选流式输出。

```bash
# 基础搜索（仅返回检索结果）
curl -X POST http://localhost:19531/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是RAG", "kb_id": "<kb_id>", "top_k": 5}'

# 搜索并 LLM 生成回答
curl -X POST http://localhost:19531/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是RAG", "kb_id": "<kb_id>", "top_k": 10, "generate_answer": true}'

# 流式搜索（SSE）
curl -N -X POST http://localhost:19531/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是RAG", "kb_id": "<kb_id>", "top_k": 5, "generate_answer": true, "stream": true}'
```

**请求体（JSON）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string | — | 搜索查询（1-4096 字符） |
| `kb_id` | string | — | 知识库 ID |
| `top_k` | int | 10 | 返回结果数（1-100） |
| `path_top_k` | int | 20 | 每路预取数（1-200） |
| `enabled_paths` | string[] | null | 启用路径列表，默认全部 |
| `generate_answer` | bool | false | 是否 LLM 生成回答 |
| `stream` | bool | false | 是否流式输出（SSE） |

**响应（`generate_answer=false`）：**
```json
{
  "results": [
    {
      "chunk_id": "uuid",
      "document_id": "uuid",
      "content": "...",
      "score": 0.85,
      "source_paths": ["vector", "bm25"],
      "chunk_type": "text",
      "metadata": {}
    }
  ],
  "answer": null,
  "source_breakdown": {"vector": 20, "bm25": 15, ...},
  "path_stats": {"vector": {"results": 20, "in_final": 8}},
  "total_fused": 10,
  "query": "..."
}
```

`generate_answer=true` 时，`answer` 字段含 LLM 生成的回答文本；`stream=true` 时返回 SSE 流式响应。

### MCP（Agent 使用）

适合 Agent / AI 工具直接调用，只返回 RRF 融合后的原始结果（不含 LLM 生成）。

#### `search_knowledge` — 多路检索

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string | — | 搜索查询 |
| `kb_id` | string | — | 知识库 ID |
| `top_k` | int | 10 | 返回结果数 |

**响应：**
```json
{
  "results": [
    {
      "chunk_id": "uuid",
      "content": "...",
      "score": 0.85,
      "source_paths": ["vector", "bm25"],
      "chunk_type": "text"
    }
  ],
  "source_breakdown": {"vector": 20, ...}
}
```

### MCP 客户端配置

以下是在 MCP 客户端中配置 Aion Knowledge 服务器的方式。

> **说明：** 确保 Aion Knowledge 服务已启动（`python -m aion_knowledge`，默认端口 `19531`），MCP Streamable HTTP 端点位于 `http://localhost:19531/mcp`。

#### Aion

在项目 `~/.aion/aion.json` 中添加：

```json
{
  "mcpServers": {
    "aion-knowledge": {
      "url": "http://localhost:19531/mcp",
      "transport": "streamable-http"
    }
  }
}
```

#### DeepSeek-Harness（dsh）

dsh 不使用 JSON `mcpServers`，而是通过 Cordis overlay 以 `@deepseek-ai/dsh-mcp-client` 插件接入。本工程根目录已提供现成的 overlay 文件 [`aion-knowledge.cordis.yml`](aion-knowledge.cordis.yml)（若 dsh 与 Aion Knowledge 不在同一台机器，把 `url` 改为实际服务地址）：

```yaml
- insert:
    - id: aion-knowledge
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: aion-knowledge
        transport: streamable-http
        url: http://localhost:19531/mcp
```

**永久挂载（推荐）：** 在 dsh 所在机器上，把上述 `insert` 条目**追加**进用户 patch 层即可——手动粘贴编辑，或拷贝文件后用 `cat >>` 合并（文件不存在则新建；已有内容勿覆盖，其中可能含其他用户 patch）：

```sh
cat aion-knowledge.cordis.yml >> ~/.dsh/profiles/web/cordis.patch.yml
```

- 仅 `web` profile 生效：`$DSH_HOME/profiles/web/cordis.patch.yml`（`DSH_HOME` 默认 `~/.dsh`）
- 本机所有 profile 生效：`$DSH_HOME/cordis.patch.yml`

之后正常启动即自动挂载，无需 `--patch`，合并完成后拷贝的文件也可删除：

```sh
npx @deepseek-ai/dsh web
```

**临时单次挂载（可选）：** 仅此方式需要把 yml 文件实际保留在 dsh 所在机器上，启动时指定路径：

```sh
npx @deepseek-ai/dsh web --patch /path/to/aion-knowledge.cordis.yml
```

连接后工具以 `mcp__<serverName>__<tool>` 形式暴露，如 `mcp__aion-knowledge__search_knowledge`。若服务未启动，dsh 会在插件激活时连接失败（可用 `failOnStartupError: false` 容忍）。

#### Claude Code

在项目 `.claude/settings.local.json` 中添加：

```json
{
  "mcpServers": {
    "aion-knowledge": {
      "url": "http://localhost:19531/mcp",
      "transport": "streamable-http"
    }
  }
}
```

#### Codex（Zed）

编辑 `~/.codex/config.json`：

```json
{
  "mcpServers": {
    "aion-knowledge": {
      "url": "http://localhost:19531/mcp",
      "transport": "streamable-http"
    }
  }
}
```

**连接后可用工具：**

| 工具 | 说明 |
|------|------|
| `search_knowledge` | 多路检索，返回 RRF 融合结果 |
| `list_knowledge_bases` | 列出所有可用知识库（id + name），供 search_knowledge 的 kb_id 参数使用 |

### 统一查询流程

```
                      ┌──────────┐         ┌──────────┐
                      │ REST API │         │ MCP 工具  │
                      │ 上层应用   │         │ Agent 检索│
                      └────┬─────┘         └────┬─────┘
                           │                    │
                           └─────────┬──────────┘
                                     │
                           ╔═════════╧══════════╗
                           ║  RetrievalRouter   ║
                           ║  ├── 多路召回       ║
                           ║  ├── RRF 融合       ║
                           ║  └── Reranker 精排  ║
                           ╚═════════╤══════════╝
                                     │
                      ┌──────────────┴──────────────┐
                      │                             │
                ┌─────┴─────┐                  ┌────┴────┐
                │ 上下文装配  │                  │  直接    │
                │ 父块加载    │                  │ 返回结果  │
                │ 截断       │                  └─────────┘
                └─────┬─────┘
                      │
                ┌─────┴─────┐
                │ LLM 生成   │
                │ └ stream? │
                │   ├ 是→SSE │
                │   └ 否→str │
                └─────┬─────┘
                      │
                      ▼
                SearchResponse
```

---

## S3 对象存储

每个文档独占一个目录，原始文档、Markdown和提取的图片均在目录内：

```
{s3_bucket}/
  └── docs/
      └── {doc_name}/
          ├── original.pdf                  # 原始文档
          ├── converted.md                  # 原始文档转成的 Markdown 文件
          ├── images/
          │   ├── img_001.png               # 文档中提取的图片
          │   └── img_002.png
          └── ...
```

---


# FAQ 模块使用手册

> **版本**: 0.1.0
> **适用范围**: FAQ 作为知识库的一种 ingestion source type，支持批量导入、结构化存储、检索增强

---

## 一、概述

FAQ（Frequently Asked Questions）模块允许用户将常见问答对以结构化方式导入知识库。FAQ 条目与文档 chunks 存储在同一个 `chunk_text` 表中（`chunk_type='faq'`），接受统一的向量检索，但在检索结果中会获得额外的后处理增强：

1. **分数提升** — FAQ 结果分数 ×1.2，优先展示
2. **负问题过滤** — 排除用户 query 匹配了负问题的 FAQ 条目
3. **直接答案通道** — 高置信度 FAQ 结果直接注入上下文，跳过 LLM 生成

### 边界说明

| 维度 | FAQ 模块 | question_gen 模块 |
|------|---------|-------------------|
| 数据来源 | 用户上传的文件（CSV/Excel/JSON） | 文档 chunks 内容 |
| 处理时机 | ingestion 时（独立导入通道） | 后处理阶段（postproc） |
| 内容 | 用户预先编写的问答对 | LLM 根据 chunk 内容生成的问题 |
| chunk_type | `'faq'` | `'text'`（写入 `chunk_vector.questions` 字段） |
| 存储位置 | `chunk_text` 表（metadata 内含结构化字段） | `chunk_vector` 表的 `questions` 字段 |

---

## 二、API 接口

### POST `/api/v1/knowledge/{kb_id}/faq/import`

批量导入 FAQ 条目。

**请求参数：**

| 参数 | 位置 | 类型 | 必填 | 说明 |
|------|------|------|------|------|
| `kb_id` | path | string | 是 | 目标知识库 UUID |
| `file` | form-data | file | 是 | FAQ 文件（CSV / XLSX / JSON） |
| `mode` | form-data | string | 否 | 导入模式：`append`（默认，追加）或 `replace`（先清空再导入） |
| `creator` | form-data | string | 否 | 创建者标识（默认 `"system"`） |

**返回结果：**

```json
{
  "status": "queued",
  "context_id": "uuid",
  "document_id": "uuid"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"queued"` 表示已入队，`"duplicate"` 表示文件已存在 |
| `context_id` | string | 消息唯一标识 |
| `document_id` | string | 文档记录 ID |

> FAQ 导入采用异步处理：API 只负责将文件入队并创建文档记录，实际的解析、分块、向量化在后台 Worker 中完成。详细统计在服务端日志中可查。

**错误响应：**

| HTTP 状态码 | 说明 |
|-------------|------|
| 409 Conflict | 同一知识库正在导入 FAQ，请稍后再试 |
| 422 | 文件格式不支持或校验失败 |
| 500 | 服务器内部错误 |

---

## 三、curl 示例

### 3.1 JSON 文件导入（append 模式）

```bash
curl -X POST "http://localhost:19531/api/v1/knowledge/{kb_id}/faq/import" \
  -F "file=@faq.json" \
  -F "mode=append"
```

### 3.2 CSV 文件导入（replace 模式）

```bash
curl -X POST "http://localhost:19531/api/v1/knowledge/{kb_id}/faq/import" \
  -F "file=@faq.csv" \
  -F "mode=replace"
```

### 3.3 Excel 文件导入

```bash
curl -X POST "http://localhost:19531/api/v1/knowledge/{kb_id}/faq/import" \
  -F "file=@faq.xlsx" \
  -F "mode=append"
```

### 3.4 指定创建者

```bash
curl -X POST "http://localhost:19531/api/v1/knowledge/{kb_id}/faq/import" \
  -F "file=@faq.json" \
  -F "mode=append" \
  -F "creator=admin"
```

---

## 四、导入文件格式

### 4.1 JSON 格式

标准的 JSON 数组，每个元素是一个 FAQ 条目：

```json
[
  {
    "standard_question": "如何重置密码？",
    "similar_questions": [
      "密码忘了怎么办",
      "忘记密码"
    ],
    "negative_questions": [
      "如何删除账号"
    ],
    "answers": [
      "进入设置页面点击重置密码",
      "联系管理员重置"
    ],
    "answer_strategy": "all",
    "tags": ["账户安全"]
  },
  {
    "standard_question": "如何修改邮箱？",
    "similar_questions": ["变更邮箱地址"],
    "answers": ["在个人设置中修改邮箱"],
    "tags": ["账户安全"]
  }
]
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `standard_question` | string | 是 | 标准问题文本 |
| `similar_questions` | array[string] | 否 | 相似问法列表 |
| `negative_questions` | array[string] | 否 | 负问题列表（匹配时排除该条目） |
| `answers` | array[string] | 是 | 答案列表（至少一个） |
| `answer_strategy` | string | 否 | `"all"` 全部返回 / `"random"` 随机返回一条（默认 `"all"`） |
| `tags` | array[string] | 否 | 分类标签 |

### 4.2 CSV 格式

UTF-8 编码，**第一行为表头**，后续每行一条 FAQ 条目。

**表头列名（按名称匹配，不依赖列顺序）：**

```
分类,问题,相似问题,负问题,答案,答案策略
```

**列说明：**

| 列名 | 必填 | 说明 |
|------|------|------|
| 分类 | 否 | 标签分类 |
| 问题 | 是 | 标准问题 |
| 相似问题 | 否 | 多个用 `##` 分隔 |
| 负问题 | 否 | 多个用 `##` 分隔 |
| 答案 | 是 | 多个用 `##` 分隔 |
| 答案策略 | 否 | `all` 或 `random`（默认 `all`） |

**示例：**

```csv
分类,问题,相似问题,负问题,答案,答案策略
网络,如何连接WiFi？,WiFi连不上##无法连接WiFi,有线网络问题,打开设置连接WiFi,all
账户,如何修改密码？,忘记密码##密码忘了,,进入设置页面修改##联系管理员重置,all
账户,如何注销账号？,删除账号,,联系客服注销,all
```

**注意事项：**
- CSV 使用 **逗号** 或 **分号** 作为分隔符（自动检测）
- 支持 UTF-8 with BOM（兼容 Excel 导出）
- 分隔符自动检测规则：统计表头行中逗号和分号的数量，取多者
- 空行自动跳过
- `##` 分隔符不可变更

### 4.3 Excel 格式（.xlsx）

第一行为表头，列名与 CSV 相同：

| 分类 | 问题 | 相似问题 | 负问题 | 答案 | 答案策略 |
|------|------|---------|-------|------|---------|
| 网络 | 如何连接WiFi？ | WiFi连不上##无法连接WiFi | 有线网络问题 | 打开设置连接WiFi | all |
| 账户 | 如何修改密码？ | 忘记密码##密码忘了 | | 进入设置页面修改##联系管理员重置 | all |

**注意事项：**
- 依赖 `openpyxl` 库（默认安装）
- 不支持旧式二进制 `.xls` 格式（需先转换为 `.xlsx`）
- 支持多个 sheet，仅读取第一个

---

## 五、校验规则

导入时每条记录会经过以下校验，任何一项不通过会导致该条导入失败（不影响其他条目）：

| 规则 | 说明 |
|------|------|
| 标准问题非空 | `standard_question` 不能为空字符串 |
| 至少一个答案 | `answers` 数组不能为空 |
| 相似问题不重复 | 相似问题不能与标准问题相同 |
| 负问题不重复 | 负问题不能与标准问题或任何相似问题相同 |

---

## 六、数据存储

FAQ 条目存储在 `chunk_text` 表中，`chunk_type='faq'`：

| chunk_text 字段 | FAQ 映射 |
|----------------|----------|
| `chunk_type` | `'faq'` |
| `content` | `"Q: {标准问题}\nA: {答案1}\n- {答案2}"`（人工可读格式） |
| `metadata` | JSON，包含完整 `FAQChunkMetadata` |
| `tags` | 分类标签数组 |

**metadata JSONB 结构：**

```json
{
  "standard_question": "如何重置密码？",
  "similar_questions": ["密码忘了怎么办"],
  "negative_questions": ["如何删除账号"],
  "answers": ["进入设置页面点击重置密码", "联系管理员重置"],
  "answer_strategy": "all",
  "version": 1,
  "source": ""
}
```

向量嵌入写入 `chunk_vector` 表，默认仅嵌入**问题文本**（标准问题 + 相似问题），不含答案。可通过配置 `faq_embed_answers=true` 开启包含答案的嵌入。

---

## 七、检索行为

FAQ 与文档 chunks 在同一检索管道中，但增加以下后处理：

### 7.1 分数提升

所有 `chunk_type='faq'` 的结果，分数乘以 `faq_score_boost`（默认 `1.2`）：

```
faq_score = original_score × 1.2
```

### 7.2 负问题过滤

对每个 FAQ 条目，检查用户 query 是否匹配其 `negative_questions` 列表。匹配规则为**双向子串匹配**（query 包含负问题文本，或负问题文本包含 query），匹配则排除该条目。

### 7.3 直接答案通道

在上下文合并阶段，`score >= faq_direct_answer_threshold`（默认 `0.85`）的 FAQ 结果以 `faq_direct` 类型直接注入上下文，不经过 LLM 生成。

---

## 八、配置参数

在 `.env` 文件中配置：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `AION_FAQ_SCORE_BOOST` | `1.2` | 检索时 FAQ 分数提升系数 |
| `AION_FAQ_DIRECT_ANSWER_THRESHOLD` | `0.85` | 直接答案通道阈值 |
| `AION_FAQ_EMBED_ANSWERS` | `false` | 向量计算是否包含答案 |
| `AION_FAQ_DEFAULT_ANSWER_STRATEGY` | `all` | 默认答案策略（`all` / `random`） |

---

## 九、限制与注意事项

1. **不支持从 PDF/DOCX 自动提取 Q&A** — 当前仅支持结构化的文件格式导入
2. **不支持迭代 TopK 召回** — 低召回时不会自动扩大搜索范围
3. **不支持二级优先级标签搜索** — FAQ 标签仅用于分类，不影响检索排序
4. **不新增数据库表** — FAQ 复用 `chunk_text` 和 `chunk_vector` 表
5. **繁简转换** — 当前未启用繁简中文归一化（依赖 opencc 库，非必须）
6. **FAQ 走基础落库管道** — 导入后经过 Text 落库（写 `chunk_text`）和 Vector 向量化（写 `chunk_vector`），但**不参与**后处理可选模块（关键词提取、问题生成、图谱提取等二次加工）

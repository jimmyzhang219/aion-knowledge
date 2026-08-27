# 文档解析依赖链

本文档说明支持的各类型文档解析流程、依赖的第三方库及外部 API。

---

## 快速一览

| 文件类型 | 首选引擎 | 回退链 | 需外部工具 | 需外部 API |
|----------|----------|--------|-----------|-----------|
| PDF | `builtin` (pypdfium2) → `markitdown` → `opendataloader` → `external_pdf` | 4 引擎独立路由 | Java 11+ (opendataloader) | MinerU / PaddleOCR-VL (可选) |
| DOCX | `markitdown` → `DocxParser` (python-docx) | 2 层 | — | — |
| DOC | `markitdown` → `DocxParser` → Antiword | 3 层 | LibreOffice, Antiword | — |
| PPTX | `markitdown` + ImageMagick 后处理 | 1 层 | ImageMagick | — |
| PPT | LibreOffice → `markitdown` + ImageMagick | 1 层 | LibreOffice, ImageMagick | — |
| XLSX | pandas + openpyxl | （ExcelParser 内多步修复） | — | — |
| XLS | LibreOffice → XLSX → pandas | 2 层 | LibreOffice | — |
| EPUB | ebooklib + BeautifulSoup + markdownify | 1 层 | — | — |
| MHTML | Python email (stdlib) + BeautifulSoup + markdownify | 1 层 | — | — |
| Markdown | markitdown → 自定义 MarkdownParser | 2 层 | — | — |
| Images | Pillow → base64 inline | 1 层 | — | — |
| Web | Playwright → trafilatura → markdownify | 3 层 | Playwright (browser) | — |

---

## PDF

PDF 是支持引擎最多的格式，4 个独立引擎通过 registry 按需选择（`parser_engine` 参数）。每个引擎在内容过少时会内部回退到内置的扫描页渲染（`PDFScannedParser`）。

### 引擎 1：builtin（pypdfium2，默认）

```
PDF → pypdfium2 文本提取
    ├── 原生文本页 → 布局分析（XY 切割 → 阅读顺序重建 → Markdown 标题检测）
    │                 ├── 内嵌图片提取（按面积/重复性过滤）
    │                 └── 隐藏文本过滤（防注入）
    └── 扫描图片页 → 渲染为 JPEG → images 字典（由后续 VLM 模块做描述）
```

**依赖：** `pypdfium2`（Python），`pypdfium2.raw`（底层 C API 常量）

**说明：**
- 逐页判断是"原生文本页"还是"扫描图片页"（依据图片框面积占比）
- 原生页提取文本层 + 嵌入图形；扫描页渲染为图片
- 所有 pypdfium2 操作串行化（全局 `_PDFIUM_LOCK`），非 PDF 格式不受影响

### 引擎 2：markitdown

```
PDF → MarkItDown → Markdown + data:image URI → _extract_data_uris() → images 字典
```

**依赖：** `markitdown[pdf]`（间接依赖 `pypdf`）

**说明：** 微软 MarkItDown 库的内部转换，结果中的 base64 图片 URI 被提取为独立 `images` 字典。

### 引擎 3：opendataloader

```
PDF → opendataloader_pdf.convert() → Markdown + 图片文件
    ├── 可选 hybrid 模式（docling-fast）：连接远程 ODL hybrid 服务
    └── 结果过少时回退 → PDFScannedParser（同 builtin 扫描页渲染）
```

**依赖：** `opendataloader-pdf`（Python）+ **Java 11+**（JVM）

**配置：**

| 环境变量 | 说明 |
|----------|------|
| `AION_ODL_MAX_WORKERS` | 限制并发 JVM 实例数 |
| `AION_ODL_HYBRID` | 混合模式引擎（如 `docling-fast`） |
| `AION_ODL_HYBRID_URL` | 混合模式服务地址 |
| `AION_ODL_HYBRID_MODE` | 模式：`auto` / 其他 |
| `AION_ODL_HYBRID_FALLBACK` | 混合失败时是否回退 |
| `AION_ODL_MARKDOWN_WITH_HTML` | 输出中保留 HTML 标签 |

### 引擎 4：external_pdf（外部 API）

```
PDF → HTTP POST (multipart/form-data) → 外部服务 → JSON {content, images}
                                                   ↓ 失败/空
                                          回退 → PDFScannedParser
```

**依赖：** `httpx`（Python），需要配置 API 地址

**配置：**

| 环境变量 | 说明 |
|----------|------|
| `AION_PDF_EXTERNAL_URL` | **必填** 外部服务地址 |
| `AION_PDF_EXTERNAL_API_KEY` | 可选，Bearer Token |
| `AION_PDF_EXTERNAL_MERGE_TABLES` | 是否启用跨页表格合并 |
| `AION_PDF_EXTERNAL_TIMEOUT` | 超时秒数（默认 300） |

**兼容服务：**
- **MinerU** — 将 API server 地址填入 `AION_PDF_EXTERNAL_URL`
- **PaddleOCR-VL** — 同上（接口兼容）

> SoMark 未集成——其异步回调协议与 `ExternalPdfParser` 的同步请求/响应模型不兼容。

### pdfium 全局锁

PDFium（pypdfium2 底层 C 库）是进程全局的且非线程安全。两个并发 PDF 解析会导致死锁。所有 pdfium 操作串行化在 `_PDFIUM_LOCK` 下。非 PDF 格式的解析不受影响。

---

## DOCX

```
DOCX → Docx2Parser
         ├── 首选: MarkitdownParser → Markdown + data:image URI → images
         └── 回退: DocxParser → python-docx → 逐段/逐表遍历
                                    └── 多进程图片提取 → ImageMagick/Pillow 格式转换
```

**依赖：** `markitdown[docx]`（首选），`python-docx` + `Pillow`（回退）

**说明：**
- `Docx2Parser` 是 `FirstParser(MarkitdownParser, DocxParser)`
- 回退解析器 `DocxParser` 使用多进程（`ProcessPoolExecutor`）逐页提取图片
- `python-docx` 的黑洞关系加载 bug 通过猴子补丁修复

---

## DOC（旧版 Word）

```
DOC → DocParser
       ├── ① LibreOffice → DOCX → Docx2Parser（同上，markitdown → python-docx）
       └── ② Antiword → 纯文本提取
```

**依赖：** `LibreOffice`（soffice，系统工具）+ `Antiword`（系统工具）

**说明：**
- LibreOffice 将 `.doc` 转换为 `.docx`，转换后的内容进入 `Docx2Parser` 管道
- `DocParser` 是 `Docx2Parser` 的子类，覆盖 `parse_into_text()` 实现 chained handler
- Antiword 作为最终回退，仅提取纯文本

**查找路径：**
- LibreOffice: `/usr/bin/soffice`, `/usr/lib/libreoffice/program/soffice`, macOS `Applications` 目录，或环境变量 `LIBREOFFICE_PATH`
- Antiword: `/usr/bin/antiword`, `/usr/local/bin/antiword`，或环境变量 `ANTIWORD_PATH`

---

## PPTX / PPT

```
PPTX → MarkitdownParser
        ├── MarkItDown → Markdown
        └── attach_pptx_media_to_markdown()
              └── 从 ZIP (ppt/media/) 提取 WMF/EMF/SVG → ImageMagick/Pillow 栅格化为 PNG → images 字典

PPT  → normalize_ppt_bytes()
        ├── 已为 PPTX → 直接过
        └── 旧版 PPT → LibreOffice → PPTX → MarkitdownParser（同上）
```

**依赖：** `markitdown[pptx]`（首选），`Pillow` + `ImageMagick`（图片栅格化）

**说明：**
- PPT **没有**回退解析器 —— markitdown 是唯一条路径（区别于 DOC/DOCX 的 2-3 层回退）
- MarkItDown 输出的未解析图片引用（`![](...)`），由 `attach_pptx_media_to_markdown()` 从 ZIP 中直接提取原始媒体文件栅格化替换
- 旧版 `.ppt` 需 `LibreOffice` 转换，若不安装则抛出明确错误

---

## XLSX / XLS

```
XLSX → repair_xlsx_bytes() —— 修复缺失的 sharedStrings.xml 引用
     → fill_merged_cells_xlsx() —— 合并单元格值展开
     → pandas.ExcelFile(engine="openpyxl") → DataFrame → 键值对文本

XLS  → detect_excel_format() → xlrd (opens pandas)
       ├── 自动检测扩展名（xlsx/xls/xlsb/ods）
       └── 无法识别的 → LibreOffice → XLSX
```

**依赖：** `pandas`（核心）, `openpyxl`（XLSX 引擎）, `xlrd`（XLS 引擎）, `LibreOffice`（格式转换）

**说明：**
- `repair_xlsx_bytes()` 处理缺少 sharedStrings.xml 但仍可读的损坏 XLSX
- `LibreOffice` 作为通用转换器：处理 `ods`、`xlsb`、`et`（WPS）等非常见格式
- `python-calamine` 可作为备用 pandas 引擎（当 openpyxl 打开失败时）

---

## EPUB

```
EPUB → ebooklib → 章节 HTML
       ├── BeautifulSoup(lxml) 解析
       ├── markdownify → Markdown
       └── 内嵌图片提取 → base64 → images 字典
```

**依赖：** `ebooklib`, `beautifulsoup4`, `lxml`, `markdownify`

---

## MHTML（Web 归档）

```
MHTML → Python email.message_from_bytes() → MIME 解包
        ├── text/html 部分 → BeautifulSoup(lxml) → 清理 → markdownify → Markdown
        └── image/* 部分 → base64 → images 字典
```

**依赖：** 标准库 `email`, `beautifulsoup4`, `lxml`, `markdownify`

---

## 图片

```
Image → Pillow 格式检查 → base64 编码 → Markdown ![](ref) → images 字典
```

**支持格式：** `jpg`, `jpeg`, `png`, `gif`, `bmp`, `tiff`, `webp`

**依赖：** `Pillow`

---

## Web 页面

```
URL → Playwright (WebKit) → 渲染 JS → HTML
       ├── trafilatura.extract() → Markdown（首选）
       ├── 失败 → Playwright 可见文本回退
       └── 再失败 → 原始 HTML textContent
```

**依赖：** `playwright`（安装需 `playwright install webkit`）, `trafilatura`, `lxml`

**说明：**
- 使用 WebKit 浏览器引擎渲染 JavaScript 单页应用（SPA）
- 内建对微信公众号文章的兼容（通过 monkey patch trafilatura 的 `IMAGE_EXTENSION` 和 `BODY_XPATH`）
- `_SPA_MIN_TEXT_LEN` (80 字符) 判断 SPA 是否完成渲染

---

## Markdown

```md
MD → MarkitdownParser → MarkdownParser（自定义拆分等逻辑）
      ├── 首选 markitdown → 文本
      └── 回退 → MarkdownParser（纯文本直接透传）
```

**依赖：** `markitdown`（首选）

---

## 系统级依赖汇总

| 工具 | 用途 | 对应文件类型 | 安装方式 |
|------|------|-------------|---------|
| **LibreOffice** (soffice) | 旧版格式转换 | DOC, PPT, XLS, XLSB, ODS | `apt install libreoffice` 或 macOS DMG |
| **Antiword** | DOC 纯文本回退 | DOC | `apt install antiword` |
| **ImageMagick** (convert) | WMF/EMF/SVG 栅格化 | PPTX, DOCX | `apt install imagemagick` |
| **Java 11+** | OpenDataLoader JVM | PDF (opendataloader) | `apt install openjdk-17-jre-headless` |
| **Playwright 浏览器** (webkit) | JS 渲染 | Web | `playwright install webkit` |

> 部署时建议使用 Docker 统一管理这些系统依赖。

---

## 配置参考

解析相关配置统一在 `src/aion_knowledge/common/config.py` 中，以 `AION_` 为前缀的环境变量驱动。详见文件中的 Settings 类定义。

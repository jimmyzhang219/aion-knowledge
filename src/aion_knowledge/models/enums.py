"""知识内核中使用的枚举类型。"""

from __future__ import annotations

import enum


class DocumentStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    finalizing = "finalizing"
    completed = "completed"
    failed = "failed"


class ChunkType(str, enum.Enum):
    """知识切片类型。"""

    # ── pipeline 基础分类（parsing 阶段直接产出） ──
    text = "text"          # → 存 chunk_text
    table = "table"        # → 存 chunk_text
    image = "image"        # → 存 chunk_text

    # ── 层级检索 ──
    parent = "parent"      # → 存 chunk_text（RAPTOR 父切片）

    # ── 图像衍生（预留，暂未写入） ──
    image_ocr = "image_ocr"
    image_caption = "image_caption"

    # ── 知识库派生 ──
    faq = "faq"            # → 存 chunk_text
    summary = "summary"    # 预留
    entity = "entity"      # 预留
    relationship = "relationship"  # 预留
    kg = "kg"              # → 存 chunk_text


class IngestionTaskStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class DocType(str, enum.Enum):
    pdf = "pdf"
    docx = "docx"
    md = "md"
    html = "html"
    epub = "epub"
    xlsx = "xlsx"
    pptx = "pptx"
    image = "image"
    audio = "audio"
    video = "video"
    email = "email"
    code = "code"
    other = "other"


class ChunkStrategy(str, enum.Enum):
    """文档分块策略。"""
    auto = "auto"             # 交由切分工具动态处理（默认）
    no_split = "no_split"     # 强制不切分，整个文档作为一个完整 chunk
    heading = "heading"       # 标题感知分块（Tier1）
    heuristic = "heuristic"   # 启发式分块（Tier2）
    recursive = "recursive"   # 递归分块（Tier3）


class StrategyName(str, enum.Enum):
    """接入/索引策略名称，ingestion 与 indexing 两层共用。"""
    faq = "faq"
    regular = "regular"
    url_import = "url_import"
    manual_entry = "manual_entry"
    connector_feishu = "connector_feishu"   # 预留
    connector_notion = "connector_notion"   # 预留
    connector_yuque = "connector_yuque"     # 预留
    connector_rss = "connector_rss"         # 预留
    api_direct = "api_direct"               # 预留

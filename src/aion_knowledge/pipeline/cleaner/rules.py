"""清洗规则常量 — 正则、字符集、配置。"""

import re

# ── Step 1：字符级净化 ──────────────────────────────────────────────

# 零宽字符（Unicode 格式控制字符，不占据视觉空间）
ZERO_WIDTH_CHARS = "​‌‍﻿"

# 控制字符（保留 \n \t \r）
CONTROL_CHAR_RE = re.compile(r"[\x00---]")

# 全角 ASCII 偏移量：全角 U+FF01-U+FF5E → 半角 U+0021-U+007E
FULLWIDTH_OFFSET = 0xff01 - 0x0021
FULLWIDTH_START = 0xff01
FULLWIDTH_END = 0xff5e

# ── Step 2：空白字符规整 ──────────────────────────────────────────

# 统一换行符：\r\n → \n, \r → \n
NEWLINE_RE = re.compile(r"\r\n|\r")

# 行尾空白
TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)

# 连续 ≥3 空行 → 2 空行（匹配4个以上换行）
COLLAPSE_BLANKS_RE = re.compile(r"\n{4,}")

# ── Step 3：Markdown 格式归一 ─────────────────────────────────────

# 标题行（atx 风格 # ~ ######）
HEADING_RE = re.compile(r"^(#{1,6})\s+.*$", re.MULTILINE)

# 代码块围栏
FENCED_CODE_RE = re.compile(r"^```", re.MULTILINE)

# 列表标记（-、*、+、数字.）
LIST_ITEM_RE = re.compile(r"^(?:[-*+]|\d+\.)\s+", re.MULTILINE)

# ── Step 4：文档噪声过滤 ─────────────────────────────────────────

# 纯页码行（全行匹配，不含其他正文）
PAGE_NUMBER_RE = re.compile(
    r"^(?:"
    r"\s*[-–—]\s*\d+\s*[-–—]\s*$"                              # - 123 -
    r"|Page\s+\d+\s+of\s+\d+\s*$"                                 # Page 5 of 10
    r"|第\s*\d+\s*页\s*$"                                         # 第 3 页
    r"|p[\.:]?\s*\d+\s*$"                                         # p.5 / p:5
    r")",
    re.MULTILINE | re.IGNORECASE,
)

# 纯分隔线（≥20 个连续的 - 或 _）
SEPARATOR_RE = re.compile(r"^[-_]{20,}\s*$", re.MULTILINE)

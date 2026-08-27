"""标记分块 —— 基于文档结构的边界信号进行分块。

检测结构化边界（分页符、编号标题、章节关键词、全大写标题、
视觉分隔线、页脚、过多空行）进行贪婪打包分块。
无边界时回退到递归分块。

=== 边界类型与优先级 ===

``_Boundary`` 按优先级排序，同位置保留最高优先级：

==== ====================== ==== ===================
优先  边界类型                 源     检出方式
==== ====================== ==== ===================
100  分页符 ``\\f``              字符  直接遍历
90   编号标题 "1.1 简介"        正则 ``NUMBERED_SECTION_RE``
85   章节关键词 "Kapitel 1"     正则 ``CHAPTER_RE[lang]``
70   全大写短标题 "INTRODUCTION" 正则 ``ALL_CAPS_HEADING_RE``
60   视觉分隔线 "---"           正则 ``VISUAL_SEPARATOR_RE``
50   页脚 "Seite 1 von 10"      正则 ``PAGE_FOOTER_RE``
40   过多空行（3+ 连续）        正则 ``EXCESSIVE_BLANKS_RE``
==== ====================== ==== ===================

=== 分块策略 ===

采用**贪婪打包**方式（``_pack_chunks``）：

1. 边界间距离 < chunk_size → 继续累积（不切分）
2. 累积内容达到 chunk_size → 在当前位置切分
3. **单一块超过 chunk_size** → 下放 ``RecursiveSplitter`` 做递归分割
   （保证绝不产生超大 chunk）
4. 尾部不足 chunk_size 的剩余内容 → 作为最后一个 chunk 输出
"""

import logging
import re
from typing import List, Tuple

from aion_knowledge.pipeline.chunker.base import ChunkConfig, ChunkResult
from aion_knowledge.pipeline.chunker.recursive_splitter import RecursiveSplitter
from aion_knowledge.pipeline.chunker.text_utils import (
    ALL_CAPS_HEADING_RE,
    CHAPTER_RE,
    EXCESSIVE_BLANKS_RE,
    NUMBERED_SECTION_RE,
    PAGE_FOOTER_RE,
    VISUAL_SEPARATOR_RE,
    count_tokens,
)

logger = logging.getLogger(__name__)


class _Boundary:
    """候选切分边界。

    ``offset`` 为边界在文本中的字符偏移量，``priority`` 用于同位置去重
    （保留最高优先级）。``MarkerSplitter`` 在收集所有候选边界后按其偏移
    排序，然后贪婪打包成 chunk。
    """

    __slots__ = ("offset", "priority")

    def __init__(self, offset: int, priority: int):
        self.offset = offset
        self.priority = priority


# 边界优先级
_BOUNDARY_FORM_FEED = 100
_BOUNDARY_NUMBERED_HEAD = 90
_BOUNDARY_CHAPTER_MARKER = 85
_BOUNDARY_ALL_CAPS_HEADING = 70
_BOUNDARY_VISUAL_SEP = 60
_BOUNDARY_PAGE_FOOTER = 50
_BOUNDARY_BLANK_BLOCK = 40


class MarkerSplitter:
    """基于结构标记的分块器 —— 检测文档结构边界进行切分。

    适用于没有 Markdown 标题但有编号章节 / 章节关键词 / 分页符等结构的文档
    （PDF 转换文本、扫描识别结果、老旧的技术文档）。

    边界检测基于正则，支持多语言（de / en / zh），通过 ``ChunkConfig.languages``
    控制启用的语言模式。

    所有边界在 ``_find_boundaries`` 中收集，经 ``_deduplicate`` 去重后
    进入 ``_pack_chunks`` 贪婪打包。单一块超大的会自动下放递归分块。
    完全无边界时回退到 ``RecursiveSplitter``。
    """

    def __init__(self, config: ChunkConfig):
        self.config = config
        self._fallback = RecursiveSplitter(config)
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[Tuple[str, int, re.Pattern[str]]]:
        """编译结构标记的正则模式。"""
        patterns: List[Tuple[str, int, re.Pattern[str]]] = []

        # 编号章节
        patterns.append(
            ("numbered", _BOUNDARY_NUMBERED_HEAD, re.compile(NUMBERED_SECTION_RE, re.MULTILINE))
        )
        # 全大写标题
        patterns.append(
            ("allcaps", _BOUNDARY_ALL_CAPS_HEADING, re.compile(ALL_CAPS_HEADING_RE, re.MULTILINE))
        )
        # 视觉分隔线
        patterns.append(
            ("visualsep", _BOUNDARY_VISUAL_SEP, re.compile(VISUAL_SEPARATOR_RE, re.MULTILINE))
        )
        # 页脚
        patterns.append(
            ("footer", _BOUNDARY_PAGE_FOOTER, re.compile(PAGE_FOOTER_RE, re.MULTILINE))
        )

        # 多语言章节关键词
        langs = self.config.languages or ["de", "en", "zh"]
        for lang in langs:
            if lang in CHAPTER_RE:
                patterns.append(
                    (
                        "chapter_" + lang,
                        _BOUNDARY_CHAPTER_MARKER,
                        re.compile(CHAPTER_RE[lang], re.MULTILINE),
                    )
                )

        return patterns

    def split(self, text: str) -> List[ChunkResult]:
        """执行结构标记分块。"""
        if not text:
            return []

        bounds = self._find_boundaries(text)
        if not bounds:
            return self._fallback.split(text)

        bounds = self._deduplicate(bounds)

        if bounds[0].offset != 0:
            bounds.insert(0, _Boundary(0, 999))
        bounds.append(_Boundary(len(text), 0))

        return self._pack_chunks(text, bounds)

    def _find_boundaries(self, text: str) -> List[_Boundary]:
        """扫描文本，返回所有候选边界。"""
        bounds: List[_Boundary] = []

        # FormFeed 分页符
        for i, ch in enumerate(text):
            if ch == "\f":
                bounds.append(_Boundary(i, _BOUNDARY_FORM_FEED))

        # 行级正则模式
        for _name, priority, pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                bounds.append(_Boundary(match.start(), priority))

        # 过多空行
        for match in re.finditer(EXCESSIVE_BLANKS_RE, text):
            bounds.append(_Boundary(match.end(), _BOUNDARY_BLANK_BLOCK))

        return bounds

    def _deduplicate(self, bounds: List[_Boundary]) -> List[_Boundary]:
        """同位置保留最高优先级。"""
        if not bounds:
            return []

        bounds.sort(key=lambda b: (b.offset, -b.priority))
        deduped = [bounds[0]]
        for b in bounds[1:]:
            if b.offset != deduped[-1].offset:
                deduped.append(b)
        return deduped

    def _pack_chunks(self, text: str, bounds: List[_Boundary]) -> List[ChunkResult]:
        """贪婪打包：边界间累积到 chunk_size 时切分。"""
        chunks: List[ChunkResult] = []
        seq = 0
        chunk_start = bounds[0].offset

        for i in range(1, len(bounds)):
            prev_offset = bounds[i - 1].offset
            block_end = bounds[i].offset

            block_text = text[prev_offset:block_end]
            block_tokens = count_tokens(block_text)

            # 单一块超过 chunk_size → 下放递归分块
            if block_tokens > self.config.chunk_size:
                sub_text = text[chunk_start:block_end]
                sub_chunks = self._fallback.split(sub_text)
                for sc in sub_chunks:
                    sc.chunk_id = f"chunk_{seq}"
                    sc.seq_num = seq
                    seq += 1
                    chunks.append(sc)
                chunk_start = block_end
                continue

            accum_text = text[chunk_start:block_end]
            accum_tokens = count_tokens(accum_text)

            if accum_tokens > self.config.chunk_size and block_tokens > 0:
                content = text[chunk_start:block_end]
                t_count = count_tokens(content)
                if content.strip():
                    chunks.append(ChunkResult(
                        content=content,
                        token_count=t_count,
                        chunk_id=f"chunk_{seq}",
                        seq_num=seq,
                    ))
                    seq += 1
                chunk_start = block_end

        # 剩余内容
        if chunk_start < len(text):
            content = text[chunk_start:]
            t_count = count_tokens(content)
            if content.strip():
                chunks.append(ChunkResult(
                    content=content,
                    token_count=t_count,
                    chunk_id=f"chunk_{seq}",
                    seq_num=seq,
                ))

        return chunks

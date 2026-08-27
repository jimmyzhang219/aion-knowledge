"""文档结构分析 —— 单遍扫描提取结构特征，辅助分块策略选择。

输出 ``DocumentFeatures`` 对象，供 ``Chunker._select_splitters()`` 决策。

=== 提取的特征维度 ===

1. **Markdown 标题体系**（``md_heading_counts`` / ``md_heading_total``）：
   - 正则检测 ``# 到 ######`` 六级标题
   - 跳过代码围栏内的 ``#``（防止误判）
   - ``dominant_heading_level``：出现次数 >= 3 的最高层级，若无则返回最低有值层级
   - ``heading_density``：标题数 / 总行数，密度阈值 0.005 用于 auto 策略判断

2. **结构化边界信号**（汇总为 ``structural_marker_total``）：
   - ``numbered_section_count``：编号标题，如 "1.1 简介"、"IV. Results"
   - ``all_caps_short_line_count``：全大写短行（4-80 字符），如 "INTRODUCTION"
   - ``german_chapter_count`` / ``english_chapter_count`` / ``chinese_chapter_count``：
     多语言章节关键词，"Kapitel 1" / "Chapter 2" / "第3章"
   - ``visual_sep_count``：视觉分隔线 --- / === / ***
   - ``form_feed_count``：换页符 \\f

3. **内容类型**：
   - ``has_tables``：检测 Markdown 表格 ``| ... |``
   - ``has_code``：检测围栏代码块

4. **页脚过滤**：
   - ``repeated_footer_count``：匹配 "Seite 1 von 10" / "Page 3 of 5" / "页码 5" 等

5. **语言检测**（``detected_langs``）：
   - 依赖多语言章节关键词匹配结果
"""

import logging
import re
from typing import Dict, List

from aion_knowledge.pipeline.chunker.text_utils import (
    ALL_CAPS_HEADING_RE,
    CHAPTER_RE,
    MARKDOWN_HEADING_RE,
    NUMBERED_SECTION_RE,
    PAGE_FOOTER_RE,
    VISUAL_SEPARATOR_RE,
)

logger = logging.getLogger(__name__)


class DocumentFeatures:
    """文档结构特征集 —— 单遍扫描的结果容器。

    由 ``DocumentAnalyzer.analyze()`` 填充，供 ``Chunker._select_splitters()``
    读取以决定分块策略。

    关键导出属性：
    - ``heading_density``：标题行占比，auto 策略使用阈值 ``0.005``
    - ``dominant_heading_level``：主导标题层级（1-6），决定 heading splitter 的切分锚点
    - ``structural_marker_total``：所有非标题结构标记的总数，auto 策略使用阈值 ``5``
    """

    def __init__(self) -> None:
        self.total_chars: int = 0
        self.total_lines: int = 0
        self.avg_line_len: float = 0.0
        self.md_heading_counts: Dict[int, int] = {}
        self.md_heading_total: int = 0
        self.numbered_section_count: int = 0
        self.all_caps_short_line_count: int = 0
        self.form_feed_count: int = 0
        self.visual_sep_count: int = 0
        self.german_chapter_count: int = 0
        self.english_chapter_count: int = 0
        self.chinese_chapter_count: int = 0
        self.repeated_footer_count: int = 0
        self.has_tables: bool = False
        self.has_code: bool = False
        self.detected_langs: List[str] = []

    @property
    def heading_density(self) -> float:
        if self.total_lines == 0:
            return 0.0
        return self.md_heading_total / self.total_lines

    @property
    def dominant_heading_level(self) -> int:
        """返回文档的主导标题层级（1-6），无标题时返回 0。

        判定规则：
        1. 从 h1 → h6 遍历，**首个出现 >= 3 次** 的层级 → 主导层级
        2. 若没有任何层级 >= 3 次，返回 **最低的有值层级**（h6 → h1 遍历）
        3. 完全无标题返回 0

        设计意图：优先选择高优先级标题（h1 > h2 > ...）作为切分锚点。
        例如 h1 出现 3 次且 h2 出现 10 次，则 h1 被选为主宰层级 ——
        因为 h1 代表更宏观的结构边界。但若 h1 只出现 2 次（不够 3），
        则降级到 h2。
        """
        if not self.md_heading_counts:
            return 0
        # 规则 1：从 h1 → h6，首个 >= 3 的层级
        for level in range(1, 7):
            if self.md_heading_counts.get(level, 0) >= 3:
                return level
        # 规则 2：没有任何层级 >= 3，返回最低的有值层级（h6 → h1）
        for level in range(6, 0, -1):
            if self.md_heading_counts.get(level, 0) > 0:
                return level
        return 0

    @property
    def structural_marker_total(self) -> int:
        """文档中结构标记的总数。"""
        return (
            self.numbered_section_count
            + self.german_chapter_count
            + self.english_chapter_count
            + self.chinese_chapter_count
            + self.all_caps_short_line_count
            + self.visual_sep_count
            + self.form_feed_count
        )


class DocumentAnalyzer:
    """文档分析器 —— 单遍扫描收集结构指标。

    使用正则扫描文本，统计标题、编号章节、章节关键词、全大写标题、
    视觉分隔符、页脚、表格、代码围栏等指标。

    注意：扫描是**惰性**的 —— ``analyze()`` 每次被调用都会重新扫描全文，
    不会缓存结果。对于超大文档（> 10MB），外部应考虑缓存。

    代码围栏（`````）处理：
    - 围栏内的 ``#`` 不会被识别为标题（防止 Markdown 代码示例中的误判）
    - ````` 行本身不计入结构标记
    """

    def __init__(self) -> None:
        self._heading_re = re.compile(MARKDOWN_HEADING_RE, re.MULTILINE)
        self._numbered_re = re.compile(NUMBERED_SECTION_RE, re.MULTILINE)
        self._allcaps_re = re.compile(ALL_CAPS_HEADING_RE, re.MULTILINE)
        self._visual_sep_re = re.compile(VISUAL_SEPARATOR_RE, re.MULTILINE)
        self._footer_re = re.compile(PAGE_FOOTER_RE, re.MULTILINE)
        self._chapter_patterns = {
            "de": re.compile(CHAPTER_RE["de"], re.MULTILINE),
            "en": re.compile(CHAPTER_RE["en"], re.MULTILINE),
            "zh": re.compile(CHAPTER_RE["zh"], re.MULTILINE),
        }

    def analyze(self, text: str) -> DocumentFeatures:
        """执行文档结构分析，返回特征集。"""
        features = DocumentFeatures()
        if not text:
            return features

        features.total_chars = len(text)
        features.form_feed_count = text.count("\f")

        lines = text.split("\n")
        features.total_lines = len(lines)
        features.avg_line_len = features.total_chars / max(features.total_lines, 1)

        heading_counts: Dict[int, int] = {}
        in_fence = False

        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith("```"):
                in_fence = not in_fence
                if in_fence:
                    features.has_code = True
                continue
            if in_fence:
                continue

            heading_match = self._heading_re.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_counts[level] = heading_counts.get(level, 0) + 1
                continue

            if self._numbered_re.match(line):
                features.numbered_section_count += 1
            elif self._allcaps_re.match(line):
                features.all_caps_short_line_count += 1
            elif self._visual_sep_re.match(line):
                features.visual_sep_count += 1
            elif self._footer_re.match(line):
                features.repeated_footer_count += 1

            if not trimmed.startswith("#"):
                for lang, pat in self._chapter_patterns.items():
                    if pat.match(line):
                        if lang == "de":
                            features.german_chapter_count += 1
                        elif lang == "en":
                            features.english_chapter_count += 1
                        elif lang == "zh":
                            features.chinese_chapter_count += 1
                        features.detected_langs.append(lang)
                        break

            if "|" in line and re.match(r"^\|.*\|$", trimmed):
                features.has_tables = True

        features.md_heading_counts = heading_counts
        features.md_heading_total = sum(heading_counts.values())
        features.detected_langs = list(dict.fromkeys(features.detected_langs))

        return features

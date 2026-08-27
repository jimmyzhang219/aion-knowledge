"""标题感知分块 —— 基于 Markdown 标题层级切分文档。

维护标题堆栈生成面包屑路径，在主导层级标题处切分，
超长章节下放到递归分块处理。

=== 分块规则 ===

1. **边界检测**（``_find_heading_boundaries``）：
   - 扫描全文的 Markdown 标题（``#`` ~ ``######``）
   - 跳过代码围栏（`````）内的标题行
   - 每个标题行成为一个候选边界

2. **章节切分**（``_split_at_boundaries``）：
   - 在标题边界处切分（该标题归属到其后的内容段）
   - 每段维护一个 ``HeadingTracker`` 堆栈生成面包屑路径
   - **超长章节处理**：如果单个章节的 token 数 > 1.5 × chunk_size，
     则用 ``RecursiveSplitter`` 对该章节进一步细切（子 chunk 继承父标题路径）

3. **碎片合并**（``merge_adjacent_chunks``）：
   - 将相同标题路径下过小的连续 chunk 合并（阈值 0.5 × chunk_size）
   - 详情见 ``fragment_utils.merge_adjacent_chunks()``

4. **无标题回退**：
   - 如果全文找不到任何标题边界，直接回退到 ``RecursiveSplitter``

=== 面包屑路径 ===

``HeadingTracker`` 维护一个标题堆栈。遇到新标题时：
- 同层级或更低层级 → 弹出直到栈顶层级 < 新标题层级
- 高一级（更深） → 直接入栈
- 例如：``# 概论`` → ``## 方法`` → ``### 实验`` → 面包屑 = "概论 > 方法 > 实验"
"""

import logging
import re
from typing import List, Tuple

from aion_knowledge.pipeline.chunker.base import ChunkConfig, ChunkResult
from aion_knowledge.pipeline.chunker.fragment_utils import merge_adjacent_chunks
from aion_knowledge.pipeline.chunker.recursive_splitter import RecursiveSplitter
from aion_knowledge.pipeline.chunker.text_utils import MARKDOWN_HEADING_RE, count_tokens

logger = logging.getLogger(__name__)

HEADING_PATTERN = re.compile(MARKDOWN_HEADING_RE, re.MULTILINE)


class HeadingTracker:
    """维护 Markdown 标题堆栈，生成面包屑路径。

    用于 ``HeadingSplitter`` 的章节切分过程，每遇到一个标题就更新堆栈状态，
    最终可以生成如 ``"概论 > 方法 > 实验设计"`` 的面包屑路径。

    堆栈维护规则（与 HTML 的 DOM 标题层级逻辑一致）：
    - 遇到 h1 → 清空栈，入栈 (1, h1_text)
    - 遇到 h2 → 弹出栈顶直到栈空或栈顶层级 < 2，然后入栈
    - 遇到 h3 → 弹出 h2/h3，保留 h1，入栈
    - 以此类推

    用法::

        tracker = HeadingTracker()
        tracker.observe("# 概述")   # 栈: [(1, "概述")]
        tracker.observe("## 方法")  # 栈: [(1, "概述"), (2, "方法")]
        tracker.observe("### 实验") # 栈: [(1, "概述"), (2, "方法"), (3, "实验")]
        tracker.breadcrumb()        # → "概述 > 方法 > 实验"
    """

    def __init__(self) -> None:
        self.stack: List[Tuple[int, str]] = []

    def observe(self, heading_line: str) -> None:
        """观察一个标题行，更新层级堆栈。"""
        match = HEADING_PATTERN.match(heading_line)
        if not match:
            return
        level = len(match.group(1))
        text = match.group(2).strip()
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))

    def breadcrumb(self, separator: str = " > ") -> str:
        """返回当前面包屑路径。"""
        if not self.stack:
            return ""
        return separator.join(text for _, text in self.stack)

    def reset(self) -> None:
        self.stack = []


class HeadingSplitter:
    """标题感知分块器 —— 基于 Markdown 标题层级做结构化切分。

    适用于有清晰 Markdown 标题体系的文档（技术文档、Wiki、手册）。
    每个标题层级内的内容作为一个 chunk，维护面包屑路径供检索使用。
    超长章节自动下放递归分块，处理后合并同路径下的微 chunk。

    使用条件（由 ``Chunker._select_splitters()`` 判断）：
    - 至少 3 个 Markdown 标题
    - 标题密度 >= 0.005
    - ``strategy="heading"`` 或 ``strategy="auto"`` 且满足上述条件
    """

    COALESCE_TARGET = 0.5

    def __init__(self, config: ChunkConfig):
        self.config = config
        self._fallback = RecursiveSplitter(config)

    def split(self, text: str) -> List[ChunkResult]:
        if not text:
            return []
        boundaries = self._find_heading_boundaries(text)
        if len(boundaries) <= 1:
            return self._fallback.split(text)
        chunks = self._split_at_boundaries(text, boundaries)
        chunks = merge_adjacent_chunks(chunks, self.config.chunk_size, self.COALESCE_TARGET)
        return chunks

    def _find_heading_boundaries(self, text: str) -> List[Tuple[int, str, int]]:
        """查找所有标题边界。"""
        boundaries: List[Tuple[int, str, int]] = [(0, "", 0)]
        for match in HEADING_PATTERN.finditer(text):
            line = match.group(0)
            level = len(match.group(1))
            prefix = text[:match.start()]
            fence_count = prefix.count("```")
            in_fence = fence_count % 2 == 1
            if not in_fence:
                boundaries.append((match.start(), line, level))
        return boundaries

    def _split_at_boundaries(
        self, text: str, boundaries: List[Tuple[int, str, int]]
    ) -> List[ChunkResult]:
        chunks: List[ChunkResult] = []
        hierarchy = HeadingTracker()
        seq = 0
        for i in range(len(boundaries)):
            start_offset = boundaries[i][0]
            end_offset = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            section_text = text[start_offset:end_offset]
            if not section_text.strip():
                continue
            if boundaries[i][1]:
                hierarchy.observe(boundaries[i][1])
            heading_path = hierarchy.breadcrumb()
            section_tokens = count_tokens(section_text)
            if section_tokens > int(self.config.chunk_size * 1.5):
                sub_chunks = self._fallback.split(section_text)
                for sc in sub_chunks:
                    sc.heading_path = heading_path
                    sc.chunk_id = f"chunk_{seq}"
                    sc.seq_num = seq
                    seq += 1
                    chunks.append(sc)
            else:
                chunks.append(ChunkResult(
                    content=section_text,
                    token_count=section_tokens,
                    heading_path=heading_path,
                    chunk_id=f"chunk_{seq}",
                    seq_num=seq,
                ))
                seq += 1
        return chunks

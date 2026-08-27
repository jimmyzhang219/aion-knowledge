"""递归分块 —— 基于分隔符优先级递归分割（安全网）。

永不分块失败，始终返回结果。作为所有其他分块器的最终兜底方案。

=== 递归分割算法 ===

``_split()`` 按 ``ChunkConfig.separators`` 列表的优先级逐步降级：

1. 如果当前段 token 数 <= chunk_size → 保留，不再分割
2. 按下一个分隔符尝试分割：
   a. ``"\\n\\n"``（段落分割）：按段落边界切分，分隔符保留在后续片段开头
   b. ``"\\n"``（行分割）：按行切分
   c. ``"。＂（句号分割）：按句子切分（中文文本）
   d. ``"；"``（分号分割）
   e. ``"，"``（逗号分割）
   f. ``""``（字符分割）：最终降级到字符级拆分
3. 每次切分均考虑保护区域（跳过落在代码块/表格/公式内的切分点）
4. 一旦所有片段都 <= chunk_size → 提前停止，不再尝试更低级的分隔符

=== 超大表格处理 ===

``_split_oversized_table()`` 处理 Markdown 表格这种特殊保护区域：
- 受保护的表格无法被任何分隔符切开（因为表格内部都是 "|" 分隔）
- 字符级拆分又会把表格拆得粉碎
- 此方法在字符级拆分前插入：识别表格结构，按行边界贪心分割
- **每段续片前置表头行 + 分隔行**，保证每个续片都能独立解析

=== 合并与重叠 ===

``_merge()`` 将分割后的片段合并为最终 chunk：
1. 贪心打包：累积片段直到超过 chunk_size，然后输出为一个 chunk
2. **语义重叠**：如果配置了 ``chunk_overlap > 0``，每个新 chunk 开头
   会包含上一个 chunk 尾部的内容（``extract_overlap_tail`` 提取，换行符对齐）
3. ``chunk_overlap = 0`` 时不做重叠处理
"""

import logging
from typing import List, Optional

from aion_knowledge.pipeline.chunker.base import ChunkConfig, ChunkResult
from aion_knowledge.pipeline.chunker.fragment_utils import extract_overlap_tail
from aion_knowledge.pipeline.chunker.protected_regions import (
    ProtectedSpan,
    find_protected_spans,
    split_avoiding_protected,
)
from aion_knowledge.pipeline.chunker.text_utils import count_tokens

logger = logging.getLogger(__name__)


class RecursiveSplitter:
    """递归分块器 —— 基于分隔符优先级递归分割。

    这是 chunker 模块的最后一道防线 —— 所有分块器回退至此。
    保证永远不会返回空列表或抛出切分异常。

    输入可以是任意语言、任意格式的纯文本文档。通过逐步降级分隔符，
    从最自然的语义边界（段落 → 行 → 句子 → 字符）逐步直到满足 chunk_size。
    """

    def __init__(self, config: ChunkConfig):
        self.config = config

    def split(self, text: str) -> List[ChunkResult]:
        if not text:
            return []

        splits = self._split(text)
        chunks = self._merge(splits)
        return chunks

    def _split(self, text: str) -> List[str]:
        """按分隔符优先级递归分割文本（保护区域不被切断）。

        算法细节：

        1. 如果整个文本 <= chunk_size，直接返回
        2. 遍历 ``self.config.separators``（默认：
           ``["\\n\\n", "\\n", "。", "；", "，", ""]``）
        3. **逐级降级**：第一轮仅当结果仍有 > chunk_size 的段时，
           对超标段使用当前分隔符切分（未超标的段保留）
        4. **早停**：一旦所有段都 <= chunk_size，立即停止，不再尝试更低级分隔符
        5. **保护区域**：每次调用 ``_split_by_sep()`` 都传入 ``find_protected_spans()``
           的结果，确保代码块/表格/公式不被切断
        6. **后处理**：对依然超大的表格片段（``_split_oversized_table``），
           做行级拆分并在续片前置表头

        Args:
            text: 原始输入文本

        Returns:
            分割后的文本片段列表（尚未合并为 ChunkResult）
        """
        if count_tokens(text) <= self.config.chunk_size:
            return [text]

        segments: list[str] = [text]
        for sep in self.config.separators:
            if all(count_tokens(s) <= self.config.chunk_size for s in segments):
                break
            next_segments: list[str] = []
            for seg in segments:
                if count_tokens(seg) <= self.config.chunk_size:
                    next_segments.append(seg)
                else:
                    protected = find_protected_spans(seg)
                    next_segments.extend(self._split_by_sep(seg, sep, protected))
            segments = next_segments
        return segments

    def _split_oversized_table(self, text: str) -> List[str]:
        """对超大的 Markdown 表格做行级拆分，续片前置表头行。

        这是递归分块的**特殊适配** —— 当一段文本被检测为 Markdown 表格但
        其 token 数超过 chunk_size 时触发。

        问题背景：受保护的表格区域被 ``protected_regions`` 整体保护，任何
        分隔符都无法将其切开。降级到 ``sep=""`` 会退化为字符级拆分，把表
        格拆成不可读的碎片。

        解决方案 —— **贪心行分割 + 表头前置**：
        1. 检测表头行（第一个 ``| ... |`` 且在下一行有 ``|---|`` 分隔行）
        2. 从第一个数据行开始，贪心追加行直到达到 chunk_size
        3. 每个续片（除了第一段）在开头插入表头行 + 分隔行
        4. 尾部非表格内容保持原样附加到最后一段

        这样每个续片的上下文都是自包含的 —— 列标题和列数关系一目了然，
        下游的 embedding 和检索可以正确理解该片段。
        """
        lines = text.split("\n")

        # 查找 Markdown 表头：第一个 | 行，紧随的是 |--- 分隔行
        header = None
        header_idx = -1
        for i in range(len(lines) - 1):
            t = lines[i].strip()
            n = lines[i + 1].strip()
            if t.startswith("|") and n.startswith("|") and set(n).intersection({"-", ":"}):
                header = t
                header_idx = i
                break

        if header is None:
            # 不是表格，退回到字符级拆分
            return list(text)

        # 收集表头前的内容、数据行、表格后内容
        pre_lines = lines[:header_idx]
        # 表头行和分隔行
        # separator 行是 header_idx + 1（| --- | --- |）
        separator_line = lines[header_idx + 1] if header_idx + 1 < len(lines) else None
        # 数据行
        data_start = header_idx + 2
        data_lines = []
        trailing_lines = []
        in_data = True
        for ln in lines[data_start:]:
            if in_data and ln.strip().startswith("|"):
                data_lines.append(ln)
            else:
                in_data = False
                trailing_lines.append(ln)

        # 表头 + 分隔行（用于续片开头）
        header_block = header
        if separator_line and separator_line.strip() != header.strip():
            header_block = header + "\n" + separator_line

        result: List[str] = []

        # 贪心打包：从 pre_lines + header + 分隔行 + 数据行 开始
        # 每段续片用 header_block 开头
        if not data_lines:
            return [text]

        # 第一段：pre_lines + header + separator + 若干数据行
        chunk = "\n".join(
            pre_lines + [header]
            + ([separator_line] if separator_line else [])
            + [data_lines[0]]
        )
        for row in data_lines[1:]:
            candidate = chunk + "\n" + row
            if count_tokens(candidate) > self.config.chunk_size:
                result.append(chunk)
                # 续片：前导换行（避免 _merge 的 "".join 吃掉行间分隔）+
                # header + separator + row
                chunk = "\n" + header_block + "\n" + row
            else:
                chunk = candidate

        # 尾部
        if chunk:
            # 如果最后一行已经有表格内容，直接加入
            result.append(chunk)

        # 如果有 trailing_lines，附加到最后一段
        if trailing_lines:
            result[-1] = result[-1] + "\n" + "\n".join(trailing_lines)

        return result

    def _split_by_sep(
        self,
        text: str,
        sep: str,
        protected_spans: Optional[List[ProtectedSpan]] = None,
    ) -> List[str]:
        """按一个分隔符切分，分隔符保留在后续片段开头。

        与 Python 的 ``str.split(sep)`` 不同，分隔符本身**保留**在
        后续片段开头。这样在 ``_merge()`` 时用 ``"".join()`` 即可无损
        恢复原始内容。

        两种特殊处理：

        1. **空分隔符**（``sep=""``）：退化为字符级拆分，但会检查文本
           是否被表格主导 —— 如果是，则调用 ``_split_oversized_table()``
           替代字符拆分（避免把表格拆得粉碎）。

        2. **受保护区域**：如果提供了 ``protected_spans``，每次找到的
           分隔符位置会调用 ``is_inside_protected()`` 检查。若落在保护
           区域内，则不在此处切分，将分隔符合并到前一段末尾。

        Args:
            text: 要切分的文本段
            sep: 分隔符字符串
            protected_spans: ``find_protected_spans()`` 的结果，可选

        Returns:
            切分后的文本列表（分隔符已保留在后续片段开头）
        """
        if not sep:
            # 字符级拆分前检查是否被表格主导 ⇒ 用行级拆分替代
            if protected_spans:
                table_span = next((s for s in protected_spans if s[0] == "table"), None)
                if table_span:
                    return self._split_oversized_table(text)
            return list(text)

        if protected_spans:
            return split_avoiding_protected(text, sep, protected_spans)

        # 原始逻辑（无保护区域时）
        parts = text.split(sep)
        result = []
        for i, part in enumerate(parts):
            if i == 0:
                if part:
                    result.append(part)
            else:
                result.append(sep + part)
        return result

    def _merge(self, splits: List[str]) -> List[ChunkResult]:
        """合并分割后的片段为最终 ChunkResult。

        贪心打包算法：
        1. 从第一个片段开始累积直到加入下一个片段会超过 chunk_size
        2. 输出当前累积块作为一个 ChunkResult（含 token 计数和 ID）
        3. 如果 ``chunk_overlap > 0``，从上一 chunk 尾部提取重叠文本作为
           新 chunk 的开头（调用 ``extract_overlap_tail()``，换行符对齐）
        4. 重置累积器，从当前片段开始新一轮累积
        5. 最后剩余的片段作为最后一个 chunk 输出

        Note:
            如果 ``chunk_overlap == 0``，每次新 chunk 开头不会包含
            重叠内容（这是 langchain 的 RecursiveCharacterTextSplitter
            的行为方式）。

        Args:
            splits: ``_split()`` 输出的文本片段列表

        Returns:
            分块结果列表（已按 seq_num 排序）
        """
        if not splits:
            return []

        chunks: List[ChunkResult] = []
        current_parts: List[str] = []
        current_token_count = 0
        seq = 0

        for part in splits:
            part_tokens = count_tokens(part)

            if current_token_count + part_tokens <= self.config.chunk_size:
                current_parts.append(part)
                current_token_count += part_tokens
            else:
                if current_parts:
                    content = "".join(current_parts)
                    chunks.append(ChunkResult(
                        content=content,
                        token_count=current_token_count,
                        chunk_id=f"chunk_{seq}",
                        seq_num=seq,
                    ))
                    seq += 1

                # 语义边界对齐重叠
                if self.config.chunk_overlap > 0 and chunks:
                    prev = chunks[-1].content
                    overlap_text = extract_overlap_tail(prev, self.config.chunk_overlap)
                    current_parts = [overlap_text] if overlap_text else []
                    current_token_count = count_tokens(overlap_text) if overlap_text else 0
                else:
                    current_parts = []
                    current_token_count = 0

                current_parts.append(part)
                current_token_count += part_tokens

        if current_parts:
            content = "".join(current_parts)
            chunks.append(ChunkResult(
                content=content,
                token_count=current_token_count,
                chunk_id=f"chunk_{seq}",
                seq_num=seq,
            ))

        return chunks

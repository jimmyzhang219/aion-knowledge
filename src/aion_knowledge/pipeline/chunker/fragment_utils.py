"""片段工具 —— 小段合并、重叠提取。

=== 功能说明 ===

1. ``merge_adjacent_chunks()`` —— 碎片合并
   将相同标题路径下连续的小 chunk 合并，减少碎片。
   规则：同标题路径 + 当前 chunk < target（chunk_size * 0.5）+ 合并后不超限。
   target 最小值为 200 tokens（防止过小阈值导致不合并）。

2. ``extract_overlap_tail()`` —— 语义重叠提取
   从 chunk 尾部提取约 target_tokens 的重叠文本用于下一个 chunk 的开头。
   特殊规则：如果截断点落在 Markdown 表格分隔行（| -- |）上，向前延伸
   到表头行，保证续片不会丢失列上下文。
   返回空串表示文本太短无需重叠（重叠部分 == 原始文本）。
"""

import logging
from typing import List

from aion_knowledge.pipeline.chunker.base import ChunkResult
from aion_knowledge.pipeline.chunker.text_utils import count_tokens

logger = logging.getLogger(__name__)


def merge_adjacent_chunks(
    chunks: List[ChunkResult],
    chunk_size: int,
    threshold_ratio: float = 0.5,
) -> List[ChunkResult]:
    """合并过小的相邻 chunk（同标题路径下），减少碎片。

    用于 ``HeadingSplitter`` 输出后处理，将被过度切分的小段重新合并。

    合并条件（**必须全部满足**）:
    1. 当前 chunk 的 token 数 < target（= chunk_size * threshold_ratio，最小 200）
    2. 当前 chunk 与下一 chunk 的 ``heading_path`` 相同（同标题层级）
    3. 合并后总 token 数不超过 chunk_size

    不满足条件时，当前 chunk 作为独立块输出，累积器指向下一个 chunk。
    """
    if len(chunks) <= 1:
        return chunks

    target = int(chunk_size * threshold_ratio)
    if target < 200:
        target = 200

    merged: List[ChunkResult] = []
    cur = chunks[0]

    for next_chunk in chunks[1:]:
        cur_too_small = cur.token_count < target
        would_not_overflow = cur.token_count + next_chunk.token_count <= chunk_size
        same_heading = cur.heading_path == next_chunk.heading_path

        if cur_too_small and would_not_overflow and same_heading:
            cur.content += next_chunk.content
            cur.token_count = count_tokens(cur.content)
            continue

        merged.append(cur)
        cur = next_chunk

    merged.append(cur)
    return merged


def extract_overlap_tail(text: str, target_tokens: int) -> str:
    """从文本尾部提取约 ``target_tokens`` 的文本，换行符对齐。

    用于 ``RecursiveSplitter`` 的 chunk 间重叠（overlap）生成。从当前 chunk
    尾部取一段文本作为下一个 chunk 的开头，实现上下文边界的平滑过渡。

    提取规则：
    1. 按 token 比 × 1.2 放大系数估算字符偏移（保守多取，对齐后截断在换行符）
    2. 从估算位置向后找第一个 ``\\n`` 对齐（避免中间切断行）
    3. **表格保护**：若对齐后首行是 Markdown 表格分隔行（``| --- | --- |``），
       向前延伸包含上一行表头，保证续片有完整的列标题上下文

    返回空字符串的情况：
    - 文本太短（重叠部分将等同于或超过原文）
    - target_tokens <= 0
    """
    if not text:
        return ""
    ratio = len(text) / max(count_tokens(text), 1)
    char_count = int(target_tokens * ratio * 1.2)
    if char_count >= len(text):
        return ""
    start = len(text) - char_count
    newline_pos = text.find("\n", start)
    if 0 <= newline_pos < len(text) - 1:
        start = newline_pos + 1

    result = text[start:]

    # 表分隔行保护：若 overlap 以 | -（表格分隔行）开头，向前延伸包含表头行
    first_line = result.split("\n")[0].strip()
    if first_line.startswith("|") and "---" in first_line:
        # 找到此分隔行之前的表头行
        sep_nl = text.rfind("\n", 0, start - 1) if start > 0 else -1
        if sep_nl >= 0:
            prev_line = text[sep_nl + 1:start - 1].strip()
            if prev_line.startswith("|") and "---" not in prev_line:
                start = sep_nl + 1
                result = text[start:]

    return result

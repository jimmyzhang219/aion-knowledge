"""受保护区域检测——在分块时保持这些区域完整不被切分。

当分块操作遇到跨越 chunk 边界的结构性元素时，若直接切断会导致该元素
失去完整性（如表格被截断后无法解析，代码块被切断后无法运行）。
受保护区域机制确保这些元素被完整保留在同一个 chunk 内。

=== 受保护区域类型 ===

按 ``PROTECTED_PATTERNS`` 的顺序检测，越靠前的模式优先级越高：

===== ================== ========================
类型  标记                 保护原因
===== ================== ========================
LaTeX ``$$ ... $$``       块级公式截断会失去语义
Code  ````` ... `````      代码块截断后不可运行
Table ``| col | col |``  表格截断后列对齐丢失
Image ``![alt](url)``    图片标记截断后渲染失效
===== ================== ========================

=== 保护机制 ===

1. ``find_protected_spans(text)`` —— 扫描全文返回 ``[(type, start, end)]``
   - 按 ``PROTECTED_PATTERNS`` 顺序匹配，后匹配的不能与前匹配重叠
   - 按起始位置排序，供后续切分步骤查询

2. ``is_inside_protected(pos, spans)`` —— 检查某字符位置是否在保护区域内

3. ``split_avoiding_protected(text, sep, spans)`` —— 受保护的切分
   - 类似 ``str.split(sep)``，但如果某个分隔符落在保护区域内，则不切分
   - 该分隔符被合并到前一段末尾（保持保护区域完整）

=== 使用场景 ===

``RecursiveSplitter._split_by_sep()`` 在递归切分时会调用 ``find_protected_spans``
获取当前段的保护区域，然后使用 ``split_avoiding_protected`` 进行受保护的切分。
"""

import re
from typing import List, Tuple

# 受保护区域的正则模式
PROTECTED_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\$\$[\s\S]*?\$\$"),           # LaTeX 块级公式
    re.compile(r"```[\s\S]*?```"),              # 围栏代码块
    re.compile(r"^\|[^\n]*\|[ \t]*(?:\n\|[^\n]*\|[ \t]*)*", re.MULTILINE),  # Markdown 表格
    re.compile(r"!\[.*?\]\(.*?\)"),             # 图片
]

# 单个保护区域：(类型, start, end)
# 类型: "code", "latex", "table", "image"
ProtectedSpan = Tuple[str, int, int]


def find_protected_spans(text: str) -> List[ProtectedSpan]:
    """扫描文本，返回所有受保护区域的列表（按起始位置排序）。

    检测顺序：LaTeX → Code → Table → Image。
    后检测的区域如果与已检测区域重叠，则跳过（保持先检测的高优先级）。

    Returns:
        ``[(type, start, end), ...]`` 列表，按 start 升序排列。
        type 取值: ``"latex"`` / ``"code"`` / ``"table"`` / ``"image"``。
    """
    spans: List[ProtectedSpan] = []
    seen: List[Tuple[int, int]] = []

    for pattern in PROTECTED_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            # 去重：跳过已被其他模式覆盖的区域
            overlaps = False
            for s, e in seen:
                if start < e and end > s:
                    overlaps = True
                    break
            if not overlaps:
                # 判断类型
                span_type = "unknown"
                matched = match.group(0)
                if matched.startswith("$$"):
                    span_type = "latex"
                elif matched.startswith("```"):
                    span_type = "code"
                elif matched.startswith("|"):
                    span_type = "table"
                elif matched.startswith("!["):
                    span_type = "image"

                spans.append((span_type, start, end))
                seen.append((start, end))

    spans.sort(key=lambda x: x[1])
    return spans


def is_inside_protected(pos: int, spans: List[ProtectedSpan]) -> bool:
    """检查字符位置是否在某个受保护区域内。

    Args:
        pos: 要检查的字符偏移量（0-indexed）
        spans: ``find_protected_spans()`` 返回的保护区域列表

    Returns:
        如果 ``pos`` 位于任意保护区域的 ``[start, end)`` 区间内则返回 True。

    注：这是在 O(n) 时间内完成的线性扫描，但保护区域数量通常很少（< 20），
    不需要二分查找优化。
    """
    for _, start, end in spans:
        if start <= pos < end:
            return True
    return False


def split_avoiding_protected(
    text: str, sep: str, protected_spans: List[ProtectedSpan]
) -> List[str]:
    """按分隔符切分，跳过落在受保护区域内的切分点。

    与 ``str.split(sep)`` 类似，但如果某个分隔符的位置位于受保护区域内，
    则该处不切分（将前后两段合并）。
    """
    if not sep:
        return list(text)

    parts = text.split(sep)
    if len(parts) <= 1:
        return parts

    result: List[str] = []
    pos = 0  # 当前 part 在原始 text 中的起始位置

    for i, part in enumerate(parts):
        part_len = len(part)

        if i == 0:
            if part:
                result.append(part)
            pos = part_len  # separator 紧跟在第一个 part 之后
        else:
            # 检查位于 pos 处的分隔符是否落在受保护区域内
            sep_inside = (
                is_inside_protected(pos, protected_spans)
                or is_inside_protected(pos + len(sep) - 1, protected_spans)
            )
            if sep_inside:
                # 合并到前一段
                if result:
                    result[-1] += sep + part
                else:
                    result.append(sep + part)
            else:
                result.append(sep + part)

            pos += len(sep) + part_len

    return result

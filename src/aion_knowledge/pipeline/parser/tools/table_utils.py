"""提取 Markdown 表格为原子 chunk，替换为占位符。"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 匹配标题行：以 # 开头（可选空白后跟随 #）
HEADING_PATTERN = re.compile(r"^#{1,6}\s")
# 匹配表格行：以 | 开头
TABLE_ROW_PATTERN = re.compile(r"^\|.+\|")
# 表格分隔行：| --- | --- | 之类的
TABLE_SEP_PATTERN = re.compile(r"^\|[\s\-:]+\|(?:[\s\-:]+\|)*$")
# 占位符模式
PLACEHOLDER_PATTERN = re.compile(r"<TABLE_(\d+)>")


def extract_tables(md_content: str) -> tuple[list[dict[str, Any]], str]:
    """从 Markdown 中提取表格，替换为占位符。

    按行扫描 markdown：
    - 跟踪标题层级（heading_path）
    - 检测以 | 开头的连续行块，至少包含表头行 + 分隔行
    - 提取表格上方的非空非标题文本行作为 table_caption
    - 每个表格替换为 <TABLE_N> 占位符

    Returns:
        (table_sections, placeholder_md)
        每个 table_section 包含:
          type, content, table_caption, heading_path, seq_num
    """
    lines = md_content.split("\n")
    heading_path: list[str] = []
    table_sections: list[dict[str, Any]] = []
    output_lines: list[str] = []
    seq_num = 0
    i = 0

    while i < len(lines):
        line = lines[i]

        # 跟踪标题层级
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            # 获取标题文本和级别
            stripped = line.lstrip()
            level = 0
            for ch in stripped:
                if ch == "#":
                    level += 1
                else:
                    break
            heading_text = stripped[level:].strip()

            # 更新 heading_path：移除同级及更深的标题
            heading_path = heading_path[: level - 1]
            heading_path.append(heading_text)

            output_lines.append(line)
            i += 1
            continue

        # 检测表格起始行
        if TABLE_ROW_PATTERN.match(line):
            # 收集连续的表格行
            table_start = i
            while i < len(lines) and TABLE_ROW_PATTERN.match(lines[i]):
                i += 1
            table_end = i

            table_lines = lines[table_start:table_end]

            # 验证至少有两行（表头 + 分隔行），且第二行是分隔行
            if len(table_lines) < 2 or not TABLE_SEP_PATTERN.match(table_lines[1]):
                # 不是有效表格，原样保留
                output_lines.extend(table_lines)
                continue

            # 提取表格上方的 caption（紧邻表格上方的非空非标题行）
            table_caption = ""
            caption_index = table_start - 1
            # 跳过表格上方的空行
            while caption_index >= 0 and lines[caption_index].strip() == "":
                caption_index -= 1
            if caption_index >= 0:
                candidate = lines[caption_index].strip()
                if candidate and not HEADING_PATTERN.match(candidate):
                    table_caption = candidate

            # 构建 table_section
            table_content = "\n".join(table_lines)
            table_sections.append({
                "type": "table",
                "content": table_content,
                "table_caption": table_caption,
                "heading_path": " > ".join(heading_path),
                "seq_num": seq_num,
            })

            # 写入占位符
            output_lines.append(f"<TABLE_{seq_num}>")
            seq_num += 1
            continue

        output_lines.append(line)
        i += 1

    placeholder_md = "\n".join(output_lines)
    return table_sections, placeholder_md


def parse_table_position(placeholder_md: str, table_seq: int) -> int:
    """在占位符文本中找到表格 seq 对应的位置。

    Args:
        placeholder_md: 包含 <TABLE_N> 占位符的 markdown 文本
        table_seq: 表格序号

    Returns:
        占位符在文本中的字符位置，找不到时返回 -1
    """
    pattern = f"<TABLE_{table_seq}>"
    pos = placeholder_md.find(pattern)
    return pos

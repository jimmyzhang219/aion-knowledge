"""测试：Markdown 表格提取为原子 chunk。"""
from aion_knowledge.pipeline.parser.tools import extract_tables, parse_table_position


class TestExtractTables:
    """extract_tables() 测试。"""

    def test_extract_tables_basic(self):
        """提取两个标准表格。"""
        md = """# 标题

| 列A | 列B |
|-----|-----|
| a1  | b1  |
| a2  | b2  |

一些文字。

| 列X | 列Y |
|-----|-----|
| x1  | y1  |"""

        table_sections, placeholder_md = extract_tables(md)

        assert len(table_sections) == 2

        # 第一个表格
        assert table_sections[0]["type"] == "table"
        assert "| 列A | 列B |" in table_sections[0]["content"]
        assert "| a1  | b1  |" in table_sections[0]["content"]
        assert "| a2  | b2  |" in table_sections[0]["content"]
        assert table_sections[0]["seq_num"] == 0

        # 第二个表格
        assert table_sections[1]["type"] == "table"
        assert "| 列X | 列Y |" in table_sections[1]["content"]
        assert "| x1  | y1  |" in table_sections[1]["content"]
        assert table_sections[1]["seq_num"] == 1

    def test_extract_tables_no_tables(self):
        """无表格时返回空列表。"""
        md = """# 纯文本

这是普通段落，没有任何表格。"""

        table_sections, placeholder_md = extract_tables(md)
        assert table_sections == []
        assert placeholder_md == md

    def test_extract_tables_placeholders(self):
        """占位符正确替换且有序。"""
        md = """前面文字。

| A | B |
|---|---|
| 1 | 2 |

中间文字。

| X | Y |
|---|---|
| 3 | 4 |

尾部文字。"""

        table_sections, placeholder_md = extract_tables(md)

        assert "<TABLE_0>" in placeholder_md
        assert "<TABLE_1>" in placeholder_md
        # 占位符顺序正确
        idx0 = placeholder_md.index("<TABLE_0>")
        idx1 = placeholder_md.index("<TABLE_1>")
        assert idx0 < idx1

        # 原表格行不应出现在 placeholder 中
        assert "|---" not in placeholder_md
        assert "| A | B |" not in placeholder_md
        assert "| 1 | 2 |" not in placeholder_md
        assert "| 3 | 4 |" not in placeholder_md

    def test_extract_tables_table_caption(self):
        """提取表格上方紧邻的非空非标题文本行作为 caption。"""
        md = """# 标题

表1：这是第一个表格的说明
| A | B |
|---|---|
| 1 | 2 |

表2：这是第二个表格的说明
| X | Y |
|---|---|
| 3 | 4 |"""

        table_sections, _ = extract_tables(md)

        assert table_sections[0]["table_caption"] == "表1：这是第一个表格的说明"
        assert table_sections[1]["table_caption"] == "表2：这是第二个表格的说明"

    def test_extract_tables_context_header(self):
        """表格继承标题层级（heading_path）。"""
        md = """# 第一章

## 第一节

介绍文字。

| A | B |
|---|---|
| 1 | 2 |

## 第二节

| X | Y |
|---|---|
| 3 | 4 |"""

        table_sections, _ = extract_tables(md)

        # 第一个表格在 "## 第一节" 下
        assert "第一章" in table_sections[0]["heading_path"]
        assert "第一节" in table_sections[0]["heading_path"]
        # heading_path 是列表
        assert isinstance(table_sections[0]["heading_path"], list)

        # 第二个表格在 "## 第二节" 下
        assert "第一章" in table_sections[1]["heading_path"]
        assert "第二节" in table_sections[1]["heading_path"]


class TestParseTablePosition:
    """parse_table_position() 测试。"""

    def test_parse_table_position(self):
        """在占位符文本中找到正确的表格位置。"""
        placeholder_md = """# 标题

前面文字。

<TABLE_0>

中间文字。

<TABLE_1>

尾部文字。"""

        pos = parse_table_position(placeholder_md, 0)
        assert pos > 0
        # 在 <TABLE_0> 附近
        assert abs(pos - placeholder_md.index("<TABLE_0>")) < 50

        pos = parse_table_position(placeholder_md, 1)
        assert pos > placeholder_md.index("<TABLE_1>") - 50

    def test_parse_table_position_not_found(self):
        """找不到占位符时返回 -1。"""
        md = """# 标题

没有表格占位符。"""
        pos = parse_table_position(md, 0)
        assert pos == -1

"""Cleaner 模块单元测试 + 集成测试。"""

import unittest

from aion_knowledge.pipeline.cleaner import Cleaner, clean_passage
from aion_knowledge.pipeline.cleaner.cleaner import (
    collapse_blank_lines,
    ensure_code_block_spacing,
    ensure_heading_spacing,
    ensure_list_spacing,
    fullwidth_to_halfwidth,
    normalize_line_endings,
    normalize_unicode,
    remove_control_chars,
    remove_page_numbers,
    remove_separators,
    remove_zero_width,
    strip_trailing_whitespace,
    trim_file_ends,
)


class TestStep1CharacterNormalization(unittest.TestCase):
    """Step 1：字符级净化"""

    def test_normalize_unicode_nfc(self):
        """复合字符 e + combining accent → é"""
        composed = "é"  # e + combining acute accent
        result = normalize_unicode(composed)
        self.assertEqual(result, "é")

    def test_remove_zero_width(self):
        """零宽字符被删除"""
        result = remove_zero_width("hello​world‍")
        self.assertEqual(result, "helloworld")

    def test_remove_zero_width_bom(self):
        """BOM (U+FEFF) 被删除"""
        result = remove_zero_width("﻿hello")
        self.assertEqual(result, "hello")

    def test_remove_control_chars(self):
        """控制字符被删除，保留 \\n \\t \\r"""
        result = remove_control_chars("a\x00b\x1bc\n\t\r")
        self.assertEqual(result, "abc\n\t\r")

    def test_fullwidth_to_halfwidth(self):
        """全角 ASCII 字母数字转半角"""
        result = fullwidth_to_halfwidth("Ｈｅｌｌｏ１２３")
        self.assertEqual(result, "Hello123")

    def test_fullwidth_unchanged(self):
        """全角非 ASCII（如中文）保持不变"""
        result = fullwidth_to_halfwidth("你好世界")
        self.assertEqual(result, "你好世界")


class TestStep2WhitespaceNormalization(unittest.TestCase):
    """Step 2：空白字符规整"""

    def test_normalize_line_endings(self):
        """\\r\\n 和 \\r 统一为 \\n"""
        result = normalize_line_endings("a\r\nb\rc")
        self.assertEqual(result, "a\nb\nc")

    def test_strip_trailing_whitespace(self):
        """行尾空格被删除"""
        result = strip_trailing_whitespace("hello   \nworld\t\n")
        self.assertEqual(result, "hello\nworld\n")

    def test_collapse_blank_lines(self):
        """连续 4 空行 → 2 空行"""
        result = collapse_blank_lines("a\n\n\n\n\nb")
        self.assertEqual(result, "a\n\n\nb")

    def test_trim_file_ends(self):
        """文件首尾空白行被裁剪"""
        result = trim_file_ends("\n\nhello\nworld\n\n")
        self.assertEqual(result, "hello\nworld")

    def test_collapse_no_change(self):
        """≤ 2 空行保持不变"""
        result = collapse_blank_lines("a\n\nb")
        self.assertEqual(result, "a\n\nb")


class TestStep3MarkdownFormatting(unittest.TestCase):
    """Step 3：Markdown 格式归一"""

    def test_heading_spacing(self):
        """标题前补充空行"""
        text = "some text\n# Heading\ncontent"
        result = ensure_heading_spacing(text)
        self.assertIn("some text\n\n# Heading", result)

    def test_heading_at_start_no_extra(self):
        """文件开头的标题前不加空行"""
        result = ensure_heading_spacing("# Title\ncontent")
        self.assertEqual(result, "# Title\ncontent")

    def test_list_spacing(self):
        """列表前后补充空行"""
        text = "text\n- item1\n- item2\nmore text"
        result = ensure_list_spacing(text)
        self.assertIn("text\n\n- item1", result)

    def test_code_block_spacing(self):
        """代码块前后补充空行"""
        text = "text\n```\ncode\n```\nmore"
        result = ensure_code_block_spacing(text)
        self.assertIn("text\n\n```", result)


class TestStep4NoiseFiltering(unittest.TestCase):
    """Step 4：文档噪声过滤"""

    def test_remove_dash_page_number(self):
        """- 123 - 格式页码行被删除"""
        result = remove_page_numbers("- 123 -")
        self.assertEqual(result, "")

    def test_remove_page_of(self):
        """Page X of Y 被删除"""
        result = remove_page_numbers("Page 5 of 10")
        self.assertEqual(result, "")

    def test_remove_chinese_page(self):
        """第 X 页 被删除"""
        result = remove_page_numbers("第 3 页")
        self.assertEqual(result, "")

    def test_keep_sentence_with_number(self):
        """正文中的数字行不被误删"""
        result = remove_page_numbers("The value is 42 and it matters")
        self.assertEqual(result, "The value is 42 and it matters")

    def test_remove_separator_line(self):
        """纯分隔线被删除"""
        result = remove_separators("________________________")
        self.assertEqual(result, "")

    def test_keep_short_separator(self):
        """少于 20 个的分隔线保留"""
        result = remove_separators("________")
        self.assertEqual(result, "________")


class TestCleanerPipeline(unittest.TestCase):
    """Cleaner 完整流水线集成测试"""

    def setUp(self):
        self.cleaner = Cleaner()

    def test_empty_string(self):
        """空字符串清洗后仍为空"""
        self.assertEqual(self.cleaner.clean(""), "")

    def test_clean_document(self):
        """模拟一个含噪声的文档，验证完整清洗流程"""
        dirty = (
            "\n﻿# Title  \n\n"
            "Some text with​zero‍width chars\n\n"
            "Ｈｅｌｌｏ\n"
            "- 123 -\n"
            "________________________\n"
            "\n\n\n"
        )
        cleaned = self.cleaner.clean(dirty)
        # 验证无零宽字符
        self.assertNotIn("​", cleaned)
        self.assertNotIn("‍", cleaned)
        self.assertNotIn("﻿", cleaned)
        # 验证全角转半角
        self.assertNotIn("Ｈｅｌｌｏ", cleaned)
        self.assertIn("Hello", cleaned)
        # 验证页码被删除
        self.assertNotIn("- 123 -", cleaned)
        # 验证分隔线被删除
        self.assertNotIn("________", cleaned)
        # 验证标题前无多余空行（文件开头）
        self.assertTrue(cleaned.startswith("# Title"))

    def test_idempotent(self):
        """多次清洗结果一致：clean(clean(x)) == clean(x)"""
        text = "# Hello\n\nSome content\n\n- list item\n\n```\ncode\n```"
        once = self.cleaner.clean(text)
        twice = self.cleaner.clean(once)
        self.assertEqual(once, twice)

    def test_no_noise_no_change(self):
        """干净文档变化最小"""
        clean_text = "Hello world\nThis is clean."
        result = self.cleaner.clean(clean_text)
        self.assertIn("Hello world", result)


class TestCleanPassage(unittest.TestCase):
    """Step 5：Markdown 语法剥离"""

    def test_removes_code_blocks(self):
        text = "前文\n```python\ncode here\n```\n后文"
        result = clean_passage(text)
        self.assertNotIn("code here", result)
        self.assertIn("前文", result)

    def test_removes_table_rows(self):
        text = "| col1 | col2 |"
        result = clean_passage(text)
        self.assertNotIn("|", result)

    def test_link_to_text(self):
        text = "见[文档](https://example.com)"
        result = clean_passage(text)
        self.assertIn("文档", result)
        self.assertNotIn("https://", result)

    def test_removes_images(self):
        text = "前文 ![alt](img.png) 后文"
        result = clean_passage(text)
        self.assertNotIn("![", result)

    def test_removes_html_tags(self):
        text = "<div class='x'>内容</div>"
        result = clean_passage(text)
        self.assertNotIn("<div", result)

    def test_inline_code_preserves_text(self):
        text = "使用 `print()` 函数"
        result = clean_passage(text)
        self.assertIn("print()", result)

    def test_removes_heading_markers(self):
        text = "## 标题\n内容"
        result = clean_passage(text)
        self.assertNotIn("##", result)
        self.assertIn("标题", result)

    def test_removes_list_markers(self):
        text = "- 项目1\n- 项目2"
        result = clean_passage(text)
        self.assertIn("项目1", result)

    def test_collapses_extra_blank_lines(self):
        text = "行1\n\n\n\n行2"
        result = clean_passage(text)
        self.assertNotIn("\n\n\n", result)

    def test_strips_whitespace(self):
        text = "  内容  \n  "
        result = clean_passage(text)
        self.assertEqual(result, "内容")

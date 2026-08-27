import unittest

from aion_knowledge.pipeline.chunker.protected_regions import (
    find_protected_spans,
    is_inside_protected,
    split_avoiding_protected,
)


class TestProtectedRegions(unittest.TestCase):
    def test_fenced_code_block(self):
        text = "before\n```python\ncode block\n```\nafter"
        spans = find_protected_spans(text)
        self.assertEqual(len(spans), 1)
        _, start, end = spans[0]
        self.assertIn("```python\ncode block\n```", text[start:end])

    def test_markdown_table(self):
        text = "before\n| a | b |\n| --- | --- |\n| 1 | 2 |\nafter"
        spans = find_protected_spans(text)
        self.assertEqual(len(spans), 1)
        _, start, end = spans[0]
        self.assertIn("| a | b |", text[start:end])

    def test_latex_block(self):
        text = "before\n$$\nformula\n$$\nafter"
        spans = find_protected_spans(text)
        self.assertEqual(len(spans), 1)

    def test_no_protected(self):
        text = "simple text\nwithout any protected regions"
        spans = find_protected_spans(text)
        self.assertEqual(len(spans), 0)

    def test_is_inside_protected(self):
        text = "a\n```\ncode\n```\nb"
        spans = find_protected_spans(text)
        self.assertTrue(is_inside_protected(5, spans))  # inside ```\ncode\n```
        self.assertFalse(is_inside_protected(0, spans))  # "a\n"
        self.assertFalse(is_inside_protected(len(text) - 1, spans))  # "b"


class TestSplitAvoidingProtected(unittest.TestCase):
    def test_split_outside_table(self):
        """分隔符在表格外时正常切分。"""
        text = "before\n| a | b |\n| c | d |\nafter"
        spans = find_protected_spans(text)
        result = split_avoiding_protected(text, "\n", spans)
        # "before" 独立为一段，"after" 前带换行符出现在末段
        self.assertEqual(result[0], "before")
        self.assertTrue(result[-1].endswith("after"))
        for r in result:
            if "| a | b |" in r and "| c | d |" in r:
                # 表格行在同一个片段中
                self.assertIn("| a | b |", r)
                self.assertIn("| c | d |", r)
                break
        else:
            self.fail("table rows not found in any single result")

    def test_no_split_inside_table(self):
        """表格行间的分隔符不应切分。"""
        text = "| col1 | col2 |\n| val1 | val2 |"
        spans = find_protected_spans(text)
        result = split_avoiding_protected(text, "\n", spans)
        self.assertEqual(len(result), 1)

    def test_split_code_block(self):
        """代码块同样受保护——代码块内部不断开，但前置/后置文本各自独立。"""
        text = "a\n```\ncode\n```\nb"
        spans = find_protected_spans(text)
        result = split_avoiding_protected(text, "\n", spans)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "a")
        self.assertIn("```", result[1])  # 代码块整体在 result[1]
        self.assertTrue(result[-1].endswith("b"))

    def test_empty_sep(self):
        text = "abc"
        spans = find_protected_spans(text)
        result = split_avoiding_protected(text, "", spans)
        self.assertEqual(result, list("abc"))

    def test_no_protected_spans(self):
        """无保护区域时退化为普通 split。"""
        text = "a\nb\nc"
        result = split_avoiding_protected(text, "\n", [])
        self.assertEqual(result, ["a", "\nb", "\nc"])

    def test_table_separated_by_blank_line(self):
        """双换行分隔的表格不应被 \n\n 切开。"""
        text = "intro\n\n| a | b |\n| c | d |\n\noutro"
        spans = find_protected_spans(text)
        result = split_avoiding_protected(text, "\n\n", spans)
        # intro, 表格段落, outro 三段
        self.assertGreaterEqual(len(result), 2)
        table_found = False
        for r in result:
            if "| a | b |" in r and "| c | d |" in r:
                table_found = True
        self.assertTrue(table_found)

import unittest

from aion_knowledge.pipeline.chunker.text_utils import count_tokens


class TestTokenCounting(unittest.TestCase):
    def test_count_tokens_english(self):
        n = count_tokens("Hello, world!")
        self.assertGreater(n, 0)
        self.assertIsInstance(n, int)

    def test_count_tokens_chinese(self):
        n = count_tokens("你好世界")
        self.assertGreater(n, 0)

    def test_count_tokens_empty(self):
        self.assertEqual(count_tokens(""), 0)

    def test_count_tokens_mixed(self):
        n = count_tokens("你好 world, 测试 token counting")
        self.assertGreater(n, 0)

    def test_count_tokens_specified_language(self):
        """指定语言时使用对应比率估算（tiktoken 不可用时回退）。"""
        n = count_tokens("你好世界", language="zh")
        self.assertGreater(n, 0)

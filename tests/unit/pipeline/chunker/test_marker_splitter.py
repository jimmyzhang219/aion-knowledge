import unittest

from aion_knowledge.pipeline.chunker.base import ChunkConfig
from aion_knowledge.pipeline.chunker.marker_splitter import MarkerSplitter


class TestMarkerSplitter(unittest.TestCase):
    def setUp(self):
        self.config = ChunkConfig(chunk_size=512, chunk_overlap=80)
        self.splitter = MarkerSplitter(self.config)

    def test_empty_text(self):
        result = self.splitter.split("")
        self.assertEqual(len(result), 0)

    def test_short_text_no_split(self):
        text = "Just a short paragraph."
        result = self.splitter.split(text)
        self.assertEqual(len(result), 1)

    def test_split_at_visual_separator(self):
        text = ("Section one content.\n\n---\n\nSection two content.\n\n") * 80
        result = self.splitter.split(text)
        self.assertGreater(len(result), 1)

    def test_split_at_numbered_section(self):
        text = ("1.1 Introduction\ncontent here.\n\n1.2 Methods\nmore content.\n\n") * 50
        result = self.splitter.split(text)
        self.assertGreater(len(result), 1)

    def test_no_marker_boundaries_fallsback(self):
        """没有结构标记边界时，应回退到递归分块（产生至少 1 个 chunk）。"""
        text = "Simple text. " * 200
        result = self.splitter.split(text)
        self.assertGreaterEqual(len(result), 1)

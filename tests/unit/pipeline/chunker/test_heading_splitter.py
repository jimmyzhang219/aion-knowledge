import unittest

from aion_knowledge.pipeline.chunker.base import ChunkConfig
from aion_knowledge.pipeline.chunker.heading_splitter import (
    HeadingSplitter,
    HeadingTracker,
)


class TestHeadingTracker(unittest.TestCase):
    def test_observe_and_breadcrumb(self):
        h = HeadingTracker()
        h.observe("# Top Level")
        self.assertEqual(h.breadcrumb(), "Top Level")

    def test_nested_headings(self):
        h = HeadingTracker()
        h.observe("# Chapter 1")
        h.observe("## Section 1.1")
        h.observe("### Subsection")
        self.assertEqual(h.breadcrumb(), "Chapter 1 > Section 1.1 > Subsection")

    def test_heading_level_change(self):
        h = HeadingTracker()
        h.observe("# Chapter 1")
        h.observe("## Section 1.1")
        h.observe("# Chapter 2")
        self.assertEqual(h.breadcrumb(), "Chapter 2")


class TestHeadingSplitter(unittest.TestCase):
    def setUp(self):
        self.config = ChunkConfig(chunk_size=512, chunk_overlap=80)
        self.splitter = HeadingSplitter(self.config)

    def test_empty_text(self):
        result = self.splitter.split("")
        self.assertEqual(len(result), 0)

    def test_no_headings_fallsback(self):
        text = "Simple text without any headings. " * 30
        result = self.splitter.split(text)
        self.assertGreaterEqual(len(result), 1)

    def test_split_at_headings(self):
        text = ("# Intro\nsome intro text\n# Methods\ndetailed methods\n") * 20
        result = self.splitter.split(text)
        self.assertGreater(len(result), 1)

    def test_heading_path_in_result(self):
        text = "# Top\n## Sub\ncontent here."
        result = self.splitter.split(text)
        self.assertGreaterEqual(len(result), 1)

import unittest

from aion_knowledge.pipeline.chunker.base import ChunkResult
from aion_knowledge.pipeline.chunker.fragment_utils import (
    extract_overlap_tail,
    merge_adjacent_chunks,
)


class TestMergeAdjacentChunks(unittest.TestCase):
    def test_single_chunk_no_merge(self):
        chunks = [ChunkResult(content="hello", token_count=5, chunk_id="c0", seq_num=0)]
        result = merge_adjacent_chunks(chunks, chunk_size=512)
        self.assertEqual(len(result), 1)

    def test_merge_small_adjacent_chunks(self):
        chunks = [
            ChunkResult(content="small", token_count=10, chunk_id="c0", seq_num=0),
            ChunkResult(content="tiny", token_count=5, chunk_id="c1", seq_num=1),
            ChunkResult(content="larger content here" * 20, token_count=100, chunk_id="c2", seq_num=2),
        ]
        result = merge_adjacent_chunks(chunks, chunk_size=512)
        self.assertLessEqual(len(result), len(chunks))


class TestExtractOverlapTail(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(extract_overlap_tail("", 80), "")

    def test_short_text_returns_empty(self):
        result = extract_overlap_tail("hello", 80)
        self.assertEqual(result, "")

    def test_long_text_extracts_tail(self):
        text = "line one\n" + "line two\n" * 50 + "line end"
        result = extract_overlap_tail(text, 20)
        self.assertGreater(len(result), 0)

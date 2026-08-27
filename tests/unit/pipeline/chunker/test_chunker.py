import unittest

from aion_knowledge.pipeline.chunker import ChunkConfig, Chunker


class TestChunkerIntegration(unittest.TestCase):
    def test_chunker_no_split(self):
        chunker = Chunker(ChunkConfig(strategy="no_split"))
        result = chunker.split("hello world")
        self.assertEqual(len(result), 1)

    def test_chunker_recursive(self):
        chunker = Chunker(ChunkConfig(strategy="recursive"))
        text = "Hello. " * 500
        result = chunker.split(text)
        self.assertGreater(len(result), 1)
        for chunk in result:
            self.assertGreater(chunk.token_count, 0)

    def test_chunker_heading(self):
        chunker = Chunker(ChunkConfig(strategy="heading"))
        text = (
            "# Introduction\n"
            + "This is the introduction section. It provides background context for the overall topic. " * 20
            + "\n\n## Background\n"
            + "The background section reviews related work and prior research in this area. " * 20
            + "\n\n## Methods\n"
            + "The methods section describes our approach and experimental setup in detail. " * 20
            + "\n\n# Results\n"
            + "The results present our key findings and observations from the experiments. " * 20
        )
        result = chunker.split(text)
        self.assertGreater(len(result), 1)

    def test_chunker_heuristic(self):
        chunker = Chunker(ChunkConfig(strategy="heuristic"))
        text = (
            "1.1 Introduction\n" + "This section describes the background and motivation for the research work. " * 100
            + "\n---\n"
            + "1.2 Methods\n" + "The methodology section explains the approach and experimental design used. " * 100
        )
        result = chunker.split(text)
        self.assertGreater(len(result), 1)

    def test_chunker_auto(self):
        chunker = Chunker(ChunkConfig(strategy="auto"))
        text = (
            "# Overview\n"
            + "Overview paragraph with enough content to be meaningful for the topic. " * 20
            + "\n## Detail 1\n"
            + "Detailed content for the first subsection with proper explanatory text. " * 20
            + "\n## Detail 2\n"
            + "Detailed content for the second subsection with proper explanatory text. " * 20
            + "\n## Detail 3\n"
            + "Detailed content for the third subsection with proper explanatory text. " * 20
        )
        result = chunker.split(text)
        self.assertGreater(len(result), 1)

    def test_empty_text(self):
        chunker = Chunker(ChunkConfig())
        result = chunker.split("")
        self.assertEqual(len(result), 0)

    def test_auto_fallback_to_recursive_for_plain_text(self):
        """Plain text without headings/markers should use recursive splitter."""
        chunker = Chunker(ChunkConfig(strategy="auto"))
        text = "Just plain text. " * 500
        result = chunker.split(text)
        self.assertGreater(len(result), 1)

    def test_quality_rejects_single_large_chunk(self):
        """A large document that produces only one chunk should trigger quality check."""
        chunker = Chunker(ChunkConfig(strategy="heading"))
        text = "No headings here. " * 500
        result = chunker.split(text)
        self.assertGreater(len(result), 1)


class TestTableIntegration(unittest.TestCase):
    """表格在完整分块流水线中的集成测试。"""

    def _make_table(self, rows: int) -> str:
        header = "| Col A | Col B | Col C |\n"
        sep = "| --- | --- | --- |\n"
        data = "\n".join(f"| val_{i}_a | val_{i}_b | val_{i}_c |" for i in range(rows))
        return header + sep + data

    def test_recursive_preserves_inline_table(self):
        """RecursiveSplitter 不应切断嵌入在段落中的小表格。"""
        chunker = Chunker(ChunkConfig(strategy="recursive", chunk_size=200))
        text = "介绍文字。\n\n" + self._make_table(5) + "\n\n结束文字。"
        result = chunker.split(text)
        self.assertGreaterEqual(len(result), 1)
        # 所有表格行应出现在同一个 chunk 中
        table_chunks = [c for c in result if "| Col A |" in c.content]
        self.assertEqual(len(table_chunks), 1)

    def test_recursive_splits_oversized_table(self):
        """超大表格应被强制分割，续片前置表头（因 overlap 机制第一行可能为分隔线，
        但表头行应出现在前几行内）。"""
        chunker = Chunker(ChunkConfig(strategy="recursive", chunk_size=80))
        text = self._make_table(50)
        result = chunker.split(text)
        self.assertGreater(len(result), 1)
        # 每个含表格行的续片 chunk 应包含表头行
        for i, chunk in enumerate(result):
            if i > 0 and "| val_" in chunk.content:
                self.assertIn(
                    "| Col A |",
                    chunk.content,
                    msg=f"chunk {i} 缺少表头",
                )
                # 验证 chunks 没有被字符级拆分破坏
                self.assertNotIn("val_|", chunk.content)

    def test_heading_splitter_fallback_protects_tables(self):
        """HeadingSplitter 下放超长章节到 RecursiveSplitter 时表格续片应包含表头。"""
        chunker = Chunker(ChunkConfig(strategy="heading", chunk_size=100))
        text = "# Section\n" + "intro\n\n" + self._make_table(30) + "\n\noutro"
        result = chunker.split(text)
        self.assertGreaterEqual(len(result), 1)
        for i, chunk in enumerate(result):
            if i > 0 and "| val_" in chunk.content:
                self.assertIn(
                    "| Col A |",
                    chunk.content,
                    msg=f"chunk {i} 缺少表头",
                )

    def test_marker_splitter_fallback_protects_tables(self):
        """MarkerSplitter 下放超长章节到 RecursiveSplitter 时表格续片应包含表头。"""
        chunker = Chunker(ChunkConfig(strategy="heuristic", chunk_size=100))
        text = "1.1 Title\nintro\n\n" + self._make_table(30) + "\n\noutro"
        result = chunker.split(text)
        self.assertGreaterEqual(len(result), 1)
        for i, chunk in enumerate(result):
            if i > 0 and "| val_" in chunk.content:
                self.assertIn(
                    "| Col A |",
                    chunk.content,
                    msg=f"chunk {i} 缺少表头",
                )

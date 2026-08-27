import io
import unittest
from pathlib import Path

from markitdown import MarkItDown

from aion_knowledge.pipeline.parser.markdown import MarkdownTableUtil


class TestMarkdownTableUtil(unittest.TestCase):
    def test_preserves_empty_cells(self):
        """Interior empty cells must not be dropped during formatting."""
        raw = "| a |  | c |\n| --- | --- | --- |\n| 1 | 2 | 3 |"
        formatted = MarkdownTableUtil().format_table(raw)
        self.assertIn("| a |  | c |", formatted)
        self.assertEqual(formatted.count("|"), raw.count("|"))

    def test_format_nonempty_table(self):
        raw = "|Name|Age|\n|---|---|\n|John|30|"
        formatted = MarkdownTableUtil().format_table(raw)
        self.assertIn("| Name | Age |", formatted)
        self.assertIn("| --- | --- |", formatted)
        self.assertIn("| John | 30 |", formatted)

    def test_normalize_markitdown_en_tables(self):
        for p in (
            Path(__file__).resolve().parents[4]
            / "testdata" / "rag_test" / "docx" / "en_tables.docx",
            Path("/tmp/en_tables.docx"),
        ):
            if p.is_file():
                docx = p
                break
        else:
            self.skipTest("en_tables.docx test fixture not available")
        raw = MarkItDown().convert(io.BytesIO(docx.read_bytes()), file_extension=".docx").text_content
        normalized = MarkdownTableUtil().format_table(raw)

        self.assertNotIn("|  |  |  |  |", normalized)
        self.assertIn("| Name | Game | Fame | Blame |", normalized)
        idx_name = normalized.index("| Name | Game | Fame | Blame |")
        idx_sep = normalized.index("| --- | --- | --- | --- |", idx_name)
        self.assertLess(idx_name, idx_sep)
        self.assertIn("| Lebron James | Basketball |", normalized)

        # Headerless 2-row tables: delimiter inserted so GFM renderers show a table
        self.assertIn(
            "| Sinple | Table |\n| --- | --- |\n| Without | Header |", normalized
        )
        self.assertIn(
            "| Simple  Multiparagraph | Table  Full |\n| --- | --- |\n"
            "| Of  Paragraphs | In each  Cell. |",
            normalized,
        )


if __name__ == "__main__":
    unittest.main()

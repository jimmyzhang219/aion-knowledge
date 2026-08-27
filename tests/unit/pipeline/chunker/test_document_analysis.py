import unittest

from aion_knowledge.pipeline.chunker.document_analysis import DocumentAnalyzer


class TestDocumentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = DocumentAnalyzer()

    def test_empty_text(self):
        features = self.analyzer.analyze("")
        self.assertEqual(features.total_chars, 0)

    def test_heading_detection(self):
        text = "# Title\npara\n## Section 1\npara\n## Section 2\npara"
        features = self.analyzer.analyze(text)
        self.assertEqual(features.md_heading_total, 3)
        self.assertEqual(features.md_heading_counts[1], 1)
        self.assertEqual(features.md_heading_counts[2], 2)

    def test_heading_density(self):
        text = "# H1\npara\n## H2\npara\n### H3\npara"
        features = self.analyzer.analyze(text)
        self.assertAlmostEqual(features.heading_density, 3 / 6)

    def test_dominant_heading_level(self):
        text = "# A\n## B\n## C\n## D\n"
        features = self.analyzer.analyze(text)
        self.assertEqual(features.dominant_heading_level, 2)

    def test_no_headings(self):
        text = "plain text\nwithout any markdown\nheadings"
        features = self.analyzer.analyze(text)
        self.assertEqual(features.md_heading_total, 0)
        self.assertEqual(features.dominant_heading_level, 0)

    def test_structural_markers(self):
        text = "1.1 First\n---\n2.2 Second\n\f\n"
        features = self.analyzer.analyze(text)
        self.assertGreater(features.structural_marker_total, 0)

    def test_code_and_tables(self):
        text = "text\n```\ncode\n```\n| table |\n|---|\n| cell |"
        features = self.analyzer.analyze(text)
        self.assertTrue(features.has_code)
        self.assertTrue(features.has_tables)

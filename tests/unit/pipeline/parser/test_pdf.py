import base64
import io
import os
import unittest
from unittest.mock import patch

from PIL import Image

from aion_knowledge.pipeline.parser.pdf import (
    PDFParser,
    PDFScannedParser,
    _classify_page,
    _filter_reading_columns,
    _group_lines,
    _is_table_line,
    _join_line_glyphs,
    _merge_cross_page_tables,
    _merge_orphan_punctuation_lines,
    _normalize_image_quality,
    _point_in_boxes,
    _segments_to_markdown,
    _select_embedded_images,
    _should_prefer_plain,
    _split_columns,
    _strip_repeating_lines,
)


def _char(ch, x0, x1, y0, y1):
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "ch": ch}


def _line(text, h):
    return {"text": text, "h": h}


def _make_image_only_pdf(num_pages: int = 2) -> bytes:
    buf = io.BytesIO()
    pages = [Image.new("RGB", (64, 64), color) for color in ("white", "black")]
    pages = (pages * ((num_pages // 2) + 1))[:num_pages]
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


# ── Unit Tests (pure function) ─────────────────────────────────────────────


class ClassifyPageTest(unittest.TestCase):
    def test_full_page_image_is_scanned_even_with_text(self):
        # Scanned newspaper: image covers the page, embedded OCR text exists.
        self.assertEqual(_classify_page(2.0, 1620), "scanned")
        # Scanned UN doc with garbled OCR text layer (ratio ~1.0).
        self.assertEqual(_classify_page(1.0, 1500), "scanned")

    def test_native_text_page_is_text(self):
        self.assertEqual(_classify_page(0.01, 673), "text")
        self.assertEqual(_classify_page(0.0, 1200), "text")

    def test_sparse_text_with_image_is_scanned(self):
        self.assertEqual(_classify_page(0.3, 2), "scanned")

    def test_blank_page_is_text(self):
        # No image, no text -> not rendered as an image.
        self.assertEqual(_classify_page(0.0, 0), "text")


class StripRepeatingLinesTest(unittest.TestCase):
    def test_removes_repeated_header_footer(self):
        header = "ACME CONFIDENTIAL"
        texts = [f"{header}\nbody page {i}\npage {i} footer" for i in range(6)]
        # Make the footer identical across pages so it is detected.
        texts = [f"{header}\nbody page {i}\nshared footer" for i in range(6)]
        classes = ["text"] * 6
        cleaned = _strip_repeating_lines(texts, classes)
        for page in cleaned:
            self.assertNotIn(header, page)
            self.assertNotIn("shared footer", page)
        self.assertIn("body page 0", cleaned[0])

    def test_keeps_lines_when_too_few_pages(self):
        texts = ["HEADER\nbody"] * 2
        classes = ["text"] * 2
        self.assertEqual(_strip_repeating_lines(texts, classes), texts)


class SelectEmbeddedImagesTest(unittest.TestCase):
    def _fig(self, page, h="fig", w=200, ht=200, area=0.2):
        return {"page": page, "width": w, "height": ht, "area_ratio": area, "hash": h}

    def test_keeps_real_figure(self):
        meta = [self._fig(0, "a")]
        self.assertEqual(_select_embedded_images(meta, 1), [0])

    def test_drops_tiny_and_small_images(self):
        meta = [
            self._fig(0, "tiny_area", area=0.001),  # too small a share
            self._fig(0, "tiny_px", w=20, ht=20),  # too few pixels
        ]
        self.assertEqual(_select_embedded_images(meta, 1), [])

    def test_drops_repeated_logo_watermark(self):
        # Same hash on 5 of 6 text pages -> running logo/watermark.
        meta = [self._fig(p, "logo") for p in range(5)]
        meta.append(self._fig(5, "unique"))
        kept = _select_embedded_images(meta, 6)
        kept_hashes = {meta[i]["hash"] for i in kept}
        self.assertNotIn("logo", kept_hashes)
        self.assertIn("unique", kept_hashes)

    def test_dedups_identical_image_on_same_page(self):
        meta = [self._fig(0, "dup"), self._fig(0, "dup")]
        self.assertEqual(len(_select_embedded_images(meta, 1)), 1)

    def test_respects_max_images_cap(self):
        meta = [self._fig(i, f"h{i}") for i in range(10)]
        self.assertEqual(len(_select_embedded_images(meta, 10, max_images=3)), 3)


class ReadingOrderTest(unittest.TestCase):
    def test_single_column_stays_single(self):
        # One column of glyphs at x~100, no full-height gutter.
        chars = [_char("a", 100, 110, 700 - i * 12, 712 - i * 12) for i in range(5)]
        cols = _split_columns(chars, scale=12.0, width=600.0)
        self.assertEqual(len(cols), 1)

    def test_two_columns_split_left_to_right(self):
        # Left column x~50-150, right column x~400-500, wide empty gutter between.
        left = [_char("L", 50, 150, 700 - i * 12, 712 - i * 12) for i in range(4)]
        right = [_char("R", 400, 500, 700 - i * 12, 712 - i * 12) for i in range(4)]
        cols = _split_columns(left + right, scale=12.0, width=600.0)
        self.assertEqual(len(cols), 2)
        # Reading order: left column before right column.
        self.assertEqual(cols[0][0]["ch"], "L")
        self.assertEqual(cols[1][0]["ch"], "R")

    def test_group_lines_orders_by_y_then_x(self):
        # Two visual lines; within a line glyphs given out of x-order.
        chars = [
            _char("B", 110, 120, 700, 712),  # adjacent to A (no word-sized gap)
            _char("A", 100, 110, 700, 712),
            _char("C", 100, 110, 680, 692),  # next line down
        ]
        lines = _group_lines(chars)
        self.assertEqual([ln["text"] for ln in lines], ["AB", "C"])

    def test_join_line_glyphs_inserts_word_spaces(self):
        # Wide gap between "copy" and "of" mimics positioned OCR / text layers.
        chars = [
            _char("c", 0, 4, 0, 10),
            _char("f", 10, 14, 0, 10),
        ]
        self.assertEqual(_join_line_glyphs(chars), "c f")

    def test_join_line_glyphs_keeps_adjacent_letters(self):
        chars = [_char("A", 100, 110, 700, 712), _char("B", 110, 120, 700, 712)]
        self.assertEqual(_join_line_glyphs(chars), "AB")


class HeadingDetectionTest(unittest.TestCase):
    def test_promotes_large_line_to_heading(self):
        lines = [_line("Big Title", 24.0)] + [_line(f"body {i}", 10.0) for i in range(6)]
        md = _segments_to_markdown(lines)
        self.assertTrue(md.startswith("# Big Title"))
        self.assertIn("\nbody 0", md)

    def test_does_not_promote_when_sizes_uniform(self):
        lines = [_line(f"line {i}", 10.0) for i in range(6)]
        md = _segments_to_markdown(lines)
        self.assertNotIn("#", md)

    def test_skips_sentence_like_long_lines(self):
        # Large but ends with a period and is long -> body text, not a heading.
        lines = [_line("x" * 90 + ".", 30.0)] + [_line("y", 10.0) for _ in range(6)]
        md = _segments_to_markdown(lines)
        self.assertFalse(md.startswith("#"))


class HiddenTextFilterTest(unittest.TestCase):
    def test_point_in_boxes(self):
        boxes = [(0.0, 0.0, 10.0, 10.0)]
        self.assertTrue(_point_in_boxes(5.0, 5.0, boxes))
        self.assertFalse(_point_in_boxes(20.0, 5.0, boxes))


class MarginColumnFilterTest(unittest.TestCase):
    def test_drops_narrow_vertical_margin_column(self):
        # Mimics arXiv sidebar: narrow x span, one glyph per line.
        margin = [
            _char(c, 20, 28, 500 - i * 14, 512 - i * 14)
            for i, c in enumerate("0202luJ22")
        ]
        body = [
            _char("L", 160, 170, 700, 712),
            _char("a", 170, 180, 700, 712),
            _char("n", 180, 190, 700, 712),
        ]
        cols = _filter_reading_columns(margin + body, scale=10.0, width=612.0)
        self.assertEqual(len(cols), 1)
        self.assertEqual(cols[0][0]["ch"], "L")

    def test_keeps_real_two_column_layout(self):
        left = [_char("L", 50, 150, 700 - i * 12, 712 - i * 12) for i in range(4)]
        right = [_char("R", 400, 500, 700 - i * 12, 712 - i * 12) for i in range(4)]
        cols = _filter_reading_columns(left + right, scale=12.0, width=600.0)
        self.assertEqual(len(cols), 2)


class PunctuationMergeTest(unittest.TestCase):
    def test_merges_orphan_periods(self):
        lines = [
            _line("Figure 1 2", 10.0),
            _line(". .", 10.0),
            _line("Next", 10.0),
        ]
        merged = _merge_orphan_punctuation_lines(lines)
        self.assertEqual([ln["text"] for ln in merged], ["Figure 1 2..", "Next"])


class PdfTextSanitizeTest(unittest.TestCase):
    def test_removes_fffe_placeholder(self):
        from aion_knowledge.pipeline.parser.pdf import _postprocess_pdf_text

        raw = "multi￾layer and non￾trivial"
        out = _postprocess_pdf_text(raw)
        self.assertEqual(out, "multilayer and nontrivial")

    def test_strips_chart_axis_run(self):
        from aion_knowledge.pipeline.parser.pdf import _postprocess_pdf_text

        raw = (
            "Deep convolutional neural networks have led to breakthroughs.\n"
            "0 1 2 3 4 5 6 0\n"
            "10\n"
            "20\n"
            "iter. (1e4)\n"
            "training error (%)\n"
            "56-layer\n"
            "20-layer\n"
            "Figure 1. Training error on CIFAR-10.\n"
        )
        out = _postprocess_pdf_text(raw)
        self.assertIn("breakthroughs", out)
        self.assertNotIn("56-layer", out)
        self.assertIn("Figure 1.", out)

    def test_strips_diagram_labels_above_caption(self):
        from aion_knowledge.pipeline.parser.pdf import _postprocess_pdf_text

        raw = (
            "Paragraph before.\n"
            "identity\n"
            "weight layer\n"
            "relu\n"
            "Figure 2. Residual learning block.\n"
            "Paragraph after.\n"
        )
        out = _postprocess_pdf_text(raw)
        self.assertIn("Paragraph before.", out)
        self.assertIn("Figure 2.", out)
        self.assertIn("Paragraph after.", out)
        self.assertNotIn("identity", out)
        self.assertNotIn("weight layer", out)

    def test_strips_arxiv_header_line(self):
        from aion_knowledge.pipeline.parser.pdf import _postprocess_pdf_text

        raw = "Body text.\n1\narXiv:1512.03385v1 [cs.CV] 10 Dec 2015\nMore body."
        out = _postprocess_pdf_text(raw)
        self.assertNotIn("arXiv:", out)
        self.assertIn("Body text.", out)


class PlainWellFormedTest(unittest.TestCase):
    def test_academic_plain_skips_layout(self):
        from aion_knowledge.pipeline.parser.pdf import _plain_is_well_formed

        plain = (
            "Recent work [DL15, MBXS17] shows progress on NLP tasks "
            "with pre-trained models."
        )
        self.assertTrue(_plain_is_well_formed(plain))

    def test_glued_scan_plain_needs_layout(self):
        from aion_knowledge.pipeline.parser.pdf import _plain_is_well_formed

        self.assertFalse(_plain_is_well_formed("Thisisadigitalcopyofabook"))


class LayoutQualityFallbackTest(unittest.TestCase):
    def test_prefers_plain_when_many_single_char_lines(self):
        plain = "Language Models are Few-Shot Learners\nTom Brown"
        layout = "0\n2\n0\n2\nl\nu\nJ\nLan ua e Models"
        self.assertTrue(_should_prefer_plain(plain, layout))

    def test_keeps_good_layout(self):
        plain = "Hello world"
        layout = "Hello world"
        self.assertFalse(_should_prefer_plain(plain, layout))


class ResNetPaperFigureTest(unittest.TestCase):
    """Regression: ResNet PDF (arXiv:1512.03385) vector figures and captions."""

    def test_resnet_figures_and_captions(self):
        from aion_knowledge.pipeline.parser.pdf import PDFParser

        for path in (
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "testdata",
                "rag_test",
                "pdf_en",
                "resnet.pdf",
            ),
            "/tmp/resnet.pdf",
        ):
            if os.path.isfile(path):
                break
        else:
            self.skipTest("resnet.pdf not available")

        with open(path, "rb") as f:
            doc = PDFParser(file_name="resnet.pdf", file_type="pdf").parse_into_text(
                f.read()
            )
        self.assertGreater(doc.metadata.get("vector_figure_count", 0), 0)
        self.assertIn("![", doc.content)
        self.assertIn("Figure 2. Residual learning", doc.content)
        self.assertNotIn("arXiv:", doc.content)
        fig2 = doc.content.find("Figure 2. Residual learning")
        before = doc.content[max(0, fig2 - 120) : fig2]
        self.assertIn("![", before)
        self.assertNotIn("identity", before)


class Gpt3PaperLayoutTest(unittest.TestCase):
    """Regression: arXiv GPT-3 paper title page must not be one-glyph-per-line."""

    def test_gpt3_page0_title_and_authors(self):
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_r

        from aion_knowledge.pipeline.parser.pdf import PDFParser, _extract_layout_text

        pdf_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "testdata",
            "rag_test",
            "pdf_en",
            "gpt3.pdf",
        )
        if not os.path.isfile(pdf_path):
            self.skipTest("gpt3.pdf not in testdata")
        with open(pdf_path, "rb") as f:
            content = f.read()
        with pdfium.PdfDocument(content) as pdf:
            page = pdf[0]
            try:
                layout = _extract_layout_text(page, pdfium_r)
            finally:
                page.close()
        # Margin sidebar must not appear as one-glyph-per-line prefix.
        self.assertNotRegex(layout[:300], r"^0\n2\n0\n2")
        self.assertIn("Few-Shot Learners", layout)

        doc = PDFParser(file_name="gpt3.pdf", file_type="pdf").parse_into_text(content)
        self.assertIn("Language Models are Few-Shot Learners", doc.content)
        self.assertIn("Tom B. Brown", doc.content[:1200])
        self.assertIn("[DL15, MBXS17, PNZtY18]", doc.content)
        self.assertIn("task-specific architectures), and more recently", doc.content)
        self.assertNotIn("k ifi hi d l", doc.content)


class ScanEnglishDictLayoutTest(unittest.TestCase):
    """Regression: Google Books-style PDFs lose spaces without gap inference."""

    def test_scan_en_dict_page0_has_word_spaces(self):
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_r

        from aion_knowledge.pipeline.parser.pdf import _extract_layout_text

        pdf_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "testdata",
            "rag_test",
            "pdf_scan",
            "scan_en_dict.pdf",
        )
        if not os.path.isfile(pdf_path):
            self.skipTest("scan_en_dict.pdf not in testdata")
        with open(pdf_path, "rb") as f:
            pdf = pdfium.PdfDocument(f.read())
        try:
            text = _extract_layout_text(pdf[0], pdfium_r)
        finally:
            pdf.close()
        self.assertIn("This is a digital copy of a book", text)
        self.assertNotIn("Thisisadigitalcopyofabook", text)


class PDFRouterIntegrationTest(unittest.TestCase):
    def test_image_only_pdf_routes_to_scanned(self):
        pdf_bytes = _make_image_only_pdf(2)
        doc = PDFParser(file_name="imgonly.pdf", file_type="pdf").parse_into_text(
            pdf_bytes
        )

        self.assertEqual(doc.metadata["image_source_type"], "scanned_pdf")
        self.assertEqual(doc.metadata["page_count"], 2)
        self.assertEqual(doc.metadata["scanned_page_count"], 2)
        self.assertEqual(len(doc.images), 2)
        self.assertIn("images/imgonly_page_1.jpg", doc.images)
        self.assertIn("![imgonly_page_1.jpg](images/imgonly_page_1.jpg)", doc.content)
        # JPEG magic bytes after decoding.
        import base64

        self.assertTrue(
            base64.b64decode(doc.images["images/imgonly_page_1.jpg"]).startswith(
                b"\xff\xd8"
            )
        )

    def test_malformed_pdf_raises_after_fallback(self):
        # Routing fails to open the PDF, falls back to full rendering which also
        # fails on garbage input; the error surfaces to the caller.
        with self.assertRaises(Exception):
            PDFParser(file_name="broken.pdf", file_type="pdf").parse_into_text(
                b"not a pdf"
            )


class ForceScannedTest(unittest.TestCase):
    """Tests for the force-scanned PDF parsing mode."""

    def test_default_text_pdf_is_not_force_scanned(self):
        """Without any override, a text PDF should NOT use scanned mode."""
        parser = PDFParser(file_name="normal.pdf", file_type="pdf")
        self.assertFalse(parser._force_scanned)

    def test_force_scanned_via_kwarg_true(self):
        """Per-upload override pdf_force_scanned=true enables scanned mode."""
        parser = PDFParser(
            file_name="test.pdf", file_type="pdf", pdf_force_scanned="true"
        )
        self.assertTrue(parser._force_scanned)

    def test_force_scanned_via_kwarg_false(self):
        """Per-upload override pdf_force_scanned=false disables scanned mode."""
        parser = PDFParser(
            file_name="test.pdf", file_type="pdf", pdf_force_scanned="false"
        )
        self.assertFalse(parser._force_scanned)

    def test_force_scanned_via_kwarg_overrides_env(self, monkeypatch=None):
        """Per-upload override takes priority over global env var."""
        import aion_knowledge.pipeline.parser.pdf as pdf_mod

        old = pdf_mod.FORCE_SCANNED_PDF
        try:
            pdf_mod.FORCE_SCANNED_PDF = True
            # Explicit false override should win over env=true
            parser = PDFParser(
                file_name="test.pdf", file_type="pdf", pdf_force_scanned="false"
            )
            self.assertFalse(parser._force_scanned)
        finally:
            pdf_mod.FORCE_SCANNED_PDF = old

    def test_force_scanned_via_env(self):
        """Global env variable enables force scanned when no override."""
        import aion_knowledge.pipeline.parser.pdf as pdf_mod

        old = pdf_mod.FORCE_SCANNED_PDF
        try:
            pdf_mod.FORCE_SCANNED_PDF = True
            parser = PDFParser(file_name="test.pdf", file_type="pdf")
            self.assertTrue(parser._force_scanned)
        finally:
            pdf_mod.FORCE_SCANNED_PDF = old

    def test_force_scanned_output_is_scanned_pdf(self):
        """Force-scanned mode produces scanned_pdf metadata and image refs."""
        pdf_bytes = _make_image_only_pdf(2)
        parser = PDFParser(
            file_name="forced.pdf", file_type="pdf", pdf_force_scanned="true"
        )
        doc = parser.parse_into_text(pdf_bytes)
        self.assertEqual(doc.metadata.get("image_source_type"), "scanned_pdf")
        self.assertEqual(doc.metadata.get("page_count"), 2)
        self.assertEqual(doc.metadata.get("scanned_page_count"), 2)
        self.assertEqual(doc.metadata.get("text_page_count"), 0)
        self.assertEqual(doc.metadata.get("embedded_image_count"), 0)
        self.assertEqual(doc.metadata.get("vector_figure_count"), 0)
        self.assertIn("![forced_page_1.jpg]", doc.content)
        self.assertIn("![forced_page_2.jpg]", doc.content)
        self.assertEqual(len(doc.images), 2)


# ── PDFScannedParser + helpers (from test_parser_concurrency) ──────────────

class PDFScannedParserTest(unittest.TestCase):
    def test_scanned_pdf_parser_outputs_jpeg_images(self):
        pdf_bytes = io.BytesIO()
        pages = [
            Image.new("RGB", (64, 64), "white"),
            Image.new("RGB", (64, 64), "black"),
        ]
        pages[0].save(
            pdf_bytes,
            format="PDF",
            save_all=True,
            append_images=pages[1:],
        )

        document = PDFScannedParser(file_name="scan.pdf").parse_into_text(
            pdf_bytes.getvalue()
        )

        image_ref = "images/scan_page_1.jpg"
        self.assertIn(f"![scan_page_1.jpg]({image_ref})", document.content)
        self.assertIn(image_ref, document.images)
        self.assertEqual(document.metadata["image_source_type"], "scanned_pdf")
        self.assertEqual(document.metadata["page_count"], 2)
        self.assertEqual(len(document.images), 2)
        self.assertIn("images/scan_page_2.jpg", document.images)
        image_bytes = base64.b64decode(document.images[image_ref])
        self.assertTrue(image_bytes.startswith(b"\xff\xd8"))

    def test_scanned_pdf_parser_logs_malformed_pdf_without_format_error(self):
        parser = PDFScannedParser(file_name="broken.pdf")
        with self.assertLogs("aion_knowledge.pipeline.parser.pdf", level="ERROR") as logs:
            with self.assertRaises(Exception):
                parser.parse_into_text(b"not a pdf")
        self.assertTrue(
            any("PDFScannedParser failed to parse PDF:" in line for line in logs.output)
        )

    def test_normalize_image_quality_bounds_jpeg_quality(self):
        self.assertEqual(_normalize_image_quality(-1), 1)
        self.assertEqual(_normalize_image_quality(90), 90)
        self.assertEqual(_normalize_image_quality(120), 95)


class TestCrossPageTableMerge(unittest.TestCase):
    """跨页表格合并测试。"""

    def test_is_table_line_detection(self):
        self.assertTrue(_is_table_line("| a | b |"))
        self.assertTrue(_is_table_line("  | a | b |  "))
        self.assertTrue(_is_table_line("|"))
        self.assertFalse(_is_table_line(""))
        self.assertFalse(_is_table_line("a | b"))
        self.assertFalse(_is_table_line(" "))

    def test_merge_two_text_pages_with_table(self):
        texts = [
            "前文内容。\n| 日期 | 金额 |\n| 1月 | 100 |",
            "| 2月 | 200 |\n| 3月 | 300 |\n后文内容。",
        ]
        classes = ["text", "text"]
        result = _merge_cross_page_tables(texts, classes)
        self.assertEqual(len(result), 1)
        self.assertIn("| 日期 | 金额 |", result[0])
        self.assertIn("| 2月 | 200 |", result[0])
        self.assertIn("| 3月 | 300 |", result[0])
        self.assertIn("前文内容", result[0])
        self.assertIn("后文内容", result[0])
        # 确保表格行之间没有 \n\n 页间分隔符
        self.assertNotIn("| 1月 | 100 |\n\n| 2月 | 200 |", result[0])

    def test_no_merge_when_first_page_not_table(self):
        texts = [
            "普通段落末尾。",
            "| 2月 | 200 |\n这个也不是表格",
        ]
        classes = ["text", "text"]
        result = _merge_cross_page_tables(texts, classes)
        self.assertEqual(len(result), 2)

    def test_no_merge_when_second_page_not_table(self):
        texts = [
            "结尾 | 表格 | 行",
            "普通段落开头。",
        ]
        classes = ["text", "text"]
        result = _merge_cross_page_tables(texts, classes)
        self.assertEqual(len(result), 2)

    def test_merge_skipped_for_scanned_pages(self):
        texts = [
            "| 表1 | 数据 |",
            "![page2](images/page2.jpg)",
        ]
        classes = ["text", "scanned"]
        result = _merge_cross_page_tables(texts, classes)
        self.assertEqual(len(result), 2)

    def test_merge_triple_page_table(self):
        """跨三页的表格应完全合并。"""
        texts = [
            "| A | B |",
            "| C | D |",
            "| E | F |",
        ]
        classes = ["text", "text", "text"]
        result = _merge_cross_page_tables(texts, classes)
        self.assertEqual(len(result), 1)
        self.assertIn("| A | B |", result[0])
        self.assertIn("| C | D |", result[0])
        self.assertIn("| E | F |", result[0])

    def test_no_merge_mixed_content(self):
        """非表格行的跨页内容不应合并。"""
        texts = [
            "第一页正文。\n还有更多内容。",
            "第二页正文。\n结束。",
        ]
        classes = ["text", "text"]
        result = _merge_cross_page_tables(texts, classes)
        self.assertEqual(len(result), 2)


class TestExternalPdfParser(unittest.TestCase):
    """外部 PDF 解析器的 mock 测试。"""

    def setUp(self):
        from aion_knowledge.pipeline.parser.external_pdf import ExternalPdfParser
        self.parser_cls = ExternalPdfParser

    @staticmethod
    def _make_test_pdf():
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), "white").save(buf, format="PDF")
        return buf.getvalue()

    def test_raises_when_no_url(self):
        """必须配置 URL，否则引发 RuntimeError。"""
        parser = self.parser_cls(file_name="test.pdf")
        # 不传 pdf_external_url → settings 中默认空字符串 → RuntimeError
        with patch("aion_knowledge.pipeline.parser.external_pdf.settings") as mock_settings:
            mock_settings.pdf_external_url = ""
            with self.assertRaises(RuntimeError):
                parser.parse_into_text(self._make_test_pdf())

    def test_falls_back_to_scanned_on_http_error(self):
        pdf_bytes = self._make_test_pdf()
        parser = self.parser_cls(
            file_name="test.pdf",
            pdf_external_url="http://nonexistent.example.com/parse",
        )
        result = parser.parse_into_text(pdf_bytes)
        self.assertIn("image_source_type", result.metadata)
        self.assertEqual(result.metadata.get("image_source_type"), "scanned_pdf")

    def test_successful_parse(self):
        import httpx
        mock_content = "# Parsed\n\n| A | B |\n| 1 | 2 |"
        mock_images = {"fig1.jpg": "AAAA"}
        pdf_bytes = self._make_test_pdf()

        def mock_post(url, **kwargs):
            class MockResponse:
                status_code = 200
                def raise_for_status(self):
                    pass
                def json(self):
                    return {"content": mock_content, "images": mock_images}
            return MockResponse()

        parser = self.parser_cls(
            file_name="test.pdf",
            pdf_external_url="http://mock.service/parse",
            pdf_external_merge_tables=True,
        )
        with patch.object(httpx, "Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post = mock_post
            result = parser.parse_into_text(pdf_bytes)

        self.assertIn("| A | B |", result.content)
        self.assertIn("fig1.jpg", result.images)
        self.assertEqual(result.metadata.get("merge_tables"), True)
        self.assertEqual(result.metadata.get("parser_engine"), "external_pdf")


if __name__ == "__main__":
    unittest.main()

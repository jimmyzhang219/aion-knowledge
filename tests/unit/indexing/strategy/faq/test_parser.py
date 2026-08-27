"""Tests for FAQ file parser."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aion_knowledge.indexing.strategy.faq.parser import ParseError, parse_faq_file


class TestParseCSV:
    def test_basic(self):
        csv_content = "分类,问题,相似问题,负问题,答案,答案策略\n安全,如何重置密码？,密码忘了, ,进入设置页面,all\n"
        result = parse_faq_file(csv_content.encode("utf-8"), "csv")
        assert len(result) == 1
        assert result[0].standard_question == "如何重置密码？"
        assert result[0].similar_questions == ["密码忘了"]
        assert result[0].answers == ["进入设置页面"]
        assert result[0].tags == ["安全"]

    def test_missing_header(self):
        with pytest.raises(ParseError):
            parse_faq_file(b"", "csv")

    def test_empty_row_skipped(self):
        csv_content = "分类,问题,相似问题,负问题,答案,答案策略\n安全,Q1,,,A1,all\n, ,,,,\n安全,Q2,,,A2,all\n"
        result = parse_faq_file(csv_content.encode("utf-8"), "csv")
        assert len(result) == 2  # empty row skipped

    def test_no_bom(self):
        csv_content = "分类,问题,相似问题,负问题,答案,答案策略\n安全,Q1,,,A1,all\n"
        result = parse_faq_file(csv_content.encode("utf-8"), "csv")
        assert len(result) == 1

    def test_semicolon_delimiter(self):
        csv_content = "分类;问题;相似问题;负问题;答案;答案策略\n安全;Q1;;;A1;all\n"
        result = parse_faq_file(csv_content.encode("utf-8"), "csv")
        assert len(result) == 1


class TestParseJSON:
    def test_basic(self):
        data = [
            {"standard_question": "Q1", "answers": ["A1"], "tags": ["安全"]},
            {"standard_question": "Q2", "answers": ["A2", "A2b"]},
        ]
        result = parse_faq_file(json.dumps(data).encode("utf-8"), "json")
        assert len(result) == 2
        assert result[0].standard_question == "Q1"
        assert result[0].answers == ["A1"]
        assert result[0].tags == ["安全"]
        assert result[1].standard_question == "Q2"

    def test_empty_list(self):
        result = parse_faq_file(b"[]", "json")
        assert result == []

    def test_invalid_json(self):
        with pytest.raises(ParseError):
            parse_faq_file(b"not json", "json")


class TestParseExcel:
    def test_basic(self):
        pytest.importorskip("openpyxl")
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(["分类", "问题", "相似问题", "负问题", "答案", "答案策略"])
        ws.append(["安全", "Q1", "", "", "A1", "all"])
        ws.append(["", "Q2", "", "", "A2", "all"])

        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        wb.save(tmp.name)
        tmp.close()

        try:
            with open(tmp.name, "rb") as f:
                result = parse_faq_file(f.read(), "xlsx")
            assert len(result) == 2
            assert result[0].standard_question == "Q1"
            assert result[1].standard_question == "Q2"
        finally:
            Path(tmp.name).unlink(missing_ok=True)


class TestValidate:
    def test_empty_question(self):
        data = [{"standard_question": "", "answers": ["A1"]}]
        with pytest.raises(ParseError, match="standard_question"):
            parse_faq_file(json.dumps(data).encode("utf-8"), "json")

    def test_empty_answers(self):
        data = [{"standard_question": "Q1", "answers": []}]
        with pytest.raises(ParseError, match="答案"):
            parse_faq_file(json.dumps(data).encode("utf-8"), "json")

    def test_similar_question_duplicates_with_standard(self):
        data = [{"standard_question": "Q1", "similar_questions": ["Q1"], "answers": ["A1"]}]
        with pytest.raises(ParseError, match="相似问题"):
            parse_faq_file(json.dumps(data).encode("utf-8"), "json")

    def test_negative_question_duplicates(self):
        data = [{"standard_question": "Q1", "negative_questions": ["Q1"], "answers": ["A1"]}]
        with pytest.raises(ParseError, match="负问题"):
            parse_faq_file(json.dumps(data).encode("utf-8"), "json")

    def test_negative_question_matches_similar(self):
        """负问题不能与相似问题相同"""
        data = [
            {
                "standard_question": "Q1",
                "similar_questions": ["SQ1"],
                "negative_questions": ["SQ1"],
                "answers": ["A1"],
            },
        ]
        with pytest.raises(ParseError, match="负问题"):
            parse_faq_file(json.dumps(data).encode("utf-8"), "json")

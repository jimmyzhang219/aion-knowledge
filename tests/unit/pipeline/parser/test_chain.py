"""Tests for chain parsers."""
from aion_knowledge.pipeline.parser.base import BaseParser, ParsedDocument
from aion_knowledge.pipeline.parser.chain import FirstParser, PipelineParser


class _FastParser(BaseParser):
    def parse_into_text(self, content: bytes) -> ParsedDocument:
        return ParsedDocument(content="fast")


class _SlowParser(BaseParser):
    def parse_into_text(self, content: bytes) -> ParsedDocument:
        return ParsedDocument(content="slow")


class _FailingParser(BaseParser):
    def parse_into_text(self, content: bytes) -> ParsedDocument:
        raise ValueError("fail")


class TestFirstParser:
    def test_first_success(self):
        parser_cls = FirstParser.create(_FailingParser, _FastParser)
        parser = parser_cls()
        doc = parser.parse(b"test")
        assert doc.content == "fast"

    def test_all_fail_returns_empty(self):
        parser_cls = FirstParser.create(_FailingParser)
        parser = parser_cls()
        doc = parser.parse(b"test")
        assert not doc.is_valid()


class TestPipelineParser:
    def test_pipeline_chain(self):
        parser_cls = PipelineParser.create(_FastParser, _SlowParser)
        parser = parser_cls()
        doc = parser.parse(b"test")
        assert doc.content == "slow"

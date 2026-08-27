"""Tests for BaseParser and ParsedDocument."""
from aion_knowledge.pipeline.parser.base import BaseParser, ParsedDocument


class _ConcreteParser(BaseParser):
    def parse_into_text(self, content: bytes) -> ParsedDocument:
        return ParsedDocument(content=content.decode("utf-8"))


class TestParsedDocument:
    def test_empty_document_is_invalid(self):
        doc = ParsedDocument()
        assert doc.is_valid() is False

    def test_document_with_content_is_valid(self):
        doc = ParsedDocument(content="hello")
        assert doc.is_valid() is True

    def test_document_images_and_metadata(self):
        doc = ParsedDocument(content="text", images={"a.png": "base64data"}, metadata={"pages": 3})
        assert doc.images["a.png"] == "base64data"
        assert doc.metadata["pages"] == 3


class TestBaseParser:
    def test_parse_delegates_to_parse_into_text(self):
        parser = _ConcreteParser(file_name="test.txt")
        doc = parser.parse(b"hello world")
        assert doc.content == "hello world"
        assert doc.is_valid()

    def test_file_type_inferred_from_name(self):
        parser = _ConcreteParser(file_name="doc.pdf")
        assert parser.file_type == "pdf"

    def test_file_type_explicit(self):
        parser = _ConcreteParser(file_name="doc", file_type="pdf")
        assert parser.file_type == "pdf"

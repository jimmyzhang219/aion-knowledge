"""链式解析器：FirstParser（首成功）和 PipelineParser（流水线）。"""
import logging
from typing import Any, Dict, List, Tuple, Type

from aion_knowledge.pipeline.parser.base import BaseParser, ParsedDocument
from aion_knowledge.pipeline.parser.utils import endecode

logger = logging.getLogger(__name__)


class FirstParser(BaseParser):
    """首成功解析器：按顺序尝试多个解析器，返回第一个成功的。"""
    _parser_cls: Tuple[Type[BaseParser], ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._parsers: List[BaseParser] = [cls(*args, **kwargs) for cls in self._parser_cls]

    def parse_into_text(self, content: bytes) -> ParsedDocument:
        for p in self._parsers:
            try:
                doc = p.parse_into_text(content)
                if doc.is_valid():
                    return doc
            except Exception:
                logger.exception("FirstParser: %s failed", p.__class__.__name__)
        return ParsedDocument()

    @classmethod
    def create(cls, *parser_classes: Type[BaseParser]) -> Type["FirstParser"]:
        names = "_".join(p.__name__ for p in parser_classes)
        return type(f"FirstParser_{names}", (cls,), {"_parser_cls": parser_classes})


class PipelineParser(BaseParser):
    """流水线解析器：多个解析器顺序处理，前一个的输出作为后一个的输入。"""
    _parser_cls: Tuple[Type[BaseParser], ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._parsers: List[BaseParser] = [cls(*args, **kwargs) for cls in self._parser_cls]

    def parse_into_text(self, content: bytes) -> ParsedDocument:
        images: Dict[str, str] = {}
        metadata: Dict[str, Any] = {}
        doc = ParsedDocument()
        for p in self._parsers:
            doc = p.parse_into_text(content)
            content = endecode.encode_bytes(doc.content)
            images.update(doc.images)
            metadata.update(doc.metadata)
        doc.images.update(images)
        doc.metadata.update(metadata)
        return doc

    @classmethod
    def create(cls, *parser_classes: Type[BaseParser]) -> Type["PipelineParser"]:
        names = "_".join(p.__name__ for p in parser_classes)
        return type(f"PipelineParser_{names}", (cls,), {"_parser_cls": parser_classes})

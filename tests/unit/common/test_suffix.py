"""Suffix 推断工具单元测试。"""
from aion_knowledge.common.suffix import infer_suffix


def test_from_filename() -> None:
    assert infer_suffix(filename="test.pdf") == "pdf"
    assert infer_suffix(filename="document.PDF") == "pdf"
    assert infer_suffix(filename="archive.tar.gz") == "gz"


def test_from_content_type() -> None:
    assert infer_suffix(content_type="text/csv") == "csv"
    assert infer_suffix(content_type="application/pdf") == "pdf"
    assert infer_suffix(content_type="image/jpeg") == "jpg"


def test_fallback_default() -> None:
    assert infer_suffix() == "txt"
    assert infer_suffix(filename="unknown") == "txt"
    assert infer_suffix(filename="", default="csv") == "csv"


def test_filename_priority() -> None:
    """filename 扩展名应优先于 MIME。"""
    assert infer_suffix(filename="data.csv", content_type="application/pdf") == "csv"

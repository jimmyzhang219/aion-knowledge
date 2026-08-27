"""错误层次结构测试。"""

from aion_knowledge.common.errors import (
    AionError,
    ConfigurationError,
    NotFoundError,
    StorageError,
)


def test_base_error_with_details() -> None:
    err = AionError("test", {"key": "value"})
    assert err.message == "test"
    assert err.details == {"key": "value"}
    assert "test" in str(err)
    assert "key" in str(err) or "value" in str(err)


def test_base_error_no_details() -> None:
    err = AionError("simple error")
    assert str(err) == "simple error"


def test_error_chain() -> None:
    assert issubclass(ConfigurationError, AionError)
    assert issubclass(StorageError, AionError)
    assert issubclass(NotFoundError, AionError)


def test_not_found_error() -> None:
    err = NotFoundError("Document not found")
    assert err.message == "Document not found"

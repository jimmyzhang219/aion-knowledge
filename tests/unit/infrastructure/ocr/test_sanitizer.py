"""测试 OCR Sanitizer。"""

from aion_knowledge.infrastructure.ocr.sanitizer import sanitize_ocr


def test_empty_input():
    assert sanitize_ocr("") == ""
    assert sanitize_ocr("   ") == ""
    assert sanitize_ocr(None) == ""


def test_strip_markdown_code_block():
    raw = "```html\n<p>Hello</p>\n```"
    result = sanitize_ocr(raw)
    # 剥离了围栏，但内容含 HTML 标签 → 转 Markdown
    assert "Hello" in result
    assert "```" not in result


def test_html_to_markdown():
    raw = "<p>Hello <strong>World</strong></p>"
    result = sanitize_ocr(raw)
    assert "**World**" in result or "World" in result
    assert "<p>" not in result


def test_known_empty_replies():
    cases = [
        "无文字内容", "无文字内容.", "No text content", "No text content!",
        "no text", "图片中没有文字", "无法识别。",
    ]
    for c in cases:
        assert sanitize_ocr(c) == "", f"should detect empty reply: {c!r}"
    # 含额外文字的应该保留
    assert sanitize_ocr("No text content found in this image") == ""  # 精确匹配
    assert sanitize_ocr("The text says Hello World") != ""


def test_consecutive_newlines():
    raw = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
    result = sanitize_ocr(raw)
    assert result == "Line 1\n\nLine 2\n\nLine 3"


def test_normal_text_preserved():
    raw = "This is a scanned document.\n\nIt has two paragraphs."
    result = sanitize_ocr(raw)
    assert result == raw


def test_strip_html_wrapper_leaves_empty():
    """VLM 有时返回仅有 HTML 结构、几乎无文字的内容。"""
    raw = '<html><body><div class="image"><img src="data:..."/></div></body></html>'
    result = sanitize_ocr(raw)
    assert result == ""

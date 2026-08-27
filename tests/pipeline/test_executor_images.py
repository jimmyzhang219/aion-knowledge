"""测试：图片 section 切分 + context_above/below 附着。"""
from aion_knowledge.pipeline.parser.tools import split_sections


def test_split_sections_basic():
    """基础图片切分：文字-图片-文字-图片-文字。"""
    content = "前面文字。![img1](https://oss/img1.png)中间文字。![img2](https://oss/img2.png)后面文字。"
    sections = split_sections(content)
    assert len(sections) == 5
    assert sections[0]["type"] == "text"
    assert sections[0]["content"] == "前面文字。"
    assert sections[1]["type"] == "image"
    assert sections[1]["image_url"] == "https://oss/img1.png"
    assert sections[2]["type"] == "text"
    assert sections[2]["content"] == "中间文字。"
    assert sections[3]["image_url"] == "https://oss/img2.png"
    assert sections[4]["type"] == "text"
    assert sections[4]["content"] == "后面文字。"


def test_split_sections_context():
    """图片 context_above/below 正确附着。"""
    content = "上文内容。![img](https://oss/img.png)下文内容。"
    sections = split_sections(content)
    img_sec = [s for s in sections if s["type"] == "image"][0]
    assert "上文内容" in img_sec["context_above"]
    assert "下文内容" in img_sec["context_below"]


def test_split_sections_no_image():
    """无图片时返回单一 text section。"""
    content = "纯文本内容，没有图片。"
    sections = split_sections(content)
    assert len(sections) == 1
    assert sections[0]["type"] == "text"
    assert sections[0]["content"] == "纯文本内容，没有图片。"


def test_split_sections_empty():
    """空输入返回空列表。"""
    sections = split_sections("")
    assert sections == []


def test_split_sections_only_image():
    """仅图片，无文字。"""
    content = "![img](https://oss/img.png)"
    sections = split_sections(content)
    assert len(sections) == 1
    assert sections[0]["type"] == "image"
    assert sections[0]["image_url"] == "https://oss/img.png"


def test_split_sections_context_truncation():
    """context_above/below 应截断为最近 500 字符。"""
    above = "A" * 1000
    below = "B" * 1000
    content = f"{above}![img](https://oss/img.png){below}"
    sections = split_sections(content)
    img_sec = sections[1]
    assert len(img_sec["context_above"]) == 500
    assert len(img_sec["context_below"]) == 500
    assert img_sec["context_above"] == above[-500:]
    assert img_sec["context_below"] == below[:500]


def test_split_sections_consecutive_images():
    """连续图片：相邻图片应各自独立，无上下文时 context 为空。"""
    content = "![img1](https://oss/img1.png)![img2](https://oss/img2.png)"
    sections = split_sections(content)
    assert len(sections) == 2
    assert sections[0]["type"] == "image"
    assert sections[1]["type"] == "image"
    assert "context_above" not in sections[0]
    assert "context_above" not in sections[1]
    assert "context_below" not in sections[0]
    assert "context_below" not in sections[1]

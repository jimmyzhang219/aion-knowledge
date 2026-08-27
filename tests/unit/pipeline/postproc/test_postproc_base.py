"""PostProcModule 基类测试。"""

from __future__ import annotations

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule


class TestPostProcModule:
    """验证基类接口约束。"""

    def test_always_on_default_is_false(self):
        """always_on 类变量默认应为 False。"""
        assert PostProcModule.always_on is False

    def test_depends_on_default_is_empty(self):
        """depends_on 类变量默认应为空列表。"""
        assert PostProcModule.depends_on == []

    def test_cannot_instantiate_abstract(self):
        """直接实例化抽象类应抛 TypeError。"""
        with pytest.raises(TypeError):
            PostProcModule()  # type: ignore

    def test_concrete_subclass_must_implement_process(self):
        """未实现 process 的子类不可实例化。"""
        class Incomplete(PostProcModule):
            pass
        with pytest.raises(TypeError):
            Incomplete()  # type: ignore

    @pytest.mark.asyncio
    async def test_concrete_subclass_can_instantiate(self):
        """实现了 process 的子类可正常实例化并调用。"""
        class Concrete(PostProcModule):
            async def process(self, ctx, chunks):
                return 42

        module = Concrete()
        assert module.always_on is False
        assert module.depends_on == []
        result = await module.process(None, [])
        assert result == 42


class TestPostProcContext:
    """验证后处理上下文数据模型。"""

    def test_required_fields(self):
        """必需字段应正确设置。"""
        ctx = PostProcContext(
            document_id="doc-1",
            kb_id="kb-1",
            doc_name="test.md",
        )
        assert ctx.document_id == "doc-1"
        assert ctx.kb_id == "kb-1"
        assert ctx.doc_name == "test.md"


class TestCheckContent:
    """验证基类 _check_content 方法"""

    def test_none_content_returns_false(self):
        class Concrete(PostProcModule):
            async def process(self, ctx, chunks):  # type: ignore[override]
                return 0
        module = Concrete()
        assert module._check_content(None) is False

    def test_empty_content_returns_false(self):
        class Concrete(PostProcModule):
            async def process(self, ctx, chunks):  # type: ignore[override]
                return 0
        module = Concrete()
        assert module._check_content("") is False

    def test_whitespace_only_returns_false(self):
        class Concrete(PostProcModule):
            async def process(self, ctx, chunks):  # type: ignore[override]
                return 0
        module = Concrete()
        assert module._check_content("   \n  \t  ") is False

    def test_pure_markdown_image_returns_false(self):
        """纯图片 Markdown 剥离后应无可读内容"""
        class Concrete(PostProcModule):
            async def process(self, ctx, chunks):  # type: ignore[override]
                return 0
        module = Concrete()
        content = "![image](https://example.com/img.png)\n\n![](local.jpg)"
        assert module._check_content(content) is False

    def test_normal_text_returns_true(self):
        class Concrete(PostProcModule):
            async def process(self, ctx, chunks):  # type: ignore[override]
                return 0
        module = Concrete()
        assert module._check_content("Hello world, this is real content") is True

    def test_text_after_stripping_markdown_still_true(self):
        """Markdown 包裹但剥离后仍有内容的文本"""
        class Concrete(PostProcModule):
            async def process(self, ctx, chunks):  # type: ignore[override]
                return 0
        module = Concrete()
        content = "**bold** and *italic* and `code`"
        assert module._check_content(content) is True

    def test_custom_min_chars(self):
        class Concrete(PostProcModule):
            async def process(self, ctx, chunks):  # type: ignore[override]
                return 0
        module = Concrete()
        # "ab" 只有 2 字符，不足 5
        assert module._check_content("ab", min_chars=5) is False
        # "abcde" 正好 5
        assert module._check_content("abcde", min_chars=5) is True

    def test_strip_markdown_false(self):
        """关闭 Markdown 剥离时，纯图片 MD 也算内容"""
        class Concrete(PostProcModule):
            async def process(self, ctx, chunks):  # type: ignore[override]
                return 0
        module = Concrete()
        content = "![image](https://example.com/img.png)"
        # strip_markdown=False 时，原始内容包含字符，所以为 True
        assert module._check_content(content, strip_markdown=False) is True

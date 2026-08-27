"""ChunkingStrategy 基类默认流水线测试。"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.indexing.strategy.base import ChunkingStrategy
from aion_knowledge.pipeline.assembler import assemble


class TestDefaultPipeline:
    """验证基类的默认 execute 按序调用 6 个步骤。"""

    @pytest.mark.asyncio
    async def test_pipeline_calls_all_steps(self):
        """默认 execute 应依次调用各子步骤。"""
        call_order = []

        class TrackingStrategy(ChunkingStrategy):
            strategy_key = "tracking"

            async def _download(self, ctx):
                call_order.append("download")
                return "/tmp/tracking.pdf"

            async def _parse(self, ctx, local_raw):
                call_order.append("parse")
                return ("/tmp/tracking.md", {})

            async def _clean(self, ctx, local_md):
                call_order.append("clean")
                return local_md

            async def _upload_md(self, ctx, local_md, image_map):
                call_order.append("upload_md")

            async def _prepare_chunks(self, ctx, local_md, image_map):
                call_order.append("prepare")
                return [], [], []

        ctx = MagicMock()
        ctx.doc_name = "tracking"

        with (
            open("/tmp/tracking.md", "w", encoding="utf-8") as f,
            patch("aion_knowledge.indexing.strategy.base.assemble") as mock_assemble,
        ):
            f.write("mock content")
            mock_assemble.return_value = []
            strategy = TrackingStrategy()
            await strategy.execute(ctx)

        if os.path.exists("/tmp/tracking.md"):
            os.remove("/tmp/tracking.md")

        assert call_order == ["download", "parse", "clean", "upload_md", "prepare"]
        mock_assemble.assert_called_once()

    def test_strategy_key_must_be_defined(self):
        """strategy_key 是抽象属性，未定义应报 TypeError。"""
        with pytest.raises(TypeError):

            class MissingKeyStrategy(ChunkingStrategy):
                pass

            MissingKeyStrategy()

    def test_assemble_sorts_by_seq_num(self):
        """assemble 应按 seq_num 排序所有 chunk。"""
        text = [{"seq_num": 3, "content": "c"}, {"seq_num": 1, "content": "a"}]
        tables = [{"seq_num": 2, "content": "b"}]
        images = []

        result = assemble(text, tables, images)
        assert [c["content"] for c in result] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_subclass_can_override_single_step(self):
        """子类可以只重写某个步骤，其余走基类默认。"""
        call_order = []

        class CustomStrategy(ChunkingStrategy):
            strategy_key = "custom"

            async def _parse(self, ctx, local_raw):
                call_order.append("parse")
                return "/tmp/custom.md", {"img1": "s3://img1"}

        # Mock resolve_storage for _upload_images
        mock_store = AsyncMock()
        mock_store.upload.return_value = "s3://bucket/img1"

        with patch(
            "aion_knowledge.indexing.strategy.base.resolve_storage", return_value=mock_store,
        ):
            ctx = MagicMock()
            ctx.doc_name = "custom_test"
            ctx.suffix = "md"
            ctx.original_file_ref = "/tmp/custom.md"
            ctx.chunk_strategy = "heading"

            strategy = CustomStrategy()
            strategy._download = AsyncMock(return_value="/tmp/custom.md")
            strategy._clean = AsyncMock(return_value="/tmp/custom.md")
            strategy._upload_md = AsyncMock()
            strategy._prepare_chunks = AsyncMock(return_value=([{"seq_num": 0}], [], []))
            strategy._assemble = MagicMock(return_value=[{"seq_num": 0}])

            result = await strategy.execute(ctx)
            assert len(result) == 1
            assert call_order == ["parse"]


class TestUploadPaths:
    """验证 md/图片上传 key 统一使用 ctx.file_dir 前缀。"""

    class _S(ChunkingStrategy):
        strategy_key = "upload_test"

    @pytest.mark.asyncio
    async def test_upload_md_uses_file_dir(self):
        from pathlib import Path

        from aion_knowledge.infrastructure.models import UnifiedContext

        md_path = "/tmp/upload_md_test.md"
        Path(md_path).write_text("# hello", encoding="utf-8")

        mock_store = AsyncMock()
        mock_store.upload.return_value = "s3://bucket/docs/kb-1/h1/converted.md"

        ctx = UnifiedContext(source="test", kb_id="kb-1", doc_name="report.pdf", suffix="pdf",
                             original_file_ref="x")
        ctx.file_dir = "kb-1/h1"
        image_map = {"images/a.png": "s3://bucket/docs/kb-1/h1/images/a.png"}

        try:
            with patch(
                "aion_knowledge.indexing.strategy.base.resolve_storage", return_value=mock_store,
            ):
                await self._S()._upload_md(ctx, md_path, image_map)
        finally:
            Path(md_path).unlink(missing_ok=True)

        mock_store.upload.assert_awaited_once_with(
            "kb-1/h1/converted.md", b"# hello", content_type="text/markdown"
        )
        assert ctx.md_file_ref == "s3://bucket/docs/kb-1/h1/converted.md"
        assert ctx.image_ref_map == image_map

    @pytest.mark.asyncio
    async def test_upload_images_uses_dir(self):
        import base64

        mock_store = AsyncMock()
        mock_store.upload.return_value = "s3://bucket/docs/kb-1/h1/images/a.png"

        with patch(
            "aion_knowledge.indexing.strategy.base.resolve_storage", return_value=mock_store,
        ):
            image_map = await self._S()._upload_images(
                {"images/a.png": base64.b64encode(b"fake-img").decode()}, "kb-1/h1",
            )

        mock_store.upload.assert_awaited_once_with(
            "kb-1/h1/images/a.png", b"fake-img",
            content_type="image/png",
        )
        assert image_map == {"images/a.png": "s3://bucket/docs/kb-1/h1/images/a.png"}

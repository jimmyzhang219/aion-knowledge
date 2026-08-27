"""ChunkingStrategy — 索引构建策略基类。

模板方法模式：execute() 定义编排骨架，各子步骤拆分独立方法。
子类只需覆盖需要变更的步骤，无需重写整个 execute。

所有子步骤直接调用 pipeline 模块，不再经过 executor 中介。
"""

from __future__ import annotations

import base64
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from aion_knowledge.infrastructure.models import UnifiedContext
from aion_knowledge.infrastructure.storage import resolve_storage
from aion_knowledge.pipeline.assembler import assemble

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """索引构建策略基类。

    默认流水线：_download → _parse → _clean → _upload_md → _prepare_chunks → _assemble
    子类可覆写任意步骤，或完全重写 execute()。
    """

    @property
    @abstractmethod
    def strategy_key(self) -> str:
        """策略注册键，对应 ctx.source。"""
        ...

    async def execute(self, ctx: UnifiedContext) -> list[dict[str, Any]]:
        """默认索引构建流水线。

        按序执行：download → parse → clean → upload_md → prepare_chunks → assemble。
        子类如不需要完整流水线（如 FAQChunkingStrategy），直接重写此方法。
        """
        # Step 1: Download
        local_raw = await self._download(ctx)

        # Step 2: Parser
        local_md, image_map = await self._parse(ctx, local_raw)

        # Step 3: Cleaner
        local_md = await self._clean(ctx, local_md)

        # Step 3.5: Upload cleaned MD + set ctx S3 references
        await self._upload_md(ctx, local_md, image_map)

        # Step 4: Prepare chunks（表格/章节/Chunker）
        text_chunks, tables, images = await self._prepare_chunks(
            ctx, local_md, image_map,
        )

        # Step 5: Assemble
        return assemble(text_chunks, tables, images)

    # ── 子步骤（子类按需覆盖） ──────────────────────────────

    async def _download(self, ctx: UnifiedContext) -> str:
        """下载 raw 文件到本地临时路径。"""
        from aion_knowledge.pipeline.downloader import Downloader
        return await Downloader().download(ctx.original_file_ref, ctx.doc_name)

    async def _parse(
        self, ctx: UnifiedContext, local_raw: str,
    ) -> tuple[str, dict[str, str]]:
        """解析原始文档为 Markdown，返回 (md_path, image_map)。"""
        from aion_knowledge.pipeline.parser import Parser
        from aion_knowledge.pipeline.parser.image_resolver import resolve_images

        parser = Parser()
        with open(local_raw, "rb") as f:
            raw_bytes = f.read()

        parsed = parser.parse_file(
            file_name=ctx.doc_name,
            file_type=ctx.suffix,
            content=raw_bytes,
        )
        parsed.content, parsed.images = resolve_images(
            parsed.content, parsed.images,
        )

        md_path = local_raw + ".md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(parsed.content)

        # 上传 base64 编码图片到存储
        image_map = await self._upload_images(parsed.images, ctx.file_dir)

        return md_path, image_map

    async def _clean(self, ctx: UnifiedContext, local_md: str) -> str:
        """清洗 Markdown 文件。"""
        from aion_knowledge.pipeline.cleaner import Cleaner

        if not os.path.isfile(local_md):
            logger.warning("【清洗】跳过：%s 文件不存在", local_md)
            return local_md

        with open(local_md, "r", encoding="utf-8") as f:
            content = f.read()

        cleaned = Cleaner().clean(content)

        if cleaned != content:
            with open(local_md, "w", encoding="utf-8") as f:
                f.write(cleaned)
            logger.info("【清洗】完成：%s（%d chars → %d chars）",
                         local_md, len(content), len(cleaned))
        else:
            logger.debug("【清洗】跳过：%s 无需变更", local_md)

        return local_md

    async def _upload_md(
        self, ctx: UnifiedContext, local_md: str, image_map: dict[str, str],
    ) -> None:
        """上传清洗后的 MD 到存储，并设置 ctx 的 S3 引用（await 等待上传完成）。"""
        md_key = f"{ctx.file_dir}/converted.md"
        with open(local_md, "r", encoding="utf-8") as f:
            md_content = f.read()

        store = resolve_storage()
        md_ref = await store.upload(
            md_key, md_content.encode("utf-8"), content_type="text/markdown",
        )

        ctx.md_file_ref = md_ref
        ctx.image_ref_map = image_map

    async def _prepare_chunks(
        self, ctx: UnifiedContext, local_md: str, image_map: dict[str, str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """表格提取 → 章节切分 → 图片分离 → 通用 Chunker。

        Returns:
            (text_chunks, table_entries, image_entries)
        """
        from aion_knowledge.pipeline.parser.tools import extract_tables, split_sections

        with open(local_md, "r", encoding="utf-8") as f:
            md_content = f.read()

        table_chunks, placeholder_md = extract_tables(md_content)
        sections = split_sections(placeholder_md)

        image_chunks_from_sections: list[dict[str, Any]] = []
        text_contents: list[str] = []
        for sec in sections:
            if sec["type"] == "image":
                orig_url = sec["image_url"]
                if orig_url in image_map:
                    sec["image_url"] = image_map[orig_url]
                    placeholder_md = placeholder_md.replace(
                        orig_url, image_map[orig_url],
                    )
                image_chunks_from_sections.append(sec)
            else:
                text_contents.append(sec["content"])

        # 写回纯文本 Markdown（移除图片标记），供 chunker 使用
        text_only_md = "\n\n".join(text_contents)
        with open(local_md, "w", encoding="utf-8") as f:
            f.write(text_only_md)

        # Chunker — 只处理纯文本
        text_chunks = await self._run_chunker(ctx, local_md)

        # 将表格 sections 转换为 table chunks
        from aion_knowledge.common.model_registry import count_tokens
        from aion_knowledge.models.enums import ChunkType

        table_entries: list[dict[str, Any]] = []
        for t in table_chunks:
            table_entries.append({
                "content": t["content"],
                "chunk_type": ChunkType.table.value,
                "table_caption": t.get("table_caption", ""),
                "heading_path": t.get("heading_path", ""),
                "token_count": count_tokens(t["content"]),
                "seq_num": t["seq_num"],
            })

        # 将图片 sections 转换为 image chunks
        image_entries: list[dict[str, Any]] = []
        for sec in image_chunks_from_sections:
            image_entries.append({
                "content": "",
                "chunk_type": ChunkType.image.value,
                "image_url": sec["image_url"],
                "context_above": sec.get("context_above", ""),
                "context_below": sec.get("context_below", ""),
                "token_count": 0,
                "seq_num": sec.get(
                    "seq_num", len(text_chunks) + len(image_entries),
                ),
            })

        return text_chunks, table_entries, image_entries

    async def _run_chunker(
        self, ctx: UnifiedContext, local_md: str,
    ) -> list[dict[str, Any]]:
        """读取本地 Markdown 文件，执行分块策略链。"""
        from aion_knowledge.pipeline.chunker.base import ChunkConfig
        from aion_knowledge.pipeline.chunker.chunker import Chunker

        config = ChunkConfig(strategy=ctx.chunk_strategy)
        chunker = Chunker(config=config)
        with open(local_md, "r", encoding="utf-8") as f:
            md_content = f.read()

        results = chunker.split(md_content)
        return [r.model_dump() for r in results]

    async def _upload_images(
        self, images: dict[str, str], file_dir: str,
    ) -> dict[str, str]:
        """上传 base64 图片到存储，返回 {ref: 完整对象存储引用} 映射。"""
        if not images:
            return {}

        store = resolve_storage()

        image_map: dict[str, str] = {}
        for ref, b64_data in images.items():
            try:
                img_bytes = base64.b64decode(b64_data)
            except Exception:
                logger.warning("图片 base64 解码失败：%s", ref)
                continue

            ext = os.path.splitext(ref)[1].lower()
            content_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".svg": "image/svg+xml",
            }.get(ext, "application/octet-stream")

            url = await store.upload(
                f"{file_dir}/{ref}", img_bytes, content_type=content_type,
            )
            image_map[ref] = url

        return image_map

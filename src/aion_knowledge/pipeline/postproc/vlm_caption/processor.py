"""VLM 图片描述 + OCR 合并处理器。

作用：
  对包含图片引用的 chunk 执行多模态理解，将图片描述和 OCR 文本
  合并写入 chunk_text.content。

流程（按优先级）：
  1. 下载图片（S3 / 本地文件存储）
  2. OCR 阶段（仅 scanned PDF 页面）：
     a. 主路径 → PaddleOCR
     b. 降级路径 → Tesseract（PaddleOCR 无结果时启用）
     c. 兜底路径 → VLM OCR（Tesseract 无结果时启用）
  3. Caption 阶段（所有图片）：
     a. 无上下文 → 标准描述 prompt
     b. 有上下文（context_above/below）→ 带上下文的描述 prompt
  4. 内容合并：context_above + OCR + VLM caption + context_below
  5. 回写 chunk_text.content，同时记录 image_s3_key / image_alt

注意事项：
  始终启用（always_on=True），并发度通过 Semaphore(3) 控制。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from sqlalchemy import text as sql_text

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.infrastructure.llm import LLMClient, get_vlm_client
from aion_knowledge.infrastructure.ocr.engine import OCRNotAvailable
from aion_knowledge.infrastructure.ocr.paddle import PaddleOCREngine, get_engine
from aion_knowledge.infrastructure.ocr.tesseract import TesseractOCREngine
from aion_knowledge.infrastructure.ocr.vlm_ocr import VLMOCREngine
from aion_knowledge.infrastructure.storage import StorageBackend, resolve_storage
from aion_knowledge.pipeline.chunker.text_utils import count_tokens
from aion_knowledge.pipeline.postproc.base import PostProcContext, PostProcModule
from aion_knowledge.pipeline.postproc.vlm_caption.prompts import (
    DESCRIBE_PROMPT,
    DESCRIBE_PROMPT_WITH_CONTEXT,
)

logger = logging.getLogger(__name__)
_sem = asyncio.Semaphore(3)

MIME_MAP = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
}


def _is_scanned_pdf(chunk: dict[str, Any]) -> bool:
    """判断 chunk 是否来自扫描 PDF 页。"""
    meta = chunk.get("metadata") or {}
    return meta.get("image_source_type") == "scanned_pdf"


class VLMCaptionModule(PostProcModule):
    """VLM 图片描述 + OCR 合并模块。首批执行，依赖 text。"""

    always_on = True
    depends_on = ["text"]

    def __init__(self) -> None:
        """惰性初始化依赖（LLM / 存储 / OCR 引擎），首次使用时才创建。"""
        self._llm: LLMClient | None = None
        self._store: StorageBackend | None = None
        self._tesseract: TesseractOCREngine | None = None
        self._paddle: PaddleOCREngine | None = None

    def _ensure_llm(self) -> LLMClient:
        """懒加载并返回 VLM 客户端。"""
        if self._llm is None:
            self._llm = get_vlm_client()
        return self._llm

    def _ensure_store(self) -> StorageBackend:
        """懒加载并返回存储后端。"""
        if self._store is None:
            self._store = resolve_storage()
        return self._store

    async def _download_images(self, refs: list[str]) -> list[tuple[bytes, str]]:
        """下载图片 bytes 列表。"""
        store = self._ensure_store()
        img_data: list[tuple[bytes, str]] = []
        for ref in refs:
            try:
                data = await store.download(ref)
                ext = ref.rsplit(".", 1)[-1].lower() if "." in ref else "png"
                mime = MIME_MAP.get(ext, "image/png")
                img_data.append((data, mime))
            except Exception as e:
                logger.warning("【VLM】下载图片失败 %s: %s", ref, e)
        return img_data

    async def _run_vlm_caption(self, prompt: str, img_data: list[tuple[bytes, str]]) -> str:
        """执行 VLM Caption。"""
        llm = self._ensure_llm()
        async with _sem:
            try:
                return await llm.generate_with_images(
                    prompt=prompt, images=img_data, max_tokens=1024
                )
            except Exception as e:
                logger.warning("【VLM】Caption 失败: %s", e)
                return ""

    async def _run_tesseract_ocr(self, image_bytes: bytes) -> str:
        """执行 Tesseract OCR（可能抛出 OCRNotAvailable）。"""
        if self._tesseract is None:
            self._tesseract = TesseractOCREngine()
        try:
            return await self._tesseract.ocr(image_bytes)
        except OCRNotAvailable:
            return ""  # 降级信号
        except Exception as e:
            logger.warning("【Tesseract】OCR 异常: %s", e)
            return ""

    async def _run_paddle_ocr(self, image_bytes: bytes) -> str:
        """执行 PaddleOCR（可能抛出 OCRNotAvailable）。"""
        if self._paddle is None:
            self._paddle = get_engine()  # 进程级单例，跨文档复用模型/失败缓存
        try:
            return await self._paddle.ocr(image_bytes)
        except OCRNotAvailable:
            return ""  # 降级信号
        except Exception as e:
            logger.warning("【PaddleOCR】OCR 异常: %s", e)
            return ""

    async def _run_vlm_ocr(self, image_bytes: bytes) -> str:
        """执行 VLM OCR 降级。"""
        llm = self._ensure_llm()
        engine = VLMOCREngine(llm=llm)
        async with _sem:
            try:
                return await engine.ocr(image_bytes)
            except Exception as e:
                logger.warning("【VLM OCR】降级失败: %s", e)
                return ""

    async def _update_chunk_content(
        self, chunk_uuid: str, new_content: str, s3_key: str, image_alt: str
    ) -> None:
        """回写 chunk_text.content 并同步 token_count / metadata / content_tokens。"""
        async with get_session() as session:
            await session.execute(
                sql_text("""
                    UPDATE chunk_text
                    SET content = :content,
                        token_count = :tc,
                        metadata = CAST(metadata AS jsonb) || CAST(:meta AS jsonb)
                    WHERE id = :cid
                """),
                {
                    "content": new_content,
                    "tc": count_tokens(new_content),
                    "meta": json.dumps({"image_s3_key": s3_key, "image_alt": image_alt}),
                    "cid": chunk_uuid,
                },
            )
            # 同步重算 content_tokens（content 已更新，需要重新分词）
            from aion_knowledge.storage.relational.chunk_repo import ChunkRepository
            repo = ChunkRepository(session)
            await repo.update_content_tokens(chunk_uuid)

    def _extract_s3_key(self, chunk: dict[str, Any], refs: list[str]) -> str:
        """从图片引用中提取 S3 key（优先 refs，其次 chunk.image_url）。"""
        for ref in refs:
            if ref.startswith("s3://") or "docs/" in ref:
                return ref
        image_url: str = chunk.get("image_url", "")
        if image_url.startswith("s3://"):
            return image_url
        return ""

    async def process(self, ctx: PostProcContext, chunks: list[dict[str, Any]]) -> int:
        """对含图片 chunk 执行 OCR + Caption 并合并回写 content，返回更新数。"""
        image_chunks = [c for c in chunks if c.get("image_refs") or c.get("image_url")]
        if not image_chunks:
            logger.info("【VLM】文档 %s 无图片", ctx.doc_name)
            return 0

        updated = 0

        for chunk in image_chunks:
            chunk_uuid = chunk.get("chunk_uuid")
            refs = list(chunk.get("image_refs", []))
            if not refs and chunk.get("image_url"):
                refs = [chunk["image_url"]]
            if not chunk_uuid or not refs:
                continue

            # 下载图片
            img_data = await self._download_images(refs)
            if not img_data:
                continue

            context_above = chunk.get("context_above", "")
            context_below = chunk.get("context_below", "")
            is_scanned = _is_scanned_pdf(chunk)

            # OCR 阶段（仅 scanned_pdf）
            ocr_text = ""
            if is_scanned:
                # 主路径：PaddleOCR
                ocr_text = await self._run_paddle_ocr(img_data[0][0])
                if not ocr_text:
                    # 降级：Tesseract
                    logger.debug("【VLM】PaddleOCR 无结果，降级到 Tesseract")
                    ocr_text = await self._run_tesseract_ocr(img_data[0][0])
                    if not ocr_text:
                        # 兜底：VLM OCR
                        logger.debug("【VLM】Tesseract 无结果，降级到 VLM OCR")
                        ocr_text = await self._run_vlm_ocr(img_data[0][0])
            if ocr_text:
                logger.info("【OCR】结果 (前20字): %s", ocr_text[:20])

            # Caption 阶段（所有图片）
            if context_above or context_below:
                prompt = DESCRIBE_PROMPT_WITH_CONTEXT.format(
                    context_above=context_above,
                    context_below=context_below,
                )
            else:
                prompt = DESCRIBE_PROMPT

            desc = await self._run_vlm_caption(prompt, img_data)
            if desc:
                logger.info("【VLM-Caption】结果 (前20字): %s", desc[:20])
            else:
                # 记录排查所需的所有上下文
                logger.warning(
                    "【VLM-Caption】返回为空: chunk=%s, refs=%s, img_count=%d, "
                    "is_scanned=%s, has_context_above=%s, has_context_below=%s, "
                    "chunk_type=%s, seq_num=%s",
                    chunk_uuid, refs, len(img_data),
                    is_scanned, bool(context_above), bool(context_below),
                    chunk.get("chunk_type"), chunk.get("seq_num"),
                )
                # 重试一次：用备用 prompt
                logger.info("【VLM-Caption】首次返回为空，使用备用 prompt 重试...")
                retry_desc = await self._run_vlm_caption(
                    "请用中文简洁描述这张图片的内容。", img_data
                )
                if retry_desc:
                    logger.info("【VLM-Caption】备用 prompt 重试成功 (前20字): %s", retry_desc[:20])
                    desc = retry_desc
                else:
                    logger.warning("【VLM-Caption】备用 prompt 重试仍为空: chunk=%s", chunk_uuid)

            # 内容合并
            parts: list[str] = []
            if context_above:
                parts.append(context_above)
            if ocr_text:
                parts.append(f"<!-- ocr-text -->\n{ocr_text}\n<!-- /ocr-text -->")
            if desc:
                parts.append(f"<!-- vlm-caption -->\n{desc}\n<!-- /vlm-caption -->")
            if context_below:
                parts.append(context_below)

            # 追加 heading_path 标签，提升 BM25/keyword 检索匹配度
            heading_path = chunk.get("heading_path", "")
            if heading_path:
                parts.append(f"<!-- vlm-tags -->\n{heading_path}\n<!-- /vlm-tags -->")

            if not parts:
                logger.info("【VLM】OCR + Caption 都为空，跳过 chunk=%s", chunk_uuid)
                continue

            new_content = "\n\n".join(parts)

            s3_key = self._extract_s3_key(chunk, refs)
            await self._update_chunk_content(chunk_uuid, new_content, s3_key, "")

            # 同步更新内存中 chunks 列表，供后续 VectorModule 等下游模块消费
            chunk["content"] = new_content
            updated += 1

        logger.info("【VLM】完成: %d/%d 个 image chunk", updated, len(image_chunks))
        return updated


def module() -> VLMCaptionModule:
    """模块工厂函数，供调度器自动发现。"""
    return VLMCaptionModule()

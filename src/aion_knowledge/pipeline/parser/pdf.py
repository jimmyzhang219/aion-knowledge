"""PDF 解析，在原生文本页和扫描图片页之间进行逐页路由。

设计理念（与 MinerU / Docling / DeepDoc 的路由方式一致）：

* 判断"此页为扫描页"的主要信号是**图片区域覆盖率**
  （图片边界框面积 / 页面面积），而非原始字符数。
  扫描页本质上是一张覆盖整个页面的大图，即使它带有（通常是低质量的）
  嵌入式 OCR 文本层。此前信任该嵌入式文本层正是产生乱码 RAG 内容的原因。
* 页面被独立分类，以便正确处理混合型 PDF（部分原生、部分扫描）。
  原生页面贡献其文本层；扫描页面被渲染为 JPEG 并标记
  ``image_source_type=scanned_pdf``，由后续 VLM 模块处理。

无需外部服务（如 MinerU）：内置引擎使用 pypdfium2 +
  后续 VLM 模块即可完全自给自足。
"""

import base64
import io
import logging
import multiprocessing as mp
import os
import re
import statistics
import threading
from typing import Any, cast

from aion_knowledge.common.config import settings
from aion_knowledge.pipeline.parser.base import BaseParser, ParsedDocument
from aion_knowledge.pipeline.parser.concurrency import parser_worker_limit

logger = logging.getLogger(__name__)

# pdfium（pypdfium2 背后的 C 库）是进程全局的，并且**不是**线程安全的：
# 两个工作线程同时解析 PDF 会破坏其共享状态，并可能导致整个进程
# 无限期死锁（表现为请求永远卡在"使用 PDFParser 解析文档"中，同时拖垮
# 所有后续上传）。所有 pdfium 操作——文本提取、页面渲染、图片提取——都
# 必须在此单一全局锁之后串行化。并发 PDF 上传因此逐个处理，而非挂起。
# 非 PDF 解析器（docx、xlsx……）不受影响，仍可在线程池中并发运行。
_PDFIUM_LOCK = threading.Lock()


# 图片对象覆盖至少该比例页面面积的页面被视为扫描页（图片主导）。
# 原生数字页面约为 ~0.0-0.05；扫描页面约为 ~1.0+，因此 0.5 留有宽裕的安全余量。
SCAN_IMAGE_AREA_RATIO = 0.5
# 低于此字符数的页面被视为无可用的文本层。
SCAN_MIN_CHARS_PER_PAGE = 10
# 近乎空文本的页面仅当实际包含某些图片内容时才渲染为图片
#（避免渲染真正空白的页面）。
_LOW_TEXT_IMAGE_RATIO = 0.1

# --- 嵌入式图形提取（文本页）----------------------------------------------
# 原生页面可以嵌入图形/图表。我们将其以图片引用形式暴露，以便后续 VLM 模块处理。
# 徽标、图标、水印和小装饰物
# 会根据尺寸、页面面积占比和跨页重复性进行过滤。
EXTRACT_EMBEDDED_IMAGES = settings.pdf_extract_embedded_images
# 保留嵌入式图片的最小像素宽度和高度。
EMBED_MIN_PIXELS = 80
# 保留嵌入式图片的最小页面面积占比。
EMBED_MIN_AREA_RATIO = 0.01
# 出现在至少该比例文本页面上的相同图片被视为重复徽标/水印并被丢弃。
EMBED_REPEAT_PAGE_FRAC = 0.5
# 每篇文档提取的嵌入式图片数量硬上限。
EMBED_MAX_IMAGES = 50

# --- 布局感知的文本提取（原生文本页）--------------------------------------
# 通过几何 XY 切割重建阅读顺序，使多栏页面逐栏线性化，而非行交错。
LAYOUT_ORDERING = settings.pdf_layout_ordering
# 当字形定位没有显式空格字符时（OCR/搜索文本层常见），如果水平间隙超过
# 行中位数字形宽度的此倍数，则插入一个空格。
WORD_GAP_WIDTH_RATIO = 0.4
# 将视觉上较大的行提升为 Markdown 标题（字号代理 = 相对于页面中位行高的矩形高度）。
DETECT_HEADINGS = settings.pdf_detect_headings
# 丢弃不可见（渲染模式 3）、页面外和异常文本——针对隐藏文本提示注入和
# OCR 伪影的廉价防护。
FILTER_HIDDEN_TEXT = settings.pdf_filter_hidden_text
# 窄侧边栏（arXiv 水印、页码标签）窄于页面宽度此比例时，若看起来像垂直/
# 单字形噪声则被丢弃。
MARGIN_COL_WIDTH_RATIO = 0.12
# 一行上的最少字符数，超过后方可被字号启发式提升为 Markdown 标题
#（避免边距字形产生 ``### C``）。
MIN_HEADING_LINE_CHARS = 8
# 去除 pdfium 占位字形（U+FFFE）和软连字符；当页面上存在 Figure 标题时，
# 从矢量图形中移除坐标轴/图例文本。
SANITIZE_PDF_TEXT = True
STRIP_CHART_TEXT_DEBRIS = True
# 将检测到的矢量图表区域（无嵌入式位图）渲染为 JPEG 供 VLM/OCR 使用。
RENDER_VECTOR_FIGURES = True
MIN_CHART_REGION_CHARS = 18
MIN_CHART_REGION_AREA_RATIO = 0.015
MAX_CHART_REGION_AREA_RATIO = 0.42
MAX_FIGURE_HEIGHT_RATIO = 0.38

# --- 强制扫描模式 ------------------------------------------------------------
# 当为 True 时，所有 PDF 页面都渲染为图片并通过 OCR/VLM 路由，跳过自动的
# 文本/扫描页面分类。适用于具有低质量或误导性文本层的 PDF（网络打印、扫描、
# 图片密集）。可通过 parser_engine_overrides.pdf_force_scanned 按上传覆盖。
FORCE_SCANNED_PDF = settings.pdf_force_scanned

# pdfium / Adobe 文本层经常因缺少连字符或连字而输出 U+FFFE。
_PDF_ARTIFACT_RE = re.compile(r"[­​-‏﻿￾￿]")
_PDF_ARTIFACT_JOIN_RE = re.compile(r"(\w)[­￾](\w)")
_CHART_DEBRIS_LINE_RE = re.compile(
    r"^(?:"
    r"[\d\s.]+|"
    r"\d{1,2}|"
    r"\d+-layer|"
    r"iter\.\s*\(1e4\)|"
    r"(?:training|test)\s+error\s*\(%\)"
    r")$",
    re.IGNORECASE,
)
_CHART_LAYER_RE = re.compile(r"^\d+-layer$", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^Figure\s+\d+\b", re.IGNORECASE)
_FIGURE_CAPTION_SEARCH_RE = re.compile(r"\bFigure\s+(\d+)\b", re.IGNORECASE)
_ARXIV_LINE_RE = re.compile(r"^arXiv:\s*\S+", re.IGNORECASE)
_PAGE_NUM_LINE_RE = re.compile(r"^\d{1,3}$")


def _close_pdfium_resource(resource: object) -> None:
    close = getattr(resource, "close", None)
    if close:
        close()


def _normalize_image_quality(quality: int) -> int:
    return min(95, max(1, quality))


def _classify_page(image_area_ratio: float, text_len: int) -> str:
    """将页面分类为 ``"scanned"`` 或 ``"text"``。

    图片区域覆盖率是主要信号；稀疏文本层结合一些图片内容是次要信号。
    """
    if image_area_ratio >= SCAN_IMAGE_AREA_RATIO:
        return "scanned"
    if text_len < SCAN_MIN_CHARS_PER_PAGE and image_area_ratio >= _LOW_TEXT_IMAGE_RATIO:
        return "scanned"
    return "text"


def _page_image_area_ratio(page: Any, raw: Any) -> float:
    """返回页面面积中被图片对象覆盖的比例。

    重叠图片可能使比例超过 1.0；调用者仅将其与阈值比较，因此这无害。
    """
    width, height = page.get_size()
    page_area = float(width) * float(height)
    if page_area <= 0:
        return 0.0

    image_area = 0.0
    for obj in page.get_objects():
        try:
            if obj.type == raw.FPDF_PAGEOBJ_IMAGE:
                left, bottom, right, top = obj.get_bounds()
                image_area += abs((right - left) * (top - bottom))
        except Exception:
            continue
    return image_area / page_area


def _extract_page_text(page: Any) -> str:
    """简单的从上到下文本提取（回退路径）。"""
    textpage = None
    try:
        textpage = page.get_textpage()
        return cast(str, textpage.get_text_range())  # pypdfium2 无 stub，返回 Any，显式断言为 str
    finally:
        _close_pdfium_resource(textpage)


def _sanitize_pdf_text(text: str) -> str:
    """移除 PDF 文本层占位符并修复损坏的连字符。"""
    if not text:
        return text
    text = _PDF_ARTIFACT_RE.sub("", text)
    text = _PDF_ARTIFACT_JOIN_RE.sub(r"\1\2", text)
    return text


def _is_chart_debris_line(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if _CHART_DEBRIS_LINE_RE.match(t):
        return True
    if _CHART_LAYER_RE.match(t):
        return True
    # 刻度标签如 "0 1 2 3 4 5 6 0"
    if re.fullmatch(r"[\d\s.()-]+", t) and len(t) <= 24 and sum(c.isdigit() for c in t) >= 3:
        return True
    return False


def _strip_chart_text_debris(text: str) -> str:
    """丢弃从矢量图形泄露到文本层中的坐标轴/图例行。"""
    if not text:
        return text
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _is_chart_debris_line(lines[i]):
            j = i
            while j < len(lines) and (
                _is_chart_debris_line(lines[j]) or not lines[j].strip()
            ):
                j += 1
            if j - i >= 3:
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _strip_arxiv_and_page_num_lines(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: list[str] = []
    for ln in lines:
        t = ln.strip()
        if _ARXIV_LINE_RE.match(t):
            continue
        if _PAGE_NUM_LINE_RE.match(t):
            continue
        if "arXiv:" in ln:
            ln = re.sub(r"\s*arXiv:\s*\S+\s*(?:\[[^\]]+\])?\s*[^\n]*", "", ln).strip()
            if not ln:
                continue
        kept.append(ln)
    return "\n".join(kept)


def _strip_lines_above_figure_captions(text: str) -> str:
    """移除紧挨在 Figure 标题上方的图表/图形标签行。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    for ln in lines:
        if _line_has_figure_caption(ln):
            while out and _is_figure_interior_line(out[-1]):
                out.pop()
            out.append(ln)
        else:
            out.append(ln)
    return "\n".join(out)


def _is_body_paragraph_line(text: str) -> bool:
    t = text.strip()
    if len(t) < 48:
        return False
    return len(t.split()) >= 8


def _is_figure_interior_line(text: str) -> bool:
    """紧挨在 Figure 标题上方的短非正文行（图形标签、刻度）。"""
    t = text.strip()
    if not t or _FIGURE_CAPTION_RE.match(t):
        return False
    if _ARXIV_LINE_RE.match(t) or _PAGE_NUM_LINE_RE.match(t):
        return True
    if _is_body_paragraph_line(t):
        return False
    if _is_chart_debris_line(t):
        return True
    # 图形上方的散文句子（换行段落尾部）——保留在文本中。
    if t.endswith((".", "。", "!", "?", "！")) and len(t) >= 15:
        return False
    if len(t.split()) >= 7:
        return False
    if len(t) <= 40:
        return True
    return False


def _postprocess_pdf_text(text: str) -> str:
    if SANITIZE_PDF_TEXT:
        text = _sanitize_pdf_text(text)
    text = _strip_arxiv_and_page_num_lines(text)
    text = _strip_lines_above_figure_captions(text)
    if STRIP_CHART_TEXT_DEBRIS:
        text = _strip_chart_text_debris(text)
    return text


def _char_looks_chart_axis_tick(ch: str) -> bool:
    """坐标轴刻度/数字图表标签（不是图中的 ``layer`` 等单词）。"""
    t = ch.strip()
    if not t:
        return False
    if len(t) == 1 and t in "0123456789.%()-":
        return True
    if _CHART_LAYER_RE.match(t):
        return True
    if re.fullmatch(r"iter\.\s*\(1e4\)", t, re.I):
        return True
    if re.fullmatch(r"(?:training|test)\s+error\s*\(%\)", t, re.I):
        return True
    return False


def _chars_bbox(char_list: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    return (
        min(c["x0"] for c in char_list),
        min(c["y0"] for c in char_list),
        max(c["x1"] for c in char_list),
        max(c["y1"] for c in char_list),
    )


def _bbox_area_ratio(
    bbox: tuple[float, float, float, float], page_w: float, page_h: float
) -> float:
    page_area = float(page_w) * float(page_h)
    if page_area <= 0:
        return 0.0
    x0, y0, x1, y1 = bbox
    return max(0.0, (x1 - x0) * (y1 - y0) / page_area)


def _chart_region_bbox(
    chars: list[dict[str, Any]], page_w: float, page_h: float
) -> tuple[float, float, float, float] | None:
    """数字图表坐标轴标签的边界框（标题遍历失败时的回退方案）。"""
    chart = [c for c in chars if _char_looks_chart_axis_tick(c["ch"])]
    if len(chart) < MIN_CHART_REGION_CHARS:
        return None
    bbox = _chars_bbox(chart)
    ratio = _bbox_area_ratio(bbox, page_w, page_h)
    if ratio < MIN_CHART_REGION_AREA_RATIO or ratio > MAX_CHART_REGION_AREA_RATIO:
        return None
    x0, y0, x1, y1 = bbox
    pad_x = max(8.0, (x1 - x0) * 0.08)
    pad_y = max(8.0, (y1 - y0) * 0.08)
    return (
        max(0.0, x0 - pad_x),
        max(0.0, y0 - pad_y),
        min(page_w, x1 + pad_x),
        min(page_h, y1 + pad_y),
    )


def _expand_chart_bbox(
    bbox: tuple[float, float, float, float],
    page_w: float,
    page_h: float,
    margin_frac: float = 0.18,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    dx = (x1 - x0) * margin_frac
    dy = (y1 - y0) * margin_frac
    return (
        max(0.0, x0 - dx),
        max(0.0, y0 - dy),
        min(page_w, x1 + dx),
        min(page_h, y1 + dy),
    )


def _render_page_clip_jpeg(
    page: Any, bbox: tuple[float, float, float, float], scale: float, quality: int, max_edge: int
) -> bytes:
    """将 PDF 页面区域渲染为 JPEG（bbox 使用 PDF 点坐标，左下角为原点）。"""
    left, bottom, right, top = bbox
    scale_eff = _effective_scale(page, scale, max_edge)
    bitmap = None
    try:
        bitmap = page.render(scale=scale_eff)
        pil = bitmap.to_pil().convert("RGB")
    finally:
        _close_pdfium_resource(bitmap)
    page_w, page_h = page.get_size()
    x0 = int(left * scale_eff)
    x1 = int(right * scale_eff)
    y0 = int((page_h - top) * scale_eff)
    y1 = int((page_h - bottom) * scale_eff)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("degenerate clip bbox")
    return _pil_to_jpeg_bytes(pil.crop((x0, y0, x1, y1)), quality)


def _pil_to_jpeg_bytes(pil: Any, quality: int) -> bytes:
    buf = io.BytesIO()
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _group_lines_with_chars(chars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将字形分组为行；每行包含其字符列表和边界框。"""
    if not chars:
        return []
    heights = [c["y1"] - c["y0"] for c in chars if c["y1"] > c["y0"]]
    med_h = statistics.median(heights) if heights else 1.0
    ordered = sorted(chars, key=lambda c: -(c["y0"] + c["y1"]) / 2)
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    ref = None
    for c in ordered:
        yc = (c["y0"] + c["y1"]) / 2
        if ref is None or abs(yc - ref) <= 0.5 * med_h:
            cur.append(c)
            ref = yc if ref is None else ref
        else:
            groups.append(cur)
            cur = [c]
            ref = yc
    if cur:
        groups.append(cur)

    lines: list[dict[str, Any]] = []
    for grp in groups:
        grp_sorted = sorted(grp, key=lambda c: c["x0"])
        text = _join_line_glyphs(grp_sorted)
        if not text:
            continue
        hs = [c["y1"] - c["y0"] for c in grp_sorted if c["y1"] > c["y0"]]
        lines.append(
            {
                "text": text,
                "h": statistics.median(hs) if hs else med_h,
                "chars": grp_sorted,
                "bbox": _chars_bbox(grp_sorted),
            }
        )
    return lines


def _line_has_figure_caption(text: str) -> bool:
    return bool(_FIGURE_CAPTION_SEARCH_RE.search((text or "").strip()))


def _bbox_above_caption(
    lines: list[dict[str, Any]], cap_i: int, page_w: float, page_h: float
) -> tuple[float, float, float, float]:
    """Figure 标题行上方的区域（PDF 坐标，左下角为原点）。"""
    cap_bbox = lines[cap_i]["bbox"]
    cap_top = cap_bbox[3]
    x0, x1 = cap_bbox[0], cap_bbox[2]
    fig_h = page_h * min(MAX_FIGURE_HEIGHT_RATIO, 0.35)
    y_bottom = cap_top
    y_top = min(page_h, cap_top + fig_h)

    for j in range(cap_i - 1, -1, -1):
        t = lines[j]["text"]
        b = lines[j]["bbox"]
        if b[3] < y_bottom - 4:
            continue
        if b[1] > y_top + 4:
            break
        if _is_body_paragraph_line(t) and not _is_figure_interior_line(t):
            break
        if _is_figure_interior_line(t) or _is_chart_debris_line(t) or not t.strip():
            x0 = min(x0, b[0])
            x1 = max(x1, b[2])
            y_top = max(y_top, min(page_h, b[3] + fig_h * 0.15))

    min_h = page_h * 0.08
    if y_top - y_bottom < min_h:
        y_top = min(page_h, y_bottom + min_h)
    margin_x = max(8.0, (x1 - x0) * 0.05)
    return (
        max(0.0, x0 - margin_x),
        y_bottom,
        min(page_w, x1 + margin_x),
        y_top,
    )


def _cap_bbox_height(
    bbox: tuple[float, float, float, float], page_h: float, cap_y_top: float
) -> tuple[float, float, float, float]:
    """限制图形边界框高度（PDF 坐标，左下角为原点）。"""
    x0, y0, x1, y1 = bbox
    max_top = min(y1, cap_y_top + page_h * MAX_FIGURE_HEIGHT_RATIO)
    if max_top <= y0:
        return bbox
    return (x0, y0, x1, max_top)


def _inject_figure_markdown_before_captions(
    text: str, clips: list[tuple[str, str, float, str]]
) -> str:
    """在页面文本中每个 Figure 标题行之前立即插入 ``![...]()``。"""
    if not clips:
        return text
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    clip_idx = 0
    for i, ln in enumerate(lines):
        if clip_idx >= len(clips):
            break
        if not _line_has_figure_caption(ln):
            continue
        if i > 0 and lines[i - 1].lstrip().startswith("!["):
            continue
        ref_path = clips[clip_idx][0]
        fname = os.path.basename(ref_path)
        img_md = f"![{fname}]({ref_path})"
        lines[i] = f"{img_md}\n\n{ln}"
        clip_idx += 1
    return "\n".join(lines)


def _extract_vector_figure_clips(
    page: Any,
    page_index: int,
    plain_text: str,
    raw: Any,
    base_name: str,
    scale: float,
    quality: int,
    max_edge: int,
) -> list[tuple[str, str, float, str]]:
    """渲染页面上每个 ``Figure N.`` 标题锚定的矢量图形区域。

    返回 ``[(ref_path, b64, y_sort, caption_line), ...]`` 用于 Markdown 注入。
    """
    if not RENDER_VECTOR_FIGURES or not re.search(r"\bFigure\s+\d+", plain_text, re.I):
        return []
    textpage = None
    try:
        textpage = page.get_textpage()
        chars, page_w = _page_chars(textpage, page, raw)
        if not chars:
            return []
        page_h = page.get_size()[1]
        lines = _merge_orphan_punctuation_lines(_group_lines_with_chars(chars))
        caption_indices = [
            i for i, ln in enumerate(lines) if _line_has_figure_caption(ln["text"])
        ]
        if not caption_indices:
            return []

        results: list[tuple[str, str, float, str]] = []
        for fig_idx, cap_i in enumerate(caption_indices):
            cap_line = lines[cap_i]["text"].strip()
            m = _FIGURE_CAPTION_SEARCH_RE.search(cap_line)
            if m:
                cap_line = cap_line[m.start() :].split("\n", 1)[0].strip()

            bbox = _bbox_above_caption(lines, cap_i, page_w, page_h)
            if bbox is None:
                bbox = _chart_region_bbox(chars, page_w, page_h)
            if bbox is None:
                continue

            ratio = _bbox_area_ratio(bbox, page_w, page_h)
            if ratio > MAX_CHART_REGION_AREA_RATIO:
                bbox = _cap_bbox_height(bbox, page_h, lines[cap_i]["bbox"][3])
                ratio = _bbox_area_ratio(bbox, page_w, page_h)
                if ratio > MAX_CHART_REGION_AREA_RATIO:
                    continue
            if ratio < MIN_CHART_REGION_AREA_RATIO:
                continue

            bbox = _expand_chart_bbox(bbox, page_w, page_h, margin_frac=0.06)
            jpeg = _render_page_clip_jpeg(page, bbox, scale, quality, max_edge)
            fname = f"{base_name}_p{page_index + 1}_fig{fig_idx + 1}.jpg"
            ref_path = f"images/{fname}"
            results.append(
                (
                    ref_path,
                    base64.b64encode(jpeg).decode("utf-8"),
                    bbox[3],
                    cap_line,
                )
            )
        return results
    except Exception:
        logger.debug("vector figure clip failed on page %d", page_index, exc_info=True)
        return []
    finally:
        _close_pdfium_resource(textpage)


def _collect_invisible_boxes(page: Any, raw: Any) -> list[tuple[float, float, float, float]]:
    """页面上不可见（渲染模式 3）文本对象的边界框。"""
    boxes: list[tuple[float, float, float, float]] = []
    try:
        for obj in page.get_objects():
            if obj.type != raw.FPDF_PAGEOBJ_TEXT:
                continue
            try:
                mode = raw.FPDFTextObj_GetTextRenderMode(obj.raw)
            except Exception:
                continue
            if mode != raw.FPDF_TEXTRENDERMODE_INVISIBLE:
                continue
            try:
                left, bottom, right, top = obj.get_bounds()
            except Exception:
                continue
            boxes.append(
                (min(left, right), min(bottom, top), max(left, right), max(bottom, top))
            )
    except Exception:
        return []
    return boxes


def _point_in_boxes(x: float, y: float, boxes: list[tuple[float, float, float, float]]) -> bool:
    for x0, y0, x1, y1 in boxes:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _page_chars(textpage: Any, page: Any, raw: Any) -> tuple[list[dict[str, Any]], float]:
    """返回 ``(chars, page_width)``，过滤了隐藏/页面外的字形。

    在字形级别工作（而非 pdfium 矩形分段）可保持混合 CJK + 拉丁/数字行的
    真实从左到右顺序，而矩形级别的 ``get_text_bounded`` API 会打乱此顺序。
    """
    n = textpage.count_chars()
    if n <= 0:
        return [], 0.0
    width, height = page.get_size()
    invisible = _collect_invisible_boxes(page, raw) if FILTER_HIDDEN_TEXT else []

    chars: list[dict[str, Any]] = []
    for i in range(n):
        try:
            left, bottom, right, top = textpage.get_charbox(i)
        except Exception:
            continue
        ch = textpage.get_text_range(i, 1)
        if ch in ("\r", "\n"):
            continue
        x0, x1 = (left, right) if left <= right else (right, left)
        y0, y1 = (bottom, top) if bottom <= top else (top, bottom)
        if FILTER_HIDDEN_TEXT:
            if x1 < 0 or x0 > width or y1 < 0 or y0 > height:
                continue  # 页面外字形
            if invisible and _point_in_boxes((x0 + x1) / 2, (y0 + y1) / 2, invisible):
                continue  # 被不可见文本对象覆盖
        chars.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "ch": ch})
    return chars, width


def _find_split(items: list[dict[str, Any]], axis: str, min_gap: float) -> float | None:
    """返回 ``axis``（'x'）上最宽干净间隙处的坐标，或 None。

    "干净"间隙意味着没有项目区间跨越它——即全高的栏间距。用于检测多栏布局。
    """
    lo, hi = ("x0", "x1") if axis == "x" else ("y0", "y1")
    intervals = sorted(((s[lo], s[hi]) for s in items), key=lambda iv: iv[0])
    cur_end = intervals[0][1]
    best_gap, best_cut = 0.0, None
    for a, b in intervals[1:]:
        gap = a - cur_end
        if gap >= min_gap and gap > best_gap:
            best_gap, best_cut = gap, cur_end + gap / 2
        if b > cur_end:
            cur_end = b
    return best_cut


def _split_columns(
    chars: list[dict[str, Any]], scale: float, width: float, depth: int = 0
) -> list[list[dict[str, Any]]]:
    """在全高间距处将字形分割为阅读顺序的栏。"""
    if len(chars) <= 1 or depth > 10:
        return [chars]
    min_gap = max(scale * 2.5, width * 0.04)
    cut = _find_split(chars, "x", min_gap)
    if cut is None:
        return [chars]
    left = [c for c in chars if (c["x0"] + c["x1"]) / 2 < cut]
    right = [c for c in chars if (c["x0"] + c["x1"]) / 2 >= cut]
    if not left or not right:
        return [chars]
    return _split_columns(left, scale, width, depth + 1) + _split_columns(
        right, scale, width, depth + 1
    )


def _column_x_span(chars: list[dict[str, Any]]) -> float:
    if not chars:
        return 0.0
    # 字形 dict 值为 pdfium 浮点坐标（Any），显式断言为 float
    return cast(float, max(c["x1"] for c in chars) - min(c["x0"] for c in chars))


def _column_single_line_fraction(lines: list[dict[str, Any]]) -> float:
    if not lines:
        return 0.0
    single = sum(1 for ln in lines if len(ln["text"]) <= 2)
    return single / len(lines)


def _is_artifact_column(chars: list[dict[str, Any]], width: float) -> bool:
    """检测边距条和垂直水印（如 arXiv 侧边栏）。

    Docling / MinerU 通过学习布局区域解决此问题；此处仅使用几何信息：
    窄栏且其行大多只有一个字形高度，则不属于阅读顺序。
    """
    if not chars or width <= 0:
        return True
    span = _column_x_span(chars)
    if span <= 0:
        return True
    lines = _group_lines(chars)
    single_frac = _column_single_line_fraction(lines)
    narrow = span / width < MARGIN_COL_WIDTH_RATIO
    if narrow and single_frac >= 0.45:
        return True
    ys = [(c["y0"] + c["y1"]) / 2 for c in chars]
    y_span = max(ys) - min(ys)
    # 垂直文本：高堆叠、窄水平范围、大多每行一个字符。
    if y_span > span * 3.5 and len(chars) >= 8 and single_frac >= 0.35:
        return True
    return False


def _filter_reading_columns(
    chars: list[dict[str, Any]], scale: float, width: float
) -> list[list[dict[str, Any]]]:
    """分割为栏并丢弃边距/水印条。"""
    cols = _split_columns(chars, scale, width)
    kept = [c for c in cols if not _is_artifact_column(c, width)]
    if kept:
        return kept
    # 所有栏都看起来像噪声——保留最宽的字形集（主体内容）。
    if len(cols) > 1:
        return [max(cols, key=_column_x_span)]
    return cols


def _merge_orphan_punctuation_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将仅包含标点符号的行附加到前一个视觉行。

    许多 PDF 将图形标签或脚注中的 ``.`` 放在略有不同的基线上；
    按 y 分组后会导致 ``Figure 1`` 和 ``2:`` 位于不同行。
    """
    if not lines:
        return []
    merged: list[dict[str, Any]] = []
    for ln in lines:
        t = ln["text"].strip()
        if (
            merged
            and t
            and len(t) <= 4
            and all(c in ".,;:!?…·" or c.isspace() for c in t)
        ):
            suffix = "".join(t.split())
            prev = merged[-1]["text"]
            if suffix and prev and not prev.endswith((" ", "-")):
                merged[-1]["text"] = prev + suffix
            else:
                merged[-1]["text"] = (prev + " " + t).strip()
            continue
        merged.append(dict(ln))
    return merged


def _join_line_glyphs(ln_sorted: list[dict[str, Any]]) -> str:
    """连接可视化行的字形，从水平间隙推断单词空格。"""
    if not ln_sorted:
        return ""
    widths = [c["x1"] - c["x0"] for c in ln_sorted if c["x1"] > c["x0"]]
    med_w = statistics.median(widths) if widths else 1.0
    gap_threshold = med_w * WORD_GAP_WIDTH_RATIO

    parts: list[str] = []
    for i, cur in enumerate(ln_sorted):
        ch = cur["ch"]
        if i == 0:
            parts.append(ch)
            continue
        prev = ln_sorted[i - 1]
        if ch.isspace() or prev["ch"].isspace():
            if not ch.isspace() or (parts and not parts[-1].endswith(" ")):
                parts.append(ch)
            continue
        if cur["x0"] - prev["x1"] > gap_threshold:
            parts.append(" ")
        parts.append(ch)
    return "".join(parts).strip()


def _group_lines(chars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将一栏的字形分组为行（从上到下，字形按 x 排序）。"""
    if not chars:
        return []
    heights = [c["y1"] - c["y0"] for c in chars if c["y1"] - c["y0"] > 0]
    med_h = statistics.median(heights) if heights else 1.0

    ordered = sorted(chars, key=lambda c: -(c["y0"] + c["y1"]) / 2)
    lines: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    ref = None
    for c in ordered:
        yc = (c["y0"] + c["y1"]) / 2
        if ref is None or abs(yc - ref) <= 0.5 * med_h:
            cur.append(c)
            ref = yc if ref is None else ref
        else:
            lines.append(cur)
            cur = [c]
            ref = yc
    if cur:
        lines.append(cur)

    out: list[dict[str, Any]] = []
    for ln in lines:
        ln_sorted = sorted(ln, key=lambda c: c["x0"])
        text = _join_line_glyphs(ln_sorted)
        if not text:
            continue
        hs = [c["y1"] - c["y0"] for c in ln_sorted if c["y1"] - c["y0"] > 0]
        out.append({"h": statistics.median(hs) if hs else med_h, "text": text})
    return out


def _segments_to_markdown(lines: list[dict[str, Any]]) -> str:
    """将合并的行渲染为文本，将视觉上较大的行提升为标题。"""
    if not lines:
        return ""
    body = statistics.median([ln["h"] for ln in lines])

    def level(ln: dict[str, Any]) -> int:
        txt = ln["text"]
        if (
            not DETECT_HEADINGS
            or body <= 0
            or len(txt) > 80
            or len(txt) < MIN_HEADING_LINE_CHARS
        ):
            return 0
        if txt[-1:] in ".。!！?？,，;；:：":
            return 0
        r = ln["h"] / body
        if r >= 2.0:
            return 1
        if r >= 1.6:
            return 2
        if r >= 1.35:
            return 3
        return 0

    levels = [level(ln) for ln in lines]
    # 如果太多行符合条件，说明字体大小过于均匀/嘈杂，不可信。
    if sum(1 for x in levels if x) > max(1, int(0.4 * len(lines))):
        levels = [0] * len(lines)

    out = []
    for ln, lv in zip(lines, levels):
        out.append(("#" * lv + " " + ln["text"]) if lv else ln["text"])
    return "\n".join(out)


def _chars_to_layout_markdown(
    chars: list[dict[str, Any]], scale: float, width: float
) -> str:
    blocks: list[str] = []
    for col in _filter_reading_columns(chars, scale, width):
        lines = _merge_orphan_punctuation_lines(_group_lines(col))
        md = _segments_to_markdown(lines)
        if md:
            blocks.append(md)
    return "\n".join(blocks)


def _layout_line_stats(text: str) -> tuple[int, int, int]:
    """返回（行数、单字符行数、仅标点行数）。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0, 0, 0
    single = sum(1 for ln in lines if len(ln) <= 2)
    punct_only = sum(
        1
        for ln in lines
        if len(ln) <= 4 and re.fullmatch(r"[\s.,;:!?…·\-–—]+", ln)
    )
    return len(lines), single, punct_only


def _layout_garbled_line_fraction(text: str) -> float:
    """看起来像损坏的 OCR 的行所占比例（许多 1-2 个字母的标记）。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    garbled = 0
    for ln in lines:
        words = ln.split()
        if len(words) >= 6 and sum(1 for w in words if len(w) <= 2) / len(words) > 0.45:
            garbled += 1
    return garbled / len(lines)


def _plain_is_well_formed(plain: str) -> bool:
    """当 pdfium 纯文本已经包含可用的单词和标点时返回 True。

    学术 PDF（arXiv）和目录已经暴露了良好的文本层；对其运行几何布局
    通常会破坏引用和单词。具有较差文本层的扫描书籍（引用中没有逗号、
    短粘合标记）仍然需要布局间隙推断。
    """
    plain = (plain or "").strip()
    if not plain:
        return False
    if re.search(r"\[\w+,\s", plain):
        return True
    if plain.count(" . . ") >= 2:
        return True
    words = re.findall(r"\S+", plain)
    if len(words) < 30:
        return False
    avg_len = sum(len(w) for w in words) / len(words)
    return avg_len >= 5.0


def _should_prefer_plain(plain: str, layout: str) -> bool:
    """当布局重建看起来有问题时，回退到 pdfium 纯文本。"""
    layout = (layout or "").strip()
    plain = (plain or "").strip()
    if not layout:
        return True
    if not plain:
        return False
    n, single, punct_only = _layout_line_stats(layout)
    if n == 0:
        return True
    if single / n >= 0.18 or punct_only / n >= 0.12:
        return True
    garbled = _layout_garbled_line_fraction(layout)
    if garbled >= 0.20 and _layout_garbled_line_fraction(plain) < 0.08:
        return True
    if re.search(r"\[\w+,\s", plain) and re.search(
        r"\[\w+\s+\w+\s+\d", layout
    ):
        return True
    # 来自纯文本的标题/引导句应在布局中保留。
    for ln in plain.splitlines():
        probe = ln.strip()
        if len(probe) < 24:
            continue
        alnum = "".join(c for c in probe if c.isalnum())[:16]
        if len(alnum) < 12:
            continue
        layout_alnum = "".join(c for c in layout if c.isalnum())
        if alnum not in layout_alnum:
            return True
        break
    return False


def _extract_layout_text(page: Any, raw: Any) -> str:
    """布局感知的提取：阅读顺序 + 标题 + 隐藏文本过滤。

    在任何失败时回退到纯文本提取，因此单个异常页面不会破坏整个文档。
    """
    textpage = None
    try:
        textpage = page.get_textpage()
        chars, width = _page_chars(textpage, page, raw)
        if not chars:
            return ""
        heights = [c["y1"] - c["y0"] for c in chars if c["y1"] - c["y0"] > 0]
        scale = (statistics.median(heights) if heights else 1.0) or 1.0
        return _chars_to_layout_markdown(chars, scale, width)
    except Exception:
        logger.debug("layout extraction failed; using plain text", exc_info=True)
        return _extract_page_text(page)
    finally:
        _close_pdfium_resource(textpage)


def _effective_scale(page: Any, scale: float, max_edge: int) -> float:
    """减小 ``scale``，使渲染的长边不超过 ``max_edge`` 像素。

    某些扫描 PDF 声明了巨大的页面框；以原始 DPI 比例渲染会产生超过
    消息限制的 100+ MP JPEG，且分辨率远超 OCR 所需。
    """
    if max_edge <= 0:
        return scale
    width, height = page.get_size()
    longest_pt = max(float(width), float(height))
    if longest_pt <= 0:
        return scale
    return min(scale, max_edge / longest_pt)


def _render_page_to_jpeg(page: Any, scale: float, quality: int, max_edge: int = 0) -> bytes:
    bitmap = None
    try:
        bitmap = page.render(scale=_effective_scale(page, scale, max_edge))
        img_obj = bitmap.to_pil()
        if img_obj.mode != "RGB":
            img_obj = img_obj.convert("RGB")
        buf = io.BytesIO()
        img_obj.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    finally:
        _close_pdfium_resource(bitmap)


# --- 并行扫描页面渲染 ------------------------------------------------------
# pdfium 不是线程安全的（对一个文档并发 get_page 会崩溃），因此我们通过
# *进程* 并行化：每个工作进程从临时文件打开自己的 PdfDocument 并渲染
# 分配的页面切片。这将串行的逐页渲染（大型扫描 PDF 的主要成本——在受
# CPU 限制的容器上需要数小时）转变为接近线性的加速。

# 每个工作进程的文档句柄，由池初始化器填充（pypdfium2 无 stub，Any 显式化）。
_WORKER_RENDER_DOC: Any = None


def _render_pool_init(pdf_path: str) -> None:
    global _WORKER_RENDER_DOC
    import pypdfium2 as pdfium

    with open(pdf_path, "rb") as f:
        _WORKER_RENDER_DOC = pdfium.PdfDocument(f.read())


def _render_pool_task(args: tuple[int, float, int, int]) -> tuple[int, bytes]:
    index, scale, quality, max_edge = args
    page = _WORKER_RENDER_DOC[index]
    try:
        return index, _render_page_to_jpeg(page, scale, quality, max_edge)
    finally:
        _close_pdfium_resource(page)


def _select_mp_context() -> mp.context.BaseContext | None:
    """选择最安全的多进程启动方法。

    ``forkserver`` 从干净的单线程服务器进程创建子进程，
    避免了多线程进程中 fork 的风险。
    回退到 ``fork``，当两者都不可用时（如 Windows/开发环境）最终返回 ``None``（串行）。
    """
    for method in ("forkserver", "fork"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue
    return None


def _render_pages_parallel(
    content: bytes, indices: list[int], scale: float, quality: int, max_edge: int, workers: int
) -> dict[int, bytes] | None:
    """并行渲染 ``indices``。返回 ``{index: jpeg_bytes}`` 或 None。

    返回 None 以提示调用者回退到串行渲染（当并行被禁用、仅请求一页、
    或没有可用的多进程启动方法时）。
    """
    if workers <= 1 or len(indices) <= 1:
        return None
    ctx = _select_mp_context()
    if ctx is None:
        return None

    import tempfile
    from concurrent.futures import ProcessPoolExecutor

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="docreader_render_", suffix=".pdf", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        max_workers = min(workers, len(indices))
        tasks = [(i, scale, quality, max_edge) for i in indices]
        result: dict[int, bytes] = {}
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_render_pool_init,
            initargs=(tmp_path,),
        ) as ex:
            for index, jpeg in ex.map(_render_pool_task, tasks, chunksize=4):
                result[index] = jpeg
        return result
    except Exception:
        logger.warning(
            "parallel page rendering failed; falling back to serial",
            exc_info=True,
        )
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _render_scanned_pages(
    pdf: Any, content: bytes, indices: list[int], scale: float, quality: int, max_edge: int
) -> dict[int, bytes]:
    """将给定（扫描）页面索引渲染为 JPEG 字节。

    首先尝试进程并行渲染（对大型扫描 PDF 收益巨大），
    当并行不可用或失败时，透明地回退到已打开的 ``pdf`` 句柄上的串行渲染。
    """
    parallel = _render_pages_parallel(
        content, indices, scale, quality, max_edge, settings.pdf_render_parallelism
    )
    if parallel is not None:
        return parallel

    out: dict[int, bytes] = {}
    for i in indices:
        page = pdf[i]
        try:
            out[i] = _render_page_to_jpeg(page, scale, quality, max_edge)
        finally:
            _close_pdfium_resource(page)
    return out


def _select_embedded_images(
    meta: list[dict[str, Any]],
    num_text_pages: int,
    *,
    min_pixels: int = EMBED_MIN_PIXELS,
    min_area_ratio: float = EMBED_MIN_AREA_RATIO,
    repeat_frac: float = EMBED_REPEAT_PAGE_FRAC,
    max_images: int = EMBED_MAX_IMAGES,
) -> list[int]:
    """决定保留哪些嵌入式图片候选（纯函数）。

    ``meta`` 是包含键 ``page``、``width``、``height``、``area_ratio`` 和
    ``hash`` 的字典列表。返回要保留的索引（指向 ``meta``），
    经过大小、页面面积占比、跨页重复（徽标/水印）、页面内精确重复和
    硬计数上限的过滤。
    """
    from collections import defaultdict

    hash_pages = defaultdict(set)
    for m in meta:
        hash_pages[m["hash"]].add(m["page"])

    repeat_threshold = max(2, int(num_text_pages * repeat_frac)) if num_text_pages else 2
    banned = {h for h, pages in hash_pages.items() if len(pages) >= repeat_threshold}

    kept: list[int] = []
    seen = set()
    for idx, m in enumerate(meta):
        if m["area_ratio"] < min_area_ratio:
            continue
        if m["width"] < min_pixels or m["height"] < min_pixels:
            continue
        if m["hash"] in banned:
            continue
        key = (m["page"], m["hash"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(idx)
        if len(kept) >= max_images:
            break
    return kept


def _extract_embedded_images(
    pdf: Any, classes: list[str], raw: Any, base_name: str, quality: int
) -> dict[int, list[tuple[str, str, float]]]:
    """从原生文本页面提取经过过滤的嵌入式图形。

    返回 ``{page_index: [(ref_path, base64_jpeg, y_top), ...]}``，按从上到下的
    顺序排列，以便调用者可以在页面文本之后放置图形。
    """
    import hashlib

    text_indices = [i for i, c in enumerate(classes) if c == "text"]
    if not text_indices:
        return {}

    candidates: list[tuple[int, Any, Any]] = []  # parallel to meta; holds heavy pixel data
    meta: list[dict[str, Any]] = []
    for i in text_indices:
        page = pdf[i]
        try:
            width, height = page.get_size()
            page_area = float(width) * float(height)
            if page_area <= 0:
                continue
            for obj in page.get_objects():
                if obj.type != raw.FPDF_PAGEOBJ_IMAGE:
                    continue
                try:
                    left, bottom, right, top = obj.get_bounds()
                except Exception:
                    continue
                area_ratio = abs((right - left) * (top - bottom)) / page_area
                if area_ratio < EMBED_MIN_AREA_RATIO:
                    continue  # cheap skip before decoding (logos/decorations)
                try:
                    pil = obj.get_bitmap().to_pil()
                except Exception:
                    continue
                content_hash = hashlib.md5(pil.tobytes()).hexdigest()
                candidates.append((i, top, pil))
                meta.append(
                    {
                        "page": i,
                        "width": pil.width,
                        "height": pil.height,
                        "area_ratio": area_ratio,
                        "hash": content_hash,
                    }
                )
        finally:
            _close_pdfium_resource(page)

    kept_idx = _select_embedded_images(meta, len(text_indices))
    if not kept_idx:
        return {}

    from collections import defaultdict

    result: dict[int, list[tuple[str, str, float]]] = defaultdict(list)
    per_page_count: dict[int, int] = defaultdict(int)
    max_edge = settings.pdf_render_max_edge
    for idx in kept_idx:
        page_i, y_top, pil = candidates[idx]
        if pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
        if max_edge > 0 and max(pil.size) > max_edge:
            ratio = max_edge / max(pil.size)
            pil = pil.resize(
                (max(1, int(pil.width * ratio)), max(1, int(pil.height * ratio)))
            )
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality, optimize=True)
        per_page_count[page_i] += 1
        fname = f"{base_name}_p{page_i+1}_img{per_page_count[page_i]}.jpg"
        ref_path = f"images/{fname}"
        result[page_i].append(
            (ref_path, base64.b64encode(buf.getvalue()).decode("utf-8"), y_top)
        )

    # 每页内从上到下排序（PDF y 向上增长，因此较大的 y 优先）。
    for page_i in result:
        result[page_i].sort(key=lambda item: item[2], reverse=True)
    return result


def _strip_repeating_lines(texts: list[str], classes: list[str]) -> list[str]:
    """移除在大多数文本页面上重复出现的页眉/页脚。

    保守策略：只有每个文本页面的第一个/最后一个非空行是候选，该行必须简短，
    并且必须至少出现在 60% 的文本页面上（并且必须有足够的页面来判断）。
    镜像 DeepDoc 的跨页"垃圾集"理念，而不冒移除真实内容的风险。
    """
    from collections import Counter

    text_indices = [i for i, c in enumerate(classes) if c == "text"]
    if len(text_indices) < 4:
        return list(texts)

    counter: Counter[str] = Counter()
    for i in text_indices:
        lines = [ln.strip() for ln in texts[i].splitlines() if ln.strip()]
        if not lines:
            continue
        for edge in {lines[0], lines[-1]}:
            if len(edge) <= 80:
                counter[edge] += 1

    threshold = max(2, int(len(text_indices) * 0.6))
    repeating = {line for line, count in counter.items() if count >= threshold}
    if not repeating:
        return list(texts)

    cleaned = []
    for i, text in enumerate(texts):
        if classes[i] != "text":
            cleaned.append(text)
            continue
        kept = [ln for ln in text.splitlines() if ln.strip() not in repeating]
        cleaned.append("\n".join(kept))
    return cleaned


def _is_table_line(line: str) -> bool:
    """判断一行是否为 Markdown 表格行（以管道符开头）。"""
    return bool(line) and line.strip().startswith("|")


def _merge_cross_page_tables(texts: list[str], classes: list[str]) -> list[str]:
    """合并跨连续文本页面的表格片段。

    当页面 N 的最后非空行和页面 N+1 的第一个非空行都是 Markdown 表格行时，
    将两页文本合并（移除页间 ``\\n\\n`` 分隔），使表格保持完整。
    支持连续三页及以上的表格链式合并。

    Markdown 表格的判定依据是行首 ``|``，该标记来自 Markdown 转换路径
    （DOCX → HTML → Markdown，或外部 OCR 输出）。
    """
    result: list[str] = []

    i = 0
    while i < len(texts):
        if classes[i] != "text" or not texts[i]:
            result.append(texts[i])
            i += 1
            continue

        # 贪心合并：从当前页开始，尽可能向后合并连续的表格片段
        combined = texts[i]
        j = i + 1
        while j < len(texts) and classes[j] == "text" and texts[j]:
            prev_lines = [ln for ln in combined.splitlines() if ln.strip()]
            curr_lines = [ln for ln in texts[j].splitlines() if ln.strip()]
            if (
                prev_lines
                and curr_lines
                and _is_table_line(prev_lines[-1].strip())
                and _is_table_line(curr_lines[0].strip())
            ):
                combined = combined.rstrip("\n") + "\n" + texts[j].lstrip("\n")
                j += 1
            else:
                break

        result.append(combined)
        i = j

    return result


class PDFScannedParser(BaseParser):
    """将每个 PDF 页面渲染为 JPEG 图片。

    用作稳健的最后手段回退方案，也用于纯图片 PDF。后续 VLM 模块对提取的页面图片执行 OCR。
    """

    def parse_into_text(self, content: bytes) -> ParsedDocument:
        import pypdfium2 as pdfium

        images = {}
        markdown_lines = []
        base_name = os.path.splitext(self.file_name or "document")[0]

        logger.info(
            "PDFScannedParser: Rendering PDF pages to JPEG images for %s",
            self.file_name,
        )

        try:
            # 先获取 pdfium 锁，然后是渲染工作线程限制——与 PDFParser._route
            # 的顺序相同，这样两者永远不会相互死锁。
            with _PDFIUM_LOCK, parser_worker_limit(
                "pdf_render", settings.pdf_render_max_workers
            ):
                pdf = pdfium.PdfDocument(content)
                try:
                    page_count = len(pdf)
                    scale = max(1, settings.pdf_render_dpi) / 72
                    quality = _normalize_image_quality(settings.pdf_jpeg_quality)

                    rendered = _render_scanned_pages(
                        pdf,
                        content,
                        list(range(page_count)),
                        scale,
                        quality,
                        settings.pdf_render_max_edge,
                    )
                finally:
                    _close_pdfium_resource(pdf)

            for i in range(page_count):
                page_filename = f"{base_name}_page_{i+1}.jpg"
                ref_path = f"images/{page_filename}"
                markdown_lines.append(f"![{page_filename}]({ref_path})")
                images[ref_path] = base64.b64encode(rendered[i]).decode("utf-8")

            text = "\n\n".join(markdown_lines)
            return ParsedDocument(
                content=text,
                images=images,
                metadata={
                    "image_source_type": "scanned_pdf",
                    "page_count": page_count,
                },
            )
        except Exception as e:
            logger.exception("PDFScannedParser failed to parse PDF: %s", e)
            raise e


class PDFParser(BaseParser):
    """在原生文本提取和扫描渲染之间的逐页路由器。

    对每页：
      * 原生文本页  -> 保留其文本层（快速，pypdfium2）
      * 扫描页      -> 渲染为 JPEG，标记 ``image_source_type=scanned_pdf``，由后续 VLM 模块处理

    混合文档在阅读顺序中交错两者。在出现任何意外错误时，
    解析器回退到将所有页面渲染为图片（安全的最後手段）。

    强制扫描模式（``pdf_force_scanned=true`` 覆盖或
    ``DOCREADER_PDF_FORCE_SCANNED=true`` 环境变量）跳过分类，
    将所有页面渲染为图片。
    """

    def __init__(
        self, file_name: str = "", file_type: str | None = None, **kwargs: Any
    ):
        # 在 BaseParser 消费 kwargs 之前捕获每次上传的覆盖值。
        raw = kwargs.pop("pdf_force_scanned", None)
        super().__init__(file_name=file_name, file_type=file_type, **kwargs)
        # 优先级：每次上传的覆盖 > 全局环境变量 > 默认值（False）。
        if raw is not None:
            self._force_scanned = str(raw).strip().lower() in {
                "1", "true", "yes", "y", "on",
            }
        else:
            self._force_scanned = FORCE_SCANNED_PDF

    def parse_into_text(self, content: bytes) -> ParsedDocument:
        # 强制扫描模式的短路路径：将所有页面渲染为图片。
        if self._force_scanned:
            logger.info(
                "PDFParser: force scanned mode enabled for %s",
                self.file_name,
            )
            doc = PDFScannedParser(
                file_name=self.file_name, file_type=self.file_type
            ).parse_into_text(content)

            # 与自动扫描路由对齐元数据字段
            page_count = doc.metadata.get("page_count", 0)
            doc.metadata.update({
                "scanned_page_count": page_count,
                "text_page_count": 0,
                "embedded_image_count": 0,
                "vector_figure_count": 0,
            })
            return doc

        try:
            return self._route(content)
        except Exception:
            logger.exception(
                "PDFParser: per-page routing failed for %s; "
                "falling back to full image rendering",
                self.file_name,
            )
            return PDFScannedParser(
                file_name=self.file_name, file_type=self.file_type
            ).parse_into_text(content)

    def _route(self, content: bytes) -> ParsedDocument:
        # 串行化所有 pdfium 工作：见 _PDFIUM_LOCK。在整个路由过程中持有该锁
        #（包括文本处理和渲染处理）正是防止并发上传死锁的关键；后续的 Markdown
        # 组装是廉价的纯 Python 操作，因此将其保留在锁内没有任何有意义的开销。
        with _PDFIUM_LOCK:
            return self._route_locked(content)

    def _route_locked(self, content: bytes) -> ParsedDocument:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_r

        base_name = os.path.splitext(self.file_name or "document")[0]
        scale = max(1, settings.pdf_render_dpi) / 72
        quality = _normalize_image_quality(settings.pdf_jpeg_quality)

        pdf = pdfium.PdfDocument(content)
        images: dict[str, str] = {}
        try:
            page_count = len(pdf)

            # 第一遍：廉价的文本提取 + 图片区域分类。
            texts: list[str] = []
            classes: list[str] = []
            vector_clips: dict[int, list[tuple[str, str, float, str]]] = {}
            for i in range(page_count):
                page = pdf[i]
                try:
                    plain = _extract_page_text(page)
                    ratio = _page_image_area_ratio(page, pdfium_r)
                    cls = _classify_page(ratio, len(plain.strip()))
                    # 布局重建仅在原生文本页面上值得投入（且仅用于此类页面）；
                    # 扫描页面被渲染，而非读取。
                    if cls == "text" and LAYOUT_ORDERING:
                        if _plain_is_well_formed(plain):
                            text = plain
                        else:
                            layout = _extract_layout_text(page, pdfium_r)
                            if layout and not _should_prefer_plain(plain, layout):
                                text = layout
                            else:
                                text = plain
                    else:
                        text = plain
                    if cls == "text":
                        clips = _extract_vector_figure_clips(
                            page,
                            i,
                            plain,
                            pdfium_r,
                            base_name,
                            scale,
                            quality,
                            settings.pdf_render_max_edge,
                        )
                        if clips:
                            vector_clips[i] = clips
                            for ref_path, b64, _y, _cap in clips:
                                images[ref_path] = b64
                    text = _postprocess_pdf_text(text)
                    if cls == "text" and vector_clips.get(i):
                        text = _inject_figure_markdown_before_captions(
                            text, vector_clips[i]
                        )
                finally:
                    _close_pdfium_resource(page)
                texts.append(text)
                classes.append(cls)

            texts = _strip_repeating_lines(texts, classes)
            texts = _merge_cross_page_tables(texts, classes)
            scanned_indices = [i for i, c in enumerate(classes) if c == "scanned"]

            # 第二遍：仅渲染扫描页面（繁重工作，速率受限）。
            if scanned_indices:
                with parser_worker_limit("pdf_render", settings.pdf_render_max_workers):
                    rendered = _render_scanned_pages(
                        pdf,
                        content,
                        scanned_indices,
                        scale,
                        quality,
                        settings.pdf_render_max_edge,
                    )
                for i, img_bytes in rendered.items():
                    ref_path = f"images/{base_name}_page_{i+1}.jpg"
                    images[ref_path] = base64.b64encode(img_bytes).decode("utf-8")

            # 第三遍：从原生文本页面提取嵌入式图形，以便后续 VLM 模块处理
            #（过滤徽标/水印/微小图片）。
            embedded: dict[int, list[tuple[str, str, float]]] = {}
            if EXTRACT_EMBEDDED_IMAGES:
                embedded = _extract_embedded_images(
                    pdf, classes, pdfium_r, base_name, quality
                )
                for refs in embedded.values():
                    for ref_path, b64, _y in refs:
                        images[ref_path] = b64
        finally:
            _close_pdfium_resource(pdf)

        # 按阅读顺序组装 Markdown。
        embedded_count = 0
        vector_figure_count = 0
        blocks = []
        for i in range(page_count):
            if classes[i] == "scanned":
                page_filename = f"{base_name}_page_{i+1}.jpg"
                blocks.append(f"![{page_filename}](images/{page_filename})")
            else:
                stripped = texts[i].strip()
                if stripped:
                    blocks.append(stripped)
                vector_figure_count += len(vector_clips.get(i, []))
                page_images = list(embedded.get(i, []))
                page_images.sort(key=lambda item: item[2], reverse=True)
                for ref_path, _b64, _y in page_images:
                    fname = os.path.basename(ref_path)
                    blocks.append(f"![{fname}]({ref_path})")
                    embedded_count += 1

        content_text = "\n\n".join(blocks).strip()

        metadata = {
            "page_count": page_count,
            "scanned_page_count": len(scanned_indices),
            "text_page_count": page_count - len(scanned_indices),
            "embedded_image_count": embedded_count,
            "vector_figure_count": vector_figure_count,
            "image_source_type": "scanned_pdf" if scanned_indices else "pdf_text_layer",
        }

        logger.info(
            "PDFParser: %s -> %d pages (%d scanned, %d text), "
            "embedded_images=%d, content_len=%d",
            self.file_name,
            page_count,
            len(scanned_indices),
            page_count - len(scanned_indices),
            embedded_count,
            len(content_text),
        )
        return ParsedDocument(content=content_text, images=images, metadata=metadata)

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, cast

from lxml.etree import (  # type: ignore[import-untyped]  # lxml 无 stub（lxml-stubs 未安装），仅用于构造 XPath 对象
    XPath,
)
from playwright.async_api import Page, async_playwright
from trafilatura import extract, utils, xpaths

from aion_knowledge.common.config import settings
from aion_knowledge.pipeline.parser.base import BaseParser, ParsedDocument
from aion_knowledge.pipeline.parser.chain import PipelineParser
from aion_knowledge.pipeline.parser.markdown import MarkdownParser
from aion_knowledge.pipeline.parser.utils import endecode

logger = logging.getLogger(__name__)

_GOTO_TIMEOUT_MS = 30_000
_NETWORK_IDLE_TIMEOUT_MS = 10_000
_SPA_WAIT_TIMEOUT_MS = 15_000
# 在将 SPA 外壳视为"已渲染"之前的最小可见字符数。
_SPA_MIN_TEXT_LEN = 80
# 当 trafilatura 失败时 Playwright 文本回退的最小可见字符数。
_MIN_FALLBACK_TEXT_LEN = 50

# 对 trafilatura 内部进行猴子补丁以更好地支持微信公众号文章，
# 其图片位于 `mmbiz.qpic.cn` 上，没有标准文件扩展名，
# 其主要内容位于 `#js_content` / `.rich_media_content` 内。
# Trafilatura 的 `utils.IMAGE_EXTENSION` 和 `xpaths.BODY_XPATH` 是内部 API，
# 因此我们对补丁进行保护，如果它们在未来的版本中被重命名/移除则静默跳过。
try:
    _WECHAT_IMAGE_EXTENSION = re.compile(
        r"[^\s]+\.(avif|bmp|gif|hei[cf]|jpe?g|png|webp)(\b|$)|"  # 标准扩展名
        r"mmbiz\.qpic\.cn/[^\s]*wx_fmt=(jpeg|jpg|png|gif|webp)"  # 微信查询格式
    )
    utils.IMAGE_EXTENSION = _WECHAT_IMAGE_EXTENSION

    _WECHAT_BODY_XPATH = XPath(
        '(.//*[@id="js_content" or contains(@class, "rich_media_content")])[1]'
    )
    _wechat_xpath_str = str(_WECHAT_BODY_XPATH)
    if not any(str(x) == _wechat_xpath_str for x in xpaths.BODY_XPATH):
        xpaths.BODY_XPATH.insert(0, _WECHAT_BODY_XPATH)
except (AttributeError, ImportError) as e:
    logger.warning(
        "Failed to patch trafilatura internals for WeChat support: %s", e
    )


@dataclass(frozen=True)
class _ScrapeResult:
    html: str
    visible_text: str
    page_title: str


def extract_markdown_from_html(html: str) -> Optional[str]:
    """在 HTML 上运行 trafilatura；返回 Markdown，如果未提取到任何内容则返回 None。"""
    if not html or not html.strip():
        return None
    md_text = extract(
        html,
        output_format="markdown",
        with_metadata=True,
        include_images=True,
        include_tables=True,
        include_links=True,
    )
    if not md_text or not md_text.strip():
        return None
    return md_text


def build_visible_text_fallback(visible_text: str, page_title: str = "") -> Optional[str]:
    """当 trafilatura 未找到文章主体时，从 Playwright 可见文本构建 Markdown。"""
    text = (visible_text or "").strip()
    if len(text) < _MIN_FALLBACK_TEXT_LEN:
        return None
    title = (page_title or "").strip()
    if title and not text.startswith(title):
        return f"# {title}\n\n{text}"
    return text


async def wait_for_rendered_content(page: Page) -> None:
    """等待 SPA/JS 页面超出初始 HTML 外壳。"""
    try:
        await page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS)
        logger.info("Network idle after navigation")
    except Exception:
        logger.info("Network idle wait timed out, continuing")

    try:
        await page.wait_for_function(
            """(minLen) => {
                const root = document.querySelector('#app')
                    || document.querySelector('main')
                    || document.body;
                return ((root?.innerText || '').trim().length >= minLen);
            }""",
            arg=_SPA_MIN_TEXT_LEN,
            timeout=_SPA_WAIT_TIMEOUT_MS,
        )
        logger.info("SPA/root visible text reached minimum length")
    except Exception:
        logger.info("SPA text wait timed out, using current DOM")


async def read_visible_text(page: Page) -> str:
    """优先使用 #app/main 的 innerText，然后回退到 body。"""
    return cast(str, await page.evaluate(  # playwright evaluate 返回 Any，脚本恒定返回字符串
        """() => {
            const root = document.querySelector('#app')
                || document.querySelector('main')
                || document.querySelector('[role="main"]')
                || document.body;
            return (root?.innerText || '').trim();
        }"""
    ))


class StdWebParser(BaseParser):
    """使用 Playwright 和 Trafilatura 的标准网页解析器。

    该解析器使用 Playwright 的 WebKit 浏览器抓取网页，并使用 Trafilatura 库
    提取干净的内容。它支持代理配置并将 HTML 内容转换为 Markdown 格式。
    """

    def __init__(self, title: str, **kwargs: Any) -> None:
        """初始化网页解析器。

        参数：
            title: 用作文件名的网页标题
            **kwargs: 传递给 BaseParser 的额外参数
        """
        self.title = title
        # 从设置中获取代理配置（如果可用）
        self.proxy = settings.external_https_proxy
        super().__init__(file_name=title, **kwargs)
        logger.info(f"Initialized WebParser with title: {title}")

    async def scrape(self, url: str) -> _ScrapeResult:
        """使用 Playwright 抓取网页内容。

        参数：
            url: 要抓取的网页 URL

        返回：
            HTML、可见文本和文档标题；严重失败时返回空字段
        """
        logger.info(f"Starting web page scraping for URL: {url}")
        empty = _ScrapeResult(html="", visible_text="", page_title="")
        try:
            async with async_playwright() as p:
                kwargs: dict[str, Any] = {}
                # 如果代理可用则进行配置
                if self.proxy:
                    kwargs["proxy"] = {"server": self.proxy}
                logger.info("Launching WebKit browser")
                browser = await p.webkit.launch(**kwargs)
                page = await browser.new_page()

                logger.info(f"Navigating to URL: {url}")
                try:
                    await page.goto(
                        url,
                        timeout=_GOTO_TIMEOUT_MS,
                        wait_until="domcontentloaded",
                    )
                    logger.info("Initial page load complete")
                except Exception as e:
                    logger.error(f"Error navigating to URL: {str(e)}")
                    await browser.close()
                    return empty

                await wait_for_rendered_content(page)

                page_title = await page.title()
                visible_text = await read_visible_text(page)
                content = await page.content()
                logger.info(
                    "Retrieved %d bytes HTML, %d chars visible text, title=%r",
                    len(content),
                    len(visible_text),
                    page_title[:80] if page_title else "",
                )

                await browser.close()
                logger.info("Browser closed")

            logger.info("Successfully retrieved HTML content")
            return _ScrapeResult(
                html=content,
                visible_text=visible_text,
                page_title=page_title or "",
            )

        except Exception as e:
            logger.error(f"Failed to scrape web page: {str(e)}")
            return empty

    def parse_into_text(self, content: bytes) -> ParsedDocument:
        """将网页内容解析为 ParsedDocument 对象。

        参数：
            content: 编码为字节的 URL

        返回：
            包含解析后的 Markdown 内容的 ParsedDocument 对象
        """
        url = endecode.decode_bytes(content)

        logger.info(f"Scraping web page: {url}")
        scrape_result = asyncio.run(self.scrape(url))
        if not scrape_result.html and not scrape_result.visible_text:
            logger.error("Failed to scrape web page (no HTML or visible text)")
            return ParsedDocument(content=f"Error parsing web page: {url}")

        md_text = extract_markdown_from_html(scrape_result.html)
        if not md_text:
            md_text = build_visible_text_fallback(
                scrape_result.visible_text,
                scrape_result.page_title,
            )
            if md_text:
                logger.info(
                    "Trafilatura empty; using Playwright visible-text fallback (%d chars)",
                    len(md_text),
                )

        if not md_text:
            logger.error("Failed to parse web page")
            return ParsedDocument(content=f"Error parsing web page: {url}")

        metadata = {}
        title_match = re.search(r"^title:\s*(.+)", md_text, re.MULTILINE)
        if title_match:
            extracted_title = title_match.group(1).strip()
            if extracted_title:
                metadata["title"] = extracted_title
                logger.info(
                    f"Extracted article title from trafilatura: {extracted_title}"
                )
        elif scrape_result.page_title:
            metadata["title"] = scrape_result.page_title.strip()
            logger.info(
                "Using page title from Playwright: %s", metadata["title"]
            )
        else:
            logger.info(
                "No title found in trafilatura output, first 200 chars: %r",
                md_text[:200],
            )
        return ParsedDocument(content=md_text, metadata=metadata)


class WebParser(PipelineParser):
    """使用管道模式的网页解析器。

    该解析器将 StdWebParser（用于网页抓取和 HTML 到 Markdown 转换）
    与 MarkdownParser（用于 Markdown 处理）链接在一起。管道按顺序通过
    两个解析器处理内容。
    """

    # 按顺序执行的解析器类
    _parser_cls = (StdWebParser, MarkdownParser)

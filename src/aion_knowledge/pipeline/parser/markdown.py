"""
Markdown 解析器模块

该模块提供全面的 Markdown 解析功能，包括：
- 表格格式化和标准化
- Base64 图片提取和转换
- 图片路径替换和 URL 生成
- 基于管道的多阶段解析

解析器采用管道方式处理 Markdown 内容，经过多个阶段：
表格格式化 -> 图片处理。
"""

import base64
import logging
import re
from typing import Any, Dict, List, Match, Optional, Tuple

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.pipeline.parser.base import BaseParser, ParsedDocument
from aion_knowledge.pipeline.parser.chain import PipelineParser
from aion_knowledge.pipeline.parser.utils import endecode

# 获取日志记录器
logger = logging.getLogger(__name__)

_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")


class MarkdownTableUtil:
    """Markdown 表格格式化工具类。

    此类通过以下方式标准化 Markdown 表格格式：
    - 规范化列对齐标记（如 :---、:---:、---:）
    - 在管道符（|）周围添加一致的间距
    - 保留缩进层级
    - 处理表头行和数据行

    示例：
        输入：  |姓名|年龄|城市|
                |:---|---:|:---:|
                |张三|25|北京|

        输出： | 姓名 | 年龄 | 城市 |
                | :--- | ---: | :---: |
                | 张三 | 25 | 北京 |
    """

    def __init__(self) -> None:
        # 匹配对齐行的模式（如 |:---|---:|:---:|）
        self.align_pattern = re.compile(
            r"^([\t ]*)\|[\t ]*[:-]+(?:[\t ]*\|[\t ]*[:-]+)*[\t ]*\|[\t ]*$",
            re.MULTILINE,
        )
        # 匹配常规表格行的模式（表头或数据）
        self.line_pattern = re.compile(
            r"^([\t ]*)\|[\t ]*[^|\r\n]*(?:[\t ]*\|[^|\r\n]*)*\|[\t ]*$",
            re.MULTILINE,
        )

    @staticmethod
    def _split_row_cells(row_line: str) -> List[str]:
        """拆分 Markdown 表格行为单元格，保留空单元格。"""
        inner = row_line.strip()
        if not inner.startswith("|"):
            return []
        parts = inner.split("|")
        if parts and parts[0].strip() == "":
            parts = parts[1:]
        if parts and parts[-1].strip() == "":
            parts = parts[:-1]
        return [part.strip() for part in parts]

    @staticmethod
    def _is_table_row(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and "|" in stripped[1:]

    @classmethod
    def _is_separator_row(cls, line: str) -> bool:
        cells = cls._split_row_cells(line)
        return bool(cells) and all(_SEPARATOR_CELL.match(cell) for cell in cells)

    @classmethod
    def _is_empty_row(cls, line: str) -> bool:
        cells = cls._split_row_cells(line)
        return bool(cells) and all(cell == "" for cell in cells)

    @classmethod
    def _separator_row_for(cls, header_line: str) -> str:
        cells = cls._split_row_cells(header_line)
        return "| " + " | ".join("---" for _ in cells) + " |"

    @classmethod
    def _normalize_table_block(cls, block: List[str]) -> List[str]:
        """修复 MarkItDown 风格的表格：去除多余的前缀行，确保 GFM 分隔符。"""
        while block and cls._is_empty_row(block[0]):
            block.pop(0)
        if block and cls._is_separator_row(block[0]):
            block.pop(0)
        # GFM/marked need "| --- |" after the first row. Headerless Word tables
        # only have data rows after we strip the fake empty+separator prefix.
        if len(block) >= 2 and not cls._is_separator_row(block[1]):
            sep = cls._separator_row_for(block[0])
            block = [block[0], sep] + block[1:]
        return block

    def normalize_spurious_table_prefixes(self, content: str) -> str:
        """去除 MarkItDown 表格输出中多余的空行/分隔符前缀行。"""
        lines = content.split("\n")
        out: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not self._is_table_row(line):
                out.append(line)
                i += 1
                continue
            block: List[str] = []
            while i < len(lines) and self._is_table_row(lines[i]):
                block.append(lines[i])
                i += 1
            out.extend(self._normalize_table_block(block))
        return "\n".join(out)

    def format_table(self, content: str) -> str:
        """格式化内容中的所有 Markdown 表格。

        参数：
            content: 包含表格的原始 Markdown 文本

        返回：
            经过标准化表格格式处理的 Markdown 文本
        """

        def process_align(match: Match[str]) -> str:
            """处理对齐行以标准化格式。"""
            columns = self._split_row_cells(match.group(0))

            processed = []
            for col in columns:
                # 保留左对齐标记 (:---)
                left_colon = ":" if col.startswith(":") else ""
                # 保留右对齐标记 (---:)
                right_colon = ":" if col.endswith(":") else ""
                processed.append(left_colon + "---" + right_colon)

            # 保留原始缩进
            prefix = match.group(1)
            return prefix + "| " + " | ".join(processed) + " |"

        def process_line(match: Match[str]) -> str:
            """处理常规表格行以标准化格式。"""
            columns = self._split_row_cells(match.group(0))

            # 保留原始缩进
            prefix = match.group(1)
            return prefix + "| " + " | ".join(columns) + " |"

        formatted_content = content
        # 首先格式化常规行（表头和数据）
        formatted_content = self.line_pattern.sub(process_line, formatted_content)
        # 然后格式化对齐行（必须在之后处理以避免冲突）
        formatted_content = self.align_pattern.sub(process_align, formatted_content)
        return self.normalize_spurious_table_prefixes(formatted_content)

    @staticmethod
    def _self_test() -> None:
        test_content = """
# 测试表格
普通文本---不会被匹配

## 表格1（无前置空格）

| 姓名   | 年龄  | 城市          |
|      :---------- | -------: | :------      |
| 张三 | 25 | 北京 |

## 表格3（前置4个空格+首尾|）
    |   产品   |   价格   |   库存   |
    | :-------------: | ----------- | :-----------: |
    | 手机 | 5999       | 100 |
"""
        util = MarkdownTableUtil()
        format_content = util.format_table(test_content)
        print(format_content)


class MarkdownTableFormatter(BaseParser):
    """用于格式化 Markdown 表格的解析器。

    该解析器标准化文档中所有 Markdown 表格的格式，确保间距和对齐标记的一致性。

    示例：
        >>> formatter = MarkdownTableFormatter()
        >>> content = b"|Name|Age|\\n|---|---|\\n|John|30|"
        >>> doc = formatter.parse_into_text(content)
        >>> print(doc.content)
        | Name | Age |
        | --- | --- |
        | John | 30 |
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.table_helper = MarkdownTableUtil()

    def parse_into_text(self, content: bytes) -> ParsedDocument:
        """解析并格式化 Markdown 表格。

        参数：
            content: 原始 Markdown 内容的字节数据

        返回：
            包含格式化表格内容的 ParsedDocument
        """
        # 将字节解码为字符串，自动检测编码
        text = endecode.decode_bytes(content)
        # 格式化内容中的所有表格
        text = self.table_helper.format_table(text)
        return ParsedDocument(content=text)


class MarkdownImageUtil:
    """Markdown 图片处理工具类。

    此类提供以下功能：
    - 从 Markdown 中提取 base64 编码的图片
    - 从 Markdown 中提取图片路径
    - 用新的 URL 替换图片路径
    - 将 base64 图片转换为二进制格式

    支持的格式：
    - Base64 嵌入式图片：![alt](data:image/png;base64,iVBORw0...)
    - 常规图片链接：![alt](path/to/image.png)
    """

    def __init__(self) -> None:
        # 匹配 base64 嵌入式图片的模式
        # 捕获组：(1) 替代文本，(2) 图片格式，(3) base64 数据
        # 替代文本使用 .*?（非贪婪）以允许字面 ]（如 Windows 路径）
        # MIME 子类型使用 [^;]+ 以处理带连字符的类型如 x-emf
        self.b64_pattern = re.compile(
            r"!\[(.*?)\]\(data:image/([^;]+);base64,([^\)]+)\)"
        )
        # 匹配常规图片语法的模式（替代文本允许 ]）
        self.image_pattern = re.compile(r"!\[(.*?)\]\(([^)]+)\)")
        # 用于替换图片路径的模式
        self.replace_pattern = re.compile(r"!\[(.*?)\]\(([^)]+)\)")

    def extract_image(
        self,
        content: str,
        path_prefix: Optional[str] = None,
        replace: bool = True,
    ) -> Tuple[str, List[str]]:
        """从 Markdown 内容中提取图片路径。

        参数：
            content: 包含图片的 Markdown 文本
            path_prefix: 可选，添加到图片路径的前缀
            replace: 是否替换内容中的图片语法

        返回：
            (处理后的文本, 图片路径列表)

        示例：
            >>> util = MarkdownImageUtil()
            >>> text, images = util.extract_image("![logo](img/logo.png)")
            >>> print(images)
            ['img/logo.png']
        """
        # 用于存储提取的图片路径的列表
        images: List[str] = []

        def repl(match: Match[str]) -> str:
            """每个图片匹配的替换函数。"""
            title = match.group(1)  # 替代文本
            image_path = match.group(2)  # 图片路径

            # 如果指定了前缀则添加
            if path_prefix:
                image_path = f"{path_prefix}/{image_path}"

            images.append(image_path)

            # 如果 replace 为 False，保留原始内容
            if not replace:
                return match.group(0)

            # 用可能带前缀的路径替换图片路径
            return f"![{title}]({image_path})"

        text = self.image_pattern.sub(repl, content)
        logger.debug(f"Extracted {len(images)} images from markdown")
        return text, images

    def extract_base64(
        self,
        content: str,
        path_prefix: Optional[str] = None,
        replace: bool = True,
    ) -> Tuple[str, Dict[str, bytes]]:
        """从 Markdown 中提取并解码 base64 嵌入式图片。

        该方法查找 Markdown 内容中所有 base64 编码的图片，
        将其解码为二进制格式，生成唯一的文件名，
        并可选择将其替换为文件路径引用。

        参数：
            content: 包含 base64 图片的 Markdown 文本
            path_prefix: 可选，生成路径的前缀目录
            replace: 是否将 base64 语法替换为文件路径

        返回：
            (处理后的文本, 路径到字节数据的字典)

        示例：
            >>> util = MarkdownImageUtil()
            >>> text = "![logo](data:image/png;base64,iVBORw0KGg...)"
            >>> new_text, images = util.extract_base64(text, "images")
            >>> print(new_text)
            ![logo](images/uuid.png)
            >>> print(len(images))
            1
        """
        # 将生成的文件路径映射到二进制图片数据的字典
        images: Dict[str, bytes] = {}

        def repl(match: Match[str]) -> str:
            """每个 base64 图片匹配的替换函数。"""
            title = match.group(1)  # 替代文本
            img_ext = match.group(2)  # 图片格式（png、jpg 等）
            img_b64 = match.group(3)  # Base64 编码的数据

            # 将 base64 字符串解码为字节
            image_byte = endecode.encode_image(img_b64, errors="ignore")
            if not image_byte:
                logger.error(f"Failed to decode base64 image skip it: {img_b64}")
                return title  # 解码失败时仅返回替代文本

            # 使用原始扩展名生成唯一文件名
            image_path = f"{uuid7()}.{img_ext}"
            if path_prefix:
                image_path = f"{path_prefix}/{image_path}"
            images[image_path] = image_byte

            # 如果 replace 为 False，保留原始 base64
            if not replace:
                return match.group(0)

            # 用文件路径引用替换 base64 数据
            return f"![{title}]({image_path})"

        text = self.b64_pattern.sub(repl, content)
        logger.debug(f"Extracted {len(images)} base64 images from markdown")
        return text, images

    def replace_path(self, content: str, images: Dict[str, str]) -> str:
        """用新的 URL 替换 Markdown 中的图片路径。

        该方法通常用于在图片存储后用已上传的 URL 替换本地文件路径。

        参数：
            content: 包含图片引用的 Markdown 文本
            images: 旧路径到新 URL 的映射

        返回：
            更新了图片 URL 的 Markdown 文本

        示例：
            >>> util = MarkdownImageUtil()
            >>> content = "![logo](temp/img.png)"
            >>> mapping = {"temp/img.png": "https://cdn.com/img.png"}
            >>> result = util.replace_path(content, mapping)
            >>> print(result)
            ![logo](https://cdn.com/img.png)
        """
        # 追踪实际被替换的路径
        content_replace: set[str] = set()

        def repl(match: Match[str]) -> str:
            """每个图片匹配的替换函数。"""
            title = match.group(1)  # 替代文本
            image_path = match.group(2)  # 当前图片路径

            # 仅当路径存在于映射中时才替换
            if image_path not in images:
                return match.group(0)  # 保留原始内容

            content_replace.add(image_path)
            # 从映射中获取新 URL
            image_path = images[image_path]
            return f"![{title}]({image_path})" if image_path else title

        text = self.replace_pattern.sub(repl, content)
        logger.debug(f"Replaced {len(content_replace)} images in markdown")
        return text

    @staticmethod
    def _self_test() -> None:
        your_content = "test![](data:image/png;base64,iVBORw0KGgoAAAA)test"
        image_handle = MarkdownImageUtil()
        text, images = image_handle.extract_base64(your_content)
        print(text)

        for image_url, image_byte in images.items():
            with open(image_url, "wb") as f:
                f.write(image_byte)


class MarkdownImageBase64(BaseParser):
    """用于从 Markdown 中提取 base64 图片的解析器。

    提取 base64 编码的图片，替换为路径引用，
    并将原始图片数据放在 ParsedDocument.images 中，
    原始图片数据由 ``executor.run_parser`` 中的 ``_upload_images`` 上传存储。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.image_helper = MarkdownImageUtil()

    def parse_into_text(self, content: bytes) -> ParsedDocument:
        text = endecode.decode_bytes(content)
        text, img_b64 = self.image_helper.extract_base64(text, path_prefix="images")

        images: Dict[str, str] = {}
        for ipath, raw_bytes in img_b64.items():
            images[ipath] = base64.b64encode(raw_bytes).decode()

        logger.debug("Extracted %d base64 images from markdown", len(images))
        return ParsedDocument(content=text, images=images)


class MarkdownParser(PipelineParser):
    """使用管道方式的完整 Markdown 解析器。

    该解析器通过多个阶段处理 Markdown 内容：
    1. MarkdownTableFormatter：标准化表格格式
    2. MarkdownImageBase64：提取并上传 base64 图片

    管道确保内容按顺序流经每个解析器，
    每个阶段的输出成为下一阶段的输入。
    """

    _parser_cls = (MarkdownTableFormatter, MarkdownImageBase64)

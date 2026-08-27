"""Chunker —— 统一分块入口。

根据文档结构自动适配分块策略，依次尝试合适的分块器，
每步进行质量检查，不合格则回退到下一级。

=== 策略选择规则 ===

策略选择由 ``_select_splitters()`` 完成，核心逻辑：

1. ``strategy="auto"``（默认）：
   a. 文档画像分析 → 判断是否存在主导标题层级
      （条件：有标题、标题总数 >= 3、标题密度 >= 0.5%）
   b. 如果有主导标题 → ``[HeadingSplitter, MarkerSplitter, RecursiveSplitter]``
   c. 否则，如果结构标记总数 >= 5 → ``[MarkerSplitter, RecursiveSplitter]``
   d. 否则 → ``[RecursiveSplitter]``（纯安全网）

2. ``strategy="heading"`` → ``[HeadingSplitter, MarkerSplitter, RecursiveSplitter]``
3. ``strategy="heuristic"`` → ``[MarkerSplitter, RecursiveSplitter]``
4. ``strategy="recursive"`` → ``[RecursiveSplitter]``
5. ``strategy="no_split"`` → 整篇作为一个 chunk 直接返回，不走分块器链

=== 质量检查规则 ===

``_check_quality()`` 验证分块结果是否合理，任何一项不通过则回退：

1. **未切开检查**：大文档（> 2x chunk_size）只产生一个 chunk → 失败
2. **微小 chunk 检查**：尾部微小 chunk（< 5 tokens）占比超过 25% 且数量 > 2 → 失败
3. **过小检查**：最大 chunk < 目标 25% 且文档本身 > 目标 → 分块器过于保守 → 失败
4. **过大检查**：存在 chunk > 2x 目标大小 → 分块器漏切 → 失败

所有分块器都不合格时，最终用 RecursiveSplitter 作为最终保险（永不失败）。
"""

import logging
from typing import List

from aion_knowledge.pipeline.chunker.base import ChunkConfig, ChunkResult
from aion_knowledge.pipeline.chunker.document_analysis import DocumentAnalyzer
from aion_knowledge.pipeline.chunker.heading_splitter import HeadingSplitter
from aion_knowledge.pipeline.chunker.marker_splitter import MarkerSplitter
from aion_knowledge.pipeline.chunker.recursive_splitter import RecursiveSplitter
from aion_knowledge.pipeline.chunker.text_utils import count_tokens

logger = logging.getLogger(__name__)

# 分块器链成员类型（供 _select_splitters 返回注解使用）
_Splitter = HeadingSplitter | MarkerSplitter | RecursiveSplitter

# 策略选择阈值
HEADING_DENSITY_THRESHOLD = 0.005
MIN_HEADING_COUNT = 3
STRUCTURAL_MARKER_THRESHOLD = 5

# 质量检查阈值
_MIN_CHUNK_TOKENS = 5
_TINY_CHUNK_RATIO = 0.25
_MIN_SIZE_RATIO = 0.25
_LARGE_DOC_MULTIPLIER = 2
_MAX_SIZE_MULTIPLIER = 2


class Chunker:
    """统一分块入口。

    根据 ``ChunkConfig.strategy`` 选择分块器链，依次尝试并做质量检查，
    全部失败时以 ``RecursiveSplitter`` 兜底。

    用法::

        config = ChunkConfig(chunk_size=512, strategy="auto")
        chunker = Chunker(config)
        results = chunker.split(document_text)

    这是 chunker 模块对外暴露的唯一切入点（见 ``__init__.py`` 的 ``__all__``）。
    """

    def __init__(self, config: ChunkConfig):
        self.config = config
        self._analyzer = DocumentAnalyzer()

    def split(self, text: str) -> List[ChunkResult]:
        if not text:
            return []

        if self.config.strategy == "no_split":
            return [ChunkResult(
                content=text,
                token_count=count_tokens(text),
                chunk_id="chunk_0",
                seq_num=0,
            )]

        # 选择分块器链
        splitter_chain = self._select_splitters(text)

        # 依次尝试，质量检查通过则返回
        for splitter in splitter_chain:
            chunks = splitter.split(text)
            if self._check_quality(chunks):
                logger.info(
                    "%s passed quality check (%d chunks)",
                    splitter.__class__.__name__, len(chunks),
                )
                return chunks
            logger.warning(
                "%s failed quality check, trying next", splitter.__class__.__name__,
            )

        # 所有分块器均不合格 → 递归分块作为最终保险
        logger.error("All splitters failed quality check, using last resort")
        return RecursiveSplitter(self.config).split(text)

    def _select_splitters(self, text: str) -> List[_Splitter]:
        """根据文档特征选择合适的分块器列表（按优先级排序）。

        根据 ``self.config.strategy`` 和文档结构特征，返回一个有序的分块器
        实例列表。``split()`` 会依次调用列表中的分块器并对结果进行质量检查，
        通过则立即返回，否则尝试下一个。

        策略树（详细规则见模块 docstring）:

        - auto: 文档画像 → heading / marker / recursive 三选一
        - heading: [HeadingSplitter, MarkerSplitter, RecursiveSplitter]
        - heuristic: [MarkerSplitter, RecursiveSplitter]
        - recursive: [RecursiveSplitter]
        """
        strategy = self.config.strategy

        if strategy == "heading":
            return [
                HeadingSplitter(self.config),
                MarkerSplitter(self.config),
                RecursiveSplitter(self.config),
            ]
        elif strategy == "heuristic":
            return [
                MarkerSplitter(self.config),
                RecursiveSplitter(self.config),
            ]
        elif strategy == "recursive":
            return [RecursiveSplitter(self.config)]

        # auto — 基于文档画像选择
        features = self._analyzer.analyze(text)

        dominant = features.dominant_heading_level or 0
        has_headings = (
            dominant > 0
            and features.md_heading_total >= MIN_HEADING_COUNT
            and (features.heading_density or 0) >= HEADING_DENSITY_THRESHOLD
        )

        if has_headings:
            return [
                HeadingSplitter(self.config),
                MarkerSplitter(self.config),
                RecursiveSplitter(self.config),
            ]

        if features.structural_marker_total >= STRUCTURAL_MARKER_THRESHOLD:
            return [
                MarkerSplitter(self.config),
                RecursiveSplitter(self.config),
            ]

        return [RecursiveSplitter(self.config)]

    def _check_quality(self, chunks: List[ChunkResult]) -> bool:
        """检查分块结果质量，决定是否接受当前分块器输出。

        四项检查（全通过返回 True）:

        1. **未切开检查**：
           大文档（总 token > chunk_size * 2）只有 1 个 chunk → 分块器未生效 → 失败
        2. **微小 chunk 检查**：
           尾部微小 chunk（< 5 tokens）占比超 25% 且数量 > 2 → 碎片过多 → 失败
        3. **过小检查**：
           最大 chunk < chunk_size * 25% 但文档总 token 超过目标 → 分块器过于保守 → 失败
        4. **过大检查**：
           存在 chunk > chunk_size * 2 → 分块器漏切合并 → 失败
        """
        if not chunks:
            return False

        chunk_token_counts = [c.token_count or count_tokens(c.content) for c in chunks]
        max_tokens = max(chunk_token_counts)
        total_tokens = sum(chunk_token_counts)

        # 大文档只有单 chunk → 没切开
        if len(chunks) == 1 and total_tokens > self.config.chunk_size * _LARGE_DOC_MULTIPLIER:
            return False

        # 过多微小 chunk
        tiny_count = 0
        for i, t in enumerate(chunk_token_counts):
            if i < len(chunks) - 1 and t < _MIN_CHUNK_TOKENS:
                tiny_count += 1
        if tiny_count > len(chunks) * _TINY_CHUNK_RATIO and tiny_count > 2:
            return False

        # 所有 chunk 远小于目标
        if (
            max_tokens < self.config.chunk_size * _MIN_SIZE_RATIO
            and total_tokens > self.config.chunk_size
        ):
            return False

        # 有 chunk 超过 2x 目标
        if max_tokens > self.config.chunk_size * _MAX_SIZE_MULTIPLIER:
            return False

        return True

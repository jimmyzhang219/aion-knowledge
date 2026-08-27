"""模型注册表 — 通过模型名称查询上下文窗口等信息。

分层设计：
  - 数据层：conf/model_registry.json（YAML 替代品，git 追踪）
  - 服务层：ModelRegistry（加载 + 查找，延迟初始化单例）
  - 消费者：只需 ``from aion_knowledge.common.model_registry import registry``

用法::

    info = registry.get("gpt-4o")
    info.context_window  # → 128000
    registry.get("qwen3.7-max-20250601")  # 未精确注册，命中 "qwen3*" 模式条目
    registry.context_window("unknown-model")  # → 8192（兜底）

"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── 默认值（registry 中找不到时的保守兜底） ────────────────────────
_FALLBACK_CONTEXT_WINDOW = 8192
_FALLBACK_MAX_OUTPUT = 4096
_FALLBACK_CHARS_PER_TOKEN: float = 3.0  # tiktoken 不可用时回退估算比率


# ── 数据模型 ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelInfo:
    """单条模型信息。"""
    provider: str = "unknown"
    context_window: int = _FALLBACK_CONTEXT_WINDOW
    max_output: int = _FALLBACK_MAX_OUTPUT


# ── 服务类 ───────────────────────────────────────────────────────

class ModelRegistry:
    """模型注册表服务。

    1. 加载 ``conf/model_registry.json``（唯一数据源，可扩展）；
    2. 文件缺失时抛 ``FileNotFoundError``（无内置兜底数据）；
    3. 通过 ``registry.get(name)`` 查询，未注册模型返回 ModelInfo() 兜底。
    """

    _DEFAULT_PATHS = [
        "conf/model_registry.json",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "conf", "model_registry.json"),
    ]

    def __init__(self, registry_path: str | None = None):
        self._models: dict[str, ModelInfo] = {}
        self._patterns: list[tuple[str, ModelInfo]] = []  # (模式, 信息)，按模式长度降序
        self._load(registry_path)
        logger.info(
            "ModelRegistry loaded %d models (%d patterns)", len(self._models), len(self._patterns),
        )

    # ── 公开查询 ──

    def get(self, name: str) -> ModelInfo:
        """查询模型信息。命中顺序：精确条目 → 星号通配符模式（多命中取最长）→ 保守兜底。

        未注册的模型返回保守兜底（context_window=8192）。
        """
        if name in self._models:
            return self._models[name]
        # 查询名不应含 *（查询侧通配符不在设计内）；fnmatch 字面语义下 "qwen3*" 会自命中模式条目
        for pattern, info in self._patterns:
            if fnmatch.fnmatch(name, pattern):
                return info
        logger.warning(
            "模型 '%s' 未在注册表中找到，使用默认值 context_window=%s。"
            "可添加至 conf/model_registry.json", name, _FALLBACK_CONTEXT_WINDOW,
        )
        return ModelInfo()

    def context_window(self, name: str) -> int:
        """快捷方式：直接返回指定模型的上下文窗口大小。"""
        return self.get(name).context_window

    def max_output(self, name: str) -> int:
        """快捷方式：直接返回指定模型的最大输出 token 数。"""
        return self.get(name).max_output

    def all_models(self) -> dict[str, ModelInfo]:
        """返回所有已注册的模型信息（只读副本）。"""
        return dict(self._models)

    # ── 内部加载 ──

    def _load(self, registry_path: str | None) -> None:
        raw: dict[str, Any] | None = None

        if registry_path:
            raw = self._load_json(registry_path)

        if raw is None:
            for path in self._DEFAULT_PATHS:
                raw = self._load_json(path)
                if raw is not None:
                    break

        if raw is None:
            raise FileNotFoundError(
                "模型注册表 JSON 未找到。conf/model_registry.json 是唯一数据源，必须存在"
                f"（尝试路径：{[registry_path] if registry_path else self._DEFAULT_PATHS}）。"
            )

        self._models = {}
        self._patterns = []
        for name, info in raw.items():
            if name.startswith("//"):
                continue  # 跳过 JSON 注释键
            model_info = ModelInfo(
                provider=info.get("provider", "unknown"),
                context_window=info.get("context_window", _FALLBACK_CONTEXT_WINDOW),
                max_output=info.get("max_output", _FALLBACK_MAX_OUTPUT),
            )
            if "*" in name:
                self._patterns.append((name, model_info))
            else:
                self._models[name] = model_info
        # 模式按长度降序，保证首个命中即最长模式
        self._patterns.sort(key=lambda item: len(item[0]), reverse=True)

    @staticmethod
    def _load_json(path: str) -> dict[str, Any] | None:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("模型注册表文件 %s 格式错误（期望 dict），忽略", path)
                return None
            return data
        except json.JSONDecodeError as e:
            logger.warning("模型注册表文件 %s JSON 解析失败：%s", path, e)
            return None


# ── 工具函数（所有截断点统一调用） ─────────────────────────────

# 缓存 tiktoken 编码器（进程内只初始化一次）
_TRUNCATE_ENCODING_INSTANCE = None


def _get_truncate_encoding() -> Any:
    global _TRUNCATE_ENCODING_INSTANCE
    if _TRUNCATE_ENCODING_INSTANCE is None:
        import tiktoken
        _TRUNCATE_ENCODING_INSTANCE = tiktoken.get_encoding("cl100k_base")
    return _TRUNCATE_ENCODING_INSTANCE


def truncate_by_tokens(text: str | None, max_tokens: int) -> str:
    """按 token 精确截断文本。

    使用 tiktoken cl100k_base 编码，回退到字符级估算。
    """
    if not text or max_tokens <= 0:
        return ""
    try:
        enc = _get_truncate_encoding()
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])  # type: ignore[no-any-return]
    except ImportError:
        return text[:int(max_tokens * _FALLBACK_CHARS_PER_TOKEN)]


def sample_head_middle_tail(
    text: str | None,
    max_tokens: int,
    head_ratio: float = 0.6,
    middle_ratio: float = 0.2,
) -> str:
    """三段采样 —— 提取文本的头/中/尾，用省略标记拼接。

    与 ``truncate_by_tokens()`` 不同（从头截断），此函数保留文本的
    头部、中部和尾部三个片段，适合**整篇文档级**的理解/摘要场景。

    当文本 token 数 <= max_tokens 时原样返回，不做采样。

    典型使用场景：

    1. **RAPTOR cluster 摘要**：代替 ``t[:len_per_chunk]`` 纯头截取，
       对 cluster 内每段文本做三段采样，保留更多语义；
    2. **文档级整体摘要**：多 chunk 拼接后用三段采样代表整篇文档；
    3. **通用工具函数**：任何需要从大文本中提取代表性内容的场景。

    Args:
        text: 输入文本。
        max_tokens: 输出允许的最大 token 数。
        head_ratio: 头部占比（默认 0.6 = 60%）。
        middle_ratio: 中部占比（默认 0.2 = 20%），尾部比例自动为
            ``1 - head_ratio - middle_ratio``。

    Returns:
        三段采样后的文本，用 ``\\n\\n[...content omitted...]\\n\\n`` 拼接；
        若输入为空或 max_tokens <= 0 返回 ``""``。
    """
    if not text or max_tokens <= 0:
        return ""

    try:
        enc = _get_truncate_encoding()
        tokens = enc.encode(text)
    except ImportError:
        # tiktoken 不可用时回退到截断
        return truncate_by_tokens(text, max_tokens)

    if len(tokens) <= max_tokens:
        return text

    # 确定分段数，预扣除省略标记的 token 开销
    tail_ratio = 1.0 - head_ratio - middle_ratio
    omitted_marker = "\n\n[...content omitted...]\n\n"
    marker_tokens = len(enc.encode(omitted_marker))
    has_middle = middle_ratio > 0
    has_tail = tail_ratio > 0
    if has_middle and has_tail:
        num_separators = 2
    elif has_middle or has_tail:
        num_separators = 1
    else:
        num_separators = 0

    # 有效内容预算 = max_tokens - 省略标记开销
    content_budget = max_tokens - marker_tokens * num_separators

    if content_budget < 1:
        # 预算不足以容纳省略标记时，直接截断头部返回
        return truncate_by_tokens(text, max_tokens)

    # 按比例分配内容预算
    head_budget = max(1, int(content_budget * head_ratio))
    middle_budget = max(1, int(content_budget * middle_ratio)) if middle_ratio > 0 else 0
    tail_budget = content_budget - head_budget - middle_budget

    if tail_budget < 1 or not has_tail:
        # 尾部预算不足时匀给 head
        # 或尾部比例=0 但整数舍入产生余额时也匀给 head，避免生成非预期的 tail 段
        head_budget = content_budget - middle_budget
        tail_budget = 0

    head = enc.decode(tokens[:head_budget])
    middle_start = (
        max(head_budget, (len(tokens) - tail_budget) // 2)
        if tail_budget > 0
        else head_budget
    )
    middle_slice = tokens[middle_start:middle_start + middle_budget]
    middle = enc.decode(middle_slice) if middle_budget > 0 else ""

    if tail_budget > 0:
        tail = enc.decode(tokens[-tail_budget:])
        if middle_budget < 1:
            # middle=0 时只用一个省略标记连接 head 和 tail
            return f"{head}\n\n[...content omitted...]\n\n{tail}"
        return f"{head}\n\n[...content omitted...]\n\n{middle}\n\n[...content omitted...]\n\n{tail}"
    elif middle_budget > 0:
        return f"{head}\n\n[...content omitted...]\n\n{middle}"
    else:
        return head  # type: ignore[no-any-return]


def get_model_max_input_tokens(
    model_name: str,
    ratio: float = 0.1,
) -> int:
    """返回指定模型 context_window 的给定比例，作为单次 prompt 输入上限。

    Args:
        model_name: 模型名称（如 gpt-4o），传给 registry 查询。
        ratio: 占用 context_window 的比例（默认 0.1 = 10%）。

    Returns:
        max_tokens，保底 512。
    """
    reg = get_registry()
    return max(512, int(reg.context_window(model_name) * ratio))


def count_tokens(text: str, language: str = "mixed") -> int:
    """计算文本的 token 数（cl100k_base 编码）。

    与 OpenAI text-embedding-3 系列 / GPT-4 使用相同的分词方式，
    确保截断边界与模型的服务端截断行为一致。

    当 tiktoken 不可用时（离线环境/受限依赖），回退到语言感知的
    字符比率估算：
    - 中文：1 token ≈ 1.7 字符
    - 英文：1 token ≈ 4.0 字符
    - 德文：1 token ≈ 4.5 字符
    - 混合（默认）：1 token ≈ 3.0 字符

    Args:
        text: 待计数的文本。
        language: tiktoken 不可用时的回退语言模式（``"zh"`` / ``"en"`` / ``"de"`` / ``"mixed"``）。

    Returns:
        token 数量（>= 0）。
    """
    if not text:
        return 0
    try:
        enc = _get_truncate_encoding()
        return len(enc.encode(text))
    except ImportError:
        ratios = {"zh": 1.7, "en": 4.0, "de": 4.5, "mixed": 3.0}
        ratio = ratios.get(language, 3.0)
        return max(1, int(len(text) / ratio))


def count_tokens_batch(texts: list[str], language: str = "mixed") -> list[int]:
    """批量计算 token 数。

    Args:
        texts: 待计数的文本列表。
        language: tiktoken 不可用时的回退语言模式。

    Returns:
        token 数量列表。
    """
    return [count_tokens(t, language) for t in texts]


# ── 模块级单例 ──────────────────────────────────────────────────

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """获取 ModelRegistry 单例（延迟初始化）。"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


# 方便导入
registry = get_registry()

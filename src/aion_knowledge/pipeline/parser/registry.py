"""解析器引擎注册表 — 按文件类型和引擎名路由到对应解析器类。

注意：parser 类的导入是惰性的（在 _build_default_registry 内部），
因为解析器模块在后续任务中才创建。
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from aion_knowledge.pipeline.parser.base import BaseParser

logger = logging.getLogger(__name__)

BUILTIN_ENGINE = "builtin"


class ParserEngineRegistry:
    """解析器引擎注册表。

    每个引擎映射文件扩展名到解析器类。
    当请求的引擎不支持某文件类型时，自动回退到 builtin 引擎。
    """

    def __init__(self) -> None:
        self._engines: Dict[str, Dict[str, Type[BaseParser]]] = {}
        self._descriptions: Dict[str, str] = {}
        self._check_available: Dict[str, Callable[..., Tuple[bool, str]]] = {}
        self._unavailable_hint: Dict[str, str] = {}

    def register(
        self,
        name: str,
        file_types: Dict[str, Type[BaseParser]],
        description: str = "",
        check_available: Optional[Callable[..., Tuple[bool, str]]] = None,
        unavailable_hint: str = "",
    ) -> None:
        self._engines[name] = file_types
        self._descriptions[name] = description
        if check_available is not None:
            self._check_available[name] = check_available
            self._unavailable_hint[name] = unavailable_hint
        logger.info("Registered parser engine '%s' with file types: %s",
                     name, ", ".join(file_types.keys()))

    def get_parser_class(self, engine: str, file_type: str) -> Type[BaseParser]:
        ft = file_type.lower()
        if engine and engine in self._engines:
            cls = self._engines[engine].get(ft)
            if cls:
                return cls
            logger.info("Engine '%s' does not support '%s', falling back to builtin",
                         engine, ft)
        builtin = self._engines.get(BUILTIN_ENGINE, {})
        cls = builtin.get(ft)
        if cls:
            return cls
        raise ValueError(f"Unsupported file type: {file_type}")

    def list_engines(self, overrides: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        result = []
        for name, parsers in self._engines.items():
            available = True
            reason = ""
            check = self._check_available.get(name)
            if check is not None:
                try:
                    available, reason = check(overrides)
                except Exception as e:
                    available = False
                    reason = str(e) or self._unavailable_hint.get(name, "")
            if not available and not reason:
                reason = self._unavailable_hint.get(name, "不可用")
            result.append({
                "name": name,
                "description": self._descriptions.get(name, ""),
                "file_types": sorted(parsers.keys()),
                "available": available,
                "unavailable_reason": reason,
            })
        return result

    def get_engine_names(self) -> List[str]:
        return list(self._engines.keys())

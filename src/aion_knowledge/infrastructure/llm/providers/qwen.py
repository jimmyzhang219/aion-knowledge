"""Qwen 系列模型自定义 LLM 实现

Qwen 模型通过 OpenAI 兼容 API 访问。与通用 ReasoningOpenAI 的差异：

- 支持 ``video_url`` 多模态内容块（qwen 专属格式）
- 默认启用 reasoning_effort
"""

from __future__ import annotations

from typing import Any

from .maas import ChatMaaS


class ChatQwen(ChatMaaS):
    """Qwen 系列模型：支持 video_url 等多模态自定义格式"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("reasoning_effort", "high")
        super().__init__(**kwargs)

    def _get_request_payload(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """获取 API 请求 payload，将 video content block 转为 qwen 格式

        Qwen 使用 ``{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}``
        替代通用的 content block 内联格式。
        """
        payload = super()._get_request_payload(*args, **kwargs)
        for msg in payload.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = self._convert_blocks(content)
        return payload

    def _convert_blocks(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将内部 content blocks 转为 qwen API 接受的格式

        内部格式: ``{"type": "video", "data": "<base64>", "mimeType": "video/mp4"}``
        Qwen 格式: ``{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,<base64>"}}``

        text / image_url 等通用 block 由 langchain 预先处理，此处原样保留。
        """
        out: list[dict[str, Any]] = []
        for block in blocks:
            if block.get("type") == "video":
                data = block.get("data", "")
                mime = block.get("mimeType", "video/mp4")
                out.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:{mime};base64,{data}"},
                    }
                )
            else:
                out.append(block)
        return out

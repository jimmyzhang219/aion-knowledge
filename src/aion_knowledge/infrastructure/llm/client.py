"""LLM 调用薄封装 — 基于 LangChain。"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


class LLMClient:
    """基于 LangChain 的 LLM 调用薄封装。

    对外提供业务需要的异步方法，内部调用 LangChain BaseChatModel。
    """

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        messages: list[Any] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs["response_format"] = {"type": "text"} if response_format is None else response_format
        resp = await self._model.ainvoke(messages, **kwargs)
        return str(resp.content or "")

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_structured(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        import json
        import logging

        logger = logging.getLogger(__name__)

        messages: list[Any] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            structured = self._model.with_structured_output(output_schema, method="json_mode")
            result = await structured.ainvoke(messages)
            if isinstance(result, dict):
                return result
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return cast(dict[str, Any], json.loads(str(result)))
        except Exception:
            resp = await self._model.ainvoke(messages, response_format={"type": "json_object"})
            raw_content = resp.content
            text = str(raw_content) if raw_content is not None else "{}"
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            try:
                return cast(dict[str, Any], json.loads(text))
            except json.JSONDecodeError:
                logger.warning("generate_structured: 无法解析 LLM 输出为 JSON: %s", text[:200])
                return {}

    async def generate_with_images(
        self,
        prompt: str,
        images: list[tuple[bytes, str]],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[Any] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img_bytes, mime in images:
            b64data = base64.b64encode(img_bytes).decode()
            content_parts.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64data}"}}
            )
        messages.append(HumanMessage(content=cast(Any, content_parts)))

        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = await self._model.ainvoke(messages, **kwargs)
        return str(resp.content or "")

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async for chunk in self._model.astream([HumanMessage(content=prompt)]):
            yield str(chunk.content or "")

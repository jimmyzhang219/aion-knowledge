"""VLM 描述图片的提示词模板。

注意：
  DESCRIBE_PROMPT 需使用简洁明确的指令。包含"详细描述"、"人物、场景等元素"
  等太复杂的引导词会导致 VLM（特别是 Qwen 系列 through 阿里云 MaaS）返回空
  内容。"请用中文描述这张图片的内容，要求简洁准确。" 经验证正常工作。
"""

DESCRIBE_PROMPT = "请用中文描述这张图片的内容，要求简洁准确。"

DESCRIBE_PROMPT_WITH_CONTEXT = """以下是一份文档中的一部分图文内容。图片的上下文如下：

上方内容：
{context_above}

下方内容：
{context_below}

请结合上下文，用中文简洁准确地描述这张图片的内容，重点说明与上下文相关的信息。"""

"""三层关键词提取模块。

Tier 1 — LLM 自由生成（开放集）
Tier 2 — 精确子串匹配 KB 预设 tags（封闭集，无 LLM）
Tier 3 — LLM 从预设 tags 中约束选取（封闭集，含 few-shot）
"""

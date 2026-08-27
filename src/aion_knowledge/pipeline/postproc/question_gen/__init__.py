"""逐 chunk 问题生成模块。

基于 chunk 内容调用 LLM 生成用户可能提出的问题，
写入 chunk_vector 表（问题文本 + 向量），用于 QA 匹配检索。
"""

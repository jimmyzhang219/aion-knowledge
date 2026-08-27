"""逐 chunk 摘要生成模块。

对 text 模块产出的每个 chunk 调用 LLM 生成摘要，
同时写入 chunk_text.summary 和 chunk_vector 的摘要向量。
"""

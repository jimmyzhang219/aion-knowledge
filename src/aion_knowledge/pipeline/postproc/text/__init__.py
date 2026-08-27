"""文本 chunk 落库模块。

将 parser 分块结果写入 chunk_text 表，并固定生成 parent 结构。
vector、summarizer 等模块依赖 text 先执行拿到 chunk_uuid。
"""

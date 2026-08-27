"""Wiki 百科页面生成模块。

通过 MAP→REDUCE→PLAN→REFINE 四阶段流程从文档中提取概念/实体，
自动生成 Wiki 百科风格的页面内容，写入 chunk_wiki 表（KB 级页面池：
一页引用多个 chunk，页面间以 [[slug]] wikilink 互链，支持按文档删除）。
"""

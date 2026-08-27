"""全局常量。"""

from aion_knowledge.models.enums import ChunkType

# 文档状态
DOCUMENT_STATUS_PENDING = "pending"
DOCUMENT_STATUS_PROCESSING = "processing"
DOCUMENT_STATUS_FINALIZING = "finalizing"
DOCUMENT_STATUS_COMPLETED = "completed"
DOCUMENT_STATUS_FAILED = "failed"

# 切片类型 —— 请优先使用 ChunkType 枚举（ChunkType.text.value 等）
# 这些常量保留仅用于向后兼容，新代码请直接引用 ChunkType
CHUNK_TYPE_TEXT = ChunkType.text.value
CHUNK_TYPE_PARENT_TEXT = ChunkType.parent.value
CHUNK_TYPE_IMAGE_OCR = ChunkType.image_ocr.value
CHUNK_TYPE_IMAGE_CAPTION = ChunkType.image_caption.value
CHUNK_TYPE_SUMMARY = ChunkType.summary.value
CHUNK_TYPE_ENTITY = ChunkType.entity.value
CHUNK_TYPE_FAQ = ChunkType.faq.value
CHUNK_TYPE_RELATIONSHIP = ChunkType.relationship.value

# 摄入任务状态
INGESTION_STATUS_PENDING = "pending"
INGESTION_STATUS_PROCESSING = "processing"
INGESTION_STATUS_COMPLETED = "completed"
INGESTION_STATUS_FAILED = "failed"

# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".md", ".html", ".htm", ".epub",
    ".xls", ".xlsx", ".csv",
    ".ppt", ".pptx",
    ".py", ".js", ".java", ".go", ".ts", ".cpp", ".sql",
    ".jpg", ".jpeg", ".png", ".gif",
    ".mp3", ".wav", ".aac",
    ".mp4", ".avi", ".mkv",
    ".eml", ".msg",
}

# 搜索常量
DEFAULT_TOP_K = 10
DEFAULT_RRF_K = 60
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3
DEFAULT_RERANK_LAMBDA = 0.7
MAX_QUERY_EXPANSIONS = 5

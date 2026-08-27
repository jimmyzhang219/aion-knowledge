# OCR 模型部署（PaddleOCR，可选）

扫描 PDF 页的 OCR 识别以 PaddleOCR 为主引擎（降级链：PaddleOCR → Tesseract → VLM OCR）。
以下为手动部署方式，可在首次使用前提前准备好模型，避免运行时联网下载。

## 1. 安装 Python 依赖

```bash
pip install -e ".[paddle-ocr]"
```

## 2. 手动下载并放置模型（约 140MB）

模型存放于 `~/.paddlex/official_models/`，服务启动后自动识别，不会重复下载：

```bash
mkdir -p ~/.paddlex/official_models && cd ~/.paddlex/official_models
curl -L -o det.tar https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_det_onnx_infer.tar
curl -L -o rec.tar https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_rec_onnx_infer.tar
tar -xf det.tar && mv PP-OCRv6_medium_det_onnx_infer PP-OCRv6_medium_det_onnx && rm det.tar
tar -xf rec.tar && mv PP-OCRv6_medium_rec_onnx_infer PP-OCRv6_medium_rec_onnx && rm rec.tar
```

最终目录结构：

```
~/.paddlex/official_models/
├── PP-OCRv6_medium_det_onnx/
│   ├── inference.onnx   # 文本检测模型（约 62MB）
│   └── inference.yml
└── PP-OCRv6_medium_rec_onnx/
    ├── inference.onnx   # 文字识别模型（约 76MB）
    └── inference.yml
```

> 说明：PaddleOCR 3.x 默认模型源为 HuggingFace（国内网络可能不稳定）。若不手动放置模型而依赖首次使用自动下载，
> 可在服务环境变量中指定 `PADDLE_PDX_MODEL_SOURCE=bos`（百度 BOS）或 `modelscope`，并可加
> `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` 跳过连通性探测。已放置模型的机器无需这些变量。

## 3. 验证

```bash
PYTHONPATH=src .venv/bin/python -c "
import asyncio
from aion_knowledge.infrastructure.ocr.paddle import PaddleOCREngine

async def main():
    await PaddleOCREngine()._ensure_available()
    print('PaddleOCR 模型就绪')

asyncio.run(main())
"
```

预期输出：`PaddleOCR 模型就绪`（约 5 秒，仅加载模型不联网）。

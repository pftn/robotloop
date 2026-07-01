"""
model_api.py
RobotLoop 模型推理 API
支持通过 model_version 参数切换不同模型版本
"""

import io
import logging
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-api")

# -------------------- 模型版本管理 --------------------
# 载入不同版本的模型，可按需扩展
MODELS = {
    "v1": fasterrcnn_mobilenet_v3_large_fpn(pretrained=True),
    # "v2": your_other_model(...),  # 示例：未来可以添加新模型
}

# 设置当前默认模型版本
DEFAULT_MODEL_VERSION = "v1"

# 确保所有模型处于评估模式
for model in MODELS.values():
    model.eval()

# COCO 类别映射
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# 图片预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

app = FastAPI(title="RobotLoop Model API", version="2.0")


@app.post("/detect")
async def detect(
        file: UploadFile = File(...),
        model_version: str = Query(DEFAULT_MODEL_VERSION, description="模型版本，例如 v1, v2")
):
    """
    接收图片，使用指定版本的模型进行目标检测。
    返回检测到的物体列表。
    """
    # 选择模型版本
    model = MODELS.get(model_version)
    if model is None:
        available = list(MODELS.keys())
        return {"error": f"Unknown model version '{model_version}'. Available: {available}"}, 400

    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            logger.warning("Invalid image file received")
            return {"detections": [], "count": 0}

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            predictions = model(img_tensor)

        results = []
        for box, label, score in zip(
                predictions[0]["boxes"].tolist(),
                predictions[0]["labels"].tolist(),
                predictions[0]["scores"].tolist(),
        ):
            if score < 0.5:
                continue
            name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else "unknown"
            results.append({
                "class": name,
                "confidence": round(score, 2),
                "bbox": {
                    "x": round(box[0], 2),
                    "y": round(box[1], 2),
                    "w": round(box[2] - box[0], 2),
                    "h": round(box[3] - box[1], 2),
                },
            })
        logger.info(f"Model {model_version} detected {len(results)} objects")
        return {"detections": results, "count": len(results), "model_version": model_version}

    except Exception as e:
        logger.error(f"Detection failed with model {model_version}: {e}")
        return {"detections": [], "count": 0, "model_version": model_version}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9002)

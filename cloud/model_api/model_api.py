import io, os, random, logging
import torch, boto3
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("model-api")

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/models")
MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "fasterrcnn_mobilenet_v3_large_fpn.pth")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

if not os.path.exists(MODEL_PATH):
    logger.info("Downloading Faster R-CNN from MinIO...")
    s3.download_file(
        "models", "detection/v1/fasterrcnn_mobilenet_v3_large_fpn.pth", MODEL_PATH
    )

model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
logger.info("Faster R-CNN model loaded")

COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

transform = transforms.Compose([transforms.ToTensor()])

app = FastAPI(title="RobotLoop Model API", version="2.0")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            preds = model(img_tensor)

        results = []
        for box, label, score in zip(
            preds[0]["boxes"].tolist(),
            preds[0]["labels"].tolist(),
            preds[0]["scores"].tolist(),
        ):
            if score < 0.5:
                continue
            name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else "unknown"
            results.append(
                {
                    "class": name,
                    "confidence": round(score, 2),
                    "bbox": {
                        "x": round(box[0], 2),
                        "y": round(box[1], 2),
                        "w": round(box[2] - box[0], 2),
                        "h": round(box[3] - box[1], 2),
                    },
                }
            )
        return {"detections": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return {"detections": [], "count": 0, "error": str(e)}


class AnnotateRequest(BaseModel):
    scene_id: str


@app.post("/annotate")
async def annotate(request: AnnotateRequest):
    scene_id = request.scene_id
    mask_emb = [random.random() for _ in range(256)]
    descriptions = [
        "a robot navigating a narrow corridor with scattered obstacles",
        "outdoor park scene with pedestrians and bicycles",
        "indoor warehouse with shelves and boxes",
        "rainy street with reflections on the ground",
        "foggy morning in a construction site",
        "robot arm picking up a red cube on a table",
    ]
    description = random.choice(descriptions)
    quality = round(random.uniform(0.6, 1.0), 2)
    logger.info(f"Annotated scene {scene_id}: {description}")
    return {
        "scene_id": scene_id,
        "description": description,
        "mask_emb": mask_emb,
        "quality": quality,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}

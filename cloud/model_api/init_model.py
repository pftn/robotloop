import os, boto3, torch, logging
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("model-init")

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
BUCKET = "models"

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

# 创建桶
try:
    s3.head_bucket(Bucket=BUCKET)
except:
    s3.create_bucket(Bucket=BUCKET)

# 1. Faster R-CNN 下载并上传
logger.info("Downloading Faster R-CNN...")
model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
local_path = "/tmp/fasterrcnn_mobilenet_v3_large_fpn.pth"
torch.save(model.state_dict(), local_path)
s3.upload_file(local_path, BUCKET, "detection/v1/fasterrcnn_mobilenet_v3_large_fpn.pth")
logger.info("Faster R-CNN uploaded")

# 2. CLIP 下载并上传
logger.info("Downloading CLIP model...")
clip = SentenceTransformer("clip-ViT-B-32")
local_clip_dir = "/tmp/clip-ViT-B-32"
clip.save(local_clip_dir)

# 递归上传
for root, _, files in os.walk(local_clip_dir):
    for file in files:
        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, "/tmp")
        s3.upload_file(full_path, BUCKET, rel_path)
logger.info("CLIP model uploaded to MinIO")

print("All models initialized successfully.")

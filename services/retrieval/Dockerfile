FROM python:3.10-slim

WORKDIR /app

# 使用 Hugging Face 镜像站，避免网络不可达
ENV HF_ENDPOINT=https://hf-mirror.com

# 安装系统依赖
RUN apt-get update && apt-get install -y curl libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载 MobileCLIP 模型（通过镜像站）
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('clip-ViT-B-32')"

# 复制源码
COPY sync_to_retrieval.py search_api.py ./
FROM rayproject/ray:2.9.0-py310

WORKDIR /app

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

USER ray

COPY requirements.txt .

# ============================================
# 关键修复：
# 1. setuptools >= 82.0.0 移除了 pkg_resources
#    必须锁定 setuptools <= 80.10.2
# 2. click 必须 <= 8.2.1（避免 Sentinel 错误）
# ============================================
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir "setuptools<=80.10.2" && \
    pip install --no-cache-dir "click<=8.2.1" && \
    pip install --no-cache-dir -r requirements.txt

# 验证关键依赖
RUN python -c "from ray.scripts.scripts import main; print('Ray CLI import: OK')" && \
    python -c "import click; print('click version:', click.__version__)" && \
    python -c "import pkg_resources; print('pkg_resources: OK')" && \
    python -c "import enum; print('enum OK')"

COPY ray_pipeline.py .
COPY entrypoint.sh /entrypoint.sh

USER root
RUN chmod +x /entrypoint.sh
USER ray

ENTRYPOINT ["/entrypoint.sh"]
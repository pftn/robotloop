FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generator.py uploader.py ./
COPY start.sh .
COPY sample.jpg .
RUN chmod +x start.sh

# 默认命令（由 docker-compose 覆盖）
CMD ["python", "generator.py"]
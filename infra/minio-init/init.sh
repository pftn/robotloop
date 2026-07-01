#!/bin/sh
echo "等待 MinIO 服务就绪..."
for i in $(seq 1 10); do
  if mc alias set minio http://minio:9000 minioadmin minioadmin 2>/dev/null; then
    echo "MinIO 连接成功"
    mc mb minio/robotloop-data --ignore-existing
    echo "存储桶 robotloop-data 已确保存在"
    exit 0
  fi
  echo "尝试 $i/10..."
  sleep 2
done
echo "MinIO 未能就绪，退出"
exit 1
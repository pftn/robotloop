#!/bin/bash
# SkyLoop Kafka 主题初始化脚本
# 用于创建数据管道所需的所有主题

KAFKA_HOST=${KAFKA_HOST:-kafka}
KAFKA_PORT=${KAFKA_PORT:-9092}
PARTITIONS=${PARTITIONS:-3}
REPLICATION=${REPLICATION:-1}

TOPICS=(
  "file-upload-events"       # 文件上传完成事件，触发后续处理
  "pre-label-tasks"          # 预标注任务分发
  "human-review-tasks"       # 人工审核任务
  "alerts"                   # 系统告警消息
)

echo "正在等待 Kafka 就绪..."
while ! nc -z $KAFKA_HOST $KAFKA_PORT; do
  sleep 1
done
echo "Kafka 已就绪，开始创建主题..."

for topic in "${TOPICS[@]}"; do
  kafka-topics --bootstrap-server $KAFKA_HOST:$KAFKA_PORT \
    --create --if-not-exists \
    --topic "$topic" \
    --partitions $PARTITIONS \
    --replication-factor $REPLICATION
  echo "主题 $topic 创建完成"
done

echo "所有主题创建完毕。"
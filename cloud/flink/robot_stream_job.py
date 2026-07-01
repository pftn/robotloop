"""
robot_stream_job.py
PyFlink 流处理作业
消费 file-upload-events，解析遥测日志，写入 PostgreSQL，并触发预标注
需要 flink-sql-connector-kafka-3.0.2-1.18.jar 在 $FLINK_HOME/lib/
"""

import json
import logging
import os
from datetime import datetime

from pyflink.common import SimpleStringSchema, Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer
from pyflink.datastream.functions import MapFunction

import boto3
import psycopg2
from kafka import KafkaProducer
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

# ---------------------------------------------------------------------
# 环境配置
# ---------------------------------------------------------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "robotloop-data")

PG_HOST = os.getenv("PG_HOST", "postgres")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "robotloop")
PG_PASSWORD = os.getenv("PG_PASSWORD", "robotloop")
PG_DB = os.getenv("PG_DB", "robotloop")


class EventProcessor(MapFunction):
    """
    处理文件上传事件：遥测日志写入 DB，媒体文件发送预标注任务。
    open/close 管理外部连接，避免序列化问题。
    """

    def open(self, configuration):
        # 初始化 S3 客户端
        self.s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )
        # 初始化 PostgreSQL 连接
        self.pg_conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DB,
        )
        # 初始化 Kafka Producer（发送预标注任务）
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            max_block_ms=5000,
            acks=1,
            retries=3,
        )
        # 初始化 Cassandra 连接
        self.cass_cluster = Cluster(['cassandra'])
        self.cass_session = self.cass_cluster.connect('robotloop')
        self.cass_insert_stmt = self.cass_session.prepare(
            "INSERT INTO robot_sensor_logs (device_id, ts, state_json, event) VALUES (?, ?, ?, ?)"
        )
        logging.info("EventProcessor 连接初始化完成")

    def map(self, event_str: str) -> str:
        try:
            event = json.loads(event_str)
            key = event.get("key", "")
            bucket = event.get("bucket", S3_BUCKET)
            logging.info(f"Processing event: {key}")

            # 1. 处理 ROS 数据消息（写入 Cassandra）
            if key.endswith("ros_data.json"):
                obj = self.s3.get_object(Bucket=bucket, Key=key)
                ros_msgs = json.loads(obj["Body"].read())
                for msg in ros_msgs:
                    state = {}
                    if msg["msg_type"] == "sensor_msgs/Imu":
                        state["imu"] = {
                            "accel": msg["data"]["linear_acceleration"],
                            "gyro": msg["data"]["angular_velocity"]
                        }
                    elif msg["msg_type"] == "sensor_msgs/NavSatFix":
                        state["gps"] = {
                            "lat": msg["data"]["latitude"],
                            "lng": msg["data"]["longitude"],
                            "alt": msg["data"]["altitude"]
                        }
                    elif msg["msg_type"] == "sensor_msgs/Image":
                        state["camera"] = msg["data"]
                    else:
                        continue

                    self._write_cassandra_record(
                        device_id="robot-001",
                        ts=datetime.utcfromtimestamp(msg["timestamp"]),
                        state_dict=state,
                        event="ros_msg"
                    )
                logging.info(f"Processed ROS data: {len(ros_msgs)} messages")

            # 2. 处理机器人状态日志（写入 Cassandra）
            elif key.endswith("robot_state.json"):
                obj = self.s3.get_object(Bucket=bucket, Key=key)
                data = json.loads(obj["Body"].read())
                self._insert_robot_logs(data)
                logging.info(f"Inserted {len(data)} robot logs")

            # 3. 媒体文件（元数据写入 PG，触发预标注）
            elif key.endswith((".jpg", ".pcd", ".png", ".jpeg", ".mp4", ".avi")):
                self._insert_media_metadata(key, bucket)
                self._trigger_prelabel(key, bucket)
                logging.info(f"Sent prelabel task for {key}")

            else:
                logging.info(f"Ignored unknown file type: {key}")

        except Exception as e:
            logging.error(f"Failed to process event: {str(event_str)[:200]}... | Error: {e}", exc_info=True)

        return ""

    def _write_cassandra_record(self, device_id: str, ts: datetime, state_dict: dict, event: str):
        """单条写入 Cassandra 传感器日志表"""
        try:
            self.cass_session.execute_async(
                self.cass_insert_stmt,
                (device_id, ts, json.dumps(state_dict), event)
            )
        except Exception as e:
            logging.warning(f"Cassandra write failed: {e}")

    def _insert_robot_logs(self, data: list):
        """批量写入机器人状态日志"""
        for record in data:
            self._write_cassandra_record(
                device_id=record.get("device_id", "unknown"),
                ts=datetime.utcfromtimestamp(record["timestamp"]),
                state_dict=record.get("state_json", {}),
                event=record.get("event", "normal")
            )

    def _insert_media_metadata(self, key: str, bucket: str):
        """写入 media_files 表"""
        cur = self.pg_conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO media_files (key, bucket, uploaded_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT DO NOTHING
                """,
                (key, bucket),
            )
            self.pg_conn.commit()
        except Exception as e:
            self.pg_conn.rollback()
            raise e
        finally:
            cur.close()

    def _trigger_prelabel(self, media_key: str, bucket: str):
        """发送预标注任务到 Kafka 主题 pre-label-tasks"""
        task = {
            "media_key": media_key,
            "bucket": bucket,
            "timestamp": datetime.utcnow().isoformat(),
        }
        future = self.kafka_producer.send("pre-label-tasks", value=task)
        future.add_callback(lambda x: logging.debug(f"Task sent: {media_key}"))
        future.add_errback(lambda exc: logging.error(f"Failed to send task for {media_key}: {exc}"))

    def close(self):
        """释放资源"""
        try:
            if hasattr(self, 'kafka_producer'):
                self.kafka_producer.flush()
                self.kafka_producer.close(timeout=10)
        except Exception as e:
            logging.warning(f"Kafka producer close error: {e}")
        try:
            if hasattr(self, 'pg_conn') and self.pg_conn and not self.pg_conn.closed:
                self.pg_conn.close()
        except Exception as e:
            logging.warning(f"PostgreSQL connection close error: {e}")
        try:
            if hasattr(self, 'cass_cluster'):
                self.cass_cluster.shutdown()
        except Exception as e:
            logging.warning(f"Cassandra connection close error: {e}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.info("Starting RobotLoop Flink Job...")

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    kafka_consumer = FlinkKafkaConsumer(
        topics="file-upload-events",
        deserialization_schema=SimpleStringSchema(),
        properties={
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": "flink-processor-v5",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": "true",
            "max.poll.records": "500",
        },
    )

    ds = env.add_source(kafka_consumer)
    ds.map(EventProcessor(), output_type=Types.STRING())

    env.execute("RobotLoop Robot Data Processing Job (Production)")


if __name__ == "__main__":
    main()

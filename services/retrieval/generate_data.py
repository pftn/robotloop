import json, random, uuid
from datetime import datetime, timedelta

scenarios = []
weather_opts = ["sunny", "rain", "fog", "snow"]
labels = ["pedestrian", "vehicle", "traffic_light", "lane_change", "sudden_brake"]
for i in range(1000):
    ts = datetime(2026, 7, 3) + timedelta(minutes=random.randint(0, 1440))
    scenarios.append({
        "id": str(uuid.uuid4()),
        "timestamp": ts.isoformat(),
        "weather": random.choice(weather_opts),
        "label": random.choice(labels),
        "image_path": f"s3://drive-data/frames/{i:06d}.jpg"
    })

with open("scenarios.json", "w") as f:
    json.dump(scenarios, f, indent=2)

# 上传到 MinIO (模拟 Hudi 表)
import boto3
import io, pandas as pd

s3 = boto3.client("s3", endpoint_url=os.getenv("MINIO_ENDPOINT"),
                  aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
                  aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"))
df = pd.DataFrame(scenarios)
buf = io.BytesIO()
df.to_parquet(buf, engine='pyarrow')
s3.put_object(Bucket='datalake', Key='hudi/scenarios.parquet', Body=buf.getvalue())
print("Hudi table mock uploaded to MinIO.")

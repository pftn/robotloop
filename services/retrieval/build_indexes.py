import os, json, numpy as np
from elasticsearch import Elasticsearch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

es = Elasticsearch(hosts=os.getenv("ES_HOST"))
milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT")

# 1. Elasticsearch
mapping = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "weather": {"type": "keyword"},
            "label": {"type": "keyword"}
        }
    }
}
es.indices.create(index="scenarios", body=mapping, ignore=400)

with open("scenarios.json") as f:
    data = json.load(f)
for doc in data:
    es.index(index="scenarios", id=doc["id"], document=doc)
print("ES index ready.")

# 2. Milvus
connections.connect(host=milvus_host, port=milvus_port)
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, "image_embeddings")
collection = Collection("scenario_images", schema)

ids = [doc["id"] for doc in data]
embeddings = np.random.rand(len(ids), 128).astype(np.float32).tolist()
collection.insert([ids, embeddings])
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
collection.load()
print("Milvus index ready.")

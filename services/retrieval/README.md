# RobotLoop 多模态检索子系统

本目录包含 RobotLoop 的可选扩展模块——**多模态联合检索服务**。  
它从 RobotLoop 的 PostgreSQL 中读取已标注的媒体数据，构建 Elasticsearch 文本索引和 Chroma 向量索引，并对外提供统一的联合查询 API。

## 快速启动

### 1. 确保 RobotLoop 核心服务已运行
```bash
# 在 robotloop 项目根目录下
docker-compose up -d
```

### 2. 启动检索子系统
```bash
cd services/retrieval
docker-compose up -d
```

### 3. 执行数据同步
同步任务（`sync` 容器）会在启动时自动运行，将 PostgreSQL 中的预标注结果导入 ES 和 Chroma。  
如需手动重新同步，可执行：
```bash
docker-compose run --rm sync python sync_to_retrieval.py
```

### 4. 测试联合查询
```bash
curl "http://localhost:8001/search?labels=person"
```

## 模块说明
- **sync_to_retrieval.py**：从 `media_files` 表提取预标注结果，生成文本标签和向量，写入 ES 和 Chroma。
- **search_api.py**：FastAPI 服务，提供 `/search` 接口，支持按标签、图像语义相似度等条件组合查询。
- **docker-compose.yml**：编排 Elasticsearch、Chroma、同步任务和 API 服务，加入 RobotLoop 的 `robotloop-net` 网络。

## 技术栈
- Elasticsearch 8.x
- Chroma 2.3
- Sentence-Transformers (clip-ViT-B-32)
- FastAPI

> 该子系统可独立运行，但需要访问 RobotLoop 的 PostgreSQL 实例。
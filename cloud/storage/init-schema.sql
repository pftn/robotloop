-- 媒体文件元数据表（key 唯一，支持 ON CONFLICT DO NOTHING）
CREATE TABLE IF NOT EXISTS media_files (
    id SERIAL PRIMARY KEY,
    key TEXT UNIQUE,
    bucket TEXT,
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    prelabel_result JSONB,
    prelabeled_at TIMESTAMPTZ
);

-- 数据集版本管理表（用于 Airflow 批处理任务）
CREATE TABLE IF NOT EXISTS dataset_versions (
    version_id SERIAL PRIMARY KEY,
    version_name TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    sample_count INTEGER,
    description TEXT
);

create database airflow;
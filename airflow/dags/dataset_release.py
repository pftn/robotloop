from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import psycopg2

default_args = {
    'owner': 'robotloop',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}


def check_prelabel_completeness():
    conn = psycopg2.connect(
        host="postgres", user="robotloop", password="robotloop", dbname="robotloop"
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM media_files")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM media_files WHERE prelabel_result IS NOT NULL")
    labeled = cur.fetchone()[0]
    cur.close()
    conn.close()

    completeness = labeled / total if total > 0 else 0
    print(f"预标注完成率: {completeness:.2%} ({labeled}/{total})")

    if completeness < 0.8:
        raise ValueError(f"预标注完成率过低 ({completeness:.2%})，请检查")


def create_dataset_version():
    conn = psycopg2.connect(
        host="postgres", user="robotloop", password="robotloop", dbname="robotloop"
    )
    cur = conn.cursor()

    cur.execute("SELECT key FROM media_files WHERE prelabel_result IS NOT NULL")
    rows = cur.fetchall()
    sample_count = len(rows)

    version_name = f"robot_dataset_{datetime.now().strftime('%Y%m%d')}"

    cur.execute(
        "INSERT INTO dataset_versions (version_name, sample_count, description) "
        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        (version_name, sample_count, "Daily auto-generated dataset")
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"数据集版本 {version_name} 已创建，包含 {sample_count} 个样本")


with DAG(
        dag_id='dataset_release',
        default_args=default_args,
        description='每日数据集质量校验与版本发布',
        schedule_interval='0 2 * * *',
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=['robotloop', 'dataset'],
) as dag:
    task_check = PythonOperator(
        task_id='check_prelabel_completeness',
        python_callable=check_prelabel_completeness,
    )

    task_create = PythonOperator(
        task_id='create_dataset_version',
        python_callable=create_dataset_version,
    )

    task_check >> task_create

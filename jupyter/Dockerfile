# syntax=docker/dockerfile:1
FROM jupyter/scipy-notebook:latest

USER root

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER jovyan

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r /tmp/requirements.txt

COPY demo.ipynb /home/jovyan/work/demo.ipynb

CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
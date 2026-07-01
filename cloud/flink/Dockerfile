FROM flink:1.18-scala_2.12

# 安装 Python 和依赖
RUN apt-get update && apt-get install -y python3 python3-pip curl && \
    ln -s /usr/bin/python3 /usr/bin/python

# 升级 pip 并安装 PyFlink 和作业依赖
RUN pip3 install --upgrade pip
RUN pip3 install apache-flink==1.18.0

RUN mkdir -p /opt/flink/lib && \
    wget -P /opt/flink/lib https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.0.2-1.18/flink-sql-connector-kafka-3.0.2-1.18.jar

WORKDIR /opt/flink
COPY robot_stream_job.py .
COPY requirements.txt .
RUN pip3 install -r requirements.txt
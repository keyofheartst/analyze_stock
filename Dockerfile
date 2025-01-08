FROM apache/airflow:2.7.2

USER root
RUN apt-get update && apt-get install -y gcc default-libmysqlclient-dev

USER airflow
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


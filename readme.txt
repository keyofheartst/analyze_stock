#pip install -r requirements.txt

docker build -t chien-airflow-stock-image:latest .

docker run -d -p 8081:8081 -p 5555:5555 -v ./dags:/opt/airflow/dags -v ./logs:/opt/airflow/logs -e AIRFLOW__CORE__EXECUTOR=LocalExecutor chien-airflow-stock-image:latest
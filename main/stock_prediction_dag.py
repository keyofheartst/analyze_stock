from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import timedelta as td

TICKER = "AAPL"
FUTURE_DAYS = 90
API_KEY = "your_alpha_vantage_api_key"

# Thu thập dữ liệu từ Alpha Vantage và Yahoo Finance
def fetch_data(**kwargs):
    symbol = TICKER
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY
    }
    response = requests.get(url, params=params)
    alpha_data = response.json() if response.status_code == 200 else None

    yf_data = yf.Ticker(symbol).history(period="1y", interval="1d")
    yf_data['Date'] = yf_data.index
    yf_data.reset_index(drop=True, inplace=True)

    return {"alpha_vantage": alpha_data, "yfinance": yf_data.to_dict()}

def prepare_data(**kwargs):
    ti = kwargs['ti']
    data = pd.DataFrame.from_dict(ti.xcom_pull(task_ids='fetch_data')['yfinance'])
    data['Target'] = data['Close'].shift(-FUTURE_DAYS)
    data.dropna(inplace=True)
    ti.xcom_push(key='prepared_data', value=data.to_dict())

def train_and_predict(**kwargs):
    ti = kwargs['ti']
    data = pd.DataFrame.from_dict(ti.xcom_pull(task_ids='prepare_data', key='prepared_data'))

    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    future_indices = np.arange(len(data), len(data) + FUTURE_DAYS).reshape(-1, 1)
    future_scaled = scaler.transform(future_indices)
    predictions = model.predict(future_scaled)

    future_dates = [data['Date'].iloc[-1] + td(days=i) for i in range(1, FUTURE_DAYS + 1)]
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": predictions})
    ti.xcom_push(key='predictions', value=pred_df.to_dict())

def plot_results(**kwargs):
    ti = kwargs['ti']
    data = pd.DataFrame.from_dict(ti.xcom_pull(task_ids='prepare_data', key='prepared_data'))
    predictions = pd.DataFrame.from_dict(ti.xcom_pull(task_ids='train_and_predict', key='predictions'))

    plt.figure(figsize=(14, 8))
    plt.plot(data['Date'], data['Close'], label="Thực tế", color='blue')
    plt.plot(predictions['Date'], predictions['Predicted_Close'], label="Dự đoán", color='red', linestyle='--')
    plt.title(f"Dự đoán giá cổ phiếu {TICKER} trong 3 tháng tới", fontsize=16)
    plt.xlabel("Ngày", fontsize=12)
    plt.ylabel("Giá đóng cửa (USD)", fontsize=12)
    plt.legend()
    plt.grid()
    plt.savefig('/tmp/stock_predictions.png')
    print("Biểu đồ được lưu tại: /tmp/stock_predictions.png")

# Cấu hình DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='stock_prediction_dag',
    default_args=default_args,
    description='Dự đoán giá cổ phiếu sử dụng Airflow',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    fetch_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_data,
    )

    prepare_task = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
        provide_context=True,
    )

    train_task = PythonOperator(
        task_id='train_and_predict',
        python_callable=train_and_predict,
        provide_context=True,
    )

    plot_task = PythonOperator(
        task_id='plot_results',
        python_callable=plot_results,
        provide_context=True,
    )

    fetch_task >> prepare_task >> train_task >> plot_task

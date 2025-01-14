import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta


TICKER = "AAPL"
FUTURE_DAYS = 90  

def get_data(ticker, period="1y", interval="1d"):
    """
    Lấy dữ liệu cổ phiếu từ yfinance.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)
    return data
    """
    Chuẩn bị dữ liệu cho mô hình dự đoán.
    - Tạo cột "Target" với giá trị giá đóng cửa của future_days tiếp theo.
    - Chỉ lấy các dòng có dữ liệu để đưa vào tập dự đoán
    """
def prepare_data(data, future_days):
    data['Target'] = data['Close'].shift(-future_days)   
    
    data.dropna(inplace=True)   
    return data
    """
    Huấn luyện mô hình Linear Regression dự đoán giá cổ phiếu dựa trên giá đóng cửa
    """
def train_model(data):
    X = np.arange(len(data)).reshape(-1, 1)   
    y = data['Close'].values   
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Đánh giá mô hình
    score = model.score(X_test_scaled, y_test)
    print(f"Độ chính xác của mô hình (R^2): {score:.2f}")
    
    return model, scaler
    """
    Dự đoán giá cổ phiếu trong future_days tiếp theo.
    """
def predict_future(model, scaler, data, future_days):

    last_index = len(data) - 1
    future_indices = np.arange(last_index + 1, last_index + future_days + 1).reshape(-1, 1)
    future_indices_scaled = scaler.transform(future_indices)
    
    predictions = model.predict(future_indices_scaled)
    future_dates = [data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
    
    return pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions})
    """
    Vẽ biểu đồ so sánh dữ liệu thực tế và dự đoán.
    """
def plot_results(data, predictions):

    plt.figure(figsize=(14, 8))
    
    # Dữ liệu thực tế
    plt.plot(data['Date'], data['Close'], label="Thực tế", color='blue')
    
    # Dự đoán
    plt.plot(predictions['Date'], predictions['Predicted_Close'], label="Dự đoán", color='red', linestyle='--')
    
    plt.title(f"Dự đoán giá cổ phiếu {TICKER} trong 3 tháng tới", fontsize=16)
    plt.xlabel("Ngày", fontsize=12)
    plt.ylabel("Giá đóng cửa (USD)", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__": 
    data = get_data(TICKER)
    data = prepare_data(data, FUTURE_DAYS)
    model, scaler = train_model(data)
    predictions = predict_future(model, scaler, data, FUTURE_DAYS)
    print(predictions)
    plot_results(data, predictions)

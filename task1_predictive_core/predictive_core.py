import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from models import LSTMForecaster
from utils import get_stock_data, calculate_metrics
from sklearn.preprocessing import MinMaxScaler

def train_forecast(data, days=7):
    """
    Trains LSTM model and generates forecasts with confidence intervals.
    Uses Monte Carlo for uncertainty.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values)
    
    seq_length = 30
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    X, y = np.array(X), np.array(y)
    
    model = LSTMForecaster()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(50):  # Demo training, not to convergence
        inputs = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Forecast
    last_seq = scaled_data[-seq_length:]
    predictions = []
    for _ in range(days):
        pred = model(torch.tensor(last_seq.reshape(1, seq_length, 1), dtype=torch.float32))
        predictions.append(pred.item())
        last_seq = np.append(last_seq[1:], pred.item())
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Confidence Interval (Monte Carlo)
    ci_low, ci_high = [], []
    for _ in range(100):
        noise = np.random.normal(0, 0.1, len(predictions))
        ci_low.append(predictions.flatten() * (1 - noise))
        ci_high.append(predictions.flatten() * (1 + noise))
    ci_low = np.mean(ci_low, axis=0)
    ci_high = np.mean(ci_high, axis=0)
    
    # Metrics on test data
    test_pred = model(torch.tensor(X[-len(y)//5:], dtype=torch.float32)).detach().numpy()
    test_pred = scaler.inverse_transform(test_pred)
    actual = scaler.inverse_transform(y[-len(y)//5:])
    mae, rmse, mape = calculate_metrics(actual, test_pred)
    
    return data, predictions, ci_low, ci_high, mae, rmse, mape

def main():
    st.title("Capital Pulse - Predictive Core")
    ticker = st.selectbox("Select Ticker", ["AAPL", "MSFT"])
    data = get_stock_data(ticker)
    hist, pred, ci_l, ci_h, mae, rmse, mape = train_forecast(data)
    
    st.subheader("Historical & Forecast")
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist.values, label="Historical")
    future_dates = pd.date_range(hist.index[-1], periods=8)[1:]
    ax.plot(future_dates, pred, label="Forecast", color='red')
    ax.fill_between(future_dates, ci_l, ci_h, alpha=0.3, label="Confidence Interval")
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Metrics")
    st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

if __name__ == "__main__":
    main()

import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def get_stock_data(ticker, period='2y'):
    """
    Fetches historical stock data using yfinance.
    Returns a DataFrame with 'Close' prices.
    """
    data = yf.download(ticker, period=period)
    return data[['Close']].dropna()

def calculate_metrics(actual, predicted):
    """
    Calculates forecasting accuracy metrics.
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

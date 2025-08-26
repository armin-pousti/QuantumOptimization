from __future__ import annotations
from typing import Dict, Any, List
from collections import defaultdict
import copy
import warnings
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from xgboost import XGBRegressor


CVAR_LEVEL = 0.95
nr_days = 100
nr_stocks = 839

def _sharpe(r: pd.Series) -> float:
    if r.std(ddof=0) == 0:
        return -float("inf")
    return r.mean() / r.std(ddof=0)

def _hist_cvar(r: pd.Series, level: float = CVAR_LEVEL) -> float:
    """Historic Conditional VaR (Expected Shortfall) at given level (positive number)."""
    if r.empty:
        return float("inf")
    var_threshold = r.quantile(1 - level)
    tail_losses = r[r <= var_threshold]
    if tail_losses.empty:
        return float("inf")
    return -tail_losses.mean()

def _downside_volatility(r: pd.Series) -> float:
    negative_returns = r[r < 0]
    if negative_returns.empty:
        return 0.0
    return negative_returns.std(ddof=0)

def _momentum(prices: pd.Series, window: int = 20) -> float:
    if len(prices) < window:
        return float("-inf")
    return prices.iloc[-1] / prices.iloc[-window] - 1

def _max_drawdown(prices: pd.Series) -> float:
    if prices.empty:
        return float("inf")
    cumulative = prices / prices.iloc[0]
    roll_max = cumulative.cummax()
    drawdown = cumulative / roll_max - 1.0
    return drawdown.min()  # negative number

def _trend_r2(prices: pd.Series) -> float:
    if len(prices) < 10:
        return 0.0
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return model.score(X, y)  # R²



def prepare_data(price_matrix):

    # Step 1: Calculate the return for each stock (last row - second to last row) / second to last row
    
    returns_matrix = (price_matrix[1:, :] - price_matrix[:-1, :]) / price_matrix[:-1, :]
    y_test = returns_matrix[-1, :]
    # Step 2: Remove the last row from the matrix for features calculation
    returns_matrix = returns_matrix[:-1, :]
    
    # Step 3: Generate features for each stock
    X_train = []
    
    for i in range(price_matrix.shape[1]):
        stock_returns = pd.Series(returns_matrix[:, i])  # Convert to pandas Series
        
        # Feature 1: Sharpe Ratio
        sharpe = _sharpe(stock_returns)

        # Feature 2: Historical Conditional VaR (CVAR)
        cvar = _hist_cvar(stock_returns)
        
        # Feature 3: Downside Volatility
        downside_volatility = _downside_volatility(stock_returns)
        
        # Feature 4: Momentum (20-day price change)
        momentum = _momentum(stock_returns)
        
        # Feature 5: Maximum Drawdown
        max_drawdown = abs(_max_drawdown(stock_returns))
        
        # Feature 6: Trend R²
        trend_r2 = _trend_r2(stock_returns)
        
        # Append the features for the stock
        features = [sharpe, cvar, downside_volatility, momentum, max_drawdown, trend_r2]
        X_train.append(features)
    
    # Convert X_train to a numpy array
    X_train = np.array(X_train)
    
    return X_train, y_test, returns_matrix


def _build_price_df(assets: Dict[str, dict]) -> pd.DataFrame:
    """Convert nested price dicts → DataFrame (index=date, columns=ticker)."""
    series: List[pd.Series] = [pd.Series(info["history"], name=tkr) for tkr, info in assets.items()]
    df = pd.concat(series, axis=1).sort_index().astype(float)
    
    # Ensure the index is in datetime format
    df.index = pd.to_datetime(df.index)
    
    return df

def load_input_data(filepath: str) -> Dict[str, Any]:
    """Load input data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]



with open("input.json", "r") as f:
    data = json.load(f)  # Load the JSON data from the file
input_data1 = data["data"] # Get the 'data' section
assets1 = input_data1.get("assets", {})  # Get the 'assets' section





# Step 2: Train the XGBoost model
def train_xgboost_model(X_train, y_train):
    """
    Train an XGBoost model on the given features and target values.
    
    Args:
    - X_train (numpy.ndarray): The feature matrix.
    - y_test (numpy.ndarray): The target vector (stock returns).
    
    Returns:
    - model: The trained XGBoost model.
    """
    # Before model fitting:
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Split the data into training and testing sets
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train an XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.15)
    model.fit(X_train_split, y_train_split)
    
    # Evaluate the model
    y_pred = model.predict(X_val_split)
    mae = mean_absolute_error(y_val_split, y_pred)
    
    print(f"Mean Absolute Error: {mae}")
    
    return model


def create_price_matrix_np(assets, num_stocks, num_days):
    selected_stocks = random.sample(list(assets.keys()), num_stocks)
    price_matrix = []
    
    for stock in selected_stocks:
        stock_data = assets[stock]['history']
        
        # Sample random 60 days of stock prices (if available)
        sampled_dates = sorted(random.sample(list(stock_data.keys()), min(num_days, len(stock_data))))
        #sampled_dates = random.sample(list(stock_data.keys()), min(num_days, len(stock_data)))
        sampled_prices = [stock_data[date] for date in sampled_dates]
        
        # Add the sampled prices as a row in the price matrix
        price_matrix.append(sampled_prices)
    
    # Convert the list to a NumPy array
    price_matrix_np = np.array(price_matrix)
    return price_matrix_np.T

# Create the price_matrix (10 stocks, 60 days)
price_matrix = create_price_matrix_np(assets1, num_stocks=nr_stocks, num_days=nr_days)

X_train, y_train, returns_matrix = prepare_data(price_matrix)

model = train_xgboost_model(X_train, y_train)

joblib.dump(model, 'trained_model.joblib')





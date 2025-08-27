#!/usr/bin/env python3
"""
Machine Learning Model Training for Portfolio Optimization
Trains XGBoost model to predict asset returns based on financial metrics
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Handles training and saving of the ML model for return prediction."""
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            'sharpe_ratio', 'cvar', 'downside_volatility', 
            'momentum', 'max_drawdown', 'trend_r2'
        ]
    
    def prepare_training_data(self, stock_list: list, days: int = 500) -> tuple:
        """Prepare training data from stock price history."""
        print(f"📊 Preparing training data for {len(stock_list)} stocks...")
        
        X_train = []
        y_train = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for i, ticker in enumerate(stock_list, 1):
            try:
                print(f"[{i:2d}/{len(stock_list)}] Processing {ticker:6s}...", end=" ")
                
                # Fetch data
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if len(hist) < 100:
                    print("✗ Insufficient data")
                    continue
                
                # Calculate features and target
                features, target = self._extract_features_and_target(hist)
                
                if features is not None and target is not None:
                    X_train.append(features)
                    y_train.append(target)
                    print("✓ Success")
                else:
                    print("✗ Feature extraction failed")
                    
            except Exception as e:
                print(f"✗ Error: {str(e)[:30]}")
                continue
        
        print(f"\n✅ Prepared {len(X_train)} training samples")
        return np.array(X_train), np.array(y_train)
    
    def _extract_features_and_target(self, hist: pd.DataFrame) -> tuple:
        """Extract features and target from historical data."""
        try:
            # Calculate features
            returns = hist['Close'].pct_change().dropna()
            prices = hist['Close']
            
            features = [
                self._calculate_sharpe_ratio(returns),
                self._calculate_cvar(returns),
                self._calculate_downside_volatility(returns),
                self._calculate_momentum(prices),
                abs(self._calculate_max_drawdown(prices)),
                self._calculate_trend_strength(prices)
            ]
            
            # Calculate target (future return)
            future_return = self._calculate_future_return(prices)
            
            # Validate data
            if any(np.isnan(features)) or np.isnan(future_return):
                return None, None
            
            return features, future_return
            
        except Exception:
            return None, None
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> XGBRegressor:
        """Train the XGBoost regression model."""
        print("🚀 Training XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train model
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"✅ Training completed!")
        print(f"   Training R²: {train_score:.4f}")
        print(f"   Testing R²:  {test_score:.4f}")
        
        return self.model
    
    def save_model(self, filename: str = 'trained_model.joblib'):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.model, filename)
        print(f"💾 Model saved to {filename}")
    
    # Financial metric calculations
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.std(ddof=0) == 0:
            return 0.0
        return returns.mean() / returns.std(ddof=0)
    
    def _calculate_cvar(self, returns: pd.Series, level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk."""
        var_threshold = returns.quantile(1 - level)
        tail_losses = returns[returns <= var_threshold]
        if tail_losses.empty:
            return 0.0
        return -tail_losses.mean()
    
    def _calculate_downside_volatility(self, returns: pd.Series) -> float:
        """Calculate downside volatility."""
        negative_returns = returns[returns < 0]
        return negative_returns.std(ddof=0) if not negative_returns.empty else 0.0
    
    def _calculate_momentum(self, prices: pd.Series, window: int = 10) -> float:
        """Calculate price momentum."""
        if len(prices) < window:
            return 0.0
        return prices.iloc[-1] / prices.iloc[-window] - 1
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if prices.empty:
            return 0.0
        cumulative = prices / prices.iloc[0]
        roll_max = cumulative.cummax()
        drawdown = cumulative / roll_max - 1.0
        return drawdown.min()
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using R²."""
        if len(prices) < 10:
            return 0.0
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.score(X, y)
    
    def _calculate_future_return(self, prices: pd.Series, days: int = 20) -> float:
        """Calculate future return for target variable."""
        if len(prices) < days + 1:
            return 0.0
        return prices.iloc[-1] / prices.iloc[-days-1] - 1


def main():
    """Main training execution."""
    print("🚀 Portfolio Optimization Model Trainer")
    print("=" * 50)
    
    # Define training stocks
    training_stocks = [
        # Technology
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK",
        # Healthcare
        "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR",
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO",
        # Consumer
        "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT",
        # Industrial
        "BA", "CAT", "GE", "MMM", "HON", "LMT", "RTX", "UNP"
    ]
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Prepare data
        X, y = trainer.prepare_training_data(training_stocks)
        
        if len(X) < 50:
            print("❌ Insufficient training data")
            return
        
        # Train model
        model = trainer.train_model(X, y)
        
        # Save model
        trainer.save_model()
        
        print("\n🎉 Model training completed successfully!")
        print("You can now run portfolio optimization with:")
        print("  python app.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")


if __name__ == "__main__":
    main()





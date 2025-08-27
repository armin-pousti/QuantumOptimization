#!/usr/bin/env python3
"""
Quantum Portfolio Optimization Engine
A hybrid quantum-classical approach to portfolio optimization using VQE algorithms.
"""

from __future__ import annotations
from typing import Dict, Any, List
from collections import defaultdict
import warnings
import json
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Quantum Computing Imports
from qiskit.circuit.library import TwoLocal
from qiskit.result import QuasiDistribution
from qiskit.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Configuration Constants
WINDOW_DAYS = 100
CVAR_LEVEL = 0.95
PER_INDUSTRY = 5
FINAL_MAX = 10
CORR_THRESHOLD = 0.8


class PortfolioOptimizer:
    """Main portfolio optimization engine with quantum and classical methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_date = config.get("evaluation_date")
        self.num_assets = config.get("num_assets", 10)
        self.use_quantum = config.get("use_quantum", True)
        
    def optimize(self, assets: Dict[str, dict]) -> Dict[str, float]:
        """Main optimization pipeline."""
        try:
            # Data preprocessing
            prices = self._build_price_dataframe(assets)
            if prices.empty:
                return {}
                
            # Ensure sufficient data
            if not self._validate_data_sufficiency(prices):
                return {}
                
            # Core optimization pipeline
            window_prices = self._get_analysis_window(prices)
            returns = self._calculate_returns(window_prices)
            
            # Asset selection
            raw_scores, candidates = self._preselect_assets(returns, assets, window_prices)
            if not candidates:
                return {}
                
            # Final selection and weighting
            chosen = self._diversify_selection(returns, candidates)
            if not chosen:
                return {}
                
            # Weight optimization
            if self.use_quantum:
                return self._quantum_weight_optimization(returns, chosen, raw_scores)
            else:
                return self._classical_weight_optimization(chosen, raw_scores)
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            return {}
    
    def _build_price_dataframe(self, assets: Dict[str, dict]) -> pd.DataFrame:
        """Build price DataFrame from asset data."""
        series_list = []
        for ticker, info in assets.items():
            price_series = pd.Series(info["history"], name=ticker)
            series_list.append(price_series)
        
        df = pd.concat(series_list, axis=1).sort_index().astype(float)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df
    
    def _validate_data_sufficiency(self, prices: pd.DataFrame) -> bool:
        """Validate that we have sufficient price data."""
        if prices.empty:
            return False
            
        eval_date = pd.to_datetime(self.evaluation_date, utc=True).tz_localize(None)
        required_date = eval_date - pd.Timedelta(days=WINDOW_DAYS)
        
        if prices.index.max() < required_date:
            warnings.warn(f"Insufficient price data before {self.evaluation_date}")
            return False
            
        return True
    
    def _get_analysis_window(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Get the analysis window of prices."""
        eval_date = pd.to_datetime(self.evaluation_date, utc=True).tz_localize(None)
        start_date = eval_date - pd.Timedelta(days=WINDOW_DAYS)
        
        # Filter data before evaluation date
        df_before = prices[prices.index < eval_date]
        
        # Get data in the window
        window_df = df_before[df_before.index >= start_date]
        
        if window_df.empty:
            warnings.warn(f"Insufficient data for {WINDOW_DAYS} day window, using available data")
            return df_before
            
        return window_df
    
    def _calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from prices."""
        return prices.bfill().pct_change(fill_method=None).dropna(how="all")
    
    def _preselect_assets(self, returns: pd.DataFrame, assets_meta: Dict[str, dict], 
                          prices: pd.DataFrame) -> tuple[Dict[str, float], List[str]]:
        """Preselect assets using ML model and industry diversification."""
        raw_scores = {}
        scores = []
        
        for ticker in returns.columns:
            if ticker not in prices.columns:
                continue
                
            try:
                # Calculate financial metrics
                metrics = self._calculate_asset_metrics(returns[ticker], prices[ticker])
                
                # Get ML prediction
                model = self._load_ml_model()
                raw_score = model.predict([list(metrics.values())])
                
                # Store results
                scalar_score = float(np.squeeze(raw_score))
                raw_scores[ticker] = scalar_score
                
                industry = assets_meta.get(ticker, {}).get("industry", "Unknown")
                scores.append((ticker, industry, -scalar_score, *metrics.values()))
                
            except Exception as e:
                warnings.warn(f"Error calculating scores for {ticker}: {e}")
                continue
        
        if not scores:
            return {}, []
        
        # Select top assets per industry
        selected = self._select_by_industry(scores)
        return raw_scores, selected
    
    def _calculate_asset_metrics(self, returns: pd.Series, prices: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive asset metrics."""
        common_idx = returns.index.intersection(prices.index)
        r = returns.loc[common_idx]
        p = prices.loc[common_idx]
        
        if r.empty or len(p) < 20:
            raise ValueError("Insufficient data for metrics calculation")
        
        return {
            'sharpe': self._calculate_sharpe_ratio(r),
            'cvar': self._calculate_cvar(r),
            'downside_vol': self._calculate_downside_volatility(r),
            'momentum': self._calculate_momentum(p),
            'max_dd': abs(self._calculate_max_drawdown(p)),
            'trend_r2': self._calculate_trend_strength(p)
        }
    
    def _select_by_industry(self, scores: List[tuple]) -> List[str]:
        """Select top assets per industry."""
        industry_assets = defaultdict(list)
        for score_tuple in scores:
            industry_assets[score_tuple[1]].append(score_tuple)
        
        selected = []
        for industry, industry_scores in industry_assets.items():
            sorted_scores = sorted(industry_scores, key=lambda x: x[2])
            top_assets = [ticker for ticker, *_ in sorted_scores[:PER_INDUSTRY]]
            selected.extend(top_assets)
        
        return selected
    
    def _diversify_selection(self, returns: pd.DataFrame, candidates: List[str]) -> List[str]:
        """Diversify selection using correlation filtering."""
        if not candidates:
            return []
        
        valid_candidates = [tkr for tkr in candidates if tkr in returns.columns]
        if not valid_candidates:
            return []
        
        # Sort by mean returns
        mean_returns = returns[valid_candidates].mean()
        ordered = mean_returns.sort_values(ascending=False).index.tolist()
        
        # Correlation-based filtering
        chosen = []
        for ticker in ordered:
            if len(chosen) >= self.num_assets:
                break
                
            if not chosen:
                chosen.append(ticker)
                continue
            
            # Check correlation with current selection
            if len(chosen) > 0:
                corr_matrix = returns[valid_candidates].corr().abs()
                max_corr = corr_matrix.loc[ticker, chosen].max()
                
                if pd.isna(max_corr) or max_corr <= CORR_THRESHOLD:
                    chosen.append(ticker)
        
        # Fill remaining slots if needed
        for ticker in ordered:
            if len(chosen) >= self.num_assets:
                break
            if ticker not in chosen:
                chosen.append(ticker)
        
        return chosen
    
    def _quantum_weight_optimization(self, returns: pd.DataFrame, chosen: List[str], 
                                   raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Quantum weight optimization using VQE."""
        try:
            quantum_weights = self._run_quantum_optimization(returns, chosen)
            quantum_selected = [tkr for tkr, wt in quantum_weights.items() if wt > 0]
            
            if not quantum_selected:
                warnings.warn("Quantum optimization selected no assets")
                return {}
            
            return self._normalize_weights(quantum_selected, raw_scores)
            
        except Exception as e:
            warnings.warn(f"Quantum optimization failed: {e}")
            return self._classical_weight_optimization(chosen, raw_scores)
    
    def _classical_weight_optimization(self, chosen: List[str], 
                                     raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Classical weight optimization."""
        return self._normalize_weights(chosen, raw_scores)
    
    def _normalize_weights(self, selected: List[str], raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights based on raw scores."""
        selected_scores = {tkr: raw_scores[tkr] for tkr in selected if tkr in raw_scores}
        
        if not selected_scores:
            return {}
        
        total_score = sum(selected_scores.values())
        if total_score == 0:
            equal_weight = 1 / len(selected)
            return {tkr: equal_weight for tkr in selected}
        
        return {tkr: score / total_score for tkr, score in selected_scores.items()}
    
    def _run_quantum_optimization(self, returns: pd.DataFrame, candidates: List[str]) -> Dict[str, float]:
        """Run quantum optimization using VQE."""
        # Prepare data for quantum optimization
        mu = returns[candidates].mean().values
        sigma = returns[candidates].cov().values
        
        # Create portfolio optimization problem
        portfolio = PortfolioOptimization(
            expected_returns=mu,
            covariances=sigma,
            risk_factor=0.5,
            budget=min(len(candidates), self.num_assets)
        )
        
        qp = portfolio.to_quadratic_program()
        
        # Configure VQE
        ansatz = TwoLocal(len(candidates), "ry", "cz", reps=3, entanglement="full")
        optimizer = COBYLA(maxiter=500)
        
        # Run VQE
        vqe = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=optimizer)
        optimizer_wrapper = MinimumEigenOptimizer(vqe)
        
        result = optimizer_wrapper.solve(qp)
        return dict(zip(candidates, result.x))
    
    def _load_ml_model(self):
        """Load the trained ML model."""
        try:
            return joblib.load('trained_model.joblib')
        except FileNotFoundError:
            raise FileNotFoundError("Trained model not found. Run model_training.py first.")
    
    # Financial metric calculations
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        if returns.std(ddof=0) == 0:
            return -float("inf")
        return returns.mean() / returns.std(ddof=0)
    
    def _calculate_cvar(self, returns: pd.Series, level: float = CVAR_LEVEL) -> float:
        var_threshold = returns.quantile(1 - level)
        tail_losses = returns[returns <= var_threshold]
        if tail_losses.empty:
            return float("inf")
        return -tail_losses.mean()
    
    def _calculate_downside_volatility(self, returns: pd.Series) -> float:
        negative_returns = returns[returns < 0]
        return negative_returns.std(ddof=0) if not negative_returns.empty else 0.0
    
    def _calculate_momentum(self, prices: pd.Series, window: int = 10) -> float:
        if len(prices) < window:
            return float("-inf")
        return prices.iloc[-1] / prices.iloc[-window] - 1
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        if prices.empty:
            return float("inf")
        cumulative = prices / prices.iloc[0]
        roll_max = cumulative.cummax()
        drawdown = cumulative / roll_max - 1.0
        return drawdown.min()
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        if len(prices) < 10:
            return 0.0
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.score(X, y)


def run(input_data: Dict[str, Any], solver_params: Dict[str, Any] = None, 
        extra_arguments: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main entry point for portfolio optimization."""
    try:
        # Initialize optimizer
        optimizer = PortfolioOptimizer(input_data)
        
        # Get assets from input
        assets = input_data.get("assets", {})
        if not assets:
            return {"selected_assets_weights": {}, "num_selected_assets": 0}
        
        # Run optimization
        weights = optimizer.optimize(assets)
        
        return {
            "selected_assets_weights": weights,
            "num_selected_assets": len(weights)
        }
        
    except Exception as e:
        warnings.warn(f"Portfolio optimization failed: {e}")
        return {"selected_assets_weights": {}, "num_selected_assets": 0}


if __name__ == "__main__":
    # Standalone execution for testing
    try:
        with open("input.json", "r") as f:
            data = json.load(f)
        
        result = run(data)
        print(f"Selected {len(result['selected_assets_weights'])} assets")
        
        for asset, weight in result["selected_assets_weights"].items():
            print(f"{asset}: {weight:.4f}")
            
    except Exception as e:
        print(f"Execution failed: {e}")
from __future__ import annotations  # For forward compatibility with type hints
from typing import Dict, Any, List
from collections import defaultdict
import copy
import warnings
import json
import random
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
"""
from qiskit_algorithms import SamplingVQE, SamplingMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer.primitives import Estimator as AerEstimator # or Sampler for SamplingVQE
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_aer import AerSimulator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_aer.primitives import SamplerV2
"""
from qiskit.circuit.library import TwoLocal
from qiskit.result import QuasiDistribution
from qiskit.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import matplotlib.pyplot as plt
import datetime


WINDOW_DAYS = 100               # ≈ two calendar months
CVAR_LEVEL = 0.95              # 95 % one‑sided VaR (historic)
PER_INDUSTRY = 5               # top N per industry in preselection
FINAL_MAX = 10                # cap on final portfolio size
CORR_THRESHOLD = 0.8           # max |ρ| between selected assets

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_input_data(filepath: str) -> Dict[str, Any]:
    """Load input data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]

def _build_price_df(assets: Dict[str, dict]) -> pd.DataFrame:
    """Convert nested price dicts → DataFrame (index=date, columns=ticker)."""
    series: List[pd.Series] = [pd.Series(info["history"], name=tkr) for tkr, info in assets.items()]
    df = pd.concat(series, axis=1).sort_index().astype(float)
    
    # Ensure the index is in datetime format with UTC timezone
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    
    return df

def _latest_window(df: pd.DataFrame, evaluation_date: str | pd.Timestamp, n_days: int) -> pd.DataFrame:
    """Slice the DataFrame to last *n_days* prior to `evaluation_date`."""
    # Ensure index is datetime and evaluation_date is a Timestamp
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    # Convert evaluation_date to Timestamp if it is a string
    if isinstance(evaluation_date, str):
        eval_ts = pd.to_datetime(evaluation_date, utc=True).tz_localize(None)
    else:
        eval_ts = evaluation_date  # Already a Timestamp

    # Ensure eval_ts is a Timestamp
    if not isinstance(eval_ts, pd.Timestamp):
        raise ValueError(f"Expected evaluation_date to be a Timestamp or a string that can be converted to Timestamp, but got {type(eval_ts)}.")

    # Filter data before evaluation date
    df_before = df[df.index < eval_ts]
    
    if df_before.empty:
        warnings.warn(f"No price data before evaluation_date {evaluation_date}.")
        return pd.DataFrame()  # Return empty DataFrame instead of raising exception
        
    # Calculate start date for the window
    start_ts = eval_ts - pd.Timedelta(days=n_days)
    
    # Get data in the window
    window_df = df_before[df_before.index >= start_ts]
    
    # If window is empty, take whatever data we have before evaluation date
    if window_df.empty:
        warnings.warn(f"Insufficient data for {n_days} day window before {evaluation_date}, using all available data.")
        return df_before
        
    return window_df

def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """bfill to handle gaps then pct_change."""
    return prices.bfill().pct_change(fill_method=None).dropna(how="all")

def load_model(model_path='trained_model.joblib'):
    """Load the pre-trained model."""
    model = joblib.load(model_path)
    #print("Model loaded successfully!")
    return model

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

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

def _momentum(prices: pd.Series, window: int = 10) -> float:
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

# -----------------------------------------------------------------------------
# Pre‑selection: top N per industry
# -----------------------------------------------------------------------------

def _preselect(returns: pd.DataFrame, assets_meta: Dict[str, dict], prices: pd.DataFrame) -> Dict[str, float]:
    raw_scores = {}  # Dictionary to store raw scores for each asset
    scores = []  # List to store selected assets for preselection

    # Ensure we have at least some data
    if returns.empty or prices.empty:
        warnings.warn("Empty returns or prices DataFrame in _preselect")
        return {}, []

    for tkr in returns.columns:
        r = returns[tkr].dropna()
        
        # Check if ticker exists in prices
        if tkr not in prices.columns:
            continue
            
        # Match time periods between returns and prices
        p = prices[tkr]
        common_idx = r.index.intersection(p.index)
        r = r.loc[common_idx]
        p = p.loc[common_idx]
        
        if r.empty or len(p) < 20:
            continue

        try:
            sharpe = _sharpe(r)
            cvar = _hist_cvar(r)
            momentum = _momentum(p)
            max_dd = _max_drawdown(p)
            trend_r2 = _trend_r2(p)
            downside_vol = _downside_volatility(r)
 
            model = load_model()
            #returns the expected return 
            raw_score = model.predict(np.array([[sharpe, cvar, downside_vol, momentum, abs(max_dd), trend_r2]]))
            #print("raw_score = ", raw_score)

            industry = assets_meta.get(tkr, {}).get("industry", "Unknown")
            scalar_score = float(np.squeeze(raw_score))
            raw_scores[tkr] = scalar_score
            scores.append((tkr, industry, -scalar_score, sharpe, cvar)) # -Score, since sorting is based on "lower is better"
            
        except Exception as e:
            warnings.warn(f"Error calculating scores for {tkr}: {e}")
            continue

    # If no scores calculated, return empty list
    if not scores:
        return raw_scores, []

    # Collect best PER_INDUSTRY per industry
    per_industry: dict[str, List[tuple]] = defaultdict(list)
    for row in scores:
        per_industry[row[1]].append(row)

    selected: List[str] = []
    for industry, rows in per_industry.items():
        rows_sorted = sorted(rows, key=lambda x: x[2])  # lower GOODNESS → better
        top = [tkr for tkr, *_ in rows_sorted[:PER_INDUSTRY]]
        selected.extend(top)

    return raw_scores, selected

# -----------------------------------------------------------------------------
# Final selection: maximise diversification + return
# -----------------------------------------------------------------------------

def _diversified_pick(returns: pd.DataFrame, candidates: List[str], max_assets: int) -> List[str]:
    if not candidates:
        return []
    
    # Make sure all candidates exist in returns
    valid_candidates = [tkr for tkr in candidates if tkr in returns.columns]
    
    if not valid_candidates:
        warnings.warn("No valid candidates for diversification")
        return []
        
    # Calculate mean returns and sort
    try:
        r_bar = returns[valid_candidates].mean()
        ordered = r_bar.sort_values(ascending=False).index.tolist()
    except Exception as e:
        warnings.warn(f"Error calculating mean returns: {e}")
        # Fallback to original order if calculation fails
        ordered = valid_candidates

    try:
        # Calculate correlation matrix
        corr = returns[valid_candidates].corr().abs()
    except Exception as e:
        warnings.warn(f"Error calculating correlation matrix: {e}")
        # If correlation fails, just return top assets by return
        return ordered[:min(max_assets, len(ordered))]

    chosen: List[str] = []
    for tkr in ordered:
        if len(chosen) >= max_assets:
            break
        if not chosen:
            chosen.append(tkr)
            continue
            
        try:
            # Max abs correlation with current basket
            max_corr = corr.loc[tkr, chosen].max()
            if pd.isna(max_corr) or max_corr <= CORR_THRESHOLD:
                chosen.append(tkr)
        except Exception as e:
            warnings.warn(f"Error checking correlation for {tkr}: {e}")
            # Add anyway if there's an error
            if len(chosen) < max_assets:
                chosen.append(tkr)
                
    # If still < max_assets, fill by highest return regardless of corr
    for tkr in ordered:
        if len(chosen) >= max_assets:
            break
        if tkr not in chosen:
            chosen.append(tkr)
            
    return chosen

# -----------------------------------------------------------------------------
# Quantum Weight Model
# -----------------------------------------------------------------------------

def quantum_optimise_vqe(returns: pd.DataFrame,
                         candidates: list[str],
                         max_assets: int,
                         risk_aversion: float = 0.5,
                         bit_precision: int = 3):
    """"
    import numpy as np
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.converters import IntegerToBinary, QuadraticProgramToQubo
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import SPSA
    from qiskit.circuit.library import RealAmplitudes
    from qiskit_aer.primitives import Sampler
    """
    μ = returns[candidates].mean().values
    Σ = returns[candidates].cov().values
    num_assets = len(μ)

    q = 0.5
    budget = 5

    portfolio = PortfolioOptimization(
        expected_returns=μ, covariances=Σ, risk_factor=q, budget=budget
    )
    qp = portfolio.to_quadratic_program()

    from qiskit_algorithms.utils import algorithm_globals

    algorithm_globals.random_seed = 1234

    cobyla = COBYLA()
    cobyla.set_options(maxiter=500)
    ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
    svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla)
    svqe = MinimumEigenOptimizer(svqe_mes)
    result = svqe.solve(qp)
    print(result.x)
    """
    app = PortfolioOptimization(expected_returns=μ,
                                covariances=Σ,
                                risk_factor=risk_aversion,
                                budget=max_assets,
                                bounds=[(0, 2**bit_precision - 1)] * len(candidates))

    qp = app.to_quadratic_program()
    qp = IntegerToBinary().convert(qp)
    qp = QuadraticProgramToQubo().convert(qp)
    operator, offset = qp.to_ising()

    num_qubits = operator.num_qubits
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2)
    optimizer = SPSA(maxiter=150)
    svqe = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=optimizer)
    result = svqe.compute_minimum_eigenvalue(operator)

    # Bitstring interpretieren
    bitstring = result.best_measurement["bitstring"]
    weights = []
    for i in range(len(candidates)):
        bits = bitstring[i*bit_precision:(i+1)*bit_precision]
        weights.append(int(bits, 2))
    
    weights = np.array(weights)
    if weights.sum() == 0:
        return {}
    """
    return dict(zip(candidates, result.x))



# -----------------------------------------------------------------------------
# Main optimization function, now calling quantum optimizer if needed
# -----------------------------------------------------------------------------
#                                                                                                  SET TRUE IF IMPLEMENTED
def _optimise(assets: Dict[str, dict], evaluation_date: str | pd.Timestamp, num_assets_cap: int, use_quantum: bool = True) -> Dict[str, float]:
    try:
        print(f"🔍 Debug: Starting optimization with {len(assets)} assets")
        print(f"🔍 Debug: Evaluation date: {evaluation_date}")
        print(f"🔍 Debug: Target assets: {num_assets_cap}")
        print(f"🔍 Debug: Use quantum: {use_quantum}")
        
        prices = _build_price_df(assets)
        print(f"🔍 Debug: Built price DataFrame with shape: {prices.shape}")
        print(f"🔍 Debug: Price data range: {prices.index.min()} to {prices.index.max()}")
        
        # Ensure sufficient price data
        if prices.empty or prices.index.max() < pd.Timestamp(evaluation_date) - pd.Timedelta(days=WINDOW_DAYS):
            warnings.warn(f"Insufficient price data before {evaluation_date}; returning empty portfolio.")
            return {}
        
        window_prices = _latest_window(prices, evaluation_date, WINDOW_DAYS)
        print(f"🔍 Debug: Window prices shape: {window_prices.shape}")
        
        returns = _daily_returns(window_prices)
        print(f"🔍 Debug: Returns shape: {returns.shape}")
        
        if returns.empty or returns.shape[0] < 10:  
            warnings.warn(f"Insufficient return data before {evaluation_date}; returning empty portfolio.")
            return {}

        raw_scores, candidates = _preselect(returns, assets, window_prices)
        print(f"🔍 Debug: Preselection found {len(candidates)} candidates")
        
        if not candidates:
            warnings.warn("No candidates after preselection; returning empty portfolio.")
            return {}

        max_final = min(num_assets_cap, FINAL_MAX)
        chosen = _diversified_pick(returns, candidates, max_final)
        print(f"🔍 Debug: Diversification selected {len(chosen)} assets: {chosen}")
        
        if not chosen:
            warnings.warn("No assets chosen after diversification; returning empty portfolio.")
            return {}

        chosen_raw_scores = {tkr: raw_scores[tkr] for tkr in chosen if tkr in raw_scores}

        if not chosen_raw_scores:
            warnings.warn("No valid raw scores for chosen assets; returning empty portfolio.")
            return {}

        if not use_quantum:
            print("🔍 Debug: Using classical optimization")
            min_score = min(chosen_raw_scores.values())
            score_range = max(chosen_raw_scores.values()) - min_score

            if score_range == 0:
                weight = 1 / len(chosen)
                weights = {tkr: weight for tkr in chosen}
            else:
                normalized_scores = {tkr: (score - min_score) / score_range for tkr, score in chosen_raw_scores.items()}
                total_score = sum(normalized_scores.values())
                weights = {tkr: score / total_score for tkr, score in normalized_scores.items()}

        else:
            print("🔍 Debug: Using quantum optimization")
            quantum_weights = quantum_optimise_vqe(returns, chosen, max_final)
            quantum_selected = [tkr for tkr, wt in quantum_weights.items() if wt > 0]
            print(f"🔍 Debug: Quantum selected: {quantum_selected}")

            if not quantum_selected:
                warnings.warn("Quantum optimisation selected no assets; returning empty portfolio.")
                return {}

            quantum_raw_scores = {tkr: chosen_raw_scores[tkr] for tkr in quantum_selected}
            total_score = sum(quantum_raw_scores.values())

            if total_score == 0:
                equal_weight = 1 / len(quantum_selected)
                weights = {tkr: equal_weight for tkr in quantum_selected}
            else:
                weights = {tkr: score / total_score for tkr, score in quantum_raw_scores.items()}

        weights = {tkr: wt for tkr, wt in weights.items() if wt > 0}
        print(f"🔍 Debug: Final weights: {weights}")

        return weights

    except Exception as e:
        warnings.warn(f"Error in optimization process: {e}; returning empty portfolio.")
        import traceback
        traceback.print_exc()
        return {}

# ---- Public API – entry point ----
def run(input_data: Dict[str, Any], solver_params: Dict[str, Any] | None = None, extra_arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Portfolio construction based on user specs."""
    # Check if evaluation_date is provided
    if 'evaluation_date' not in input_data:
        # For standalone testing - log warning instead of raising error
        if __name__ == "__main__":
            print("Warning: No evaluation_date provided, using today's date")
            import datetime
            input_data["evaluation_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
        else:
            # When called from main.py, evaluation_date should be present
            raise KeyError("evaluation_date is missing from input_data. Ensure that evaluation_date is provided.")
    
    num_assets_cap = input_data["num_assets"]
    assets = input_data["assets"]  # Assets directly from input
    evaluation_date = input_data["evaluation_date"]
    
    # Ensure evaluation_date is a string
    if not isinstance(evaluation_date, str):
        evaluation_date = pd.to_datetime(evaluation_date).strftime("%Y-%m-%d")
        input_data["evaluation_date"] = evaluation_date
    
    print(f"Evaluating for date: {evaluation_date}")
    
    # Check if the quantum optimizer should be used
    use_quantum = input_data.get("use_quantum", True)  # Default to False if not specified
    
    try:
        # Calculate weights using the optimization function
        weights = _optimise(assets, evaluation_date, num_assets_cap, use_quantum)

        # Prepare data for plotting
        prices = _build_price_df(assets)
        returns = _daily_returns(prices)
        
        #selected_assets = list(weights.keys())
        #if len(selected_assets) > 0:
            #plot_selected_assets(prices, returns, selected_assets)


        return {
            "selected_assets_weights": weights,
            "num_selected_assets": len(weights),
        }
    except Exception as e:
        import traceback
        print(f"Error in run(): {e}")
        traceback.print_exc()
        # Return empty portfolio on error
        return {
            "selected_assets_weights": {},
            "num_selected_assets": 0,
        }

if __name__ == "__main__":
    try:
        # Load input data (from your input file or any other source)
        with open("input.json", "r") as f:
            data = json.load(f)
            input_data = data["data"]

            # Set evaluation_date for testing purposes (if not already set)
            if "from" in input_data:
                input_data["evaluation_date"] = input_data["from"]
                print(f"Setting evaluation_date to {input_data['evaluation_date']} for testing")
            
            # Call the run function to get the result
            result = run(input_data)
            
            # Use the result now that it is properly defined
            print(f"Selected {len(result['selected_assets_weights'])} assets")
            
            # Extract the selected assets and their weights
            for asset, weight in result["selected_assets_weights"].items():
                print(f"{asset}: {weight:.4f}")

            
                
    except Exception as e:
        print(f"Error during execution: {e}")
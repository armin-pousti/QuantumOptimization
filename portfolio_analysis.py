#!/usr/bin/env python3
"""
Portfolio Analysis and Visualization Script
Generates comprehensive graphs and comparisons for portfolio optimization report
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import portfolio functions
import portfolio
import main

def load_input_data():
    """Load input data from JSON file."""
    with open("input.json", "r") as f:
        data = json.load(f)
    return data["data"]

def run_portfolio_optimization(input_data, use_quantum=True):
    """Run portfolio optimization with specified quantum setting."""
    # Create a copy to avoid modifying original
    test_data = input_data.copy()
    test_data["use_quantum"] = use_quantum
    
    try:
        result = main.run(test_data, {}, {})
        return result
    except Exception as e:
        print(f"Error running optimization with quantum={use_quantum}: {e}")
        return None

def calculate_portfolio_metrics(weights, assets, evaluation_date):
    """Calculate portfolio performance metrics."""
    if not weights:
        return {}
    
    # Build price dataframe
    prices = portfolio._build_price_df(assets)
    returns = portfolio._daily_returns(prices)
    
    # Get latest window for analysis
    window_prices = portfolio._latest_window(prices, evaluation_date, 100)
    window_returns = portfolio._daily_returns(window_prices)
    
    # Calculate metrics for selected assets
    selected_assets = list(weights.keys())
    portfolio_returns = window_returns[selected_assets]
    
    # Portfolio weights as array
    weight_array = np.array([weights[asset] for asset in selected_assets])
    
    # Calculate portfolio metrics
    portfolio_return = np.sum(portfolio_returns.mean() * weight_array) * 252  # Annualized
    portfolio_volatility = np.sqrt(np.dot(weight_array.T, np.dot(portfolio_returns.cov() * 252, weight_array)))
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    # Maximum drawdown
    portfolio_cumulative = (1 + portfolio_returns.dot(weight_array)).cumprod()
    rolling_max = portfolio_cumulative.expanding().max()
    drawdown = (portfolio_cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_assets': len(selected_assets),
        'concentration': np.sum(weight_array ** 2)  # Herfindahl index
    }

def create_comparison_analysis():
    """Create comprehensive comparison analysis."""
    print("Loading input data...")
    input_data = load_input_data()
    
    print("Running classical optimization...")
    classical_result = run_portfolio_optimization(input_data, use_quantum=False)
    
    print("Running quantum optimization...")
    quantum_result = run_portfolio_optimization(input_data, use_quantum=True)
    
    if not classical_result or not quantum_result:
        print("Error: Could not run both optimizations")
        return
    
    # Calculate metrics for both approaches
    evaluation_date = input_data.get("evaluation_date", "2024-04-01")
    assets = input_data["assets"]
    
    classical_metrics = calculate_portfolio_metrics(
        classical_result["selected_assets_weights"], 
        assets, 
        evaluation_date
    )
    
    quantum_metrics = calculate_portfolio_metrics(
        quantum_result["selected_assets_weights"], 
        assets, 
        evaluation_date
    )
    
    # Create comprehensive visualizations
    create_portfolio_composition_charts(classical_result, quantum_result)
    create_performance_comparison_charts(classical_metrics, quantum_metrics)
    create_risk_return_charts(classical_result, quantum_result, assets, evaluation_date)
    create_asset_correlation_heatmap(classical_result, quantum_result, assets, evaluation_date)
    
    # Print summary statistics
    print_summary_statistics(classical_result, quantum_result, classical_metrics, quantum_metrics)

def create_portfolio_composition_charts(classical_result, quantum_result):
    """Create portfolio composition comparison charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Classical portfolio
    classical_weights = classical_result["selected_assets_weights"]
    if classical_weights:
        assets = list(classical_weights.keys())
        weights = list(classical_weights.values())
        ax1.pie(weights, labels=assets, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Classical Portfolio Composition')
    
    # Quantum portfolio
    quantum_weights = quantum_result["selected_assets_weights"]
    if quantum_weights:
        assets = list(quantum_weights.keys())
        weights = list(quantum_weights.values())
        ax2.pie(weights, labels=assets, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Quantum Portfolio Composition')
    
    plt.tight_layout()
    plt.savefig('portfolio_composition_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_comparison_charts(classical_metrics, quantum_metrics):
    """Create performance comparison charts."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Metrics to compare
    metrics = ['return', 'volatility', 'sharpe_ratio', 'max_drawdown']
    metric_names = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown']
    
    classical_values = [classical_metrics.get(m, 0) for m in metrics]
    quantum_values = [quantum_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, classical_values, width, label='Classical', alpha=0.8)
    ax1.bar(x + width/2, quantum_values, width, label='Quantum', alpha=0.8)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Values')
    ax1.set_title('Portfolio Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Risk-return scatter plot
    ax2.scatter(classical_metrics.get('volatility', 0), classical_metrics.get('return', 0), 
                s=200, label='Classical', alpha=0.7, color='blue')
    ax2.scatter(quantum_metrics.get('volatility', 0), quantum_metrics.get('return', 0), 
                s=200, label='Quantum', alpha=0.7, color='red')
    ax2.set_xlabel('Volatility (Risk)')
    ax2.set_ylabel('Return')
    ax2.set_title('Risk-Return Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Portfolio size comparison
    portfolio_sizes = [classical_metrics.get('num_assets', 0), quantum_metrics.get('num_assets', 0)]
    labels = ['Classical', 'Quantum']
    colors = ['blue', 'red']
    ax3.bar(labels, portfolio_sizes, color=colors, alpha=0.7)
    ax3.set_ylabel('Number of Assets')
    ax3.set_title('Portfolio Diversification')
    ax3.grid(True, alpha=0.3)
    
    # Concentration comparison (Herfindahl index)
    concentrations = [classical_metrics.get('concentration', 0), quantum_metrics.get('concentration', 0)]
    ax4.bar(labels, concentrations, color=colors, alpha=0.7)
    ax4.set_ylabel('Concentration Index')
    ax4.set_title('Portfolio Concentration (Lower = More Diversified)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_risk_return_charts(classical_result, quantum_result, assets, evaluation_date):
    """Create risk-return analysis charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Build price and return data
    prices = portfolio._build_price_df(assets)
    returns = portfolio._daily_returns(prices)
    window_returns = portfolio._daily_returns(portfolio._latest_window(prices, evaluation_date, 100))
    
    # Calculate individual asset metrics
    asset_metrics = {}
    for asset in assets.keys():
        if asset in window_returns.columns:
            asset_returns = window_returns[asset].dropna()
            if len(asset_returns) > 0:
                asset_metrics[asset] = {
                    'return': asset_returns.mean() * 252,
                    'volatility': asset_returns.std() * np.sqrt(252)
                }
    
    # Plot all assets
    returns_list = [metrics['return'] for metrics in asset_metrics.values()]
    volatilities_list = [metrics['volatility'] for metrics in asset_metrics.values()]
    asset_names = list(asset_metrics.keys())
    
    ax1.scatter(volatilities_list, returns_list, alpha=0.6, s=50, color='gray', label='All Assets')
    
    # Highlight selected assets for classical
    if classical_result["selected_assets_weights"]:
        classical_assets = list(classical_result["selected_assets_weights"].keys())
        classical_returns = [asset_metrics[asset]['return'] for asset in classical_assets if asset in asset_metrics]
        classical_volatilities = [asset_metrics[asset]['volatility'] for asset in classical_assets if asset in asset_metrics]
        ax1.scatter(classical_volatilities, classical_returns, s=100, color='blue', alpha=0.8, label='Classical Selected')
        
        # Add labels for classical assets
        for i, asset in enumerate(classical_assets):
            if asset in asset_metrics:
                ax1.annotate(asset, (classical_volatilities[i], classical_returns[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Highlight selected assets for quantum
    if quantum_result["selected_assets_weights"]:
        quantum_assets = list(quantum_result["selected_assets_weights"].keys())
        quantum_returns = [asset_metrics[asset]['return'] for asset in quantum_assets if asset in asset_metrics]
        quantum_volatilities = [asset_metrics[asset]['volatility'] for asset in quantum_assets if asset in asset_metrics]
        ax1.scatter(quantum_volatilities, quantum_returns, s=100, color='red', alpha=0.8, label='Quantum Selected')
        
        # Add labels for quantum assets
        for i, asset in enumerate(quantum_assets):
            if asset in asset_metrics:
                ax1.annotate(asset, (quantum_volatilities[i], quantum_returns[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Volatility (Risk)')
    ax1.set_ylabel('Annual Return')
    ax1.set_title('Risk-Return Profile: All Assets vs Selected')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Portfolio weights comparison
    if classical_result["selected_assets_weights"] and quantum_result["selected_assets_weights"]:
        all_assets = set(classical_result["selected_assets_weights"].keys()) | set(quantum_result["selected_assets_weights"].keys())
        
        classical_weights = [classical_result["selected_assets_weights"].get(asset, 0) for asset in all_assets]
        quantum_weights = [quantum_result["selected_assets_weights"].get(asset, 0) for asset in all_assets]
        
        x = np.arange(len(all_assets))
        width = 0.35
        
        ax2.bar(x - width/2, classical_weights, width, label='Classical', alpha=0.8, color='blue')
        ax2.bar(x + width/2, quantum_weights, width, label='Quantum', alpha=0.8, color='red')
        ax2.set_xlabel('Assets')
        ax2.set_ylabel('Weight')
        ax2.set_title('Portfolio Weights Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(list(all_assets), rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('risk_return_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_asset_correlation_heatmap(classical_result, quantum_result, assets, evaluation_date):
    """Create correlation heatmap for selected assets."""
    if not classical_result["selected_assets_weights"] or not quantum_result["selected_assets_weights"]:
        return
    
    # Build return data
    prices = portfolio._build_price_df(assets)
    returns = portfolio._daily_returns(prices)
    window_returns = portfolio._daily_returns(portfolio._latest_window(prices, evaluation_date, 100))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Classical portfolio correlation
    classical_assets = list(classical_result["selected_assets_weights"].keys())
    classical_returns = window_returns[classical_assets].dropna()
    if len(classical_returns) > 0:
        classical_corr = classical_returns.corr()
        sns.heatmap(classical_corr, annot=True, cmap='coolwarm', center=0, ax=ax1, 
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax1.set_title('Classical Portfolio Asset Correlations')
    
    # Quantum portfolio correlation
    quantum_assets = list(quantum_result["selected_assets_weights"].keys())
    quantum_returns = window_returns[quantum_assets].dropna()
    if len(quantum_returns) > 0:
        quantum_corr = quantum_returns.corr()
        sns.heatmap(quantum_corr, annot=True, cmap='coolwarm', center=0, ax=ax2, 
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax2.set_title('Quantum Portfolio Asset Correlations')
    
    plt.tight_layout()
    plt.savefig('correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(classical_result, quantum_result, classical_metrics, quantum_metrics):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("PORTFOLIO OPTIMIZATION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nClassical Portfolio Results:")
    print(f"  Selected Assets: {list(classical_result['selected_assets_weights'].keys())}")
    print(f"  Number of Assets: {classical_metrics.get('num_assets', 0)}")
    print(f"  Annual Return: {classical_metrics.get('return', 0):.4f}")
    print(f"  Annual Volatility: {classical_metrics.get('volatility', 0):.4f}")
    print(f"  Sharpe Ratio: {classical_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown: {classical_metrics.get('max_drawdown', 0):.4f}")
    print(f"  Concentration Index: {classical_metrics.get('concentration', 0):.4f}")
    
    print(f"\nQuantum Portfolio Results:")
    print(f"  Selected Assets: {list(quantum_result['selected_assets_weights'].keys())}")
    print(f"  Number of Assets: {quantum_metrics.get('num_assets', 0)}")
    print(f"  Annual Return: {quantum_metrics.get('return', 0):.4f}")
    print(f"  Annual Volatility: {quantum_metrics.get('volatility', 0):.4f}")
    print(f"  Sharpe Ratio: {quantum_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown: {quantum_metrics.get('max_drawdown', 0):.4f}")
    print(f"  Concentration Index: {quantum_metrics.get('concentration', 0):.4f}")
    
    print(f"\nPerformance Comparison:")
    if classical_metrics.get('sharpe_ratio', 0) > 0 and quantum_metrics.get('sharpe_ratio', 0) > 0:
        sharpe_improvement = ((quantum_metrics.get('sharpe_ratio', 0) - classical_metrics.get('sharpe_ratio', 0)) / 
                             classical_metrics.get('sharpe_ratio', 0)) * 100
        print(f"  Sharpe Ratio Improvement: {sharpe_improvement:+.2f}%")
    
    if classical_metrics.get('volatility', 0) > 0 and quantum_metrics.get('volatility', 0) > 0:
        risk_reduction = ((classical_metrics.get('volatility', 0) - quantum_metrics.get('volatility', 0)) / 
                         classical_metrics.get('volatility', 0)) * 100
        print(f"  Risk Reduction: {risk_reduction:+.2f}%")
    
    print(f"\nDiversification Analysis:")
    print(f"  Classical Portfolio Size: {classical_metrics.get('num_assets', 0)} assets")
    print(f"  Quantum Portfolio Size: {quantum_metrics.get('num_assets', 0)} assets")
    print(f"  Classical Concentration: {classical_metrics.get('concentration', 0):.4f}")
    print(f"  Quantum Concentration: {quantum_metrics.get('concentration', 0):.4f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("Starting Portfolio Analysis and Visualization...")
    create_comparison_analysis()
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

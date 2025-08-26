#!/usr/bin/env python3
"""
Report Generation Script for Portfolio Optimization Analysis
Creates a comprehensive PDF report with all findings and visualizations
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
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
    
    prices = portfolio._build_price_df(assets)
    returns = portfolio._daily_returns(prices)
    window_returns = portfolio._daily_returns(portfolio._latest_window(prices, evaluation_date, 100))
    
    selected_assets = list(weights.keys())
    portfolio_returns = window_returns[selected_assets]
    weight_array = np.array([weights[asset] for asset in selected_assets])
    
    # Calculate metrics
    portfolio_return = np.sum(portfolio_returns.mean() * weight_array) * 252
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
        'concentration': np.sum(weight_array ** 2)
    }

def create_portfolio_composition_chart(classical_result, quantum_result, pdf):
    """Create portfolio composition comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Classical portfolio
    classical_weights = classical_result["selected_assets_weights"]
    if classical_weights:
        assets = list(classical_weights.keys())
        weights = list(classical_weights.values())
        ax1.pie(weights, labels=assets, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Classical Portfolio Composition', fontsize=14, fontweight='bold')
    
    # Quantum portfolio
    quantum_weights = quantum_result["selected_assets_weights"]
    if quantum_weights:
        assets = list(quantum_weights.keys())
        weights = list(quantum_weights.values())
        ax2.pie(weights, labels=assets, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Quantum Portfolio Composition', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_comparison_chart(classical_metrics, quantum_metrics, pdf):
    """Create performance comparison chart."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Metrics comparison
    metrics = ['return', 'volatility', 'sharpe_ratio', 'max_drawdown']
    metric_names = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown']
    
    classical_values = [classical_metrics.get(m, 0) for m in metrics]
    quantum_values = [quantum_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, classical_values, width, label='Classical', alpha=0.8, color='blue')
    ax1.bar(x + width/2, quantum_values, width, label='Quantum', alpha=0.8, color='red')
    ax1.set_xlabel('Metrics', fontsize=12)
    ax1.set_ylabel('Values', fontsize=12)
    ax1.set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Risk-return scatter
    ax2.scatter(classical_metrics.get('volatility', 0), classical_metrics.get('return', 0), 
                s=200, label='Classical', alpha=0.7, color='blue')
    ax2.scatter(quantum_metrics.get('volatility', 0), quantum_metrics.get('return', 0), 
                s=200, label='Quantum', alpha=0.7, color='red')
    ax2.set_xlabel('Volatility (Risk)', fontsize=12)
    ax2.set_ylabel('Return', fontsize=12)
    ax2.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Portfolio size
    portfolio_sizes = [classical_metrics.get('num_assets', 0), quantum_metrics.get('num_assets', 0)]
    labels = ['Classical', 'Quantum']
    colors = ['blue', 'red']
    ax3.bar(labels, portfolio_sizes, color=colors, alpha=0.7)
    ax3.set_ylabel('Number of Assets', fontsize=12)
    ax3.set_title('Portfolio Diversification', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Concentration
    concentrations = [classical_metrics.get('concentration', 0), quantum_metrics.get('concentration', 0)]
    ax4.bar(labels, concentrations, color=colors, alpha=0.7)
    ax4.set_ylabel('Concentration Index', fontsize=12)
    ax4.set_title('Portfolio Concentration (Lower = More Diversified)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

def create_risk_return_analysis(classical_result, quantum_result, assets, evaluation_date, pdf):
    """Create risk-return analysis charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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
    
    ax1.scatter(volatilities_list, returns_list, alpha=0.6, s=50, color='gray', label='All Assets')
    
    # Highlight selected assets
    if classical_result["selected_assets_weights"]:
        classical_assets = list(classical_result["selected_assets_weights"].keys())
        classical_returns = [asset_metrics[asset]['return'] for asset in classical_assets if asset in asset_metrics]
        classical_volatilities = [asset_metrics[asset]['volatility'] for asset in classical_assets if asset in asset_metrics]
        ax1.scatter(classical_volatilities, classical_returns, s=100, color='blue', alpha=0.8, label='Classical Selected')
        
        for i, asset in enumerate(classical_assets):
            if asset in asset_metrics:
                ax1.annotate(asset, (classical_volatilities[i], classical_returns[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    if quantum_result["selected_assets_weights"]:
        quantum_assets = list(quantum_result["selected_assets_weights"].keys())
        quantum_returns = [asset_metrics[asset]['return'] for asset in quantum_assets if asset in asset_metrics]
        quantum_volatilities = [asset_metrics[asset]['volatility'] for asset in quantum_assets if asset in asset_metrics]
        ax1.scatter(quantum_volatilities, quantum_returns, s=100, color='red', alpha=0.8, label='Quantum Selected')
        
        for i, asset in enumerate(quantum_assets):
            if asset in asset_metrics:
                ax1.annotate(asset, (quantum_volatilities[i], quantum_returns[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Volatility (Risk)', fontsize=12)
    ax1.set_ylabel('Annual Return', fontsize=12)
    ax1.set_title('Risk-Return Profile: All Assets vs Selected', fontsize=14, fontweight='bold')
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
        ax2.set_xlabel('Assets', fontsize=12)
        ax2.set_ylabel('Weight', fontsize=12)
        ax2.set_title('Portfolio Weights Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(list(all_assets), rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmaps(classical_result, quantum_result, assets, evaluation_date, pdf):
    """Create correlation heatmaps."""
    if not classical_result["selected_assets_weights"] or not quantum_result["selected_assets_weights"]:
        return
    
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
        ax1.set_title('Classical Portfolio Asset Correlations', fontsize=14, fontweight='bold')
    
    # Quantum portfolio correlation
    quantum_assets = list(quantum_result["selected_assets_weights"].keys())
    quantum_returns = window_returns[quantum_assets].dropna()
    if len(quantum_returns) > 0:
        quantum_corr = quantum_returns.corr()
        sns.heatmap(quantum_corr, annot=True, cmap='coolwarm', center=0, ax=ax2, 
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax2.set_title('Quantum Portfolio Asset Correlations', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report():
    """Generate comprehensive PDF report."""
    print("Loading input data...")
    input_data = load_input_data()
    
    print("Running classical optimization...")
    classical_result = run_portfolio_optimization(input_data, use_quantum=False)
    
    print("Running quantum optimization...")
    quantum_result = run_portfolio_optimization(input_data, use_quantum=True)
    
    if not classical_result or not quantum_result:
        print("Error: Could not run both optimizations")
        return
    
    # Calculate metrics
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
    
    # Generate PDF report
    report_filename = f"Portfolio_Optimization_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    with PdfPages(report_filename) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        ax.text(0.5, 0.7, 'Portfolio Optimization Analysis Report', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(0.5, 0.6, 'Classical vs Quantum Optimization Comparison', 
                ha='center', va='center', fontsize=16)
        ax.text(0.5, 0.5, f'Generated on: {datetime.now().strftime("%B %d, %Y at %H:%M")}', 
                ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.4, f'Evaluation Date: {evaluation_date}', 
                ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.3, f'Total Assets Available: {len(assets)}', 
                ha='center', va='center', fontsize=12)
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Executive Summary
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        summary_text = f"""
EXECUTIVE SUMMARY

Classical Portfolio Results:
• Selected Assets: {list(classical_result['selected_assets_weights'].keys())}
• Number of Assets: {classical_metrics.get('num_assets', 0)}
• Annual Return: {classical_metrics.get('return', 0):.4f}
• Annual Volatility: {classical_metrics.get('volatility', 0):.4f}
• Sharpe Ratio: {classical_metrics.get('sharpe_ratio', 0):.4f}
• Max Drawdown: {classical_metrics.get('max_drawdown', 0):.4f}

Quantum Portfolio Results:
• Selected Assets: {list(quantum_result['selected_assets_weights'].keys())}
• Number of Assets: {quantum_metrics.get('num_assets', 0)}
• Annual Return: {quantum_metrics.get('return', 0):.4f}
• Annual Volatility: {quantum_metrics.get('volatility', 0):.4f}
• Sharpe Ratio: {quantum_metrics.get('sharpe_ratio', 0):.4f}
• Max Drawdown: {quantum_metrics.get('max_drawdown', 0):.4f}

Key Findings:
• Portfolio Size Difference: {abs(quantum_metrics.get('num_assets', 0) - classical_metrics.get('num_assets', 0))} assets
• Sharpe Ratio Improvement: {((quantum_metrics.get('sharpe_ratio', 0) - classical_metrics.get('sharpe_ratio', 0)) / max(classical_metrics.get('sharpe_ratio', 0), 0.001)) * 100:+.2f}%
• Risk Reduction: {((classical_metrics.get('volatility', 0) - quantum_metrics.get('volatility', 0)) / max(classical_metrics.get('volatility', 0), 0.001)) * 100:+.2f}%
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace')
        ax.set_title('Executive Summary', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create all charts
        print("Creating portfolio composition charts...")
        create_portfolio_composition_chart(classical_result, quantum_result, pdf)
        
        print("Creating performance comparison charts...")
        create_performance_comparison_chart(classical_metrics, quantum_metrics, pdf)
        
        print("Creating risk-return analysis...")
        create_risk_return_analysis(classical_result, quantum_result, assets, evaluation_date, pdf)
        
        print("Creating correlation heatmaps...")
        create_correlation_heatmaps(classical_result, quantum_result, assets, evaluation_date, pdf)
        
        # Methodology page
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        methodology_text = """
METHODOLOGY

1. Data Preparation:
   • Historical price data loaded from input.json
   • 100-day rolling window analysis
   • Daily returns calculation and normalization

2. Classical Optimization:
   • XGBoost regression model for return prediction
   • Industry-based preselection (top 5 per industry)
   • Correlation threshold filtering (max |ρ| = 0.8)
   • Risk-adjusted scoring and weight normalization

3. Quantum Optimization:
   • Variational Quantum Eigensolver (VQE) implementation
   • Binary asset selection (include/exclude)
   • Budget constraint optimization
   • COBYLA classical optimizer for parameter tuning

4. Performance Metrics:
   • Annualized return and volatility
   • Sharpe ratio (risk-adjusted return)
   • Maximum drawdown
   • Portfolio concentration (Herfindahl index)
   • Asset correlation analysis

5. Comparison Framework:
   • Side-by-side portfolio composition
   • Risk-return profile analysis
   • Diversification metrics
   • Correlation structure comparison
        """
        
        ax.text(0.05, 0.95, methodology_text, transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace')
        ax.set_title('Methodology', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nReport generated successfully: {report_filename}")
    print("The report contains:")
    print("• Executive Summary")
    print("• Portfolio Composition Comparison")
    print("• Performance Metrics Analysis")
    print("• Risk-Return Analysis")
    print("• Correlation Heatmaps")
    print("• Methodology Description")

if __name__ == "__main__":
    print("Starting Comprehensive Report Generation...")
    generate_comprehensive_report()
    print("\nReport generation complete!")

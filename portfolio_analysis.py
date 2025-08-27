#!/usr/bin/env python3
"""
Portfolio Analysis and Visualization
Interactive analysis comparing classical and quantum portfolio optimization
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PortfolioAnalyzer:
    """Handles portfolio analysis and visualization."""
    
    def __init__(self, input_file: str = "input.json"):
        self.input_file = input_file
        self.data = self._load_data()
        self.assets = self.data.get("assets", {})
        
    def _load_data(self) -> dict:
        """Load portfolio data from JSON file."""
        try:
            with open(self.input_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {self.input_file} not found")
    
    def run_analysis(self):
        """Run complete portfolio analysis."""
        print("🚀 Starting Portfolio Analysis...")
        
        # Run optimizations
        classical_result = self._run_classical_optimization()
        quantum_result = self._run_quantum_optimization()
        
        # Generate visualizations
        self._create_portfolio_composition_charts(classical_result, quantum_result)
        self._create_performance_comparison_charts(classical_result, quantum_result)
        self._create_risk_return_charts(classical_result, quantum_result)
        self._create_correlation_heatmap()
        
        # Print summary
        self._print_summary_statistics(classical_result, quantum_result)
        
        print("✅ Analysis completed! Check the generated charts.")
    
    def _run_classical_optimization(self) -> dict:
        """Run classical portfolio optimization."""
        print("📊 Running classical optimization...")
        
        # Temporarily disable quantum optimization
        temp_data = self.data.copy()
        temp_data["use_quantum"] = False
        
        import main
        return main.run(temp_data)
    
    def _run_quantum_optimization(self) -> dict:
        """Run quantum portfolio optimization."""
        print("⚛️  Running quantum optimization...")
        
        # Ensure quantum optimization is enabled
        temp_data = self.data.copy()
        temp_data["use_quantum"] = True
        
        import main
        return main.run(temp_data)
    
    def _create_portfolio_composition_charts(self, classical_result: dict, quantum_result: dict):
        """Create portfolio composition comparison charts."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Classical portfolio
        classical_weights = classical_result.get("selected_assets_weights", {})
        if classical_weights:
            assets = list(classical_weights.keys())
            weights = list(classical_weights.values())
            
            ax1.pie(weights, labels=assets, autopct='%1.1f%%', startangle=90)
            ax1.set_title("Classical Portfolio Composition", fontsize=14, fontweight='bold')
        
        # Quantum portfolio
        quantum_weights = quantum_result.get("selected_assets_weights", {})
        if quantum_weights:
            assets = list(quantum_weights.keys())
            weights = list(quantum_weights.values())
            
            ax2.pie(weights, labels=assets, autopct='%1.1f%%', startangle=90)
            ax2.set_title("Quantum Portfolio Composition", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("portfolio_composition.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_performance_comparison_charts(self, classical_result: dict, quantum_result: dict):
        """Create performance comparison charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Prepare data
        classical_weights = classical_result.get("selected_assets_weights", {})
        quantum_weights = quantum_result.get("selected_assets_weights", {})
        
        if classical_weights and quantum_weights:
            # Weight comparison
            all_assets = set(classical_weights.keys()) | set(quantum_weights.keys())
            classical_data = [classical_weights.get(asset, 0) for asset in all_assets]
            quantum_data = [quantum_weights.get(asset, 0) for asset in all_assets]
            
            x = np.arange(len(all_assets))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, classical_data, width, label='Classical', alpha=0.8)
            axes[0, 0].bar(x + width/2, quantum_data, width, label='Quantum', alpha=0.8)
            axes[0, 0].set_xlabel('Assets')
            axes[0, 0].set_ylabel('Weights')
            axes[0, 0].set_title('Asset Weight Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(all_assets, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Portfolio size comparison
            portfolio_sizes = ['Classical', 'Quantum']
            asset_counts = [len(classical_weights), len(quantum_weights)]
            
            axes[0, 1].bar(portfolio_sizes, asset_counts, color=['skyblue', 'lightcoral'])
            axes[0, 1].set_ylabel('Number of Assets')
            axes[0, 1].set_title('Portfolio Size Comparison')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Weight distribution
            axes[1, 0].hist(list(classical_weights.values()), bins=10, alpha=0.7, label='Classical', color='skyblue')
            axes[1, 0].hist(list(quantum_weights.values()), bins=10, alpha=0.7, label='Quantum', color='lightcoral')
            axes[1, 0].set_xlabel('Weight Values')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Weight Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Diversification comparison
            classical_diversification = 1 - sum(w**2 for w in classical_weights.values())
            quantum_diversification = 1 - sum(w**2 for w in quantum_weights.values())
            
            diversification_data = [classical_diversification, quantum_diversification]
            axes[1, 1].bar(portfolio_sizes, diversification_data, color=['skyblue', 'lightcoral'])
            axes[1, 1].set_ylabel('Diversification Index')
            axes[1, 1].set_title('Portfolio Diversification')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_risk_return_charts(self, classical_result: dict, quantum_result: dict):
        """Create risk-return analysis charts."""
        if not self.assets:
            return
        
        # Calculate historical returns
        prices_df = self._build_price_dataframe()
        returns_df = prices_df.pct_change().dropna()
        
        classical_weights = classical_result.get("selected_assets_weights", {})
        quantum_weights = quantum_result.get("selected_assets_weights", {})
        
        if classical_weights and quantum_weights:
            # Portfolio returns
            classical_portfolio_returns = self._calculate_portfolio_returns(returns_df, classical_weights)
            quantum_portfolio_returns = self._calculate_portfolio_returns(returns_df, quantum_weights)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Cumulative returns
            classical_cumulative = (1 + classical_portfolio_returns).cumprod()
            quantum_cumulative = (1 + quantum_portfolio_returns).cumprod()
            
            axes[0, 0].plot(classical_cumulative.index, classical_cumulative.values, label='Classical', linewidth=2)
            axes[0, 0].plot(quantum_cumulative.index, quantum_cumulative.values, label='Quantum', linewidth=2)
            axes[0, 0].set_title('Cumulative Portfolio Returns')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Cumulative Return')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Return distribution
            axes[0, 1].hist(classical_portfolio_returns, bins=30, alpha=0.7, label='Classical', color='skyblue')
            axes[0, 1].hist(quantum_portfolio_returns, bins=30, alpha=0.7, label='Quantum', color='lightcoral')
            axes[0, 1].set_xlabel('Daily Returns')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Return Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Risk-return scatter
            classical_risk = classical_portfolio_returns.std() * np.sqrt(252)
            classical_return = classical_portfolio_returns.mean() * 252
            quantum_risk = quantum_portfolio_returns.std() * np.sqrt(252)
            quantum_return = quantum_portfolio_returns.mean() * 252
            
            axes[1, 0].scatter(classical_risk, classical_return, s=200, label='Classical', color='skyblue', alpha=0.8)
            axes[1, 0].scatter(quantum_risk, quantum_return, s=200, label='Quantum', color='lightcoral', alpha=0.8)
            axes[1, 0].set_xlabel('Annualized Risk (Volatility)')
            axes[1, 0].set_ylabel('Annualized Return')
            axes[1, 0].set_title('Risk-Return Profile')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Sharpe ratio comparison
            classical_sharpe = classical_return / classical_risk if classical_risk > 0 else 0
            quantum_sharpe = quantum_return / quantum_risk if quantum_risk > 0 else 0
            
            sharpe_data = [classical_sharpe, quantum_sharpe]
            axes[1, 1].bar(['Classical', 'Quantum'], sharpe_data, color=['skyblue', 'lightcoral'])
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].set_title('Risk-Adjusted Returns')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("risk_return_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def _create_correlation_heatmap(self):
        """Create asset correlation heatmap."""
        if not self.assets:
            return
        
        prices_df = self._build_price_dataframe()
        returns_df = prices_df.pct_change().dropna()
        
        if returns_df.empty:
            return
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Asset Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _build_price_dataframe(self) -> pd.DataFrame:
        """Build price DataFrame from asset data."""
        series_list = []
        for ticker, info in self.assets.items():
            price_series = pd.Series(info["history"], name=ticker)
            series_list.append(price_series)
        
        df = pd.concat(series_list, axis=1).sort_index().astype(float)
        df.index = pd.to_datetime(df.index)
        return df
    
    def _calculate_portfolio_returns(self, returns_df: pd.DataFrame, weights: dict) -> pd.Series:
        """Calculate portfolio returns given weights."""
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        
        for asset, weight in weights.items():
            if asset in returns_df.columns:
                portfolio_returns += weight * returns_df[asset]
        
        return portfolio_returns
    
    def _print_summary_statistics(self, classical_result: dict, quantum_result: dict):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("PORTFOLIO ANALYSIS SUMMARY")
        print("="*80)
        
        classical_weights = classical_result.get("selected_assets_weights", {})
        quantum_weights = quantum_result.get("selected_assets_weights", {})
        
        print(f"\n📊 Portfolio Composition:")
        print(f"   Classical: {len(classical_weights)} assets")
        print(f"   Quantum:   {len(quantum_weights)} assets")
        
        if classical_weights and quantum_weights:
            print(f"\n⚖️  Weight Statistics:")
            classical_weights_list = list(classical_weights.values())
            quantum_weights_list = list(quantum_weights.values())
            
            print(f"   Classical - Min: {min(classical_weights_list):.4f}, Max: {max(classical_weights_list):.4f}")
            print(f"   Quantum   - Min: {min(quantum_weights_list):.4f}, Max: {max(quantum_weights_list):.4f}")
            
            print(f"\n🎯 Diversification:")
            classical_div = 1 - sum(w**2 for w in classical_weights_list)
            quantum_div = 1 - sum(w**2 for w in quantum_weights_list)
            print(f"   Classical: {classical_div:.4f}")
            print(f"   Quantum:   {quantum_div:.4f}")
        
        print(f"\n📈 Generated Charts:")
        print("   - portfolio_composition.png")
        print("   - performance_comparison.png")
        print("   - risk_return_analysis.png")
        print("   - correlation_heatmap.png")


def main():
    """Main execution function."""
    try:
        analyzer = PortfolioAnalyzer()
        analyzer.run_analysis()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    main()

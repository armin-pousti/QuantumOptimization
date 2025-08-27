#!/usr/bin/env python3
"""
Professional PDF Report Generator for Portfolio Optimization
Creates comprehensive reports with executive summary and analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style for professional appearance
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ReportGenerator:
    """Handles generation of professional PDF reports."""
    
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
    
    def generate_report(self, output_file: str = "portfolio_optimization_report.pdf"):
        """Generate comprehensive PDF report."""
        print("📄 Generating professional PDF report...")
        
        # Run optimizations
        classical_result = self._run_classical_optimization()
        quantum_result = self._run_quantum_optimization()
        
        # Generate report
        with PdfPages(output_file) as pdf:
            self._create_title_page(pdf)
            self._create_executive_summary(pdf, classical_result, quantum_result)
            self._create_methodology_section(pdf)
            self._create_portfolio_composition_charts(pdf, classical_result, quantum_result)
            self._create_performance_comparison_charts(pdf, classical_result, quantum_result)
            self._create_risk_return_charts(pdf, classical_result, quantum_result)
            self._create_correlation_heatmap(pdf)
            self._create_conclusion_section(pdf, classical_result, quantum_result)
        
        print(f"✅ Report generated successfully: {output_file}")
    
    def _run_classical_optimization(self) -> dict:
        """Run classical portfolio optimization."""
        print("📊 Running classical optimization...")
        
        temp_data = self.data.copy()
        temp_data["use_quantum"] = False
        
        import main
        return main.run(temp_data)
    
    def _run_quantum_optimization(self) -> dict:
        """Run quantum portfolio optimization."""
        print("⚛️  Running quantum optimization...")
        
        temp_data = self.data.copy()
        temp_data["use_quantum"] = True
        
        import main
        return main.run(temp_data)
    
    def _create_title_page(self, pdf):
        """Create professional title page."""
        fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 size
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.7, 'QUANTUM PORTFOLIO OPTIMIZATION', 
                fontsize=24, fontweight='bold', ha='center', va='center',
                transform=ax.transAxes)
        
        # Subtitle
        ax.text(0.5, 0.6, 'A Hybrid Quantum-Classical Approach to Financial Portfolio Optimization',
                fontsize=16, ha='center', va='center', transform=ax.transAxes)
        
        # Date
        ax.text(0.5, 0.4, f'Generated on: {datetime.now().strftime("%B %d, %Y")}',
                fontsize=12, ha='center', va='center', transform=ax.transAxes)
        
        # Company/Project info
        ax.text(0.5, 0.3, 'Portfolio Optimization Engine',
                fontsize=14, ha='center', va='center', transform=ax.transAxes)
        
        # Technical details
        ax.text(0.5, 0.2, f'Assets Analyzed: {len(self.assets)} | Target Portfolio Size: {self.data.get("num_assets", "N/A")}',
                fontsize=10, ha='center', va='center', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_executive_summary(self, pdf, classical_result: dict, quantum_result: dict):
        """Create executive summary page."""
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        
        # Title
        ax.text(0.1, 0.95, 'EXECUTIVE SUMMARY', fontsize=18, fontweight='bold')
        
        # Summary text
        summary_text = [
            "This report presents a comprehensive analysis of portfolio optimization using both classical",
            "and quantum computing approaches. The analysis demonstrates the potential advantages of",
            "quantum algorithms in financial portfolio selection and weighting.",
            "",
            "Key Findings:",
            f"• Classical optimization selected {len(classical_result.get('selected_assets_weights', {}))} assets",
            f"• Quantum optimization selected {len(quantum_result.get('selected_assets_weights', {}))} assets",
            "• Both approaches provide diversified portfolio solutions",
            "• Quantum algorithms offer alternative optimization strategies",
            "",
            "Methodology:",
            "• Asset preselection using machine learning models",
            "• Industry-based diversification constraints",
            "• Correlation-based risk management",
            "• VQE (Variational Quantum Eigensolver) for quantum optimization",
            "",
            "The analysis reveals insights into portfolio construction strategies and demonstrates",
            "the practical application of quantum computing in financial optimization."
        ]
        
        y_pos = 0.85
        for line in summary_text:
            ax.text(0.1, y_pos, line, fontsize=11, va='top')
            y_pos -= 0.05
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_methodology_section(self, pdf):
        """Create methodology section."""
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        
        # Title
        ax.text(0.1, 0.95, 'METHODOLOGY', fontsize=18, fontweight='bold')
        
        # Methodology text
        methodology_text = [
            "Portfolio Optimization Pipeline:",
            "",
            "1. Data Collection & Preprocessing:",
            "   • Historical price data from Yahoo Finance",
            "   • 200-day lookback period for analysis",
            "   • Data quality validation and cleaning",
            "",
            "2. Asset Preselection:",
            "   • XGBoost ML model for return prediction",
            "   • Financial metrics calculation (Sharpe ratio, CVaR, momentum)",
            "   • Industry-based diversification constraints",
            "   • Top 5 assets per industry selection",
            "",
            "3. Portfolio Construction:",
            "   • Correlation-based diversification",
            "   • Maximum correlation threshold: 0.8",
            "   • Target portfolio size optimization",
            "",
            "4. Weight Optimization:",
            "   • Classical: Score-based normalization",
            "   • Quantum: VQE with TwoLocal ansatz",
            "   • COBYLA optimizer with 500 iterations",
            "",
            "5. Risk Management:",
            "   • Conditional Value at Risk (CVaR)",
            "   • Downside volatility analysis",
            "   • Maximum drawdown calculation",
            "   • Trend strength assessment"
        ]
        
        y_pos = 0.85
        for line in methodology_text:
            ax.text(0.1, y_pos, line, fontsize=10, va='top')
            y_pos -= 0.04
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_portfolio_composition_charts(self, pdf, classical_result: dict, quantum_result: dict):
        """Create portfolio composition charts for PDF."""
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
        
        plt.suptitle("Portfolio Composition Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_performance_comparison_charts(self, pdf, classical_result: dict, quantum_result: dict):
        """Create performance comparison charts for PDF."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
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
        
        plt.suptitle("Performance Comparison Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_risk_return_charts(self, pdf, classical_result: dict, quantum_result: dict):
        """Create risk-return analysis charts for PDF."""
        if not self.assets:
            return
        
        prices_df = self._build_price_dataframe()
        returns_df = prices_df.pct_change().dropna()
        
        classical_weights = classical_result.get("selected_assets_weights", {})
        quantum_weights = quantum_result.get("selected_assets_weights", {})
        
        if classical_weights and quantum_weights:
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
            
            plt.suptitle("Risk-Return Analysis", fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    def _create_correlation_heatmap(self, pdf):
        """Create correlation heatmap for PDF."""
        if not self.assets:
            return
        
        prices_df = self._build_price_dataframe()
        returns_df = prices_df.pct_change().dropna()
        
        if returns_df.empty:
            return
        
        corr_matrix = returns_df.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Asset Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(plt.gcf(), bbox_inches='tight')
        plt.close()
    
    def _create_conclusion_section(self, pdf, classical_result: dict, quantum_result: dict):
        """Create conclusion section."""
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        
        # Title
        ax.text(0.1, 0.95, 'CONCLUSION & RECOMMENDATIONS', fontsize=18, fontweight='bold')
        
        classical_weights = classical_result.get("selected_assets_weights", {})
        quantum_weights = quantum_result.get("selected_assets_weights", {})
        
        # Conclusion text
        conclusion_text = [
            "Analysis Summary:",
            "",
            f"• Classical optimization successfully selected {len(classical_weights)} assets",
            f"• Quantum optimization successfully selected {len(quantum_weights)} assets",
            "• Both approaches demonstrate effective diversification strategies",
            "",
            "Key Insights:",
            "• Quantum algorithms provide alternative optimization pathways",
            "• Machine learning enhances asset preselection accuracy",
            "• Industry-based diversification improves portfolio stability",
            "• Correlation analysis supports risk management",
            "",
            "Recommendations:",
            "• Consider hybrid approaches combining classical and quantum methods",
            "• Implement regular portfolio rebalancing based on market conditions",
            "• Monitor correlation changes for dynamic risk management",
            "• Explore additional quantum algorithms (QAOA, VQE variants)",
            "",
            "Future Work:",
            "• Extend analysis to larger asset universes",
            "• Implement real-time portfolio optimization",
            "• Explore quantum advantage in larger-scale problems",
            "• Develop hybrid classical-quantum optimization frameworks"
        ]
        
        y_pos = 0.85
        for line in conclusion_text:
            ax.text(0.1, y_pos, line, fontsize=10, va='top')
            y_pos -= 0.04
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
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


def main():
    """Main execution function."""
    try:
        generator = ReportGenerator()
        generator.generate_report()
        print("🎉 Report generation completed successfully!")
    except Exception as e:
        print(f"❌ Report generation failed: {e}")


if __name__ == "__main__":
    main()

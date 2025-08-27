#!/usr/bin/env python3
"""
Data Generation Script for Portfolio Optimization
Creates input.json with fresh stock data from Yahoo Finance
"""

import yfinance as yf
import json
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataGenerator:
    """Handles generation of portfolio data from Yahoo Finance."""
    
    def __init__(self):
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=200)
    
    def create_comprehensive_portfolio(self) -> dict:
        """Create comprehensive portfolio with diverse sectors."""
        print("🚀 Creating comprehensive portfolio...")
        
        config = {
            "evaluation_date": "2024-06-01",
            "num_assets": 10,
            "use_quantum": True,
            "solver_params": {},
            "extra_arguments": {}
        }
        
        stock_list = [
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
        
        assets = self._fetch_stock_data(stock_list)
        return self._create_input_structure(config, assets)
    
    def create_focused_portfolio(self) -> dict:
        """Create focused portfolio with balanced sectors."""
        print("🎯 Creating focused portfolio...")
        
        config = {
            "evaluation_date": "2024-06-01",
            "num_assets": 8,
            "use_quantum": True,
            "solver_params": {},
            "extra_arguments": {}
        }
        
        focused_stocks = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
            "Finance": ["JPM", "BAC", "GS", "MS", "WFC"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB"],
            "Consumer": ["KO", "PEP", "WMT", "HD", "MCD"]
        }
        
        assets = self._fetch_focused_data(focused_stocks)
        return self._create_input_structure(config, assets)
    
    def _fetch_stock_data(self, stock_list: list) -> dict:
        """Fetch data for a list of stocks."""
        print(f"📊 Fetching data for {len(stock_list)} stocks...")
        print("=" * 60)
        
        assets = {}
        successful_stocks = 0
        
        for i, ticker in enumerate(stock_list, 1):
            try:
                print(f"[{i:2d}/{len(stock_list)}] Fetching {ticker:6s}...", end=" ")
                
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date)
                
                if not hist.empty and len(hist) > 50:
                    info = stock.info
                    assets[ticker] = {
                        "industry": info.get('industry', 'Unknown'),
                        "sector": info.get('sector', 'Unknown'),
                        "market_cap": info.get('marketCap', 0),
                        "history": {str(date): float(price) for date, price in hist['Close'].to_dict().items()}
                    }
                    print(f"✓ {len(hist)} days")
                    successful_stocks += 1
                else:
                    print("✗ Insufficient data")
                    
            except Exception:
                print("✗ Error")
                continue
        
        print("=" * 60)
        print(f"✅ Successfully fetched data for {successful_stocks} stocks")
        return assets
    
    def _fetch_focused_data(self, focused_stocks: dict) -> dict:
        """Fetch data for focused sector allocation."""
        assets = {}
        total_stocks = sum(len(stocks) for stocks in focused_stocks.values())
        current_stock = 0
        
        for sector, stocks in focused_stocks.items():
            print(f"\n📊 Fetching {sector} stocks...")
            for ticker in stocks:
                current_stock += 1
                try:
                    print(f"[{current_stock:2d}/{total_stocks}] {ticker:6s}...", end=" ")
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=self.start_date, end=self.end_date)
                    
                    if not hist.empty and len(hist) > 50:
                        info = stock.info
                        assets[ticker] = {
                            "industry": info.get('industry', sector),
                            "sector": sector,
                            "market_cap": info.get('marketCap', 0),
                            "history": {str(date): float(price) for date, price in hist['Close'].to_dict().items()}
                        }
                        print(f"✓ {len(hist)} days")
                    else:
                        print("✗ Insufficient data")
                        
                except Exception:
                    print("✗ Error")
                    continue
        
        return assets
    
    def _create_input_structure(self, config: dict, assets: dict) -> dict:
        """Create the final input data structure."""
        input_data = {
            "evaluation_date": config["evaluation_date"],
            "num_assets": config["num_assets"],
            "use_quantum": config["use_quantum"],
            "solver_params": config["solver_params"],
            "extra_arguments": config["extra_arguments"],
            "assets": assets
        }
        
        # Save to file
        with open("input.json", "w") as f:
            json.dump(input_data, f, indent=2, default=str)
        
        self._print_summary(input_data, assets)
        return input_data
    
    def _print_summary(self, input_data: dict, assets: dict):
        """Print summary of generated data."""
        print(f"\n💾 Saved to input.json")
        print(f"📅 Data range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"📊 Portfolio target: {input_data['num_assets']} assets")
        print(f"⚛️  Quantum optimization: {'Enabled' if input_data['use_quantum'] else 'Disabled'}")
        
        if assets:
            industries = [asset.get('industry', 'Unknown') for asset in assets.values()]
            sectors = [asset.get('sector', 'Unknown') for asset in assets.values()]
            
            print(f"\n📈 Portfolio Diversity:")
            print(f"   Industries: {len(set(industries))} unique")
            print(f"   Sectors: {len(set(sectors))} unique")
            
            sample_stocks = list(assets.keys())[:5]
            print(f"   Sample stocks: {', '.join(sample_stocks)}")


def main():
    """Main execution function."""
    print("🚀 Portfolio Data Generator")
    print("=" * 50)
    
    generator = DataGenerator()
    
    print("\nChoose an option:")
    print("1. Create comprehensive portfolio (many stocks, diverse sectors)")
    print("2. Create focused portfolio (balanced sectors, fewer stocks)")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == "2":
        generator.create_focused_portfolio()
    else:
        generator.create_comprehensive_portfolio()
    
    print("\n🎉 Data generation complete!")
    print("You can now run:")
    print("  python app.py                    # Basic portfolio optimization")
    print("  python portfolio_analysis.py     # Analysis and charts")
    print("  python generate_report.py        # PDF report")


if __name__ == "__main__":
    main()

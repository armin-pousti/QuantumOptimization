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

def create_input_json():
    """Create input.json with fresh stock data."""
    
    # Configuration
    config = {
        "evaluation_date": "2024-06-01",  # Use a date within the data range
        "num_assets": 10,
        "use_quantum": True,
        "solver_params": {},
        "extra_arguments": {}
    }
    
    # Define stock list with diverse sectors
    stock_list = [
        # Technology
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
        
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK",
        
        # Healthcare
        "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR",
        
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "EOG", "PSX", "VLO",
        
        # Consumer
        "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT",
        
        # Industrial
        "BA", "CAT", "GE", "MMM", "HON", "LMT", "RTX", "UNP"
    ]
    
    print(f"🚀 Fetching fresh data for {len(stock_list)} stocks...")
    print("=" * 60)
    
    # Fetch historical data
    assets = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)  # 200 days of data
    
    successful_stocks = 0
    
    for i, ticker in enumerate(stock_list, 1):
        try:
            print(f"[{i:2d}/{len(stock_list)}] Fetching {ticker:6s}...", end=" ")
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(start=start_date, end=end_date)
            
            if not hist.empty and len(hist) > 50:  # Ensure sufficient data
                # Get company info
                info = stock.info
                
                # Create asset entry
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
                
        except Exception as e:
            print(f"✗ Error")
            continue
    
    print("=" * 60)
    print(f"✅ Successfully fetched data for {successful_stocks} stocks")
    
    # Create final structure
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
    
    print(f"💾 Saved to input.json")
    print(f"📅 Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📊 Portfolio target: {config['num_assets']} assets")
    print(f"⚛️  Quantum optimization: {'Enabled' if config['use_quantum'] else 'Disabled'}")
    
    # Show some statistics
    if assets:
        industries = [asset.get('industry', 'Unknown') for asset in assets.values()]
        sectors = [asset.get('sector', 'Unknown') for asset in assets.values()]
        
        print(f"\n📈 Portfolio Diversity:")
        print(f"   Industries: {len(set(industries))} unique")
        print(f"   Sectors: {len(set(sectors))} unique")
        
        # Show sample stocks
        sample_stocks = list(assets.keys())[:5]
        print(f"   Sample stocks: {', '.join(sample_stocks)}")
    
    return input_data

def create_focused_portfolio():
    """Create input.json with focused sector allocation."""
    
    # Focus on specific sectors for better comparison
    focused_stocks = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        "Finance": ["JPM", "BAC", "GS", "MS", "WFC"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB"],
        "Consumer": ["KO", "PEP", "WMT", "HD", "MCD"]
    }
    
    print("🎯 Creating focused portfolio with balanced sectors...")
    
    # Configuration
    config = {
        "evaluation_date": "2024-06-01",  # Use a date within the data range
        "num_assets": 8,  # Will select from multiple sectors
        "use_quantum": True,
        "solver_params": {},
        "extra_arguments": {}
    }
    
    # Fetch data for focused stocks
    assets = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)
    
    total_stocks = sum(len(stocks) for stocks in focused_stocks.values())
    current_stock = 0
    
    for sector, stocks in focused_stocks.items():
        print(f"\n📊 Fetching {sector} stocks...")
        for ticker in stocks:
            current_stock += 1
            try:
                print(f"[{current_stock:2d}/{total_stocks}] {ticker:6s}...", end=" ")
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
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
                    
            except Exception as e:
                print(f"✗ Error")
                continue
    
    # Create final structure
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
    
    print(f"\n✅ Created focused input.json with {len(assets)} assets")
    return input_data

if __name__ == "__main__":
    print("🚀 Portfolio Data Generator")
    print("=" * 50)
    
    print("\nChoose an option:")
    print("1. Create comprehensive portfolio (many stocks, diverse sectors)")
    print("2. Create focused portfolio (balanced sectors, fewer stocks)")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == "2":
        create_focused_portfolio()
    else:
        create_input_json()
    
    print("\n🎉 Data generation complete!")
    print("You can now run:")
    print("  python app.py                    # Basic portfolio optimization")
    print("  python portfolio_analysis.py     # Analysis and charts")
    print("  python generate_report.py        # PDF report")

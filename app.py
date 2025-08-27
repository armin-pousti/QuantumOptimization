#!/usr/bin/env python3
"""
Quantum Portfolio Optimization Application
Main application entry point for portfolio optimization
"""

import json
import main


def main():
    """Main application execution."""
    # Load configuration
    input_file_name = "input.json"
    
    try:
        with open(input_file_name, "r") as f:
            input_data = json.load(f)
        
        # Extract configuration
        data = input_data
        
        # Optional parameters
        solver_params = input_data.get("solver_params", {})
        extra_arguments = input_data.get("extra_arguments", {})
        
        # Run portfolio optimization
        result = main.run(data, solver_params, extra_arguments)
        
        # Display results
        print("\n" + "="*60)
        print("PORTFOLIO OPTIMIZATION RESULTS")
        print("="*60)
        
        if result["num_selected_assets"] > 0:
            print(f"✅ Successfully selected {result['num_selected_assets']} assets")
            print("\nAsset Allocation:")
            print("-" * 40)
            
            for asset, weight in result["selected_assets_weights"].items():
                print(f"{asset:8s}: {weight:6.2%}")
                
            print("-" * 40)
            print(f"{'Total':8s}: {sum(result['selected_assets_weights'].values()):6.2%}")
        else:
            print("❌ No assets were selected")
            print("This may indicate insufficient data or optimization constraints")
        
        print("\n" + "="*60)
        
    except FileNotFoundError:
        print(f"❌ Error: {input_file_name} not found")
        print("Please run generate_data.py first to create the input file")
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {input_file_name}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
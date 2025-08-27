#!/usr/bin/env python3
"""
Main Entry Point for Quantum Portfolio Optimization
Handles data flow and configuration between app.py and portfolio.py
"""

import portfolio
from datetime import datetime


def run(input_data: dict, solver_params: dict = None, extra_arguments: dict = None) -> dict:
    """
    Main execution function that prepares data and calls portfolio optimization.
    
    Args:
        input_data: Portfolio configuration and asset data
        solver_params: Solver-specific parameters (optional)
        extra_arguments: Additional arguments (optional)
    
    Returns:
        Portfolio optimization results
    """
    # Handle evaluation date
    if 'evaluation_date' not in input_data:
        if 'evaluation_date' in extra_arguments:
            input_data['evaluation_date'] = extra_arguments['evaluation_date']
        elif 'from' in input_data:
            input_data['evaluation_date'] = input_data['from']
        else:
            input_data['evaluation_date'] = datetime.now().strftime("%Y-%m-%d")
    
    return portfolio.run(input_data)


if __name__ == "__main__":
    # Standalone testing
    try:
        import json
        with open("input.json", "r") as f:
            data = json.load(f)
        
        result = run(data)
        print(f"Portfolio optimization completed successfully")
        print(f"Selected assets: {result['num_selected_assets']}")
        
    except Exception as e:
        print(f"Execution failed: {e}")

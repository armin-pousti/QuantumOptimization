# Quantum Portfolio Optimization Engine

A hybrid quantum-classical approach to financial portfolio optimization using VQE algorithms and machine learning.

## 🚀 Overview

This project implements a sophisticated portfolio optimization system that combines:
- **Machine Learning**: XGBoost regression for asset return prediction
- **Quantum Computing**: VQE (Variational Quantum Eigensolver) for portfolio weight optimization
- **Classical Methods**: Traditional financial metrics and diversification strategies
- **Risk Management**: Advanced risk metrics including CVaR, downside volatility, and correlation analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  ML Model      │───▶│  Optimization   │
│   (input.json)  │    │  Training      │    │  Engine         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Trained Model │    │  Portfolio      │
                       │  (joblib)      │    │  Results        │
                       └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
Quantum-Portfolio-Optimization/
├── app.py                 # Main application entry point
├── main.py               # Core execution logic
├── portfolio.py          # Portfolio optimization engine
├── model_training.py     # ML model training
├── generate_data.py      # Data generation from Yahoo Finance
├── portfolio_analysis.py # Interactive analysis and charts
├── generate_report.py    # Professional PDF report generation
├── input.json           # Configuration and asset data
├── trained_model.joblib # Trained ML model
└── requirements.txt      # Dependencies
```

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Quantum-Portfolio-Optimization
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import qiskit, pandas, xgboost; print('All dependencies installed successfully!')"
   ```

## 🚀 Quick Start

### 1. Generate Data
```bash
python generate_data.py
```
Choose between comprehensive or focused portfolio options.

### 2. Train ML Model
```bash
python model_training.py
```
Trains XGBoost model for asset return prediction.

### 3. Run Portfolio Optimization
```bash
python app.py
```
Executes portfolio optimization with current configuration.

### 4. Generate Analysis
```bash
python portfolio_analysis.py    # Interactive charts
python generate_report.py       # Professional PDF report
```

## 📊 Key Features

### Portfolio Optimization
- **Asset Preselection**: ML-based scoring with industry diversification
- **Correlation Filtering**: Maximum correlation threshold of 0.8
- **Weight Optimization**: Both classical and quantum approaches
- **Risk Management**: CVaR, downside volatility, maximum drawdown

### Quantum Computing
- **VQE Algorithm**: Variational Quantum Eigensolver implementation
- **TwoLocal Ansatz**: Ry rotations with CZ entanglement
- **COBYLA Optimizer**: Classical optimization for quantum parameters
- **Portfolio Constraints**: Budget and risk factor constraints

### Machine Learning
- **XGBoost Regression**: Return prediction model
- **Feature Engineering**: 6 comprehensive financial metrics
- **Model Persistence**: Joblib serialization for reuse

### Analysis & Reporting
- **Interactive Charts**: Matplotlib and Seaborn visualizations
- **Performance Metrics**: Sharpe ratio, risk-return analysis
- **PDF Reports**: Professional reports with executive summary
- **Comparison Analysis**: Classical vs. quantum optimization

## ⚙️ Configuration

Edit `input.json` to customize:
- `evaluation_date`: Analysis date
- `num_assets`: Target portfolio size
- `use_quantum`: Enable/disable quantum optimization
- `assets`: Stock tickers and historical data

## 🔬 Technical Details

### Financial Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **CVaR**: Conditional Value at Risk (95% level)
- **Downside Volatility**: Negative return volatility
- **Momentum**: Price trend strength
- **Maximum Drawdown**: Peak-to-trough decline
- **Trend R²**: Linear trend strength

### Quantum Implementation
- **Qiskit Finance**: Portfolio optimization problems
- **Qiskit Optimization**: Quadratic program solving
- **Qiskit Aer**: Quantum simulation backend
- **TwoLocal Circuit**: 3 repetitions, full entanglement

### Data Sources
- **Yahoo Finance**: Real-time stock data via `yfinance`
- **Historical Data**: 200-day lookback period
- **Data Validation**: Quality checks and filtering

## 📈 Usage Examples

### Basic Optimization
```python
import main
import json

# Load configuration
with open("input.json", "r") as f:
    data = json.load(f)

# Run optimization
result = main.run(data)
print(f"Selected {result['num_selected_assets']} assets")
```

### Custom Analysis
```python
from portfolio_analysis import PortfolioAnalyzer

# Create analyzer
analyzer = PortfolioAnalyzer("input.json")

# Run analysis
analyzer.run_analysis()
```

### Report Generation
```python
from generate_report import ReportGenerator

# Generate report
generator = ReportGenerator("input.json")
generator.generate_report("custom_report.pdf")
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Performance Tests
```bash
python -m pytest tests/performance/
```

## 📊 Performance Metrics

- **Training Time**: ~2-5 minutes for 50+ stocks
- **Optimization Time**: ~30 seconds (classical), ~2-5 minutes (quantum)
- **Memory Usage**: ~500MB for typical portfolios
- **Accuracy**: R² > 0.7 for return prediction

## 🔮 Future Enhancements

- **Real-time Optimization**: Live market data integration
- **Additional Quantum Algorithms**: QAOA, Grover's algorithm
- **Multi-objective Optimization**: Risk-return trade-off analysis
- **Alternative Data Sources**: News sentiment, economic indicators
- **Cloud Deployment**: AWS, Azure, or Google Cloud integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qiskit Team**: Quantum computing framework
- **Yahoo Finance**: Financial data API
- **XGBoost**: Machine learning library
- **Financial Research Community**: Portfolio optimization methodologies

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example notebooks

---

**Note**: This is a research and educational project. Results should not be used for actual investment decisions without proper financial advice.

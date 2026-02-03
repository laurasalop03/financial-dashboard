# 📈 Financial Data Analysis Dashboard

A comprehensive interactive dashboard built with Python and Streamlit for stock market analysis, quantitative finance, and algorithmic trading simulation.

This tool allows users to visualize financial data, perform technical analysis, calculate risk metrics, and forecast future prices using AI.

## 🚀 Key Features
### 1. General Summary

Real-time Data: Fetches historical data using Yahoo Finance (yfinance).

Performance Metrics: Calculates Total Return, Annualized Volatility, and current prices.

Visualizations: Relative performance comparison and interactive Candlestick charts.

### 2. Technical Analysis & Strategy

Indicators: Moving Averages (SMA 50/200) and Drawdown visualization.

Algorithmic Trading: Automatic detection of Golden Cross (Bullish) and Death Cross (Bearish) signals.

Backtesting: Simulates the SMA strategy vs. a "Buy & Hold" benchmark with initial capital management.

### 3. Risk & Statistics

Market Exposure: Calculates Alpha, Beta, and VaR (95%) against the S&P 500.

Distribution Analysis: Compares actual return distribution vs. normal distribution.

Correlation Matrix: Heatmap to analyze asset diversification.

### 4. AI Forecasting

Time Series Prediction: Uses Facebook Prophet to predict future prices with confidence intervals.

## 🛠️ Tech Stack

Core: Python 3.10+

Frontend: Streamlit

Data Processing: Pandas, NumPy, Scipy

Visualization: Plotly (Interactive charts)

Financial Data: yfinance

Machine Learning: Prophet

## 💻 Installation & Usage

Clone the repository:

```bash
git clone https://github.com/laurasalop03/financial-dashboard.git
cd financial-dashboard
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run dashboard.py
```

## ⚠️ Disclaimer

This project is for educational and informational purposes only. It does not constitute financial advice. Past performance is not indicative of future results.

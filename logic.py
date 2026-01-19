import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm, linregress
import streamlit as st

@st.cache_data
def get_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Downloads historical financial data for given tickers and computes log returns.
    """
    if not tickers:
        return pd.DataFrame()
    
    # Download historical data
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Calculate log returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    log_returns.columns = pd.MultiIndex.from_product([ ['log_return'], log_returns.columns ])
    df = pd.concat([df, log_returns], axis=1)
    df.dropna(inplace=True)

    return df


@st.cache_data
def get_benchmark_data(start: str, end: str) -> pd.DataFrame:
    """
    Downloads historical data for S&P 500 and computes log returns.
    """
    bench = yf.download('^GSPC', start=start, end=end, auto_adjust=True)

    # Calculate S&P 500 log returns
    bench['log_return'] = np.log(bench['Close'] / bench['Close'].shift(1))
    bench.dropna(inplace=True)

    return bench


def calculate_metrics(df: pd.DataFrame, ticker: str) -> dict:
    """
    Calculates financial metrics including total return, volatility and last price.
    """
    prices = df['Close',ticker]
    log_ret = df['log_return',ticker]

    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    volatility = log_ret.std() * np.sqrt(252)
    last_price = prices.iloc[-1]

    return {
        'Ticker': ticker, 
        'Total Return': f"{total_return:.2%}",
        'Volatility (Ann.)': f"{volatility:.2%}",
        "Last Price": f"${last_price:.2f}"
    }


def calculate_technical_indicators(series_price: pd.Series) -> pd.DataFrame:
    """
    Calculates, given a series of prices, Drawdown and Moving Averages.
    """
    # Drawdown
    rolling_max = series_price.cummax()
    drawdown = (series_price - rolling_max) / rolling_max
    
    # SMAs
    sma_50 = series_price.rolling(window=50).mean()
    sma_200 = series_price.rolling(window=200).mean()
    
    tech_df = pd.DataFrame({
        'Drawdown': drawdown,
        'SMA_50': sma_50,
        'SMA_200': sma_200
    })
    return tech_df


def calculate_risk_stats(asset_returns: pd.Series, benchmark_returns: pd.Series):
    """
    Calculates Alpha, Beta and VaR.
    """
    # Alineate dates
    data = pd.DataFrame({
        'asset': asset_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if data.empty:
        return None

    # Linear Regression
    beta, alpha, r_value, p_value, std_err = linregress(data['benchmark'], data['asset'])
    alpha_annualized = alpha * 252
    
    # VaR 
    confidence_level = 0.95
    mu, std = norm.fit(data['asset'])
    var_95 = norm.ppf(1 - confidence_level, mu, std)
    
    return {
        "alpha": alpha_annualized,
        "beta": beta,
        "var_95": var_95,
        "combined_df": data,
        "mu": mu,
        "std": std
    }
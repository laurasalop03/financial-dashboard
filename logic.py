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


def calculate_drawdown(series_price: pd.Series) -> pd.Series:
    """
    Calculates, given a series of prices, the Drawdown.
    """
    rolling_max = series_price.cummax()
    drawdown = (series_price - rolling_max) / rolling_max
    
    return drawdown


def calculate_trading_signals(series_price: pd.Series):
    """
    Calculates Moving Averages and detects signals of Golden Cross and Death Cross.
    Returns a DataFrame with the signals and the actual position.
    """

    # SMAs
    sma_50 = series_price.rolling(window=50).mean()
    sma_200 = series_price.rolling(window=200).mean()
    
    # create signals df
    signals = pd.DataFrame(index=series_price.index)
    signals['price'] = series_price
    signals['SMA_50'] = sma_50
    signals['SMA_200'] = sma_200
    signals['Signal'] = 0   # 0: do nothing, 1: buy, -1: sell

    # detect crosses
    sma_50_prev = sma_50.shift(1)
    sma_200_prev = sma_200.shift(1)

    # buy signal (today sma50 > sma200 and yesterdar sma50 < sma200)
    buy_condition = (sma_50 > sma_200) & (sma_50_prev < sma_200_prev)
    signals.loc[buy_condition, 'Signal'] = 1

    # sell signal (today sma50 < sma200 and yesterdar sma50 > sma200)
    sell_condition = (sma_50 < sma_200) & (sma_50_prev > sma_200_prev)
    signals.loc[sell_condition, 'Signal'] = -1

    # current state
    curr_state = "BULLISH 🟢" if sma_50.iloc[-1] > sma_200.iloc[-1] else "BEARISH 🔴"

    return signals, curr_state


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


def calculate_correlation(df: pd.DataFrame):
    """
    Calculates de correlation matrix for the assets we have.
    """
    return df['log_return'].corr()
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm, linregress
import streamlit as st
from prophet import Prophet

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


def predict_forecast(prices: pd.Series, n_days: int) -> pd.DataFrame:
    """
    Predicts the next n_days using the information of prices.
    """
    # adapt pd.Series to prophet's format
    df = prices.reset_index()
    df.columns = ['ds', 'y']

    # train model
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=n_days)

    prediction = model.predict(future)

    return prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def run_backtest(prices: pd.Series, signals: pd.Series, initial_capital: float = 10000.0):
    """
    Simulates a trading strategy.
    Param:
        - prices: Series with the daily closing prices.
        - signals: Series with 1 (buy), -1 (sell), 0 (do nothing).
        - initial_capital
    Return:
        - Series with the total value of the portfolio daily.
    """
    
    # inicial state
    cash = initial_capital
    shares = 0
    portfolio_history = []  # for the daily total value
    
    # in case signals and prices start in different days
    aligned_data = pd.DataFrame({'price': prices, 'signal': signals}).dropna()

    # daily loop
    for date, row in aligned_data.iterrows():
        price = row['price']
        signal = row['signal']
        
        # buy signal and we have no shares
        # we assume we buy everything we can with the cash available
        if signal == 1 and shares == 0:
            shares_to_buy = cash / price
            cash -= shares_to_buy * price
            shares += shares_to_buy

        # sell signal and we have shares to sell
        elif signal == -1 and shares > 0:
            cash += shares * price
            shares = 0
        
        # total value today
        total_value = cash + (shares * price)
        portfolio_history.append(total_value)
    
    # turn list to Series with correct dates
    return pd.Series(portfolio_history, index=aligned_data.index)


def simulate_monte_carlo(prices: pd.Series, days_to_project: int, n_simulations: int):
    """
    Simulates n_simulations possible futures using the Geometric Brownian motion.
    """

    log_returns = np.log(prices / prices.shift(1)).dropna()

    mu = log_returns.mean()
    sigma = log_returns.std()
    drift = mu - (0.5 * sigma**2)

    # matrix with random numbers
    Z = np.random.normal(0, 1, (days_to_project, n_simulations))
    
    daily_returns = np.exp(drift + sigma * Z)

    # we want first day to be actual price, so return = 1
    daily_returns[0] = 1

    # accumulative prices
    last_price = prices.iloc[-1]
    prices_path = last_price * np.cumprod(daily_returns, axis=0)

    return prices_path


def simulate_portfolio_optimization (prices: pd.DataFrame, n_simulations: int = 5000):
    '''
    Simulates thousands of portfolio combinations to find the Efficient Frontier.
    '''

    n_stocks = len(prices.columns)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # covariance matrix annualized
    cov_matrix = log_returns.cov() * 252

    returns, volatilities, weights = [], [], []

    for _ in range(n_simulations):
        w = np.random.random(n_stocks)
        w /= np.sum(w)

        returns.append(np.sum(w * log_returns.mean()) * 252)
        volatilities.append(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
        weights.append(w)

    return {
        'returns': returns,
        'volatility': volatilities,
        'weights': weights
    }
import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, linregress

st.title("Financial Data Analysis Dashboard")

# streatmlit caching to avoid re-downloading data on every run
@st.cache_data
def get_data(ticker='AAPL', start='2020-01-01', end='2023-12-31'):

    # get info from ticker
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # calculate returns
    df['simple_return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)

    # calculate annualized volatility
    df['volatility'] = df['log_return'].std() * np.sqrt(252)

    # VaR
    confidence_level = 0.95
    df['VaR_95'] = norm.ppf(1 - confidence_level) * df['volatility'].iloc[-1]

    
    # get info from S&P 500
    df2 = yf.download('^GSPC', start=start, end=end, auto_adjust=True)

    # calculate S&P 500 returns
    df2['log_return_sp500'] = np.log(df2['Close'] / df2['Close'].shift(1))
    df2.dropna(inplace=True)

    # clean dataframe for merging
    df_sp500_clean = df2[['log_return_sp500']] # dual [] to keep as DataFrame

    # merge both dataframes
    df_total = df.join(df_sp500_clean, how='inner') 

    # linear regression to find alpha and beta
    beta, alpha, r_value, p_value, std_err = linregress(x=df_total['log_return_sp500'], y=df_total['log_return'])
    df_total['alpha_anualized'] = alpha * 252
    df_total['beta'] = beta

    # Drawdown calculation
    df_total['Rolling_Max'] = df_total['Close'].cummax()
    close_series = df_total[('Close', ticker)]
    max_series = df_total[('Rolling_Max', '')]
    df_total['Drawdown'] = (close_series - max_series) / max_series
    df_total['max_drawdown'] = df_total['Drawdown'].min()

    # SMA 50 and 200
    df_total['SMA_50'] = df_total['Close', ticker].rolling(window=50).mean()
    df_total['SMA_200'] = df_total['Close', ticker].rolling(window=200).mean()

    return df_total


# ask user for ticker symbol, start date, end date
ticker = st.text_input("Enter Ticker Symbol", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))


# Load data
df = get_data(ticker=ticker, start=start_date, end=end_date)

col1, col2, col3 = st.columns(3)

# show total return
total_return_acc = (df['Close', ticker].iloc[-1] - df['Close', ticker].iloc[0]) / df['Close', ticker].iloc[0]
col1.metric("Total Return", f"{total_return_acc:.2%}")

# show annualized volatility
annualized_volatility = df['volatility'].iloc[-1]
col2.metric("Annualized Volatility", f"{annualized_volatility:.2%}")

# show max drawdown
max_drawdown = df['max_drawdown'].iloc[-1]
col3.metric("Max Drawdown", f"{max_drawdown:.2%}")


# time series plot of closing prices
st.subheader("Closing Prices Over Time")
fig, ax = plt.subplots()
ax.plot(df.index, df['Close', ticker], label= ticker + "'Close Price")
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.legend()
st.pyplot(fig)

# histogram of log returns
st.subheader("Histogram of Log Returns")
fig2, ax2 = plt.subplots()
ax2.hist(df['log_return'], bins=50, alpha=0.7, color='blue', density=True)
# overlay normal distribution curve
mu, std = norm.fit(df['log_return'])
x = np.linspace(df['log_return'].min(), df['log_return'].max(), 100)
y = norm.pdf(x, mu, std)
ax2.plot(x, y, color='red', linewidth=2)
ax2.set_xlabel('Log Return')
ax2.set_ylabel('Frequency')
st.pyplot(fig2)


# scatter plot of returns with regression line
st.subheader("Alpha and Beta from CAPM Regression")

col1, col2 = st.columns(2)
col1.metric("Alpha", f"{df['alpha_anualized'].iloc[-1]:.4f}")
col2.metric("Beta", f"{df['beta'].iloc[-1]:.4f}")

fig3, ax3 = plt.subplots()
ax3.scatter(df['log_return_sp500'], df['log_return'], alpha=0.5)
# regression line
x = np.linspace(df['log_return_sp500'].min(), df['log_return_sp500'].max(), 100)
y = df['beta'].iloc[-1] * x + df['alpha_anualized'].iloc[-1] / 252
ax3.plot(x, y, color='red', label='Regression Line')
ax3.set_xlabel('S&P 500 Log Return')
ax3.set_ylabel(ticker + ' Log Return')
ax3.legend()
st.pyplot(fig3)


# SMA 50 and 200 plot
st.subheader("Price vs Moving Averages (SMA 50 and SMA 200)")
fig4, ax4 = plt.subplots()
ax4.plot(df.index, df['Close', ticker], label=ticker + ' Close Price', color='blue', alpha=0.5)
ax4.plot(df.index, df['SMA_50'], label='SMA 50', color='orange')
ax4.plot(df.index, df['SMA_200'], label='SMA 200', color='green')
ax4.set_xlabel('Date')
ax4.set_ylabel('Price ($)')
ax4.legend()
st.pyplot(fig4)


# Drawdown plot
st.subheader("Drawdown Over Time")
fig5, ax5 = plt.subplots()
ax5.fill_between(df.index, df['Drawdown'], 0, label='Drawdown', color='red', alpha=0.3)
ax5.plot(df.index, df['Drawdown'], color='red', alpha=0.8)
ax5.set_xlabel('Date')
ax5.set_ylabel('Drawdown')
ax5.legend()
st.pyplot(fig5)
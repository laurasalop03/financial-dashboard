import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, linregress
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(layout="wide")

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
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-12-31"))



st.sidebar.markdown("---") 
option = st.sidebar.radio("Navegation", [ticker + "'s Summary", "Technical Analysis", "Risk & Statistics"])


# Load data
df = get_data(ticker=ticker, start=start_date, end=end_date)


if option == ticker + "'s Summary":
    st.header(ticker + "'s Financial Summary")

    st.write(f"Displaying data for **{ticker}** from **{start_date}** to **{end_date}**.")

    st.dataframe(df.tail())

    st.markdown("### Key Metrics")
    col1, col2, col3 = st.columns(3)

    total_return_acc = (df['Close', ticker].iloc[-1] - df['Close', ticker].iloc[0]) / df['Close', ticker].iloc[0]
    col1.metric("Total Return", f"{total_return_acc:.2%}")

    annualized_volatility = df['volatility'].iloc[-1]
    col2.metric("Annualized Volatility", f"{annualized_volatility:.2%}")

    max_drawdown = df['max_drawdown'].iloc[-1]
    col3.metric("Max Drawdown", f"{max_drawdown:.2%}")

    # calculate busy days between start and end date
    dt_all = pd.bdate_range(start=df.index[0], end=df.index[-1])
    dt_obs = df.index
    dt_breaks = dt_all.difference(dt_obs)

    # time series plot of closing prices
    fig_closing = go.Figure()
    fig_closing.add_trace(go.Candlestick(
        x=df.index, 
        open=df['Open', ticker], 
        high=df['High', ticker], 
        low=df['Low', ticker], 
        close=df['Close', ticker],
        name='Candlestick'
    ))
    fig_closing.update_layout(
        title=f"{ticker} Daily Prices (Candlestick)",
        xaxis_title='Date', 
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(values=dt_breaks)  # hide holidays and missing days
            ]
        )
    )
    st.plotly_chart(fig_closing, use_container_width=True)

    st.stop()  # Stop further execution if in General Summary mode



if option == "Technical Analysis":
    st.header("Technical Indicators")

    st.write(f"Technical analysis for **{ticker}** from **{start_date}** to **{end_date}**.")

    # Drawdown plot
    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(
        x=df.index, 
        y=df['Drawdown'], 
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=2),
        fill='tozeroy'
    ))
    fig_drawdown.update_layout(
        title='Drawdown Over Time',
        xaxis_title='Date',
        yaxis_title='Drawdown'
    )
    st.plotly_chart(fig_drawdown, use_container_width=True)

    # SMA 50 and 200 plot
    fig_MA = go.Figure()
    fig_MA.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close', ticker], 
        mode='lines',
        name='Close Price',
        line=dict(color='red', width=1),
        opacity=0.5
    ))
    fig_MA.add_trace(go.Scatter(
        x=df.index, 
        y=df['SMA_50'], 
        mode='lines',
        name='SMA 50',
        line=dict(color='orange', width=2)
    ))
    fig_MA.add_trace(go.Scatter(
        x=df.index, 
        y=df['SMA_200'], 
        mode='lines',
        name='SMA 200',
        line=dict(color='green', width=2)
    ))
    fig_MA.update_layout(
        title='Price vs Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)'
    )
    st.plotly_chart(fig_MA, use_container_width=True)

    st.stop()



if option == "Risk & Statistics":
    st.header("Risk & Statistical Analysis")

    st.write(f"Risk and statistical analysis for **{ticker}** from **{start_date}** to **{end_date}**.")
    
    # histogram of log returns with normal distribution overlay
    mu, std = norm.fit(df['log_return'])
    x = np.linspace(df['log_return'].min(), df['log_return'].max(), 100)
    y = norm.pdf(x, mu, std)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df['log_return'],
        histnorm='probability density',
        name='Log Returns',
        marker_color='blue',
        opacity=0.7
    ))
    fig_hist.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Normal Distribution Fit',
        line=dict(color='red', width=2)
    ))
    fig_hist.update_layout(
        title='Histogram of Log Returns with Normal Distribution Fit',
        xaxis_title='Log Return',
        yaxis_title='Density'
    )
    st.plotly_chart(fig_hist, use_container_width=True)


    # scatter plot of returns with regression line
    col1, col2 = st.columns(2)
    col1.metric("Alpha", f"{df['alpha_anualized'].iloc[-1]:.4f}")
    col2.metric("Beta", f"{df['beta'].iloc[-1]:.4f}")

    fig_linreg = go.Figure()
    fig_linreg.add_trace(go.Scatter(
        x=df['log_return_sp500'], 
        y=df['log_return'], 
        mode='markers',
        name='Data Points',
        marker=dict(opacity=0.5)
    ))
    # regression line
    x = np.linspace(df['log_return_sp500'].min(), df['log_return_sp500'].max(), 100)
    y = df['beta'].iloc[-1] * x + df['alpha_anualized'].iloc[-1] / 252

    fig_linreg.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Regression Line',
        line=dict(color='red', width=2)
    ))
    fig_linreg.update_layout(
        title='Scatter Plot of Returns with Regression Line',
        xaxis_title='S&P 500 Log Return',
        yaxis_title=ticker + ' Log Return'
    )
    fig_linreg.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        text=f"<b>Alpha:</b> {df['alpha_anualized'].iloc[-1]:.4f}<br><b>Beta:</b> {df['beta'].iloc[-1]:.4f}",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=15, color="black")
    )
    st.plotly_chart(fig_linreg, use_container_width=True)

    st.stop()


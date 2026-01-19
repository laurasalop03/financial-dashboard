import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import logic

COLOR_PALETTE = {
    "primary": "#2563EB", 
    "secondary": "#38BDF8",  
    "accent": "#818CF8",    
    "dark": "#167D6A",    
    "danger": "#F87171",    
    "success": "#34D399"
}

st.set_page_config(layout="wide", page_title="Financial Dashboard")

st.title("Financial Data Analysis Dashboard")


# --- SIDEBAR ---

# ask user for ticker symbol, start date, end date

posible_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'META', 'NVDA']
tickers = st.sidebar.multiselect("Select Tickers", options=posible_tickers, default=["AAPL"])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-12-31"))


st.sidebar.markdown("---") 
option = st.sidebar.radio("Navegation", ["General Summary", "Technical Analysis", "Risk & Statistics"])


# Validation
if not tickers:
    st.warning("Please select at least one ticker.")
    st.stop()

# Load data
try:
    df = logic.get_data(tickers, start=start_date, end=end_date)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# --- GENERAL SUMARY ---

if option == "General Summary":

    st.header("Financial Summary & Comparison")

    # METRICS
    st.subheader("Market Overview")

    metrics_list = [logic.calculate_metrics(df, t) for t in tickers]
    st.dataframe(pd.DataFrame(metrics_list))


    # GRAPHS
    st.markdown("### Price Comparison")

    # time series plot of closing prices
    fig_closing = go.Figure()

    for t in tickers:
        prices = df['Close', t]
        normalized_prices = (prices / prices.iloc[0]) * 100
        fig_closing.add_trace(go.Scatter(
            x=df.index, 
            y=normalized_prices,
            mode='lines',
            name=t
        ))
    fig_closing.update_layout(
        title='Relative Performance (Base 100)',
        xaxis_title='Date', 
        yaxis_title='Growth (Base 100)',
        hovermode='x unified',
        xaxis_rangeslider_visible=True,
    )
    st.plotly_chart(fig_closing, use_container_width=True)

    st.markdown("---")

    # candlestick plot doesn't make sense for multiple tickers, so only for the choosen one
    st.subheader("Candlestick Chart")

    selected_ticker = st.selectbox("Select asset to inspect:", options=tickers, key="summary_select")

    fig_candlestick = go.Figure()
    fig_candlestick.add_trace(go.Candlestick(
        x=df.index, 
        open=df['Open', selected_ticker], 
        high=df['High', selected_ticker], 
        low=df['Low', selected_ticker], 
        close=df['Close', selected_ticker],
        name=selected_ticker
    ))

    # calculate busy days between start and end date
    dt_all = pd.bdate_range(start=df.index[0], end=df.index[-1])
    dt_obs = df.index
    dt_breaks = dt_all.difference(dt_obs)

    fig_candlestick.update_layout(
        title=f"{selected_ticker} Daily Prices (Candlestick)",
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
    st.plotly_chart(fig_candlestick, use_container_width=True)

    st.stop()  # Stop further execution if in General Summary mode


# --- TECHNICAL ANALYSIS ---

if option == "Technical Analysis":
    st.header("Technical Indicators")

    selected_ticker = st.selectbox("Select asset to inspect:", options=tickers, key="tech_select")

    prices = df['Close', selected_ticker]
    tech_data = logic.calculate_technical_indicators(prices)

    # Drawdown plot
    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(
        x=df.index, 
        y=tech_data['Drawdown'], 
        mode='lines',
        name='Drawdown',
        line=dict(color=COLOR_PALETTE["danger"], width=2),
        fill='tozeroy'
    ))
    fig_drawdown.update_layout(
        title='Drawdown Over Time',
        xaxis_title='Date',
        yaxis_title='Drawdown'
    )
    st.plotly_chart(fig_drawdown, use_container_width=True)

    # SMA 50 and 200 plot
    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(
        x=df.index, 
        y=prices, 
        mode='lines',
        name='Close Price',
        line=dict(color=COLOR_PALETTE["secondary"], width=1),
        opacity=0.5
    ))
    fig_sma.add_trace(go.Scatter(
        x=df.index, 
        y=tech_data['SMA_50'], 
        mode='lines',
        name='SMA 50',
        line=dict(color=COLOR_PALETTE["primary"], width=2)
    ))
    fig_sma.add_trace(go.Scatter(
        x=df.index, 
        y=tech_data['SMA_200'], 
        mode='lines',
        name='SMA 200',
        line=dict(color=COLOR_PALETTE["accent"], width=2)
    ))
    fig_sma.update_layout(
        title='Price vs Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)'
    )
    st.plotly_chart(fig_sma, use_container_width=True)

    st.stop()


# --- RISK & STATISTICS ---

if option == "Risk & Statistics":
    st.header("Risk & Statistical Analysis")
        
    selected_ticker = st.selectbox("Select asset to inspect:", options=tickers, key="risk_select")
    
    # Get S&P 500 data and calculate risk
    df_spy = logic.get_benchmark_data(start_date, end_date)
    risk_data = logic.calculate_risk_stats(df['log_return', selected_ticker], df_spy['log_return'])

    if risk_data:
        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Alpha", f"{risk_data['alpha']:.4f}")
        c2.metric("Beta", f"{risk_data['beta']:.4f}")
        c3.metric("VaR (95%)", f"{risk_data['var_95']:.2%}")

        # Regression
        comb = risk_data['combined_df']

        fig_linreg = go.Figure()
        fig_linreg.add_trace(go.Scatter(
            x=comb['benchmark'], 
            y=comb['asset'], 
            mode='markers',
            name='Data Points',
            opacity=0.5
        ))

        # regression line
        x_line = np.linspace(comb['benchmark'].min(), comb['benchmark'].max(), 100)
        y_line = risk_data['beta'] * x_line + (risk_data['alpha']/252)
        fig_linreg.add_trace(go.Scatter(
            x=x_line, 
            y=y_line,
            mode='lines',  
            name='Regression',
            marker=dict(color=COLOR_PALETTE["danger"])
        ))

        fig_linreg.update_layout(
            title='Scatter Plot of Returns with Regression Line',
            xaxis_title='S&P 500 Log Return',
            yaxis_title=selected_ticker + ' Log Return'
        )
        st.plotly_chart(fig_linreg, use_container_width=True)

        st.markdown("---")
        
        # Histogram
        st.subheader("Distribution Analysis (Reality vs. Theory)")
        
        mu = risk_data['mu']
        std = risk_data['std']
        asset_returns = risk_data['combined_df']['asset'] 

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=asset_returns,
            histnorm='probability density',
            name='Actual Returns',
            marker_color=COLOR_PALETTE["primary"],
            opacity=0.7
        ))

        # normal distribution plot
        x_range = np.linspace(asset_returns.min(), asset_returns.max(), 100)
        y_pdf = norm.pdf(x_range, mu, std)

        fig_hist.add_trace(go.Scatter(
            x=x_range,
            y=y_pdf,
            mode='lines',
            name='Normal Assumption',
            line=dict(color=COLOR_PALETTE["danger"], width=2, dash='dash')
        ))

        fig_hist.update_layout(
            title=f"Return Distribution: {selected_ticker}",
            xaxis_title="Log Returns",
            yaxis_title="Density",
            bargap=0.1
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.error("Error performing risk analysis.")

    st.stop()


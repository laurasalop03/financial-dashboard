import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import logic

COLOR_PALETTE = {
    "primary": "#3B82F6", 
    "secondary": "#22D3EE",  
    "accent": "#A487F9",    
    "dark": "#94A3B8",    
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
option = st.sidebar.radio("Navegation", ["General Summary", "Technical Analysis", "Risk & Statistics", "AI Forecasting"])


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
        name=selected_ticker,
        increasing_line_color=COLOR_PALETTE["success"], 
        decreasing_line_color=COLOR_PALETTE["danger"]
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
    
    # Drawdown plot
    drawdown = logic.calculate_drawdown(prices)

    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(
        x=df.index, 
        y=drawdown, 
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


    # SMA 50 and 200 plot with buy and sell signals
    signals_df, current_state = logic.calculate_trading_signals(prices)
    
    # delete the first 200 days without sma_200 so that the plot doesn't have gaps
    signals_plot = signals_df.dropna()

    st.metric(label="**Current Trend Strategy (SMA 50/200)**", value=current_state)

    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(
        x=signals_plot.index, 
        y=signals_plot['price'], 
        mode='lines',
        name='Close Price',
        line=dict(color=COLOR_PALETTE["secondary"], width=1),
        opacity=0.5
    ))
    fig_sma.add_trace(go.Scatter(
        x=signals_plot.index, 
        y=signals_plot['SMA_50'], 
        mode='lines',
        name='SMA 50',
        line=dict(color=COLOR_PALETTE["primary"], width=2)
    ))
    fig_sma.add_trace(go.Scatter(
        x=signals_plot.index, 
        y=signals_plot['SMA_200'], 
        mode='lines',
        name='SMA 200',
        line=dict(color=COLOR_PALETTE["accent"], width=2)
    ))

    # buy signals
    buy_signals = signals_plot[signals_plot['Signal'] == 1]
    fig_sma.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['SMA_50'],
        mode='markers',
        marker=dict(symbol='triangle-up', color=COLOR_PALETTE['success'], size=12),
        name='Buy Signal'
    ))
    # sell signals
    sell_signals = signals_plot[signals_plot['Signal'] == -1]
    fig_sma.add_trace(go.Scatter(
        x=sell_signals.index, 
        y=sell_signals['SMA_50'], 
        mode='markers', 
        marker=dict(symbol='triangle-down', color=COLOR_PALETTE['danger'], size=12),
        name='Sell Signal'
    ))

    fig_sma.update_layout(
        title=f"Trading Signals: Golden Cross / Death Cross ({selected_ticker})",
        xaxis_title='Date',
        yaxis_title='Price ($)'
    )
    st.plotly_chart(fig_sma, use_container_width=True)

    # explanation for the user
    with st.expander("ℹ️ How to read this chart?"):
        st.write("""
        * **Golden Cross (Triangle Up 🟢):** The short-term average (50 days) crosses *above* the long-term average (200 days). Usually indicates the start of a **Bull Market**.
        * **Death Cross (Triangle Down 🔴):** The short-term average crosses *below* the long-term average. Usually indicates the start of a **Bear Market**.
        """)

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


        # heatmap for correlation matrix, just if more than one ticker
        if (len(tickers) > 1): 
            corr_matrix = logic.calculate_correlation(df)
            fig_heatmap = go.Figure()
            fig_heatmap.add_trace(go.Heatmap(
                z=corr_matrix, 
                x=corr_matrix.columns, 
                y=corr_matrix.index,
                texttemplate="%{z:.2f}",
                colorscale=[
                    [0.0, COLOR_PALETTE["danger"]],
                    [0.5, "white"],                 
                    [1.0, COLOR_PALETTE["primary"]]
                ],
                zmin=-1, 
                zmax=1
            ))
            fig_heatmap.update_layout(
                title='Correlation Matrix Heatmap',
                height=600 
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)


    else:
        st.error("Error performing risk analysis.")

    st.stop()


# --- AI FORECASTING ---

if option == "AI Forecasting":
    st.header("AI Price Prediction (Prophet)")

    # ask user for ticker and number of days for prediction
    selected_ticker = st.selectbox("Select asset to predict:", tickers, key="ai_select")
    n_days = st.slider("Days to predict:", min_value=30, max_value=365, value=90)

    # button so that it doesn't calculate all the time
    if st.button("Generate Prediction"):
        
        # visual feedback while it thinks
        with st.spinner(f"Training AI model for {selected_ticker}..."):
            
            prices = df['Close', selected_ticker]
            forecast = logic.predict_forecast(prices, n_days)
            
            # plot
            fig_ai = go.Figure()
            
            fig_ai.add_trace(go.Scatter(
                x=prices.index, 
                y=prices,
                mode='lines', 
                name='Historical Data',
                line=dict(width=2), opacity=0.5
            ))
            
            # just add the future data 
            future_data = forecast[forecast['ds'] > prices.index[-1]]
            
            # prediction line
            fig_ai.add_trace(go.Scatter(
                x=future_data['ds'], 
                y=future_data['yhat'],
                mode='lines', 
                name='AI Prediction',
                line=dict(color=COLOR_PALETTE['primary'], width=2)
            ))

            # confidence area
            fig_ai.add_trace(go.Scatter(
                x=pd.concat([future_data['ds'], future_data['ds'][::-1]]),
                y=pd.concat([future_data['yhat_upper'], future_data['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor=COLOR_PALETTE['dark'], 
                opacity=0.3,
                line=dict(color='rgba(255,255,255,0)'),  # invisible line
                name='Confidence Interval'
            ))

            fig_ai.update_layout(
                title=f"AI Price Forecast: {selected_ticker} (+{n_days} days)",
                xaxis_title="Date", yaxis_title="Price",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_ai, use_container_width=True)
            
            st.info("Note: The shaded area represents the uncertainty. The wider the area, the less sure the AI is about the price.")


    st.stop()

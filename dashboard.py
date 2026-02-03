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

st.sidebar.header("Data Selection")

TICKER_NAMES = {
    # US Tech
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corp.',
    'META': 'Meta Platforms',
    'NFLX': 'Netflix Inc.',
    # Indices / ETFs
    'SPY': 'S&P 500 ETF',
    'QQQ': 'Nasdaq 100 ETF',
    'GLD': 'Gold Trust',
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    # Finance & Retail
    'JPM': 'JPMorgan Chase',
    'V': 'Visa Inc.',
    'WMT': 'Walmart Inc.',
    'KO': 'Coca-Cola Co.',
    # Spanish
    'SAN.MC': 'Banco Santander',
    'BBVA.MC': 'BBVA',
    'ITX.MC': 'Inditex',
    'TEF.MC': 'Telefónica'
}

def format_func(option):
    # if we have the ticker in our dict, show the name
    return f"{option} - {TICKER_NAMES[option]}" if option in TICKER_NAMES else option

# let the user select tickers
dropdown_tickers = st.sidebar.multiselect(
    "Select Popular Tickers", 
    options=list(TICKER_NAMES.keys()), 
    default=["SPY", "NVDA"],
    format_func=format_func
)

# let the user input its own tickers
st.sidebar.markdown("---")
st.sidebar.write("Don't see your ticker? Add it manually:")
custom_tickers_input = st.sidebar.text_input(
    "Enter Tickers (comma separated)", 
    placeholder="e.g. GME, AMC, EURUSD=X"
).upper()

# clean user input
if custom_tickers_input:
    custom_list = [t.strip() for t in custom_tickers_input.split(",")]
else:
    custom_list = []

tickers = list(set(dropdown_tickers + custom_list))


start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))


st.sidebar.markdown("---") 
option = st.sidebar.radio("Navegation", ["General Summary", "Technical Analysis", "Risk & Statistics", "AI Forecasting", "Portfolio Optimization"])


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

    # calculate busy days between start and end date (only if not crypto)
    is_crypto = df.index.dayofweek.isin([5, 6]).any()

    layout_args = dict(
        title=f"{selected_ticker} Daily Prices (Candlestick)",
        xaxis_title='Date', 
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=True
    )

    if not is_crypto:
        dt_all = pd.bdate_range(start=df.index[0], end=df.index[-1])
        dt_obs = df.index
        dt_breaks = dt_all.difference(dt_obs)

        layout_args["xaxis"] = dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]), # hide weekends
                dict(values=dt_breaks)       # hide holidays
            ]
        )

    fig_candlestick.update_layout(**layout_args)
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


    # backtesting
    st.markdown("---")
    st.subheader("Backtesting: Strategy vs. Buy & Hold")

    initial_capital = 10000
    
    portfolio_value = logic.run_backtest(prices, signals_df['Signal'], initial_capital)
    
    # calculate Benchmark (Buy & Hold)
    # (price today / initial price) * initial capital
    # we use the same dates as the portfolio has
    buy_and_hold_value = (prices[portfolio_value.index] / prices[portfolio_value.index][0]) * initial_capital

    # calculate returns
    strat_return = (portfolio_value.iloc[-1] - initial_capital) / initial_capital
    bh_return = (buy_and_hold_value.iloc[-1] - initial_capital) / initial_capital

    # show metrics
    c1, c2 = st.columns(2)
    c1.metric("Strategy Return", f"{strat_return:.2%}", delta=f"${portfolio_value.iloc[-1] - initial_capital:.0f}")
    c2.metric("Buy & Hold Return", f"{bh_return:.2%}", delta=f"${buy_and_hold_value.iloc[-1] - initial_capital:.0f}")

    # evolution graph
    fig_backtest = go.Figure()

    # my strategy
    fig_backtest.add_trace(go.Scatter(
        x=portfolio_value.index, 
        y=portfolio_value, 
        mode='lines', 
        name='My Strategy (SMA Cross)',
        line=dict(color=COLOR_PALETTE['success'], width=2)
    ))

    # Buy & Hold 
    fig_backtest.add_trace(go.Scatter(
        x=buy_and_hold_value.index, 
        y=buy_and_hold_value, 
        mode='lines', 
        name='Buy & Hold (Benchmark)',
        line=dict(color=COLOR_PALETTE['dark'], width=2, dash='dash')
    ))

    fig_backtest.update_layout(
        title=f"Portfolio Value Over Time (Start: ${initial_capital})",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)"
    )
    
    st.plotly_chart(fig_backtest, use_container_width=True)


    # monte carlo simulation
    st.markdown("---")
    st.subheader("Monte Carlo Simulation (Future Risk)")

    n_sim_days = st.slider("Days to simulate:", min_value=30, max_value=365, value=252, key="mc_slider")

    # button to re-run
    if st.button("🔄 Run Simulation Again"):
        st.rerun()

    simulations = logic.simulate_monte_carlo(prices, days_to_project=n_sim_days, n_simulations=1000)

    # median each day
    median_path = np.median(simulations, axis=1)

    # graph simulations and conclusions
    fig_mc = go.Figure()
    for i in range(50):
        fig_mc.add_trace(go.Scatter(
            y=simulations[:, i],
            mode='lines',
            opacity=0.1, 
            line=dict(color=COLOR_PALETTE['secondary'], width=1),
            showlegend=False
        ))
        
    fig_mc.add_trace(go.Scatter(
        x = list(range(n_sim_days)),
        y = median_path,
        name='Median Prediction (P50)',
        line=dict(color=COLOR_PALETTE['primary'], width=3)
    ))

    fig_mc.update_layout(
        title=f"Monte Carlo Simulation ({n_sim_days} days ahead)",
        xaxis_title="Days into the future",
        yaxis_title="Price ($)"
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # percentiles and conclusions
    final_prices = simulations[-1, :]
    start_price = prices.iloc[-1]
    p5 = np.percentile(final_prices, 5)
    p50 = np.percentile(final_prices, 50)
    p95 = np.percentile(final_prices, 95)

    st.subheader("Probabilistic Analysis (VaR)")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Bearish Case (P5)", f"${p5:.2f}", delta=f"{p5-start_price:.2f}")
    col2.metric("Base Case (Median)", f"${p50:.2f}", delta=f"{p50-start_price:.2f}")
    col3.metric("Bullish Case (P95)", f"${p95:.2f}", delta=f"{p95-start_price:.2f}")

    # winning probability
    st.markdown("---")
    
    prob_profit = (final_prices > start_price).mean()
    
    st.caption(f"Based on 1000 simulations, there is a **{prob_profit:.1%}** probability that the price will be higher than today in {n_sim_days} days.")

    # histogram with the final prices
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_prices,
        nbinsx=50,
        name='Final Price Distribution',
        marker_color=COLOR_PALETTE['secondary'],
        opacity=0.7
    ))

    fig_hist.add_vline(x=start_price, line_width=3, line_dash="dash", line_color=COLOR_PALETTE['danger'], annotation_text="Start Price")
    
    fig_hist.add_vline(x=p50, line_width=3, line_color=COLOR_PALETTE['success'], annotation_text="Median Prediction")

    fig_hist.update_layout(
        title="Distribution of Ending Prices",
        xaxis_title="Price ($)",
        yaxis_title="Frequency (Number of Simulations)",
        bargap=0.1
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

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


# --- PORTFOLIO OPTIMIZATION ---

if option == "Portfolio Optimization":
    st.header("Modern Portfolio Theory (Markowitz)")

    # let user select the tickers desired
    opt_tickers = st.multiselect(
        "Select assets to include in the optimization:",
        options=tickers,
        default=tickers,
        format_func=format_func
    )

    # we need at least 2 stocks to optimize
    if len(tickers) < 2:
        st.warning("⚠️ To optimize a portfolio, you need to select at least 2 tickers in the sidebar.")
        st.stop()

    st.write(f"Simulating {5000} different combinations for: **{', '.join(tickers)}**...")

    # button to re-run
    if st.button("🔄 Run Optimization Again"):
        st.rerun()

    prices_subset = df['Close'][tickers]
    
    with st.spinner("Running Monte Carlo Optimization..."):
        results = logic.simulate_portfolio_optimization(prices_subset, n_simulations=5000)

    returns = np.array(results['returns'])
    volatility = np.array(results['volatility'])
    weights = np.array(results['weights'])

    sharpe_ratios = returns / volatility

    # index of the portfolio with max sharp ratio
    max_sharpe_idx = np.argmax(sharpe_ratios)
    
    best_return = returns[max_sharpe_idx]
    best_volatility = volatility[max_sharpe_idx]
    best_weights = weights[max_sharpe_idx]

    # winning porfolio
    st.subheader("Optimal Portfolio (Max Sharpe Ratio)")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Return", f"{best_return:.2%}")
    c2.metric("Annual Volatility", f"{best_volatility:.2%}")
    c3.metric("Sharpe Ratio", f"{sharpe_ratios[max_sharpe_idx]:.2f}")

    # grahps
    col_chart1, col_chart2 = st.columns([2, 1])

    with col_chart1:
        # efficient frontier
        fig_eff = go.Figure()

        fig_eff.add_trace(go.Scatter(
            x=volatility, 
            y=returns,
            mode='markers',
            marker=dict(
                size=5,
                color=sharpe_ratios,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Portfolios',
            text=[f"Sharpe: {s:.2f}" for s in sharpe_ratios],
            hoverinfo='text+x+y'
        ))

        # star in the best one
        fig_eff.add_trace(go.Scatter(
            x=[best_volatility],
            y=[best_return],
            mode='markers',
            marker=dict(symbol='star', size=18, color=COLOR_PALETTE['danger'], line=dict(width=2, color='white')),
            name='Max Sharpe Portfolio'
        ))

        fig_eff.update_layout(
            title="Efficient Frontier",
            xaxis_title="Risk (Volatility)",
            yaxis_title="Return",
            height=500
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    with col_chart2:
        # pie chart with the best option
        labels = []
        values = []
        for ticker, weight in zip(tickers, best_weights):
            if weight > 0.01:
                labels.append(ticker)
                values.append(weight)

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['accent'], COLOR_PALETTE['success']])
        )])
        
        fig_pie.update_layout(title="Optimal Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.stop()
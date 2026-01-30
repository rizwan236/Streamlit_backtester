import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import traceback

# Page config
st.set_page_config(page_title="Tech Analysis", layout="wide", initial_sidebar_state="expanded")

# Popular symbols for autocomplete
POPULAR_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "JPM", "JNJ", "V", "WMT", "PG", "MA", "UNH", "HD", "BAC", "DIS", "ADBE"
]

def calculate_rsi(close, period=14):
    """Calculate RSI - fixed for 2D array issue"""
    # Ensure we have a 1D Series
    close_series = pd.Series(close.values.flatten() if hasattr(close, 'values') else close)
    
    delta = close_series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Convert to 1D arrays
    gain_1d = np.squeeze(gain) if gain.ndim > 1 else gain
    loss_1d = np.squeeze(loss) if loss.ndim > 1 else loss
    
    # Simple moving average for first period values
    avg_gain = pd.Series(gain_1d).rolling(window=period).mean()
    avg_loss = pd.Series(loss_1d).rolling(window=period).mean()
    
    # EMA calculation
    for i in range(period, len(gain_1d)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain_1d[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss_1d[i]) / period
    
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_cci(high, low, close, period=20):
    """Calculate CCI"""
    # Ensure 1D arrays
    high_1d = high.values.flatten() if hasattr(high, 'values') else high
    low_1d = low.values.flatten() if hasattr(low, 'values') else low
    close_1d = close.values.flatten() if hasattr(close, 'values') else close
    
    tp = (high_1d + low_1d + close_1d) / 3
    sma = pd.Series(tp).rolling(window=period).mean()
    
    # Mean Deviation
    def mean_deviation(x):
        return np.abs(x - x.mean()).mean()
    
    mad = pd.Series(tp).rolling(window=period).apply(mean_deviation, raw=False)
    cci = (pd.Series(tp) - sma) / (0.015 * mad.replace(0, np.nan))
    return cci.fillna(0)

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    # Ensure 1D Series
    close_series = pd.Series(close.values.flatten() if hasattr(close, 'values') else close)
    
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_adx(high, low, close, period=14):
    """Calculate ADX - simplified"""
    # Ensure 1D arrays
    high_series = pd.Series(high.values.flatten() if hasattr(high, 'values') else high)
    low_series = pd.Series(low.values.flatten() if hasattr(low, 'values') else low)
    close_series = pd.Series(close.values.flatten() if hasattr(close, 'values') else close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift())
    tr3 = abs(low_series - close_series.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_series - high_series.shift()
    down_move = low_series.shift() - low_series
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth using rolling mean
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
    tr_smooth = tr.rolling(window=period).mean()
    
    # DI lines
    plus_di = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window=period).mean()
    return adx.fillna(0)

def calculate_drawdown(prices):
    """Calculate drawdown"""
    # Ensure 1D Series
    prices_series = pd.Series(prices.values.flatten() if hasattr(prices, 'values') else prices)
    cummax = prices_series.cummax()
    drawdown = ((prices_series - cummax) / cummax * 100).fillna(0)
    return drawdown
    
# Trading logic
def calculate_trades(df, buy_rsi, buy_cci, sell_rsi, sell_cci):
    """Calculate trading signals and performance"""
    df = df.copy()
    df['Signal'] = ''
    trades = []
    in_position, entry_price, entry_date = False, 0, None
    
    for i in range(1, len(df)):
        date = df.index[i]
        close = df['Close'].iloc[i]
        rsi = df['RSI'].iloc[i]
        cci = df['CCI'].iloc[i]
        
        if not in_position and rsi > buy_rsi and cci > buy_cci:
            in_position, entry_price, entry_date = True, close, date
            df.loc[date, 'Signal'] = 'BUY'
        elif in_position and rsi < sell_rsi and cci < sell_cci:
            pnl = ((close - entry_price) / entry_price * 100)
            holding = (date - entry_date).days
            df.loc[date, 'Signal'] = 'SELL'
            trades.append({
                'Entry': entry_date.date(), 'Exit': date.date(),
                'Buy Price': f"${entry_price:.2f}", 'Sell Price': f"${close:.2f}",
                'P&L': f"{pnl:+.2f}%", 'Days': holding
            })
            in_position = False
    
    # Close any open position
    if in_position:
        date, close = df.index[-1], df['Close'].iloc[-1]
        pnl = ((close - entry_price) / entry_price * 100)
        holding = (date - entry_date).days
        df.loc[date, 'Signal'] = 'SELL (End)'
        trades.append({
            'Entry': entry_date.date(), 'Exit': date.date(),
            'Buy Price': f"${entry_price:.2f}", 'Sell Price': f"${close:.2f}",
            'P&L': f"{pnl:+.2f}%", 'Days': holding
        })
    
    return df, trades

# Chart creation
def create_chart(df, buy_rsi, buy_cci, sell_rsi, sell_cci, symbol):
    """Create interactive chart"""
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} - Price & Volume', 'RSI & CCI', 'ADX', 'MACD', 'Drawdown'),
        row_heights=[0.4, 0.15, 0.1, 0.15, 0.1]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                       low=df['Low'], close=df['Close'], name='Price'),
        row=1, col=1
    )
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', 
               marker_color=colors, opacity=0.3),
        row=1, col=1
    )
    
    # Signals
    buy_dates = df[df['Signal'] == 'BUY'].index
    sell_dates = df[df['Signal'].str.contains('SELL', na=False)].index
    
    if len(buy_dates) > 0:
        fig.add_trace(
            go.Scatter(x=buy_dates, y=df.loc[buy_dates, 'Low'] * 0.98,
                       mode='markers', name='BUY',
                       marker=dict(symbol='triangle-up', size=12, color='green')),
            row=1, col=1
        )
    
    if len(sell_dates) > 0:
        fig.add_trace(
            go.Scatter(x=sell_dates, y=df.loc[sell_dates, 'High'] * 1.02,
                       mode='markers', name='SELL',
                       marker=dict(symbol='triangle-down', size=12, color='red')),
            row=1, col=1
        )
    
    # Indicators
    indicators = [
        (df['RSI'], 'RSI', 'blue', 2),
        (df['CCI'], 'CCI', 'orange', 2),
        (df['ADX'], 'ADX', 'purple', 3),
        (df['MACD'], 'MACD', 'blue', 4),
        (df['MACD_signal'], 'Signal', 'red', 4),
        (df['Drawdown'], 'Drawdown', 'red', 5)
    ]
    
    for data, name, color, row in indicators:
        fig.add_trace(
            go.Scatter(x=df.index, y=data, name=name, line=dict(color=color)),
            row=row, col=1
        )
    
    # MACD Histogram
    colors_hist = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_hist'], marker_color=colors_hist,
               opacity=0.5, showlegend=False, name='MACD Hist'),
        row=4, col=1
    )
    
    # Horizontal lines
    levels = [
        (buy_rsi, 'green', f'Buy RSI > {buy_rsi}', 2),
        (sell_rsi, 'red', f'Sell RSI < {sell_rsi}', 2),
        (buy_cci, 'green', f'Buy CCI > {buy_cci}', 2),
        (sell_cci, 'red', f'Sell CCI < {sell_cci}', 2),
        (25, 'orange', 'Strong Trend', 3),
        (0, 'gray', None, 4)
    ]
    
    for level, color, text, row in levels:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.7,
                     annotation_text=text, row=row, col=1)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=True, row=5, col=1)
    
    return fig

# Main App
st.title("📈 Technical Analysis Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Symbol selection
    selected_symbol = st.selectbox(
        "Stock Symbol",
        options=POPULAR_SYMBOLS,
        index=POPULAR_SYMBOLS.index("GOOGL") if "GOOGL" in POPULAR_SYMBOLS else 0,
        help="Select from popular symbols or enter custom below"
    )
    
    custom_symbol = st.text_input("Or enter custom symbol", "")
    symbol = custom_symbol.upper() if custom_symbol.strip() else selected_symbol
    
    # Period
    period = st.selectbox("Period", ["1y", "2y", "3y", "5y", "10y", "max"], index=0)
    
    # Trading rules
    st.subheader("📊 Trading Rules")
    col1, col2 = st.columns(2)
    with col1:
        buy_rsi = st.number_input("Buy RSI >", 50, 80, 60)
        sell_rsi = st.number_input("Sell RSI <", 20, 50, 30)
    with col2:
        buy_cci = st.number_input("Buy CCI >", -100, 100, 0)
        sell_cci = st.number_input("Sell CCI <", -100, 100, 0)
    
    # Indicator periods
    st.subheader("⚙️ Indicator Settings")
    rsi_period = st.slider("RSI Period", 5, 30, 14, key='rsi')
    cci_period = st.slider("CCI Period", 5, 30, 20, key='cci')
    adx_period = st.slider("ADX Period", 5, 30, 14, key='adx')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        macd_fast = st.slider("MACD Fast", 5, 20, 12, key='fast')
    with col2:
        macd_slow = st.slider("MACD Slow", 15, 35, 26, key='slow')
    with col3:
        macd_signal = st.slider("MACD Signal", 5, 15, 9, key='signal')
    
    analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

# Main content
if analyze_button:
    try:
        # Download data
        with st.spinner(f"Fetching {symbol} data..."):
            data = yf.download(symbol, period=period, progress=False)
            if data.empty:
                st.error(f"❌ No data found for {symbol}")
                st.stop()


        
        # Calculate indicators
        with st.spinner("Calculating indicators..."):
            data['RSI'] = calculate_rsi(data['Close'], rsi_period)
            data['CCI'] = calculate_cci(data['High'], data['Low'], data['Close'], cci_period)
            data['ADX'] = calculate_adx(data['High'], data['Low'], data['Close'], adx_period)
            
            macd, signal, hist = calculate_macd(data['Close'], macd_fast, macd_slow, macd_signal)
            data['MACD'] = macd.values
            data['MACD_signal'] = signal.values
            data['MACD_hist'] = hist.values
            
            data['Drawdown'] = calculate_drawdown(data['Close'])
        
        # Calculate trades
        data, trades = calculate_trades(data, buy_rsi, buy_cci, sell_rsi, sell_cci)
        
        # Display metrics
        st.subheader("📊 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price = data['Close'].iloc[-1]
            change = ((price - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100)
            st.metric("Current Price", f"${price:.2f}", f"{change:+.2f}%")
        
        with col2:
            total_trades = len(trades)
            winning_trades = len([t for t in trades if float(t['P&L'].replace('%', '').replace('+', '')) > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%", f"{winning_trades}/{total_trades}")
        
        with col3:
            if trades:
                pnls = [float(t['P&L'].replace('%', '').replace('+', '')) for t in trades]
                avg_pnl = np.mean(pnls)
                st.metric("Avg P&L", f"{avg_pnl:+.2f}%")
            else:
                st.metric("Avg P&L", "0.00%")
        
        with col4:
            if trades:
                avg_days = np.mean([t['Days'] for t in trades])
                st.metric("Avg Days Held", f"{avg_days:.1f}")
            else:
                st.metric("Avg Days Held", "0")
        
        # Display chart
        st.subheader("📈 Interactive Chart")
        fig = create_chart(data, buy_rsi, buy_cci, sell_rsi, sell_cci, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display trades
        if trades:
            st.subheader("📋 Trade History")
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df, use_container_width=True)
            
            # Performance metrics
            st.subheader("🎯 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pnls = [float(t['P&L'].replace('%', '').replace('+', '')) for t in trades]
                st.metric("Best Trade", f"{max(pnls):+.2f}%")
            
            with col2:
                st.metric("Worst Trade", f"{min(pnls):+.2f}%")
            
            with col3:
                st.metric("Total Return", f"{sum(pnls):+.2f}%")
            
            with col4:
                if len(pnls) > 1 and np.std(pnls) > 0:
                    sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252/avg_days)
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                else:
                    st.metric("Sharpe Ratio", "N/A")
        
        # Export data
        with st.expander("📁 Export Data"):
            csv = data.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Tip: Try a different stock symbol or check your internet connection")
        # Optionally show more details
        with st.expander("Show Traceback"):
            st.code(traceback.format_exc())        

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h3>📊 Technical Analysis Dashboard</h3>
        <p>Analyze stocks with RSI, CCI, ADX, and MACD indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Quick Start - Select a stock:")
    cols = st.columns(5)
    popular_subset = POPULAR_SYMBOLS[:10]  # Show first 10
    
    for idx, sym in enumerate(popular_subset):
        with cols[idx % 5]:
            if st.button(sym, use_container_width=True):
                st.session_state.custom_symbol = sym
                st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### 📈 How to Use:
    1. **Select** a stock symbol from the dropdown or enter custom
    2. **Set** your trading rules (RSI/CCI thresholds)
    3. **Adjust** indicator periods if needed
    4. **Click** "Analyze" to run the analysis
    
    ### 📊 Trading Strategy:
    - **BUY** when RSI > [Buy RSI] AND CCI > [Buy CCI]
    - **SELL** when RSI < [Sell RSI] AND CCI < [Sell CCI]
    """)

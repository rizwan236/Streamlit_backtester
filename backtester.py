import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Page config
st.set_page_config(page_title="Tech Analysis", layout="wide", initial_sidebar_state="expanded")

# Popular symbols for autocomplete
POPULAR_SYMBOLS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "JPM", "JNJ", "V", "WMT", "PG", "MA", "UNH", "HD", "BAC", "DIS", "ADBE"
]

# Indicator calculations
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs)).fillna(50)

def calculate_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
    return ((tp - sma) / (0.015 * mad)).fillna(0)

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def calculate_adx(high, low, close, period=14):
    tr = pd.concat([high - low, 
                   abs(high - close.shift()), 
                   abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    up_move = high - high.shift()
    down_move = low.shift() - low
    pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
    neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=high.index)
    
    pos_di = 100 * pos_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    neg_di = 100 * neg_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)

def calculate_drawdown(prices):
    cummax = prices.cummax()
    return (prices - cummax) / cummax * 100

# Trading logic
def calculate_trades(df, buy_rsi, buy_cci, sell_rsi, sell_cci):
    df = df.copy()
    df['Signal'] = ''
    trades = []
    in_position, entry_price, entry_date = False, 0, None
    
    for i in range(1, len(df)):
        date, close, rsi, cci = df.index[i], df['Close'].iloc[i], df['RSI'].iloc[i], df['CCI'].iloc[i]
        
        if not in_position and rsi > buy_rsi and cci > buy_cci:
            in_position, entry_price, entry_date = True, close, date
            df.loc[date, 'Signal'] = 'BUY'
        elif in_position and rsi < sell_rsi and cci < sell_cci:
            pnl = (close - entry_price) / entry_price * 100
            holding = (date - entry_date).days
            df.loc[date, 'Signal'] = 'SELL'
            trades.append({
                'Entry_Date': entry_date, 'Exit_Date': date,
                'Entry_Price': entry_price, 'Exit_Price': close,
                'P_L': pnl, 'Holding_Days': holding
            })
            in_position = False
    
    # Close open position at end
    if in_position:
        date, close = df.index[-1], df['Close'].iloc[-1]
        pnl = (close - entry_price) / entry_price * 100
        holding = (date - entry_date).days
        df.loc[date, 'Signal'] = 'SELL (End)'
        trades.append({
            'Entry_Date': entry_date, 'Exit_Date': date,
            'Entry_Price': entry_price, 'Exit_Price': close,
            'P_L': pnl, 'Holding_Days': holding
        })
    
    return df, trades

# Chart creation
def create_chart(df, buy_rsi, buy_cci, sell_rsi, sell_cci, symbol):
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol}', 'RSI & CCI', 'ADX', 'MACD', 'Drawdown'),
        row_heights=[0.4, 0.15, 0.1, 0.15, 0.1]
    )
    
    # Price & Volume
    colors = ['red' if close < open else 'green' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                 low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                         marker_color=colors, opacity=0.3), row=1, col=1)
    
    # Signals
    for signal, color, symbol in [('BUY', 'green', 'triangle-up'), ('SELL', 'red', 'triangle-down')]:
        dates = df[df['Signal'].str.contains(signal, na=False)].index
        if len(dates) > 0:
            y = df.loc[dates, 'Low'] * 0.98 if signal == 'BUY' else df.loc[dates, 'High'] * 1.02
            fig.add_trace(go.Scatter(x=dates, y=y, mode='markers', name=signal,
                                    marker=dict(symbol=symbol, size=12, color=color)), row=1, col=1)
    
    # Indicators
    indicators = [
        ('RSI', 'blue', 2), ('CCI', 'orange', 2),
        ('ADX', 'purple', 3), ('MACD', 'blue', 4),
        ('MACD_signal', 'red', 4), ('Drawdown', 'red', 5)
    ]
    
    for indicator, color, row in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df[indicator], name=indicator,
                                line=dict(color=color)), row=row, col=1)
    
    # MACD Histogram
    colors_hist = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], marker_color=colors_hist,
                        opacity=0.5, showlegend=False), row=4, col=1)
    
    # Levels
    levels = [
        (buy_rsi, 'green', f'BUY RSI>{buy_rsi}', 2),
        (sell_rsi, 'red', f'SELL RSI<{sell_rsi}', 2),
        (buy_cci, 'green', f'BUY CCI>{buy_cci}', 2),
        (sell_cci, 'red', f'SELL CCI<{sell_cci}', 2),
        (25, 'orange', 'Strong Trend', 3),
        (0, 'gray', None, 4)
    ]
    
    for level, color, text, row in levels:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.7,
                     annotation_text=text, row=row, col=1)
    
    fig.update_layout(height=1000, showlegend=True, hovermode='x unified')
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=True, row=5, col=1)
    
    return fig

# Main App
st.title("📈 Technical Analysis Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Symbol with autocomplete dropdown
    selected_symbol = st.selectbox(
        "Stock Symbol",
        options=POPULAR_SYMBOLS,
        index=POPULAR_SYMBOLS.index("GOOG") if "GOOG" in POPULAR_SYMBOLS else 0
    )
    
    # Also allow custom input
    custom_symbol = st.text_input("Or enter custom symbol", "")
    symbol = custom_symbol.upper() if custom_symbol else selected_symbol
    
    # Period
    period = st.selectbox("Period", ["1y", "2y", "3y", "5y", "10y", "max"], index=0)
    
    # Trading rules
    st.subheader("Trading Rules")
    buy_rsi = st.slider("Buy: RSI >", 50, 80, 60)
    buy_cci = st.slider("Buy: CCI >", -100, 100, 0)
    sell_rsi = st.slider("Sell: RSI <", 20, 50, 30)
    sell_cci = st.slider("Sell: CCI <", -100, 100, 0)
    
    # Indicator settings
    st.subheader("Indicator Periods")
    col1, col2 = st.columns(2)
    with col1:
        rsi_period = st.slider("RSI", 5, 30, 14)
        cci_period = st.slider("CCI", 5, 30, 20)
    with col2:
        adx_period = st.slider("ADX", 5, 30, 14)
        macd_fast = st.slider("MACD Fast", 5, 20, 12)
    
    col1, col2 = st.columns(2)
    with col1:
        macd_slow = st.slider("MACD Slow", 15, 35, 26)
    with col2:
        macd_signal = st.slider("MACD Signal", 5, 15, 9)
    
    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        st.session_state.analyze = True
    else:
        st.session_state.analyze = False

# Main content
if st.session_state.get('analyze', False):
    try:
        # Download data
        with st.spinner(f"Fetching {symbol} data..."):
            data = yf.download(symbol, period=period, progress=False)
            if data.empty:
                st.error(f"No data found for {symbol}")
                st.stop()
        
        # Calculate indicators
        data['RSI'] = calculate_rsi(data['Close'], rsi_period)
        data['CCI'] = calculate_cci(data['High'], data['Low'], data['Close'], cci_period)
        data['ADX'] = calculate_adx(data['High'], data['Low'], data['Close'], adx_period)
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = calculate_macd(
            data['Close'], macd_fast, macd_slow, macd_signal
        )
        data['Drawdown'] = calculate_drawdown(data['Close'])
        
        # Calculate trades
        data, trades = calculate_trades(data, buy_rsi, buy_cci, sell_rsi, sell_cci)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            price = data['Close'].iloc[-1]
            change = (price - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
            st.metric("Current Price", f"${price:.2f}", f"{change:+.2f}%")
        with col2:
            win_rate = (len([t for t in trades if t['P_L'] > 0]) / len(trades) * 100) if trades else 0
            st.metric("Win Rate", f"{win_rate:.1f}%", f"{len(trades)} trades")
        with col3:
            avg_pnl = np.mean([t['P_L'] for t in trades]) if trades else 0
            st.metric("Avg P&L", f"{avg_pnl:+.2f}%")
        with col4:
            avg_days = np.mean([t['Holding_Days'] for t in trades]) if trades else 0
            st.metric("Avg Days", f"{avg_days:.1f}")
        
        # Chart
        st.plotly_chart(create_chart(data, buy_rsi, buy_cci, sell_rsi, sell_cci, symbol), 
                       use_container_width=True)
        
        # Trade details
        if trades:
            st.subheader("📋 Trade History")
            trades_df = pd.DataFrame(trades)
            
            # Format for display
            display_df = trades_df.copy()
            display_df['Entry_Date'] = display_df['Entry_Date'].dt.strftime('%Y-%m-%d')
            display_df['Exit_Date'] = display_df['Exit_Date'].dt.strftime('%Y-%m-%d')
            display_df['Entry_Price'] = display_df['Entry_Price'].apply(lambda x: f"${x:.2f}")
            display_df['Exit_Price'] = display_df['Exit_Price'].apply(lambda x: f"${x:.2f}")
            display_df['P_L'] = display_df['P_L'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Performance
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Trade", f"{trades_df['P_L'].max():+.2f}%")
            with col2:
                st.metric("Worst Trade", f"{trades_df['P_L'].min():+.2f}%")
            with col3:
                st.metric("Total Return", f"{trades_df['P_L'].sum():+.2f}%")
            with col4:
                sharpe = (trades_df['P_L'].mean() / trades_df['P_L'].std() * np.sqrt(252/avg_days)) if len(trades_df) > 1 and trades_df['P_L'].std() != 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        # Export
        with st.expander("📊 Data Export"):
            csv = data.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h3>Welcome to Technical Analysis Dashboard</h3>
        <p>Select a stock and parameters, then click Analyze</p>
        <div style='display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin: 2rem 0;'>
    """, unsafe_allow_html=True)
    
    cols = st.columns(5)
    for idx, sym in enumerate(POPULAR_SYMBOLS[:10]):
        with cols[idx % 5]:
            if st.button(sym, use_container_width=True):
                st.session_state.custom_symbol = sym
                st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)

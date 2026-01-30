import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Technical Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .negative {
        color: #f44336;
        font-weight: bold;
    }
    .buy-signal {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .sell-signal {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #f44336;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to calculate drawdown
def calculate_drawdown(prices):
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max * 100
    return drawdown

# Function to calculate position metrics
def calculate_position_metrics(df, buy_rsi, buy_cci, sell_rsi, sell_cci):
    df = df.copy()
    df['Position'] = None
    df['Entry_Price'] = np.nan
    df['Exit_Price'] = np.nan
    df['P_L'] = np.nan
    df['Holding_Days'] = np.nan
    df['Signal'] = ''
    
    in_position = False
    entry_date = None
    entry_price = 0
    trades = []
    
    for i in range(1, len(df)):
        current_date = df.index[i]
        
        # BUY signal conditions
        if not in_position and df['RSI'].iloc[i] > buy_rsi and df['CCI'].iloc[i] > buy_cci:
            in_position = True
            entry_date = current_date
            entry_price = df['Close'].iloc[i]
            df.loc[current_date, 'Signal'] = 'BUY'
            df.loc[current_date, 'Position'] = 'LONG'
            df.loc[current_date, 'Entry_Price'] = entry_price
        
        # SELL signal conditions
        elif in_position and df['RSI'].iloc[i] < sell_rsi and df['CCI'].iloc[i] < sell_cci:
            in_position = False
            exit_price = df['Close'].iloc[i]
            holding_days = (current_date - entry_date).days
            
            # Calculate P&L
            pnl = (exit_price - entry_price) / entry_price * 100
            
            # Record trade details
            df.loc[current_date, 'Signal'] = 'SELL'
            df.loc[current_date, 'Position'] = 'EXIT'
            df.loc[current_date, 'Exit_Price'] = exit_price
            df.loc[current_date, 'P_L'] = pnl
            df.loc[current_date, 'Holding_Days'] = holding_days
            
            # Mark the entry with the final P&L
            df.loc[entry_date, 'P_L'] = pnl
            df.loc[entry_date, 'Holding_Days'] = holding_days
            
            # Add to trades list
            trades.append({
                'Entry_Date': entry_date,
                'Exit_Date': current_date,
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'P_L': pnl,
                'Holding_Days': holding_days
            })
    
    # Close any open position at the end
    if in_position:
        current_date = df.index[-1]
        exit_price = df['Close'].iloc[-1]
        holding_days = (current_date - entry_date).days
        pnl = (exit_price - entry_price) / entry_price * 100
        
        df.loc[current_date, 'Signal'] = 'SELL (End)'
        df.loc[current_date, 'P_L'] = pnl
        df.loc[current_date, 'Holding_Days'] = holding_days
        
        trades.append({
            'Entry_Date': entry_date,
            'Exit_Date': current_date,
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'P_L': pnl,
            'Holding_Days': holding_days
        })
    
    return df, trades

# Function to create interactive chart
def create_chart(data, buy_rsi, buy_cci, sell_rsi, sell_cci):
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Volume', 'RSI & CCI', 'ADX', 'MACD', 'Drawdown'),
        row_heights=[0.4, 0.15, 0.1, 0.15, 0.1]
    )
    
    # 1. Candlestick chart with volume
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['red' if close < open else 'green' 
              for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.3,
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add buy/sell markers
    buy_dates = data[data['Signal'] == 'BUY'].index
    sell_dates = data[data['Signal'].str.contains('SELL', na=False)].index
    
    if len(buy_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=data.loc[buy_dates, 'Low'] * 0.98,
                mode='markers',
                name='BUY',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                hovertemplate='BUY<extra></extra>'
            ),
            row=1, col=1
        )
    
    if len(sell_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=data.loc[sell_dates, 'High'] * 1.02,
                mode='markers',
                name='SELL',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                hovertemplate='SELL<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. RSI and CCI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='blue')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['CCI'], name='CCI', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Add RSI levels
    fig.add_hline(y=buy_rsi, line_dash="dash", line_color="green", opacity=0.7, 
                  annotation_text=f"BUY RSI > {buy_rsi}", row=2, col=1)
    fig.add_hline(y=sell_rsi, line_dash="dash", line_color="red", opacity=0.7,
                  annotation_text=f"SELL RSI < {sell_rsi}", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
    
    # Add CCI levels
    fig.add_hline(y=buy_cci, line_dash="dash", line_color="green", opacity=0.5,
                  annotation_text=f"BUY CCI > {buy_cci}", row=2, col=1)
    fig.add_hline(y=sell_cci, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text=f"SELL CCI < {sell_cci}", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
    
    # 3. ADX
    fig.add_trace(
        go.Scatter(x=data.index, y=data['ADX'], name='ADX', line=dict(color='purple')),
        row=3, col=1
    )
    fig.add_hline(y=25, line_dash="dash", line_color="orange", opacity=0.7,
                  annotation_text="Strong Trend > 25", row=3, col=1)
    
    # 4. MACD
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD_signal'], name='Signal', line=dict(color='red')),
        row=4, col=1
    )
    
    # MACD histogram
    colors_hist = ['green' if val >= 0 else 'red' for val in data['MACD_hist']]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['MACD_hist'],
            name='MACD Hist',
            marker_color=colors_hist,
            opacity=0.5,
            showlegend=False
        ),
        row=4, col=1
    )
    fig.add_hline(y=0, line_color="gray", opacity=0.3, row=4, col=1)
    
    # 5. Drawdown
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Drawdown'], name='Drawdown', 
                  line=dict(color='red'), fill='tozeroy'),
        row=5, col=1
    )
    fig.add_hline(y=0, line_color="gray", opacity=0.3, row=5, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'Technical Analysis - Buy: RSI>{buy_rsi} & CCI>{buy_cci}, Sell: RSI<{sell_rsi} & CCI<{sell_cci}',
        height=1000,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI/CCI", row=2, col=1)
    fig.update_yaxes(title_text="ADX", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=5, col=1)
    
    # Remove rangeslider from all except bottom plot
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=True, row=5, col=1)
    
    return fig

# Main Streamlit app
def main():
    st.markdown('<h1 class="main-header">📈 Technical Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Parameters</h2>', unsafe_allow_html=True)
        
        # Stock selection
        st.markdown("### Stock Selection")
        symbol = st.text_input("Stock Symbol", value="GOOGL").upper()
        
        # Period selection
        st.markdown("### Data Period")
        period_options = {
            "1 Year": "1y",
            "2 Years": "2y",
            "3 Years": "3y",
            "5 Years": "5y",
            "10 Years": "10y",
            "Max": "max"
        }
        selected_period = st.selectbox(
            "Select Period",
            list(period_options.keys()),
            index=0
        )
        period = period_options[selected_period]
        
        # RSI Parameters
        st.markdown("### RSI Parameters")
        buy_rsi = st.slider("BUY when RSI >", 50, 80, 60)
        sell_rsi = st.slider("SELL when RSI <", 20, 50, 30)
        
        # CCI Parameters
        st.markdown("### CCI Parameters")
        buy_cci = st.slider("BUY when CCI >", -100, 100, 0)
        sell_cci = st.slider("SELL when CCI <", -100, 100, 0)
        
        # Indicator Parameters
        st.markdown("### Indicator Settings")
        rsi_period = st.slider("RSI Period", 5, 30, 14)
        cci_period = st.slider("CCI Period", 5, 30, 20)
        adx_period = st.slider("ADX Period", 5, 30, 14)
        
        macd_fast = st.slider("MACD Fast Period", 5, 20, 12)
        macd_slow = st.slider("MACD Slow Period", 15, 35, 26)
        macd_signal = st.slider("MACD Signal Period", 5, 15, 9)
        
        # Action buttons
        st.markdown("---")
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
        st.markdown("---")
        
        # Information
        with st.expander("ℹ️ About this Dashboard"):
            st.write("""
            This dashboard performs technical analysis using:
            - **RSI (Relative Strength Index)**: Momentum oscillator
            - **CCI (Commodity Channel Index)**: Trend indicator
            - **ADX (Average Directional Index)**: Trend strength
            - **MACD (Moving Average Convergence Divergence)**: Trend following
            
            **Trading Strategy**:
            - **BUY** when RSI > [Buy RSI] AND CCI > [Buy CCI]
            - **SELL** when RSI < [Sell RSI] AND CCI < [Sell CCI]
            """)
    
    # Main content area
    if analyze_button:
        with st.spinner(f"Fetching {selected_period} of data for {symbol}..."):
            try:
                # Download data
                data = yf.download(symbol, period=period, progress=False)
                
                if data.empty:
                    st.error(f"No data found for {symbol}. Please check the symbol.")
                    return
                
                # Calculate technical indicators
                data['RSI'] = talib.RSI(data['Close'], timeperiod=rsi_period)
                data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=cci_period)
                data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=adx_period)
                data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(
                    data['Close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
                )
                data['Drawdown'] = calculate_drawdown(data['Close'])
                
                # Calculate trading signals
                data_with_signals, trades = calculate_position_metrics(
                    data, buy_rsi, buy_cci, sell_rsi, sell_cci
                )
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = data['Close'].iloc[-1]
                    price_change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        f"{price_change:.2f}%"
                    )
                
                with col2:
                    total_trades = len(trades)
                    winning_trades = len([t for t in trades if t['P_L'] > 0])
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%", f"{winning_trades}/{total_trades}")
                
                with col3:
                    if total_trades > 0:
                        total_pnl = sum(trade['P_L'] for trade in trades)
                        avg_pnl = total_pnl / total_trades
                        st.metric("Avg P&L per Trade", f"{avg_pnl:.2f}%")
                
                with col4:
                    if total_trades > 0:
                        avg_holding_days = np.mean([trade['Holding_Days'] for trade in trades])
                        st.metric("Avg Holding Days", f"{avg_holding_days:.1f}")
                
                # Create chart
                st.markdown('<h2 class="sub-header">📊 Interactive Chart</h2>', unsafe_allow_html=True)
                fig = create_chart(data_with_signals, buy_rsi, buy_cci, sell_rsi, sell_cci)
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade Details
                st.markdown('<h2 class="sub-header">📋 Trade History</h2>', unsafe_allow_html=True)
                
                if trades:
                    trades_df = pd.DataFrame(trades)
                    trades_df['Entry_Date'] = pd.to_datetime(trades_df['Entry_Date'])
                    trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date'])
                    
                    # Format the dataframe for display
                    display_df = trades_df.copy()
                    display_df['Entry_Price'] = display_df['Entry_Price'].apply(lambda x: f"${x:.2f}")
                    display_df['Exit_Price'] = display_df['Exit_Price'].apply(lambda x: f"${x:.2f}")
                    display_df['P_L'] = display_df['P_L'].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(
                        display_df.style.apply(
                            lambda x: ['background-color: rgba(76, 175, 80, 0.1)' if float(x['P_L'].replace('%', '')) > 0 
                                      else 'background-color: rgba(244, 67, 54, 0.1)' for _ in x],
                            axis=1
                        ),
                        use_container_width=True
                    )
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        best_trade = trades_df['P_L'].max()
                        st.metric("Best Trade", f"{best_trade:.2f}%")
                    
                    with col2:
                        worst_trade = trades_df['P_L'].min()
                        st.metric("Worst Trade", f"{worst_trade:.2f}%")
                    
                    with col3:
                        total_return = trades_df['P_L'].sum()
                        st.metric("Total Return", f"{total_return:.2f}%")
                    
                    with col4:
                        sharpe_ratio = (trades_df['P_L'].mean() / trades_df['P_L'].std()) * np.sqrt(252/avg_holding_days) if len(trades_df) > 1 and trades_df['P_L'].std() != 0 else 0
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                else:
                    st.info("No trades executed with the current parameters.")
                
                # Raw Data (collapsible)
                with st.expander("📁 View Raw Data"):
                    st.dataframe(data_with_signals.tail(20), use_container_width=True)
                    
                    # Download button
                    csv = data_with_signals.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Data (CSV)",
                        data=csv,
                        file_name=f"{symbol}_technical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Make sure you have TA-Lib installed. If not, try: pip install TA-Lib")
    
    else:
        # Welcome message
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h2>Welcome to the Technical Analysis Dashboard! 🚀</h2>
            <p style='font-size: 1.2rem;'>
                Configure your parameters in the sidebar and click "Analyze" to get started.
            </p>
            <div style='margin-top: 2rem;'>
                <h4>🎯 Popular Symbols to Try:</h4>
                <div style='display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 1rem;'>
                    <div class='metric-card'>AAPL</div>
                    <div class='metric-card'>MSFT</div>
                    <div class='metric-card'>GOOGL</div>
                    <div class='metric-card'>TSLA</div>
                    <div class='metric-card'>AMZN</div>
                    <div class='metric-card'>NFLX</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample analysis
        st.markdown("### 📊 Sample Analysis (GOOGL - 1 Year)")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.png", 
                    caption="Sample Technical Analysis Chart", use_column_width=True)
        
        with col2:
            st.markdown("""
            ### Key Features:
            
            ✅ **Interactive Controls**
            - Adjust all indicator parameters
            - Customize buy/sell thresholds
            
            ✅ **Multiple Timeframes**
            - From 1 year to max available data
            
            ✅ **Comprehensive Analysis**
            - Real-time calculations
            - Performance metrics
            - Trade history
            
            ✅ **Export Options**
            - Download data as CSV
            - Save charts as images
            
            ⚠️ **Note**: This is for educational purposes only.
            """)

if __name__ == "__main__":
    main()

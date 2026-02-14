import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mplfinance as mpf

#[Symbol, Date, Close, High, Low, Open, Volume, Stock_Cumulative_Return, MRP, MRP13, MRP25, Exp, DD_LOG, DD, DD_PCT, ta_DD_LOG, ST, OBV, AD, Beta, weighted_excessMR, weighted_MR, Score, SMA_200C, RS
#I_e, niftyOpen, niftyHigh, niftyLow, niftyClose]

# Page config   
st.set_page_config(page_title="Tech Analysis", layout="wide", initial_sidebar_state="expanded")

# Popular symbols for autocomplete
try:
    combined_data = pd.read_pickle(
        r"https://raw.githubusercontent.com/rizwan236/Streamlit_backtester/main/combined_ticker_data.pkl.gz", compression="gzip")
    #print(combined_data.columns.tolist())
    combined_data.fillna(0, inplace=True)
    POPULAR_SYMBOLS = combined_data["Symbol"].dropna().unique().tolist()
    latest_data = combined_data.groupby("Symbol").tail(1)
    top_25 = latest_data.nlargest(25, 'Score')[['Symbol', 'Score']]
    top_symbols = top_25['Symbol'].tolist()
    
except:
    POPULAR_SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX","JPM", "JNJ", "V", "WMT", "PG", "MA", "UNH", "HD", "BAC", "DIS", "ADBE"]
    top_symbols =["AAPL"]
    

def calculate_drawdown(prices):
    """Calculate drawdown using pandas operations"""
    # Ensure we have a Series
    if isinstance(prices, pd.Series):
        price_series = prices
    elif isinstance(prices, pd.DataFrame):
        # If it's a DataFrame, take the first column or flatten
        price_series = prices.iloc[:, 0] if prices.shape[1] > 0 else prices.iloc[:, 0]
    else:
        price_series = pd.Series(prices)
    
    # Calculate cumulative max
    cummax = price_series.expanding().max()
    
    # Calculate drawdown percentage
    drawdown = ((price_series - cummax) / cummax * 100)
    
    # Fill any NaN values (should only be the first value)
    return drawdown.fillna(0)
    
# Trading logic
def calculate_trades(df, buy_rsi, buy_cci, sell_rsi, sell_cci):
    """Calculate trading signals and performance"""
    df = df.copy()
    df['Signal'] = ''
    trades = []
    in_position, entry_price, entry_date = False, 0, None
    
    for i in range(0, len(df)):
        date = df.index[i]
        # Get scalar values using .iloc and convert to float
        close_val = float(df['Close'].iloc[i])
        opene_val = float(df['Open'].iloc[i])
        high_val = float(df['High'].iloc[i])
        low_val = float(df['Low'].iloc[i])
        volume_val = float(df['Volume'].iloc[i])
        rsi_val = float(df['RSI'].iloc[i])
        cci_val = float(df['CCI'].iloc[i])
        niftyClose = float(df['niftyClose'].iloc[i])
        SMAAD = float(df['SMAAD'].iloc[i])
        SMAAD2 = float(df['SMAAD'].iloc[i-5])
        AD = float(df['AD'].iloc[i])
        ADX= float(df['ADX'].iloc[i])
        ADX3= float(df['ADX'].iloc[i-5])
        PLUS_DI= float(df['PLUS_DI'].iloc[i])
        PLUS_DI3= float(df['PLUS_DI'].iloc[i-5])
        MINUS_DI= float(df['MINUS_DI'].iloc[i])
        MOMScore = float(df['MOMScore'].iloc[i])
        weighted_excessMR = float(df['weighted_excessMR'].iloc[i])
        SMAScore24 = float(df['SMAScore24'].iloc[i])
        ST= (df['ST'].iloc[i])
        EMAMRP24= float(df['EMAMRP24'].iloc[i])
        MRP= float(df['MRP'].iloc[i])
        EMAOBV= float(df['EMAOBV'].iloc[i])
        OBV= float(df['OBV'].iloc[i])
        SMAClose10= float(df['SMAClose10'].iloc[i])
        SMAClose30= float(df['SMAClose30'].iloc[i])
        SMAClose40= float(df['SMAClose40'].iloc[i])
        SMAClose40_21= float(df['SMAClose40'].iloc[i-21])
        sma_based_sma200= float(df['sma_based_sma200'].iloc[i])
        wkH52= float(df['wkH52'].iloc[i])
        wkL52= float(df['wkL52'].iloc[i])
        SMA_200C = float(df['SMA_200C'].iloc[i])
        MRP =float(df['MRP'].iloc[i])
        #OBV=float(df['OBV'].iloc[i])
        MOMScore =float(df['MOMScore'].iloc[i])
        RSI_e =float(df['RSI_e'].iloc[i])
        weighted_excessMR =float(df['weighted_excessMR'].iloc[i])
        #ST =pd.Series(data['ST'].values.flatten(), index=data.index) 
        DD_LOG =float(df['DD_LOG'].iloc[i])
        RSI_max = float(df['RSI_max'].iloc[i])
        RSI_min = float(df['RSI_min'].iloc[i])
        
        
        

        #if ctx.bars >= 250 and ctx.indicator("HT_TRENDMODE")[-1] >= 1 and ctx.indicator("LINEARREG_SLOPE_OBV")[-1] > 2 and ctx.niftyClose[-1] > ctx.indicator("niftyClosewkH52")[-1] * 0.0 and ctx.niftyClose[-1]*100 > ctx.indicator("SMAniftyClose10")[-1]*10 > ctx.indicator("SMAniftyClose30")[-1]*0 and ctx.indicator("SMAAD")[-2] * 1.0 < ctx.AD[-1] and ctx.indicator("ADX")[-3] < ctx.indicator("ADX")[-1] > 25 and ctx.indicator("PLUS_DI")[-1] > ctx.indicator("PLUS_DI")[-3] and (ctx.indicator("PLUS_DI")[-1] - ctx.indicator("MINUS_DI")[-1]) > 15 and ctx.volume[-90] >= 1000 and ctx.volume[-30] >= 1000 and ctx.volume[-3] >= 1000 and (0 != ctx.MOMScore[-1] > 1.5 or 0 != ctx.weighted_excessMR[-1] > 0.4) and ctx.indicator("SMAScore24")[-1] > -10 and (ctx.ST[-1] == 1 and ctx.indicator("EMAMRP24")[-1] < ctx.MRP[-1] and ctx.indicator("EMAOBV")[-1] * 1.0 < ctx.OBV[-1] and ctx.close[-1] > ctx.indicator("SMAClose10")[-1] > ctx.indicator("SMAClose30")[-1] > ctx.indicator("SMAClose40")[-1] and ctx.indicator("SMAClose40")[-21] < ctx.indicator("SMAClose40")[-1] > ctx.indicator("sma_based_sma200")[-1] and ctx.indicator("RSI_14")[-1] > 50 and ctx.close[-1] > ctx.indicator("wkH52")[-1] * .75 and ctx.close[-1] > ctx.indicator("wkL52")[-1] * 1.35):
        #if not in_position and rsi_val > buy_rsi and cci_val > buy_cci  and ST ==1:
        if not in_position and  ADX > 25 and (PLUS_DI- MINUS_DI) > 5 and volume_val >= 500 and ( MOMScore > -1.5 or weighted_excessMR > -0.4)  and (ST == 1 and EMAMRP24 < MRP*100 and EMAOBV * 1.0 < OBV and close_val > SMAClose10 > SMAClose30 > SMAClose40 and SMAClose40_21 < SMAClose40 > sma_based_sma200 and RSI_e > buy_rsi and  RSI_min > 40 and RSI_max > 58 and close_val > wkH52 * .65 and close_val > wkL52 * 1.35):    
            in_position, entry_price, entry_date = True, close_val, date
            df.loc[date, 'Signal'] = 'BUY'
        #if (ctx.bars >= 250 and (ctx.close[-1] < ctx.indicator("SMAClose40")[-1] and ctx.ST[-1] == -1 and ctx.DD_LOG[-1] > 15)) or ((0 != ctx.MOMScore[-1] < 1.5 or 0 != ctx.weighted_excessMR[-1] < 0.4 or ctx.DD_LOG[-1] > 25 or ctx.indicator("EMAOBV")[-1] * 0.95 > ctx.OBV[-1]) and ctx.ST[-1] == -1 and ctx.DD_LOG[-1] > 15 and (ctx.indicator("RSI_20")[-1] < 35 or ctx.indicator("CCI_34")[-1] < 0 or ctx.indicator("SMAClose10")[-1] < ctx.indicator("SMAClose30")[-1])):    
        #elif in_position and rsi_val < sell_rsi and cci_val < sell_cci :
        elif in_position and ((close_val < SMAClose40 and ST == -1 and DD_LOG > 15) or (( MOMScore < 1.5 or weighted_excessMR < 0.4 or DD_LOG > 25 or EMAOBV * 0.95 > OBV) and ST == -1 and DD_LOG > 15 and (sell_rsi < 35 or cci_val < 0 or SMAClose10 < SMAClose30))):    
            pnl = ((close_val - entry_price) / entry_price * 100) #if entry_price !=0 else 0
            holding = (date - entry_date).days
            df.loc[date, 'Signal'] = 'SELL'
            trades.append({
                'Entry': entry_date.date(), 'Exit': date.date(),
                'Buy Price': f"${entry_price:.2f}", 
                'Sell Price': f"${close_val:.2f}",
                'P&L': f"{pnl:+.2f}%", 
                'Days': holding
            })
            in_position = False
    
    # Close any open position
    if in_position:
        date = df.index[-1]
        close_val = float(df['Close'].iloc[-1])
        pnl = ((close_val - entry_price) / entry_price * 100)
        holding = (date - entry_date).days
        df.loc[date, 'Signal'] = 'SELL (End)'
        trades.append({
            'Entry': entry_date.date(), 'Exit': date.date(),
            'Buy Price': f"${entry_price:.2f}", 
            'Sell Price': f"${close_val:.2f}",
            'P&L': f"{pnl:+.2f}%", 
            'Days': holding
        })
    
    return df, trades



# Chart creation
def create_chart(df, buy_rsi, buy_cci, sell_rsi, sell_cci, symbol,):    
    """
    Simplified matplotlib chart that avoids candlestick plotting issues
    """
    # Ensure the index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), 
                             gridspec_kw={'height_ratios': [4, 1, 1, 1, 1]},
                             sharex=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Price Chart (Simplified - just close price)
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], color='blue', linewidth=2, label='Close Price')
    ax1.plot(df.index, df['SMAClose10'], color='green', linewidth=2, label='SMAClose10')
    ax1.plot(df.index, df['SMAClose30'], color='yellow', linewidth=2, label='SMAClose30')
    ax1.plot(df.index, df['SMAClose40'], color='pink', linewidth=2, label='SMAClose40')
    df['sma_based_sma200'] = df['sma_based_sma200'].replace(0, np.nan)
    ax1.plot(df.index, df['sma_based_sma200'], color='red', linewidth=2, label='sma_based_sma200')
    
    
    #mpf.plot(df, type='candle', ax=ax1, style='charles', show_nontrading=False)
    
    # Mark buy/sell signals
    buy_mask = df['Signal'] == 'BUY'
    sell_mask = df['Signal'].str.contains('SELL', na=False)
    
    if buy_mask.any():
        ax1.scatter(df.index[buy_mask], df.loc[buy_mask, 'Close'], 
                   marker='^', color='green', s=100, label='BUY', zorder=5)
    if sell_mask.any():
        ax1.scatter(df.index[sell_mask], df.loc[sell_mask, 'Close'], 
                   marker='v', color='red', s=100, label='SELL', zorder=5)
    
    ax1.set_ylabel('Price ($)', fontweight='bold')
    ax1.set_title(f'{symbol} - Price with Trading Signals', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    '''
    # Volume overlay
    ax1_vol = ax1.twinx()
    volume_colors = df['Close'].diff().apply(
        lambda x: 'green' if x >= 0 else 'red'
    )
    ax1_vol.bar(
        df.index,
        df['Volume']/1000,
        color=volume_colors,
        width=1.0,
        alpha=0.3,
        zorder=1
    )
    ax1_vol.plot(df.index, df['OBV'], color='black', linewidth=1.5, label='OBV', zorder=2)
    ax1_vol.set_yticks([])
    #ax1_vol.set_ylim(0, df['Volume'].max() * 4)  
    combined_max = max(df['Volume'].max(), df['OBV'].max()) *1.1
    ax1_vol.set_ylim(0, combined_max)
    '''
    ax1_vol = ax1.twinx()
    volume_colors = df['Close'].diff().apply(lambda x: 'green' if x >= 0 else 'red')    
    volume_norm = df['Volume'] / df['Volume'].max() * 100
    obc_norm = (df['OBV'] - df['OBV'].min()) / (df['OBV'].max() - df['OBV'].min()) * 100
    
    ax1_vol.bar(df.index, volume_norm, color=volume_colors, width=1.0, alpha=0.3, zorder=1)
    ax1_vol.plot(df.index, obc_norm, color='grey', linewidth=1.5, label='OBV (norm)', zorder=2)
    ax1_vol.set_ylabel('Normalised %', fontweight='bold')
    ax1_vol.set_ylim(0, 250)    
    
    # 2. RSI Chart
    ax2 = axes[1]
    ax2.plot(df.index, df['RSI'], color='purple', linewidth=1.5)
    ax2.axhline(y=70, color='green', linestyle='--', alpha=0.7) #buy_rsi
    ax2.axhline(y=30, color='red', linestyle='--', alpha=0.7) #sell_rsi
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax2.fill_between(df.index, 70, 100, alpha=0.1, color='green') #buy_rsi
    ax2.fill_between(df.index, 0, 30, alpha=0.1, color='red') #sell_rsi
    ax2.set_ylabel('RSI', fontweight='bold')
    ax2.set_title('RSI Indicator', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 3. CCI Chart
    ax3 = axes[2]
    ax3.plot(df.index, df['CCI'], color='orange', linewidth=1.5)
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7) #buy_cci
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7) #sell_cci
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_ylabel('CCI', fontweight='bold')
    ax3.set_title('CCI Indicator', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. MACD Chart
    ax4 = axes[3]
    ax4.plot(df.index, df['MACD'], color='blue', linewidth=1.5, label='MACD')
    ax4.plot(df.index, df['MACD_signal'], color='red', linewidth=1.5, label='Signal')
    ax4.bar(df.index, df['MACD_hist'], alpha=0.5, width=1, 
            color=['green' if h >= 0 else 'red' for h in df['MACD_hist']])
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.set_ylabel('MACD', fontweight='bold')
    ax4.set_title('MACD Indicator', fontsize=10, fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. ADX & Drawdown Chart
    ax5 = axes[4]
    ax5.plot(df.index, df['ADX'], color='blue', linewidth=1.5, label='ADX')
    # +DI (green)
    ax5.plot(df.index, df['PLUS_DI'], color='green', linewidth=1, label='+DI')
    # -DI (blue)
    ax5.plot(df.index, df['MINUS_DI'], color='red', linewidth=1, label='-DI')    
    ax5.plot(df.index, df['Drawdown'], color='black', linewidth=1.5, label='Drawdown', alpha=0.7)
    ax5.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Strong Trend')
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax5.set_ylabel('ADX / Drawdown', fontweight='bold')
    ax5.set_title('ADX & Drawdown', fontsize=10, fontweight='bold')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Format x-axis
    fig.autofmt_xdate(rotation=10)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

  
    # Add overall title
    fig.suptitle(f'{symbol} Technical Analysis Dashboard', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main App
st.title("📈 Technical Analysis Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    selected_symbol = st.selectbox(
        "Lookup Top25 highest score Symbol",
        options=top_symbols,
        #index=POPULAR_SYMBOLS.index("GOOGL") if "GOOGL" in POPULAR_SYMBOLS else 0,
        help="list of Top 25 highest score Symbols"
    )    
    
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
    period = st.selectbox("Period", ["1y"], index=0)  #, "2y", "3y", "5y", "10y", "max"
    
    # Trading rules
    st.subheader("📊 Trading Rules")
    col1, col2 = st.columns(2)
    with col1:
        buy_rsi = st.number_input("Buy RSI >", 0, 100, 50)
        sell_rsi = st.number_input("Sell RSI <", 0, 100, 35)
    with col2:
        buy_cci = st.number_input("Buy CCI >", -100, 100, 0)
        sell_cci = st.number_input("Sell CCI <", -100, 100, 0)

   
    # Indicator periods
    st.subheader("⚙️ Indicator Settings")
    #rsi_period = st.slider("RSI Period", 5, 30, 14, key='rsi')
    rsi_period = st.number_input("RSI Period",min_value=7,max_value= 64, value=21,key="rsi_input")
    #cci_period = st.slider("CCI Period", 5, 30, 20, key='cci')
    cci_period = st.number_input("CCI Period",min_value=7,max_value= 42, value=34, key="cci_input")
    #adx_period = st.slider("ADX Period", 5, 30, 14, key='adx')
    adx_period = st.number_input("ADX Period",min_value=7,max_value= 34, value= 21, key="adx_input")

    #macd_fast =14
    #macd_slow=26
    #macd_signal=9
    col1, col2, col3 = st.columns(3)
    with col1:
        #macd_fast = st.slider("MACD Fast", 5, 20, 12, key='fast')
        macd_fast = 14 #st.number_input("MACD Fast",min_value=5,max_value= 20, value= 14, key="fast")
    with col2:
        #macd_slow = st.slider("MACD Slow", 15, 35, 26, key='slow')
        macd_slow = 26# st.number_input("MACD Slow",min_value=15,max_value= 35, value= 26, key="slow")
    with col3:
        #macd_signal = st.slider("MACD Signal", 5, 15, 9, key='signal')
        macd_signal = 9#st.number_input("MACD Signal",min_value=5,max_value= 15, value= 9, key="signal")
     
    
    analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

# Main content
if analyze_button:
    try:
        # Download data
        with st.spinner(f"Fetching {symbol} data..."):
            #combined_data["Symbol"].dropna().unique().tolist()          
            data = combined_data.loc[combined_data["Symbol"] == symbol]
            data = data.reset_index(drop=True)
            data = data.copy()
            data["Date"] = pd.to_datetime(data["Date"])  
            data.set_index("Date", inplace=True)
            data.rename(columns={"Score": "MOMScore"}, inplace=True)   
            #pybroker.register_columns("Stock_Cumulative_Return", "MRP", "Exp", "DD_PCT", "DD_LOG", "ta_DD_LOG", "ST", "OBV", "AD", "MOMScore", "niftyOpen", "niftyHigh",
            #                          "niftyLow", "niftyClose", "SMA_200C", "AD", "weighted_MR", "weighted_excessMR", "Beta")         
            #data = yf.download(symbol, period=period, progress=False)
            if data.empty:
                st.error(f"❌ No data found for {symbol}")
                st.stop()


        
        # Calculate indicators
        with st.spinner("Calculating indicators..."):
            # Ensure we have proper Series objects (1D arrays)
            close = pd.Series(data['Close'].values.flatten(), index=data.index)
            high = pd.Series(data['High'].values.flatten(), index=data.index)
            low = pd.Series(data['Low'].values.flatten(), index=data.index)
            open_price = pd.Series(data['Open'].values.flatten(), index=data.index)
            volume = pd.Series(data['Volume'].values.flatten(), index=data.index)
            df['SMA_200C'] = df['SMA_200C'].replace(0, np.nan)
            SMA_200C = pd.Series(data['SMA_200C'].values.flatten(), index=data.index)  
            MRP =pd.Series(data['MRP'].values.flatten(), index=data.index) 
            OBV=pd.Series(data['OBV'].values.flatten(), index=data.index) 
            MOMScore =pd.Series(data['MOMScore'].values.flatten(), index=data.index) 
            RSI_e =pd.Series(data['RSI_e'].values.flatten(), index=data.index) 
            weighted_excessMR =pd.Series(data['weighted_excessMR'].values.flatten(), index=data.index) 
            ST =pd.Series(data['ST'].values.flatten(), index=data.index) 
            DD_LOG =pd.Series(data['DD_LOG'].values.flatten(), index=data.index) 
            
            
            data['RSI'] = ta.momentum.RSIIndicator(close, window=rsi_period).rsi()  #calculate_rsi(data['Close'], rsi_period)
            data['RSI_20'] = ta.momentum.RSIIndicator(close, window=24).rsi() 
            data['sma_based_sma200'] = ta.trend.sma_indicator(SMA_200C, window=42, fillna=False)
            data['SMAScore24'] = ta.trend.sma_indicator(MOMScore, window=7, fillna=False)
            data['EMAOBV']= ta.trend.sma_indicator(OBV, window=42, fillna=False)
            data['EMAClose12']= ta.trend.ema_indicator(close, window=50, fillna=False)
            data['EMAClose24']= ta.trend.ema_indicator(close, window=150, fillna=False)
            data['EMAClose52']= ta.trend.ema_indicator(close, window=200, fillna=False)
            data['SMAClose40'] = ta.trend.sma_indicator(close, window=200, fillna=False)
            data['SMAClose30'] = ta.trend.ema_indicator(close, window=150, fillna=False)
            data['SMAClose10']= ta.trend.ema_indicator(close, window=50, fillna=False)
            data['wkH52'] = close.rolling(250, min_periods=1).max()
            data['wkL52'] = close.rolling(250, min_periods=1).min()
            data['EMAMRP12'] = ta.trend.ema_indicator(MRP, window=50, fillna=False)
            data['EMAMRP52'] = ta.trend.ema_indicator(MRP, window=200, fillna=False)
            data['EMAMRP24'] = ta.trend.ema_indicator(MRP, window=150, fillna=False)
            data['ADX']  = ta.trend.adx(high, low, close, window=21, fillna=False)            
            data['PLUS_DI']=ta.trend.adx_pos(high, low, close, window=14, fillna=False)
            data['MINUS_DI']= ta.trend.adx_neg(high, low, close, window=14, fillna=False)
            data['RSI_max'] = RSI_e.rolling(64, min_periods=1).max()
            data['RSI_min'] = RSI_e.rolling(18, min_periods=1).min()            
            AD= data['AD'] = ta.volume.acc_dist_index(high, low, close, volume, fillna=False)
            data['SMAAD']= ta.trend.ema_indicator(AD, window=34, fillna=False)
            
            
            
            data['CCI'] = ta.trend.CCIIndicator(
            high, 
            low, 
            close, 
            window=cci_period
            ).cci() #calculate_cci(data['High'], data['Low'], data['Close'], cci_period)

            #ta.trend.sma_indicator(close, window=12, fillna=False)
            #ta.trend.ema_indicator(close, window=12, fillna=False)         
                
            data['ADX'] = ta.trend.ADXIndicator(
            high, 
            low, 
            close, 
            window=adx_period
            ).adx() #calculate_adx(data['High'], data['Low'], data['Close'], adx_period)

            macd_indicator = ta.trend.MACD(
                close,
                window_slow=macd_slow,
                window_fast=macd_fast,
                window_sign=macd_signal
            )
            data['MACD'] = macd_indicator.macd()
            data['MACD_signal'] = macd_indicator.macd_signal()
            data['MACD_hist'] = macd_indicator.macd_diff()  # This is the histogram
            
            
            #macd, signal, hist = calculate_macd(data['Close'], macd_fast, macd_slow, macd_signal)
            #data['MACD'] = macd.values
            #data['MACD_signal'] = signal.values
            #data['MACD_hist'] = hist.values
            
            data['Drawdown'] = calculate_drawdown(data['Close'])
        
        # Calculate trades
        data, trades = calculate_trades(data, buy_rsi, buy_cci, sell_rsi, sell_cci)
        
        # Display metrics
        # Display metrics - CORRECTED VERSION
        st.subheader("📊 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Get scalar values from the series
            current_price = float(data['Close'].iloc[-1])
            initial_price = float(data['Close'].iloc[0])
            price_change = ((current_price - initial_price) / initial_price * 100)
            st.metric("Current Price", f"{current_price:.2f}", f"1yr {price_change:+.2f}%")
        
        with col2:
            total_trades = len(trades)
            if trades:
                winning_trades = len([t for t in trades if float(t['P&L'].strip('%').replace('+', '')) > 0])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%", f"{winning_trades}/{total_trades}")
            else:
                st.metric("Win Rate", "0.0%", "0/0")
        
        with col3:
            if trades:
                pnls = [float(t['P&L'].strip('%').replace('+', '')) for t in trades]
                avg_pnl = float(np.mean(pnls))
                st.metric("Avg P&L", f"{avg_pnl:+.2f}%")
            else:
                st.metric("Avg P&L", "0.00%")
        
        with col4:
            if trades:
                avg_days = float(np.mean([t['Days'] for t in trades]))
                st.metric("Avg Days Held", f"{avg_days:.1f}")
            else:
                st.metric("Avg Days Held", "0")
        
        # In your analyze button section, replace the plotly chart with:
        # In your analyze section, replace the plotly chart with:
        stock1 = yf.Ticker(symbol)
        
        company_name = stock1.info.get("shortName", symbol)
        
        st.subheader(f"📈 Technical Analysis Chart — {company_name} ({symbol})")
        
        #st.subheader("📈 Technical Analysis Chart")
        
        try:
            # Use the simple chart to avoid issues
            fig = create_chart(data, buy_rsi, buy_cci, sell_rsi, sell_cci, symbol,)
            
            # Display in Streamlit
            st.pyplot(fig)
            
            # Optional: Add download button
            if st.button("💾 Save Chart as PNG"):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{symbol}_chart_{timestamp}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                st.success(f"Chart saved as {filename}")
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            st.info("Showing data table instead...")
            st.dataframe(data.tail(10))
        
        # Display trades
        # Performance metrics - CORRECTED VERSION
        if trades:
            st.subheader("🎯 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            # Convert P&L strings to floats
            pnl_values = []
            for t in trades:
                pnl_str = t['P&L'].strip('%')
                # Handle both positive and negative values
                if pnl_str.startswith('+'):
                    pnl_values.append(float(pnl_str[1:]))
                elif pnl_str.startswith('-'):
                    pnl_values.append(float(pnl_str))
                else:
                    pnl_values.append(float(pnl_str))
            
            with col1:
                best_trade = float(np.max(pnl_values))
                st.metric("Best Trade", f"{best_trade:+.2f}%")
            
            with col2:
                worst_trade = float(np.min(pnl_values))
                st.metric("Worst Trade", f"{worst_trade:+.2f}%")
            
            with col3:
                total_return = float(np.sum(pnl_values))
                st.metric("Total Return", f"{total_return:+.2f}%")
            
            with col4:
                if len(pnl_values) > 1 and np.std(pnl_values) > 0:
                    avg_days = float(np.mean([t['Days'] for t in trades]))
                    sharpe = (np.mean(pnl_values) / np.std(pnl_values)) * np.sqrt(252/avg_days)
                    st.metric("Sharpe Ratio", f"{float(sharpe):.2f}")
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

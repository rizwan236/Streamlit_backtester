import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TJI Symbol Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data (cached) ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data …")
def load_data():
    url = (
        "https://raw.githubusercontent.com/rizwan236/Streamlit_backtester/"
        "main/combined_ticker_data.pkl.gz"
    )
    df = pd.read_pickle(url, compression="gzip")
    df.fillna(0, inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    df = df[
        df["Symbol"].str.startswith("TJI_", na=False)
        | (df["Symbol"] == "^NSEI")
    ].copy()

    df = df.sort_values(["Symbol", "Date"])
    first_close = df.groupby("Symbol")["Close"].transform("first")
    df["Close_Base1000"] = (df["Close"] / first_close) * 1000.0
    df["Close"] = df["Close_Base1000"]
    return df

df = load_data()

if df.empty:
    st.error("No data found after filtering (TJI_* and ^NSEI).")
    st.stop()

all_symbols = sorted(df["Symbol"].unique())
BENCHMARK = "^NSEI"

# ── Session state defaults ──────────────────────────────────────────────────
default_symbols = {BENCHMARK} | set(
    [s for s in all_symbols if s != BENCHMARK][:3]
)
for s in all_symbols:
    if f"sym_{s}" not in st.session_state:
        st.session_state[f"sym_{s}"] = s in default_symbols

def get_checked_symbols():
    return [s for s in all_symbols if st.session_state.get(f"sym_{s}", False)]

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

checked = get_checked_symbols()
with st.sidebar.popover(
    f"📌 Symbols  ({len(checked)}/{len(all_symbols)} selected)",
    use_container_width=True,
):
    b1, b2 = st.columns(2)
    if b1.button("✅ Select All", use_container_width=True):
        for s in all_symbols:
            st.session_state[f"sym_{s}"] = True
        st.rerun()
    if b2.button("❌ Clear All", use_container_width=True):
        for s in all_symbols:
            st.session_state[f"sym_{s}"] = False
        st.rerun()

    st.markdown("---")

    if BENCHMARK in all_symbols:
        st.checkbox(
            f"{BENCHMARK}  (benchmark — always shown)",
            value=True,
            disabled=True,
            key="benchmark_pinned",
        )
        st.session_state[f"sym_{BENCHMARK}"] = True

    st.markdown("---")

    for s in all_symbols:
        if s == BENCHMARK:
            continue
        st.checkbox(s, key=f"sym_{s}")

selected_symbols = get_checked_symbols()

# ── Latest Score filter ─────────────────────────────────────────────────────
st.sidebar.subheader("Score Filter (latest value)")
score_mode = st.sidebar.radio(
    "Condition", ["Greater than", "Less than"], horizontal=True
)
score_value = st.sidebar.number_input(
    "Score threshold", value=0.0, step=0.1, format="%.4f"
)

latest_scores = (
    df.sort_values("Date").groupby("Symbol")["Score"].last().reset_index()
)
if score_mode == "Greater than":
    valid_symbols = latest_scores.loc[
        latest_scores["Score"] > score_value, "Symbol"
    ].tolist()
else:
    valid_symbols = latest_scores.loc[
        latest_scores["Score"] < score_value, "Symbol"
    ].tolist()

valid_symbols_set = set(valid_symbols) | {BENCHMARK}

final_symbols = [s for s in selected_symbols if s in valid_symbols_set]
if BENCHMARK in all_symbols and BENCHMARK not in final_symbols:
    final_symbols.insert(0, BENCHMARK)

# ── Metric selector ─────────────────────────────────────────────────────────
metric_options = [
    #"Close_Base1000",
    "Close",
    "Stock_Cumulative_Return",
    "MRP",
    "MRP13",
    "MRP25",
    "DD",
    "DD_PCT",
    "OBV",
    "AD",
    "Beta",
    "Score",
    "SMA_200C",
    "RS",
    "niftyClose",
]
metric = st.sidebar.selectbox("📊 Metric to plot", metric_options, index=0)

# ── Main panel ──────────────────────────────────────────────────────────────
st.title("📈 TJI Symbol Interactive Dashboard")

if not final_symbols:
    st.warning("No symbols match the current filters. Adjust the sidebar.")
    st.stop()

plot_df = df[df["Symbol"].isin(final_symbols)].copy().sort_values("Date")

# ── Columns to show in hover tooltip ────────────────────────────────────────
hover_metrics = [
    "Close", "Volume", "Score",
    "MRP", "MRP13", "MRP25", "DD", "DD_PCT",
    "Beta", "RS", "SMA_200C", "Stock_Cumulative_Return",
]
hover_metrics = [m for m in hover_metrics if m in plot_df.columns]

# ── Build chart with full control ───────────────────────────────────────────
fig = go.Figure()

for sym in final_symbols:
    sub = plot_df[plot_df["Symbol"] == sym].sort_values("Date").copy()
    if sub.empty:
        continue

    # Build hover template with all metrics for this symbol only
    hovertemplate = (
        f"<b style='font-size:14px'>{sym}</b><br>"
        "<span style='color:#888'>Date</span>: %{{x|%Y-%m-%d}}<br>"
        "<span style='color:#888'>Plotted</span>: %{{y:.4f}}<br>"
    )
    for i, m in enumerate(hover_metrics):
        hovertemplate += f"<span style='color:#888'>{m}</span>: %{{customdata[{i}]:.4f}}<br>"
    hovertemplate += "<extra></extra>"  # hide trace-name box

    customdata = sub[hover_metrics].values

    # ^NSEI rendered as a distinct dotted black line
    if sym == BENCHMARK:
        line_style = dict(color="#000000", width=3, dash="dot")
        opacity = 0.9
        legend_rank = 0
    else:
        line_style = dict(width=2)
        opacity = 1.0
        legend_rank = 1

    fig.add_trace(
        go.Scatter(
            x=sub["Date"],
            y=sub[metric],
            mode="lines",
            name=sym,
            line=line_style,
            opacity=opacity,
            customdata=customdata,
            hovertemplate=hovertemplate,
            legendrank=legend_rank,
        )
    )

# ── Mark ^NSEI min & max on the plotted metric ──────────────────────────────
nsei = plot_df[plot_df["Symbol"] == BENCHMARK].sort_values("Date")
if not nsei.empty and nsei[metric].notna().any():
    idx_max = nsei[metric].idxmax()
    idx_min = nsei[metric].idxmin()
    x_max, y_max = nsei.loc[idx_max, "Date"], nsei.loc[idx_max, metric]
    x_min, y_min = nsei.loc[idx_min, "Date"], nsei.loc[idx_min, metric]

    fig.add_trace(
        go.Scatter(
            x=[x_max, x_min],
            y=[y_max, y_min],
            mode="markers+text",
            name=f"{BENCHMARK} High/Low",
            marker=dict(
                size=15,
                color=["#2ca02c", "#d62728"],   # green=high, red=low
                symbol=["triangle-up", "triangle-down"],
                line=dict(width=2, color="black"),
            ),
            text=[f"HIGH  {y_max:,.2f}", f"LOW  {y_min:,.2f}"],
            textposition=["top center", "bottom center"],
            textfont=dict(size=12, color="black"),
            hovertemplate=(
                f"<b>{BENCHMARK}</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Value: %{y:,.4f}<extra></extra>"
            ),
            showlegend=True,
        )
    )

# ── Layout: bigger chart, per-trace hover ───────────────────────────────────
fig.update_layout(
    title=f"{metric} over Time"
    + (" (base 1000)" if metric == "Close_Base1000" else ""),
    hovermode="closest",       # ← hover shows ONLY the trace under the cursor
    template="plotly_white",
    height=900,                # ← bigger chart window
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12),
    ),
    margin=dict(l=60, r=40, t=80, b=60),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#444",
        font=dict(size=12, family="monospace"),
        align="left",
    ),
)
fig.update_xaxes(showgrid=True, gridcolor="#eee")
fig.update_yaxes(showgrid=True, gridcolor="#eee")

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ── Raw data expander ───────────────────────────────────────────────────────
with st.expander("📋 Show raw data for selected symbols"):
    st.dataframe(
        plot_df[["Date", "Symbol", "Close", metric]]
        .drop_duplicates(subset=["Date", "Symbol"])
        .sort_values(["Symbol", "Date"]),
        use_container_width=True,
    )

# ── Sidebar footer ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Showing **{len(final_symbols)}** symbol(s) · "
    f"Metric: **{metric}** · "
    f"Score {score_mode.lower()} **{score_value}**"
)

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

# ── Latest score per symbol (for legend sorting/labels) ─────────────────────
latest_scores_map = (
    df.sort_values("Date").groupby("Symbol")["Score"].last().to_dict()
)

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

    # Sort TJI symbols by latest Score descending in the checkbox list too
    tjis_sorted = sorted(
        [s for s in all_symbols if s != BENCHMARK],
        key=lambda s: latest_scores_map.get(s, float("-inf")),
        reverse=True,
    )
    for s in tjis_sorted:
        score_val = latest_scores_map.get(s, 0.0)
        st.checkbox(f"{s}  ·  Score {score_val:.2f}", key=f"sym_{s}")

selected_symbols = get_checked_symbols()


# ── Latest Score filter: BETWEEN a range ────────────────────────────────────
st.sidebar.subheader("Score Filter (latest value)")

# Global min / max (excluding benchmark) to size the slider sensibly
tji_scores = [v for s, v in latest_scores_map.items() if s != BENCHMARK]
if tji_scores:
    score_min_data = float(min(tji_scores))
    score_max_data = float(max(tji_scores))
else:
    score_min_data, score_max_data = 0.0, 1.0

if score_min_data == score_max_data:
    score_min_data -= 0.5
    score_max_data += 0.5

range_mode = st.sidebar.radio(
    "Range input", ["Slider", "Manual"], horizontal=True
)

if range_mode == "Slider":
    score_range = st.sidebar.slider(
        "Score between",
        min_value=float(score_min_data),
        max_value=float(score_max_data),
        value=(float(score_min_data), float(score_max_data)),
        step=0.01,
        format="%.2f",
    )
else:
    c1, c2 = st.sidebar.columns(2)
    score_lo = c1.number_input(
        "Min", value=float(score_min_data), step=0.1, format="%.4f"
    )
    score_hi = c2.number_input(
        "Max", value=float(score_max_data), step=0.1, format="%.4f"
    )
    score_range = (min(score_lo, score_hi), max(score_lo, score_hi))

score_lo, score_hi = score_range

valid_symbols = [
    s for s, v in latest_scores_map.items()
    if score_lo <= v <= score_hi
]
valid_symbols_set = set(valid_symbols) | {BENCHMARK}

final_symbols = [s for s in selected_symbols if s in valid_symbols_set]
if BENCHMARK in all_symbols and BENCHMARK not in final_symbols:
    final_symbols.insert(0, BENCHMARK)
'''
final_symbols = sorted(
    final_symbols,
    key=lambda s: (
        0 if s == BENCHMARK else 1,
        -latest_scores_map.get(s, float("-inf")),
    ),
)


# ── Latest Score filter ─────────────────────────────────────────────────────
st.sidebar.subheader("Score Filter (latest value)")
score_mode = st.sidebar.radio(
    "Condition", ["Greater than", "Less than"], horizontal=True
)
score_value = st.sidebar.number_input(
    "Score threshold", value=0.0, step=0.1, format="%.4f"
)

if score_mode == "Greater than":
    valid_symbols = [
        s for s, v in latest_scores_map.items() if v > score_value
    ]
else:
    valid_symbols = [
        s for s, v in latest_scores_map.items() if v < score_value
    ]

valid_symbols_set = set(valid_symbols) | {BENCHMARK}

final_symbols = [s for s in selected_symbols if s in valid_symbols_set]
if BENCHMARK in all_symbols and BENCHMARK not in final_symbols:
    final_symbols.insert(0, BENCHMARK)
'''
# ── Sort final_symbols: benchmark first, then by score descending ───────────
final_symbols = sorted(
    final_symbols,
    key=lambda s: (
        0 if s == BENCHMARK else 1,                       # benchmark first
        -latest_scores_map.get(s, float("-inf")),         # then desc score
    ),
)

# ── Metric selector ─────────────────────────────────────────────────────────
metric_options = [
    "Close_Base1000",
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

# ── Hover metrics ───────────────────────────────────────────────────────────
hover_metrics = [
    "Close_Base1000", "Close", "Volume", "Score",
    "MRP", "MRP13", "MRP25", "DD", "DD_PCT",
    "Beta", "RS", "SMA_200C", "Stock_Cumulative_Return",
]
hover_metrics = [m for m in hover_metrics if m in plot_df.columns]

# ── Build chart ─────────────────────────────────────────────────────────────
fig = go.Figure()

# Palette for TJI symbols (assign by index so it's stable)
palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]
color_idx = 0

for sym in final_symbols:
    sub = plot_df[plot_df["Symbol"] == sym].sort_values("Date").copy()
    if sub.empty:
        continue

    # Build the hover template (all metrics for THIS symbol only)
    # Use explicit font colors inside the tooltip for readability
    hovertemplate = (
        f"<b style='font-size:14px;color:#ffffff'>{sym}</b><br>"
        f"<span style='color:#c8c8c8'>Date</span>: "
        f"<span style='color:#ffffff'>%{{x|%Y-%m-%d}}</span><br>"
        f"<span style='color:#c8c8c8'>Plotted</span>: "
        f"<span style='color:#ffffff'>%{{y:.4f}}</span><br>"
    )
    for i, m in enumerate(hover_metrics):
        hovertemplate += (
            f"<span style='color:#c8c8c8'>{m}</span>: "
            f"<span style='color:#ffffff'>%{{customdata[{i}]:.4f}}</span><br>"
        )
    hovertemplate += "<extra></extra>"

    customdata = sub[hover_metrics].values

    score_val = latest_scores_map.get(sym, 0.0)
    legend_label = f"{sym}_{score_val:.2f}"

    if sym == BENCHMARK:
        line_style = dict(color="#FFFFFF", width=3, dash="dot")
        opacity = 0.95
        legend_rank = 0
    else:
        line_style = dict(color=palette[color_idx % len(palette)], width=2)
        color_idx += 1
        opacity = 1.0
        legend_rank = None

    fig.add_trace(
        go.Scatter(
            x=sub["Date"],
            y=sub[metric],
            mode="lines",
            name=legend_label,
            line=line_style,
            opacity=opacity,
            customdata=customdata,
            hovertemplate=hovertemplate,
            legendrank=legend_rank,
        )
    )

# ── Mark ^NSEI high / low ───────────────────────────────────────────────────
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
                color=["#2ca02c", "#d62728"],
                symbol=["triangle-up", "triangle-down"],
                line=dict(width=2, color="black"),
            ),
            text=[f"HIGH  {y_max:,.2f}", f"LOW  {y_min:,.2f}"],
            textposition=["top center", "bottom center"],
            textfont=dict(size=12, color="black"),
            hovertemplate=(
                f"<b style='color:#ffffff'>{BENCHMARK}</b><br>"
                "<span style='color:#c8c8c8'>Date</span>: "
                "<span style='color:#ffffff'>%{x|%Y-%m-%d}</span><br>"
                "<span style='color:#c8c8c8'>Value</span>: "
                "<span style='color:#ffffff'>%{y:,.4f}</span>"
                "<extra></extra>"
            ),
            showlegend=True,
        )
    )

# ── Layout ──────────────────────────────────────────────────────────────────
fig.update_layout(
    #title=f"{metric} over Time"
    title=f"{metric}"
    + (" (base 1000)" if metric == "Close_Base1000xx" else ""),
    hovermode="closest",
    template="plotly_white",
    height=900,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=13, color="#FFFFFF"),
        #bgcolor="rgba(0,0,0,0)",   
        traceorder="normal",           # honor trace insertion order (= score desc)
    ),
    margin=dict(l=60, r=40, t=90, b=60),
    hoverlabel=dict(
        bgcolor="#1f2430",             # dark background
        bordercolor="#000000",
        font=dict(size=13, family="monospace", color="#ffffff"),
        align="left",
    ),
)
fig.update_xaxes(showgrid=True, gridcolor="#eee")
fig.update_yaxes(showgrid=True, gridcolor="#eee")

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False},
)

# ── Raw data expander ───────────────────────────────────────────────────────
with st.expander("📋 Show raw data for selected symbols"):
    st.dataframe(
        plot_df[["Date", "Symbol", "Close", "Close_Base1000", metric]]
        .drop_duplicates(subset=["Date", "Symbol"])
        .sort_values(["Symbol", "Date"]),
        use_container_width=True,
    )

# ── Sidebar footer ──────────────────────────────────────────────────────────

st.sidebar.markdown("---")
#st.sidebar.caption(
#    f"Showing **{len(final_symbols)}** symbol(s) · "
#    f"Metric: **{metric}** · "
#    f"Score {score_mode.lower()} **{score_value}**"
#)

st.sidebar.caption(
    f"Showing **{len(final_symbols)}** symbol(s) · "
    f"Metric: **{metric}** · "
    f"Score between **{score_lo:.2f}** and **{score_hi:.2f}**"
)


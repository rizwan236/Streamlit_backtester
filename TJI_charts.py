import streamlit as st
import pandas as pd
import plotly.express as px

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

    # Keep TJI_* symbols and ^NSEI benchmark
    df = df[
        df["Symbol"].str.startswith("TJI_", na=False)
        | (df["Symbol"] == "^NSEI")
    ].copy()

    # Normalize Close to start at 1000 per symbol
    df = df.sort_values(["Symbol", "Date"])
    first_close = df.groupby("Symbol")["Close"].transform("first")
    df["Close_Base1000"] = (df["Close"] / first_close) * 1000.0
    df["Close"] =df["Close_Base1000"] 

    return df

df = load_data()

if df.empty:
    st.error("No data found after filtering (TJI_* and ^NSEI).")
    st.stop()

all_symbols = sorted(df["Symbol"].unique())
BENCHMARK = "^NSEI"

# ── Initialize session_state for each symbol checkbox ────────────────────────
# Default: ^NSEI always checked; first 3 TJI_ symbols checked
default_symbols = {BENCHMARK} | set(
    [s for s in all_symbols if s != BENCHMARK][:3]
)

for s in all_symbols:
    key = f"sym_{s}"
    if key not in st.session_state:
        st.session_state[key] = s in default_symbols

# ── Helper: get currently checked symbols ───────────────────────────────────
def get_checked_symbols():
    return [s for s in all_symbols if st.session_state.get(f"sym_{s}", False)]

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

# ── Dropdown (popover) with checkbox list ───────────────────────────────────
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

    # Benchmark pinned at the top (cannot be unchecked)
    if BENCHMARK in all_symbols:
        st.checkbox(
            f"{BENCHMARK}  (benchmark — always shown)",
            value=True,
            disabled=True,
            key="benchmark_pinned",
        )
        st.session_state[f"sym_{BENCHMARK}"] = True  # force-on

    st.markdown("---")

    # TJI_ symbols — tick / untick
    for s in all_symbols:
        if s == BENCHMARK:
            continue
        st.checkbox(s, key=f"sym_{s}")

selected_symbols = get_checked_symbols()

# ── Latest Score filter (greater / less than) ───────────────────────────────
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

# ^NSEI bypasses the Score filter (benchmark)
valid_symbols_set = set(valid_symbols) | {BENCHMARK}

# Build final list from checked symbols, honoring the Score filter
# (^NSEI is always kept if it was checked — it's already forced True above)
final_symbols = [s for s in selected_symbols if s in valid_symbols_set]

# Safety: ensure benchmark is always in the final list
if BENCHMARK in all_symbols and BENCHMARK not in final_symbols:
    final_symbols.insert(0, BENCHMARK)

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

# ── Chart ───────────────────────────────────────────────────────────────────
fig = px.line(
    plot_df,
    x="Date",
    y=metric,
    color="Symbol",
    title=f"{metric} over Time"
    + (" (base 1000)" if metric == "Close_Base1000" else ""),
    labels={metric: metric, "Date": "Date"},
    template="plotly_white",
)
fig.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=600,
)
st.plotly_chart(fig, use_container_width=True)

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
st.sidebar.caption(
    f"Showing **{len(final_symbols)}** symbol(s) · "
    f"Metric: **{metric}** · "
    f"Score {score_mode.lower()} **{score_value}**"
)

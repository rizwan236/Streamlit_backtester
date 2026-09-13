import streamlit as st
import pandas as pd
import plotly.express as px
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
    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# ── Initial filter: only symbols starting with TJI_ ─────────────────────────
#df = df[df["Symbol"].str.startswith("TJI_", na=False)].copy()
df = df[
    df["Symbol"].str.startswith("TJI_", na=False)
    | df["Symbol"].str.startswith("^NSEI", na=False)
].copy()

if df.empty:
    st.error("No TJI_ symbols found in the dataset.")
    st.stop()

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

# 1. Symbol multi‑select
all_symbols = sorted(df["Symbol"].unique())
selected_symbols = st.sidebar.multiselect(
    "Select Symbols",
    options=all_symbols,
    default=all_symbols[:3],          # first 3 by default
    help="Choose one or more TJI_ symbols to display.",
)

# 2. Latest Score filter (greater / less than)
st.sidebar.subheader("Score Filter (latest value)")
score_mode = st.sidebar.radio(
    "Condition", ["Greater than", "Less than"], horizontal=True
)
score_value = st.sidebar.number_input(
    "Score threshold",
    value=0.0,
    step=0.1,
    format="%.4f",
)

# Compute latest score per symbol
latest_scores = (
    df.sort_values("Date")
    .groupby("Symbol")["Score"]
    .last()
    .reset_index()
)

if score_mode == "Greater than":
    valid_symbols = latest_scores.loc[
        latest_scores["Score"] > score_value, "Symbol"
    ].tolist()
else:
    valid_symbols = latest_scores.loc[
        latest_scores["Score"] < score_value, "Symbol"
    ].tolist()

# Intersect with manual selection (if any)
if selected_symbols:
    final_symbols = [s for s in selected_symbols if s in valid_symbols]
else:
    final_symbols = valid_symbols

# ── Metric selector ──────────────────────────────────────────────────────────
metric_options = [
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

# ── Main panel ───────────────────────────────────────────────────────────────
st.title("📈 TJI Symbol Interactive Dashboard")

if not final_symbols:
    st.warning("No symbols match the current filters. Adjust the sidebar.")
    st.stop()

# Filter data for the chosen symbols
plot_df = df[df["Symbol"].isin(final_symbols)].copy()

# Ensure Date is sorted
plot_df = plot_df.sort_values("Date")

# ── Chart ────────────────────────────────────────────────────────────────────
fig = px.line(
    plot_df,
    x="Date",
    y=metric,
    color="Symbol",
    title=f"{metric} over Time",
    labels={metric: metric, "Date": "Date"},
    template="plotly_white",
)
fig.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=600,
)
st.plotly_chart(fig, use_container_width=True)

# ── Optional: raw data table ─────────────────────────────────────────────────
with st.expander("📋 Show raw data for selected symbols"):
    st.dataframe(
        plot_df[["Date", "Symbol", metric]].sort_values(["Symbol", "Date"]),
        use_container_width=True,
    )

# ── Sidebar info ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Showing **{len(final_symbols)}** symbol(s) · "
    f"Metric: **{metric}** · "
    f"Score {score_mode.lower()} **{score_value}**"
)

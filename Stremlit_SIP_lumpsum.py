# [Symbol, Date, Close, High, Low, Open, Volume, Stock_Cumulative_Return, MRP, Rolling_52DDpct, DD_PCT, DD_LOG, ST, #OBV, AD, maxBrk, SMA_200, Volume_50_Median, RSI_21, weighted_excessMR, weighted_MR, Score, ZScore3m, #finmagine_mom_score, avgRSI, niftyOpen, niftyHigh, niftyLow, niftyClose]


"""
Advanced Investment Dashboard - Lumpsum & SIP Analyzer
=======================================================
Run:       streamlit run invest_dashboard.py
Required:  pip install streamlit pandas numpy plotly openpyxl
Optional:  pip install yfinance fpdf2     (historical backtest / PDF export)

Model notes (all simplifications are deliberate and stated in the UI):
- SIP instalments are invested at the START of each period; the first one at t=0.
- Net return = CAGR - expense ratio; periodic rate = (1+net)^(1/periods_per_year) - 1.
- Tax is a flat rate on total gains (above an exemption) applied once at the end.
"""

import io
from dataclasses import astuple, dataclass, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

'''
# Allow `python invest_dashboard.py` as well as `streamlit run invest_dashboard.py`
if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as _st_running

        if not _st_running():
            import subprocess
            import sys

            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
            sys.exit(0)
    except ImportError:
        pass
'''
st.set_page_config(page_title="Lumpsum & SIP Analyzer", page_icon="📈", layout="wide")

try:
    _ver = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _ver = (1, 50)
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

CURRENCIES = {
    "INR (₹)": ("₹", "IN", "INR"),
    "USD ($)": ("$", "INTL", "USD"),
    "EUR (€)": ("€", "INTL", "EUR"),
    "GBP (£)": ("£", "INTL", "GBP"),
}
FREQ = {"Weekly": 52, "Monthly": 12, "Yearly": 1}
CUR = CURRENCIES["INR (₹)"]
TEMPLATE = "plotly_white"


# ───────────────────────── formatting helpers ─────────────────────────
def _group(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail, parts = s[:-3], s[-3:], []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def money(x, short=False, plain=False):
    sym, style, code = CUR
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    neg = "-" if x < 0 else ""
    a = abs(float(x))
    prefix = f"{code} " if plain else sym
    if short:
        units = [(1e7, " Cr"), (1e5, " L")] if style == "IN" else [(1e9, " B"), (1e6, " M"), (1e3, " K")]
        for div, suf in units:
            if a >= div:
                return f"{neg}{prefix}{a / div:,.2f}{suf}"
    n = int(round(a))
    body = _group(n) if style == "IN" else f"{n:,}"
    return f"{neg}{prefix}{body}"


def pct(x):
    return "–" if x is None or np.isnan(x) else f"{x * 100:.2f}%"


def style(fig, title, ytitle="Value", xtitle="Years", h=420):
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        height=h,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_yaxes(title=ytitle, tickprefix=CUR[0], tickformat=".2~s")
    fig.update_xaxes(title=xtitle)
    return fig


# ───────────────────────── calculation engine ─────────────────────────
@dataclass(frozen=True)
class Plan:
    lump: float
    sip: float
    ppy: int  # periods per year: 52 / 12 / 1
    years: int
    cagr: float  # gross annual return, decimal
    expense: float = 0.0  # annual fee, decimal
    step_mode: str = "None"  # None | Percent | Amount
    step_val: float = 0.0
    pause_start_m: int = 0
    pause_len_m: int = 0
    shock_pct: float = 0.0  # one-off market crash, %
    shock_year: float = 0.0


@dataclass(frozen=True)
class Econ:
    infl: float
    tax_on: bool
    tax_rate: float
    exempt: float


def step_amount(p: Plan, yr):
    yr = np.asarray(yr)
    if p.step_mode == "Percent":
        return p.sip * (1 + p.step_val / 100) ** yr
    if p.step_mode == "Amount":
        return p.sip + p.step_val * yr
    return np.full(yr.shape, float(p.sip))


def contributions(p: Plan) -> np.ndarray:
    n = p.years * p.ppy
    k = np.arange(n)
    c = step_amount(p, k // p.ppy)
    if p.pause_len_m > 0:
        t_m = k / p.ppy * 12
        c = np.where((t_m >= p.pause_start_m) & (t_m < p.pause_start_m + p.pause_len_m), 0.0, c)
    return np.maximum(c, 0.0)


def growth_factors(p: Plan) -> np.ndarray:
    n = p.years * p.ppy
    net = max(p.cagr - p.expense, -0.99)
    g = np.full(n, (1 + net) ** (1 / p.ppy))
    if p.shock_pct > 0 and p.shock_year > 0:
        idx = int(round(p.shock_year * p.ppy)) - 1
        if 0 <= idx < n:
            g[idx] *= 1 - p.shock_pct / 100
    return g


def simulate(p: Plan) -> pd.DataFrame:
    """Vectorised path: V[k+1] = (V[k] + c[k]) * g[k]."""
    n = p.years * p.ppy
    c, g = contributions(p), growth_factors(p)
    P = np.concatenate([[1.0], np.cumprod(g)])
    value = np.empty(n + 1)
    value[0] = p.lump
    value[1:] = P[1:] * (p.lump + np.cumsum(c / P[:-1]))
    invested = p.lump + np.concatenate([[0.0], np.cumsum(c)])
    return pd.DataFrame({"t": np.arange(n + 1) / p.ppy, "invested": invested, "value": value})


def xirr_flows(times, flows):
    times, flows = np.asarray(times, float), np.asarray(flows, float)
    m = flows != 0
    times, flows = times[m], flows[m]
    if not (flows < 0).any() or not (flows > 0).any():
        return np.nan
    f = lambda r: np.sum(flows / (1 + r) ** times)
    lo, hi = -0.99, 10.0
    flo = f(lo)
    if flo * f(hi) > 0:
        return np.nan
    for _ in range(80):
        mid = (lo + hi) / 2
        fm = f(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def plan_xirr(p: Plan, end_value: float) -> float:
    c = contributions(p)
    times = np.concatenate([[0.0], np.arange(len(c)) / p.ppy, [float(p.years)]])
    flows = np.concatenate([[-p.lump], -c, [end_value]])
    return xirr_flows(times, flows)


def summarize(p: Plan, e: Econ) -> dict:
    df = simulate(p)
    inv, val = df.invested.iloc[-1], df.value.iloc[-1]
    gain = val - inv
    tax = max(0.0, gain - e.exempt) * e.tax_rate / 100 if e.tax_on else 0.0
    post = val - tax
    return dict(
        df=df,
        invested=inv,
        value=val,
        gain=gain,
        abs_ret=gain / inv if inv else np.nan,
        xirr=plan_xirr(p, val),
        tax=tax,
        post_tax=post,
        real=post / (1 + e.infl) ** p.years,
    )


def yearly_table(df, p: Plan, e: Econ) -> pd.DataFrame:
    d = df.iloc[np.arange(p.years + 1) * p.ppy].reset_index(drop=True)
    out = pd.DataFrame({"Year": np.arange(p.years + 1), "Invested": d.invested, "Value": d.value})
    out["Gain"] = out.Value - out.Invested
    out["Return %"] = np.where(out.Invested > 0, out.Gain / out.Invested * 100, np.nan)
    out["Real value"] = out.Value / (1 + e.infl) ** out.Year
    return out.round(2)


def goal_numbers(p: Plan, target: float) -> dict:
    """Value is linear in lump and in SIP amount, so required amounts are closed-form."""
    G = simulate(
        replace(p, lump=1.0, sip=0.0, step_mode="None", step_val=0.0, pause_len_m=0, shock_pct=0.0)
    ).value.iloc[-1]
    v0 = simulate(replace(p, lump=0.0, sip=0.0)).value.iloc[-1]
    A = simulate(replace(p, lump=0.0, sip=1.0)).value.iloc[-1] - v0
    sip_val = simulate(replace(p, lump=0.0)).value.iloc[-1]
    return dict(
        sip_given_lump=(target - p.lump * G - v0) / A if A else np.nan,
        lump_given_sip=(target - sip_val) / G,
        sip_only=(target - v0) / A if A else np.nan,
        lump_only=target / G,
    )


def breakeven_crash(sip_p: Plan, lump_p: Plan):
    f = lambda x: (
        simulate(replace(sip_p, shock_pct=x)).value.iloc[-1] - simulate(replace(lump_p, shock_pct=x)).value.iloc[-1]
    )
    if f(0) >= 0:
        return 0.0
    if f(90) < 0:
        return None
    lo, hi = 0.0, 90.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return hi


@st.cache_data(show_spinner=False)
def monte_carlo(plan_t: tuple, vol: float, paths: int, seed: int):
    p = Plan(*plan_t)
    n = p.years * p.ppy
    c = contributions(p)
    net = max(p.cagr - p.expense, -0.99)
    rng = np.random.default_rng(seed)
    g = np.exp(rng.normal(np.log1p(net) / p.ppy, vol / np.sqrt(p.ppy), size=(paths, n)))
    P = np.concatenate([np.ones((paths, 1)), np.cumprod(g, axis=1)], axis=1)
    V = np.empty((paths, n + 1))
    V[:, 0] = p.lump
    V[:, 1:] = P[:, 1:] * (p.lump + np.cumsum(c / P[:, :-1], axis=1))
    return np.percentile(V, [5, 25, 50, 75, 95], axis=0), V[:, -1].copy()


def swp(corpus, monthly, ret, step_pct, max_years=60):
    r = (1 + ret) ** (1 / 12) - 1
    bal, w, total, rows, lasted = corpus, monthly, 0.0, [(0, corpus, 0.0)], None
    for m in range(1, max_years * 12 + 1):
        if m > 1 and (m - 1) % 12 == 0:
            w *= 1 + step_pct / 100
        pay = min(w, bal)
        bal -= pay
        total += pay
        bal *= 1 + r
        rows.append((m, bal, total))
        if bal <= 1e-6:
            lasted = m
            break
    return pd.DataFrame(rows, columns=["month", "balance", "withdrawn"]), lasted, total


@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(ticker: str, start: str, end: str):
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().astype(float)


def read_price_csv(f):
    d = pd.read_csv(f)
    dc = next(c for c in d.columns if "date" in c.lower())
    vc = next((c for c in d.columns if c.lower() in ("close", "adj close", "price")), d.columns[-1])
    s = pd.Series(pd.to_numeric(d[vc], errors="coerce").values, index=pd.to_datetime(d[dc], errors="coerce"))
    return s[~s.index.isna()].dropna().sort_index()


def backtest(prices: pd.Series, p: Plan):
    prices = prices.sort_index()
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    start, end = prices.index[0], prices.index[-1]
    freq = {52: "W-MON", 12: "MS", 1: "YS"}[p.ppy]
    rng = pd.date_range(start, end, freq=freq)
    dates = pd.DatetimeIndex([start]).append(rng[rng > start])
    yrs_elapsed = np.floor((dates - start).days / 365.25).astype(int)
    c = step_amount(p, yrs_elapsed)
    idx = prices.index.searchsorted(dates)
    ok = idx < len(prices)
    idx, dates, c = idx[ok], dates[ok], c[ok]
    px, tdates = prices.values[idx], prices.index[idx]

    def cum(vals):
        return pd.Series(vals, index=tdates).groupby(level=0).sum().reindex(prices.index, fill_value=0).cumsum()

    inv = cum(c) + p.lump
    units = cum(c / px) + p.lump / prices.iloc[0]
    out = pd.DataFrame(
        {
            "price": prices,
            "invested": inv,
            "plan_value": units * prices,
            "all_in_value": inv.iloc[-1] / prices.iloc[0] * prices,
        }
    )
    yrs = (end - start).days / 365.25
    times = np.concatenate([[0.0], (tdates - start).days / 365.25, [yrs]])
    flows = np.concatenate([[-p.lump], -c, [out.plan_value.iloc[-1]]])
    stats = dict(
        invested=inv.iloc[-1],
        final=out.plan_value.iloc[-1],
        all_in=out.all_in_value.iloc[-1],
        xirr=xirr_flows(times, flows),
        index_cagr=(prices.iloc[-1] / prices.iloc[0]) ** (1 / yrs) - 1,
        max_dd=(prices / prices.cummax() - 1).min(),
        years=yrs,
    )
    return out, stats


def to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def make_pdf(title, inputs: dict, summ: dict, yearly: pd.DataFrame):
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    finally:
        return None
    nl = dict(new_x="LMARGIN", new_y="NEXT")
    clean = lambda x: str(x).encode("latin-1", "replace").decode("latin-1")  # core fonts are latin-1 only
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean(title), **nl)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Inputs", **nl)
    pdf.set_font("Helvetica", "", 10)
    for k, v in inputs.items():
        pdf.cell(0, 5, clean(f"{k}: {v}"), **nl)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Results", **nl)
    pdf.set_font("Helvetica", "", 10)
    for k, v in summ.items():
        pdf.cell(0, 5, clean(f"{k}: {v}"), **nl)
    pdf.ln(3)
    widths, heads = [18, 42, 42, 42, 40], ["Year", "Invested", "Value", "Gain", "Return %"]
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(widths, heads):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for _, r in yearly.iterrows():
        vals = [
            int(r.Year),
            money(r.Invested, plain=True),
            money(r.Value, plain=True),
            money(r.Gain, plain=True),
            "-" if np.isnan(r["Return %"]) else f"{r['Return %']:.1f}%",
        ]
        for w, v in zip(widths, vals):
            pdf.cell(w, 6, clean(v), border=1)
        pdf.ln()
    return bytes(pdf.output())


# ───────────────────────── sidebar inputs ─────────────────────────
with st.sidebar:
    st.title("⚙️ Inputs")
    cur_name = st.selectbox("Currency", list(CURRENCIES))
    CUR = CURRENCIES[cur_name]
    dark = st.toggle("Dark mode", value=False)
    TEMPLATE = "plotly_dark" if dark else "plotly_white"
    if dark:
        st.markdown(
            "<style>.stApp{background-color:#0e1117;color:#fafafa}"
            "[data-testid=stSidebar]{background-color:#161a23}</style>",
            unsafe_allow_html=True,
        )

    st.subheader("💰 Investment")
    lump = st.number_input("Lumpsum amount", min_value=0.0, value=4000000.0, step=10000.0, format="%.0f")
    sip = st.number_input("SIP amount per instalment", min_value=0.0, value=10000.0, step=500.0, format="%.0f")
    freq_name = st.selectbox("SIP frequency", list(FREQ), index=1)
    years = st.slider("Tenure (years)", 1, 50, 15)
    cagr = st.number_input("Expected CAGR (%)", min_value=-20.0, max_value=60.0, value=20.0, step=0.5)
    scen_txt = st.text_input("Compare CAGRs (%, comma-separated)", "12, 15, 18,20,22,24,26,28,30")

    st.subheader("📈 Step-up SIP")
    step_mode = st.radio("Annual step-up", ["None", "Percent", "Amount"], horizontal=True)
    step_val = 0.0
    if step_mode == "Percent":
        step_val = st.number_input("Step-up (% per year)", 0.0, 100.0, 10.0, 1.0)
    elif step_mode == "Amount":
        step_val = st.number_input("Step-up (amount per year)", min_value=0.0, value=500.0, step=100.0)

    st.subheader("🧾 Costs, tax & inflation")
    expense = st.number_input("Expense ratio / fees (% p.a.)", 0.0, 5.0, 0.5, 0.05)
    infl = st.number_input("Inflation (% p.a.)", 0.0, 20.0, 7.0, 0.25)
    tax_on = st.toggle("Apply capital-gains tax at the end", value=False)
    tax_rate = st.number_input("Tax rate on gains (%)", 0.0, 60.0, 12.5, 0.5, disabled=not tax_on)
    exempt = st.number_input("Tax-free gains (amount)", min_value=0.0, value=125000.0, step=5000.0, disabled=not tax_on)
    st.caption("Tax defaults are editable placeholders - set them for your jurisdiction.")

    st.subheader("⏸️ SIP pause")
    pause_start = st.number_input("Pause starts after (months)", 0, 600, 0)
    pause_len = st.number_input("Pause length (months)", 0, 120, 0)

if lump == 0 and sip == 0:
    st.warning("Enter a lumpsum and/or a SIP amount in the sidebar.")
    st.stop()

ppy = FREQ[freq_name]
base = Plan(lump, sip, ppy, years, cagr / 100, expense / 100, step_mode, step_val, int(pause_start), int(pause_len))
econ = Econ(infl / 100, tax_on, tax_rate, exempt)
S = summarize(base, econ)
df = S["df"]
ytab = yearly_table(df, base, econ)

try:
    scen_cagrs = sorted({float(x) for x in scen_txt.replace(";", ",").split(",") if x.strip()})
except ValueError:
    scen_cagrs = [6.0, 8.0, 10.0, 12.0, 15.0, 18.0]
    st.sidebar.error("Couldn't parse CAGR list - using defaults.")

scen_rows, scen_dfs = [], {}
for cg in scen_cagrs:
    r = summarize(replace(base, cagr=cg / 100), econ)
    scen_dfs[cg] = r["df"]
    scen_rows.append(
        {
            "CAGR %": cg,
            "Invested": r["invested"],
            "End value": r["value"],
            "Gain": r["gain"],
            "Return %": r["abs_ret"] * 100,
            "XIRR %": r["xirr"] * 100,
            "Post-tax": r["post_tax"],
            "Real (post-tax)": r["real"],
        }
    )
scen_df = pd.DataFrame(scen_rows).round(2)

st.title("📈 Lumpsum & SIP Analyzer")
st.caption(
    f"{money(lump, True)} lumpsum + {money(sip)} {freq_name.lower()} SIP · {years} years · "
    f"{cagr:.1f}% CAGR (net {cagr - expense:.2f}% after fees)"
)

tabs = st.tabs(
    [
        "📊 Overview",
        "🎛️ What-if",
        "⚖️ CAGR Scenarios",
        "🆚 SIP vs Lumpsum",
        "🎯 Goal Planner",
        "🔥 Sensitivity",
        "🎲 Monte Carlo",
        "💸 Fees·Tax·Inflation",
        "🏖️ Withdrawal (SWP)",
        "📜 Backtest",
        "💾 Saved & Export",
    ]
)

# ───────────────────────── 1. Overview ─────────────────────────
with tabs[0]:
    a = st.columns(4)
    a[0].metric("Total invested", money(S["invested"], True), help=money(S["invested"]))
    a[1].metric(
        "End value (net of fees)", money(S["value"], True), f"+{money(S['gain'], True)}", help=money(S["value"])
    )
    a[2].metric("Absolute return", pct(S["abs_ret"]), help="Gain ÷ invested")
    a[3].metric("XIRR", pct(S["xirr"]), help="Money-weighted annual return (accounts for instalment timing)")
    b = st.columns(4)
    b[0].metric("Wealth multiple", f"{S['value'] / S['invested']:.2f}x")
    b[1].metric("Net CAGR after fees", f"{cagr - expense:.2f}%")
    b[2].metric(
        "Post-tax value",
        money(S["post_tax"], True),
        f"-{money(S['tax'], True)} tax" if tax_on else None,
        delta_color="off",
    )
    b[3].metric("Real value (today's money)", money(S["real"], True), help="Post-tax value deflated by inflation")

    if pause_len > 0:
        np_ = summarize(replace(base, pause_len_m=0), econ)
        loss = np_["value"] - S["value"]
        st.info(
            f"⏸️ Pausing for {pause_len} months (from month {pause_start}) costs {money(loss)} "
            f"({loss / np_['value'] * 100:.1f}%) in end value versus never pausing."
        )
    st.caption(
        f"Annualised SIP outlay in year 1: {money(sip * ppy)}"
        + (f" · rising by {step_val:g}{'%' if step_mode == 'Percent' else ''} per year" if step_mode != "None" else "")
    )

    fig = go.Figure()
    fig.add_scatter(x=df.t, y=df.invested, name="Invested", line=dict(dash="dash"))
    fig.add_scatter(x=df.t, y=df.value, name="Portfolio value", line=dict(width=3))
    fig.add_scatter(
        x=df.t, y=df.value / (1 + econ.infl) ** df.t, name="Real value (inflation-adj.)", line=dict(dash="dot")
    )
    st.plotly_chart(style(fig, "Growth over time"), **STRETCH)

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=df.t, y=df.invested, name="Invested", stackgroup="s")
    fig.add_scatter(x=df.t, y=df.value - df.invested, name="Gains", stackgroup="s")
    c1.plotly_chart(style(fig, "Invested vs gains (stacked)", h=380), **STRETCH)

    if lump > 0 and sip > 0:
        l_df = simulate(replace(base, sip=0.0, step_mode="None", step_val=0.0))
        s_df = simulate(replace(base, lump=0.0))
        fig = go.Figure()
        fig.add_scatter(x=l_df.t, y=l_df.value, name="From lumpsum", stackgroup="s")
        fig.add_scatter(x=s_df.t, y=s_df.value, name="From SIP", stackgroup="s")
        c2.plotly_chart(style(fig, "Where the value comes from", h=380), **STRETCH)
    else:
        fig = go.Figure(go.Pie(labels=["Invested", "Gains"], values=[S["invested"], max(S["gain"], 0)], hole=0.55))
        fig.update_layout(template=TEMPLATE, title="Split of end value", height=380)
        c2.plotly_chart(fig, **STRETCH)

    st.subheader("Year-by-year breakdown")
    st.dataframe(ytab, hide_index=True, **STRETCH)
    st.download_button("⬇️ Download table (CSV)", ytab.to_csv(index=False), "yearly_breakdown.csv", "text/csv")

# ───────────────────────── What-if (interactive) ─────────────────────────
with tabs[1]:
    st.caption(
        "Drag any slider - results update live. **Baseline** = your sidebar inputs; "
        "**What-if** = baseline plus the changes you make here."
    )
    WI_DEFAULTS = {
        "wi_sip": 0,
        "wi_lump": 0,
        "wi_cagr": 0.0,
        "wi_tenure": 0,
        "wi_step": 0.0,
        "wi_fee": 0.0,
        "wi_pause_len": 0,
        "wi_pause_start": 0,
    }
    for k_, v_ in WI_DEFAULTS.items():
        st.session_state.setdefault(k_, v_)

    def _wi_reset():
        for k_, v_ in WI_DEFAULTS.items():
            st.session_state[k_] = v_

    st.button("↺ Reset all sliders", on_click=_wi_reset)
    r1 = st.columns(4)
    d_sip = r1[0].slider("SIP amount change (%)", -100, 300, step=5, key="wi_sip")
    d_lump = r1[1].slider("Lumpsum change (%)", -100, 300, step=5, key="wi_lump")
    d_cagr = r1[2].slider("CAGR change (pp)", -10.0, 10.0, step=0.5, key="wi_cagr")
    d_ten = r1[3].slider("Tenure change (years)", -20, 20, key="wi_tenure")
    r2 = st.columns(4)
    d_step = r2[0].slider(
        "Step-up change (pp / yr)",
        -20.0,
        30.0,
        step=1.0,
        key="wi_step",
        help="Switches the step-up to a % rule. Non-zero overrides an 'Amount' step-up.",
    )
    d_fee = r2[1].slider("Fees change (pp)", -2.0, 3.0, step=0.05, key="wi_fee")
    d_pl = r2[2].slider("Add SIP pause (months)", 0, 60, key="wi_pause_len", help="0 keeps the sidebar pause setting.")
    d_ps = r2[3].slider("Pause starts after (months)", 0, 600, key="wi_pause_start", disabled=d_pl == 0)

    sv0 = base.step_val if base.step_mode == "Percent" else 0.0
    W = replace(
        base,
        lump=base.lump * (1 + d_lump / 100),
        sip=base.sip * (1 + d_sip / 100),
        years=int(np.clip(years + d_ten, 1, 50)),
        cagr=max(base.cagr + d_cagr / 100, -0.2),
        expense=max(0.0, base.expense + d_fee / 100),
        step_mode="Percent" if d_step != 0 else base.step_mode,
        step_val=max(0.0, sv0 + d_step) if d_step != 0 else base.step_val,
        pause_len_m=int(d_pl) if d_pl > 0 else base.pause_len_m,
        pause_start_m=int(d_ps) if d_pl > 0 else base.pause_start_m,
    )
    if W.lump == 0 and W.sip == 0:
        st.warning("These changes leave nothing invested - raise the SIP or lumpsum.")
    else:
        SW = summarize(W, econ)
        sg = lambda x: ("+" if x >= 0 else "") + money(x, True)
        m = st.columns(5)
        m[0].metric("Invested", money(SW["invested"], True), sg(SW["invested"] - S["invested"]), delta_color="off")
        m[1].metric("End value", money(SW["value"], True), sg(SW["value"] - S["value"]))
        m[2].metric("Gain", money(SW["gain"], True), sg(SW["gain"] - S["gain"]))
        m[3].metric("XIRR", pct(SW["xirr"]), f"{(SW['xirr'] - S['xirr']) * 100:+.2f} pp")
        m[4].metric("Real value (post-tax)", money(SW["real"], True), sg(SW["real"] - S["real"]))
        st.caption(
            f"Baseline → invested {money(S['invested'], True)} · end value {money(S['value'], True)} · "
            f"XIRR {pct(S['xirr'])} · real value {money(S['real'], True)}  |  "
            f"What-if plan: {money(W.lump, True)} lumpsum + {money(W.sip)} {freq_name.lower()} SIP, "
            f"{W.years}y at {W.cagr * 100:.1f}% (fees {W.expense * 100:.2f}%)."
        )

        wdf = SW["df"]
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=df.t, y=df.value, name="Baseline value", line=dict(color="gray", width=2))
        fig.add_scatter(x=df.t, y=df.invested, name="Baseline invested", line=dict(color="gray", dash="dot"))
        fig.add_scatter(x=wdf.t, y=wdf.value, name="What-if value", line=dict(width=3))
        fig.add_scatter(x=wdf.t, y=wdf.invested, name="What-if invested", line=dict(dash="dash"))
        c1.plotly_chart(style(fig, "Baseline vs what-if", h=400), **STRETCH)
        n_common = min(len(df), len(wdf))
        diff = wdf.value.values[:n_common] - df.value.values[:n_common]
        fig = go.Figure()
        fig.add_scatter(x=df.t[:n_common], y=diff, fill="tozeroy", name="What-if minus baseline")
        c2.plotly_chart(style(fig, "Value gained / lost vs baseline", h=400), **STRETCH)

        sv = W.step_val if W.step_mode == "Percent" else 0.0
        levers = [
            ("SIP ±20%", replace(W, sip=W.sip * 0.8), replace(W, sip=W.sip * 1.2)),
            ("Lumpsum ±20%", replace(W, lump=W.lump * 0.8), replace(W, lump=W.lump * 1.2)),
            ("CAGR ±2 pp", replace(W, cagr=W.cagr - 0.02), replace(W, cagr=W.cagr + 0.02)),
            ("Tenure ±3 yrs", replace(W, years=max(1, W.years - 3)), replace(W, years=min(50, W.years + 3))),
            ("Fees ±0.5 pp", replace(W, expense=W.expense + 0.005), replace(W, expense=max(0.0, W.expense - 0.005))),
        ]
        if W.step_mode != "Amount":
            levers.append(
                (
                    "Step-up ±5 pp",
                    replace(W, step_mode="Percent", step_val=max(0.0, sv - 5)),
                    replace(W, step_mode="Percent", step_val=sv + 5),
                )
            )
        rows = []
        for nm_, lo_p, hi_p in levers:
            a_, b_ = simulate(lo_p).value.iloc[-1] - SW["value"], simulate(hi_p).value.iloc[-1] - SW["value"]
            rows.append((nm_, min(a_, b_), max(a_, b_)))
        rows.sort(key=lambda r: r[2] - r[1])
        names_ = [r[0] for r in rows]
        fig = go.Figure()
        fig.add_bar(
            y=names_,
            x=[r[1] for r in rows],
            orientation="h",
            name="Downside",
            marker_color="#EF553B",
            text=[money(r[1], True) for r in rows],
            textposition="outside",
        )
        fig.add_bar(
            y=names_,
            x=[r[2] for r in rows],
            orientation="h",
            name="Upside",
            marker_color="#00CC96",
            text=[money(r[2], True) for r in rows],
            textposition="outside",
        )
        fig.update_layout(
            template=TEMPLATE,
            barmode="relative",
            height=380,
            hovermode="y unified",
            title="Which lever matters most? (change in end value around the what-if plan)",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", y=-0.2),
        )
        fig.update_xaxes(tickprefix=CUR[0], tickformat=".2~s", title="Change in end value")
        st.plotly_chart(fig, **STRETCH)

        if W == base:
            st.info("Move a slider to see how the outcome changes.")
        else:
            dv = SW["value"] - S["value"]
            st.success(
                f"With these changes you'd invest {money(SW['invested'], True)} "
                f"(baseline {money(S['invested'], True)}) and end with {money(SW['value'], True)} - "
                f"{abs(dv) / S['value'] * 100:.1f}% {'more' if dv >= 0 else 'less'} than the baseline. "
                f"Around this scenario, **{names_[-1]}** moves the result the most."
            )

# ───────────────────────── 2. CAGR scenarios ─────────────────────────
with tabs[2]:
    st.dataframe(scen_df, hide_index=True, **STRETCH)
    fig = go.Figure()
    for cg, d in scen_dfs.items():
        fig.add_scatter(x=d.t, y=d.value, name=f"{cg:g}% CAGR")
    fig.add_scatter(x=df.t, y=df.invested, name="Invested", line=dict(color="gray", dash="dash"))
    st.plotly_chart(style(fig, "Portfolio value by CAGR scenario"), **STRETCH)
    fig = go.Figure()
    fig.add_bar(x=[f"{c:g}%" for c in scen_df["CAGR %"]], y=scen_df["Invested"], name="Invested")
    fig.add_bar(x=[f"{c:g}%" for c in scen_df["CAGR %"]], y=scen_df["Gain"], name="Gain")
    fig.update_layout(barmode="stack")
    st.plotly_chart(style(fig, "End value split: invested + gain", xtitle="CAGR"), **STRETCH)

# ───────────────────────── 3. SIP vs lumpsum ─────────────────────────
with tabs[3]:
    st.caption(
        "Deploys the same total capital two ways: spread out as your SIP plan vs. all at day 1. "
        "Your lumpsum input is ignored here. Optionally add a one-off market crash to stress-test both."
    )
    k1, k2 = st.columns(2)
    crash = k1.slider("One-off market crash (%)", 0, 70, 0)
    crash_year = k2.number_input("Crash happens at year", 1, years, min(3, years))
    sip_p = replace(base, lump=0.0, shock_pct=float(crash), shock_year=float(crash_year))
    total_cap = simulate(replace(sip_p, shock_pct=0.0)).invested.iloc[-1]
    lump_p = replace(
        base,
        lump=total_cap,
        sip=0.0,
        step_mode="None",
        step_val=0.0,
        pause_len_m=0,
        shock_pct=float(crash),
        shock_year=float(crash_year),
    )
    sd, ld = simulate(sip_p), simulate(lump_p)
    se, le = sd.value.iloc[-1], ld.value.iloc[-1]
    G = simulate(replace(lump_p, lump=1.0)).value.iloc[-1]
    m = st.columns(4)
    m[0].metric("Capital deployed", money(total_cap, True))
    m[1].metric("SIP end value", money(se, True), f"{pct(plan_xirr(sip_p, se))} XIRR", delta_color="off")
    m[2].metric(
        "Lumpsum end value", money(le, True), f"{pct((le / total_cap) ** (1 / years) - 1)} CAGR", delta_color="off"
    )
    m[3].metric("Winner", "Lumpsum" if le > se else "SIP", f"by {money(abs(le - se), True)}", delta_color="off")
    st.write(
        f"**Equivalent lumpsum today:** {money(se / G)} invested at day 1 would match the SIP's end value "
        f"(vs. {money(total_cap)} spread over the SIP)."
    )
    be = breakeven_crash(replace(sip_p, shock_year=float(crash_year)), replace(lump_p, shock_year=float(crash_year)))
    if be is None:
        st.write(f"**Break-even:** no crash up to 90% at year {crash_year} lets the SIP overtake the lumpsum.")
    elif be == 0:
        st.write("**Break-even:** the SIP already matches or beats the lumpsum with no crash.")
    else:
        st.write(
            f"**Break-even:** a crash of ~{be:.1f}% at year {crash_year} is where the SIP catches up with the lumpsum."
        )
    fig = go.Figure()
    fig.add_scatter(x=sd.t, y=sd.value, name="SIP")
    fig.add_scatter(x=ld.t, y=ld.value, name="Lumpsum (day 1)")
    fig.add_scatter(x=sd.t, y=sd.invested, name="SIP invested", line=dict(dash="dash"))
    st.plotly_chart(style(fig, "SIP vs lumpsum - same capital"), **STRETCH)
    grid = list(range(0, 71, 5))
    gs = [simulate(replace(sip_p, shock_pct=float(x))).value.iloc[-1] for x in grid]
    gl = [simulate(replace(lump_p, shock_pct=float(x))).value.iloc[-1] for x in grid]
    fig = go.Figure()
    fig.add_scatter(x=grid, y=gs, name="SIP")
    fig.add_scatter(x=grid, y=gl, name="Lumpsum")
    st.plotly_chart(style(fig, f"End value vs crash size (crash at year {crash_year})", xtitle="Crash %"), **STRETCH)

# ───────────────────────── 4. Goal planner ─────────────────────────
with tabs[4]:
    g1, g2 = st.columns(2)
    target = g1.number_input("Target corpus", min_value=0.0, value=10_000_000.0, step=100_000.0, format="%.0f")
    in_today = g2.checkbox("Target is in today's money (inflate it to the future)", value=False)
    tgt = target * (1 + econ.infl) ** years if in_today else target
    if in_today:
        st.caption(f"Inflation-adjusted nominal target after {years} years: {money(tgt)}")
    gn = goal_numbers(replace(base, shock_pct=0.0), tgt)
    nz = lambda x: "Already achieved" if x < 0 else money(x)
    r = st.columns(4)
    r[0].metric(f"SIP needed ({freq_name.lower()})", nz(gn["sip_given_lump"]), help="Given your lumpsum input")
    r[1].metric("Lumpsum needed today", nz(gn["lump_given_sip"]), help="Given your SIP input")
    r[2].metric("SIP only (no lumpsum)", nz(gn["sip_only"]))
    r[3].metric("Lumpsum only (no SIP)", nz(gn["lump_only"]))
    st.caption("Uses your CAGR, fees, step-up and pause settings; figures are pre-tax and nominal.")
    cg_list = [10, 12, 15, 18, 20, 22, 24, 26]
    vals = [max(goal_numbers(replace(base, cagr=c / 100, shock_pct=0.0), tgt)["sip_only"], 0) for c in cg_list]
    fig = go.Figure(go.Bar(x=[f"{c}%" for c in cg_list], y=vals, text=[money(v, True) for v in vals]))
    st.plotly_chart(
        style(
            fig,
            f"SIP-only amount needed per {freq_name.lower()[:-2] if ppy != 1 else 'year'} instalment, by CAGR",
            xtitle="CAGR",
            h=380,
        ),
        **STRETCH,
    )

# ───────────────────────── 5. Sensitivity ─────────────────────────
with tabs[5]:
    h1, h2 = st.columns(2)
    axes = h1.radio("Grid", ["CAGR × Tenure", "CAGR × SIP amount"], horizontal=True)
    metric = h2.radio("Metric", ["Final value", "Total gain", "Wealth multiple"], horizontal=True)
    # cgs = list(range(4, 21, 2))
    cgs = list(range(10, 32, 2))
    if axes == "CAGR × Tenure":
        cols = [5, 10, 15, 20, 25, 30, 35]
        mk = lambda c, x: replace(base, cagr=c / 100, years=x)
        xlabels, xtitle = [f"{x}y" for x in cols], "Tenure"
    else:
        cols = [round(sip * f, 2) for f in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4)]
        mk = lambda c, x: replace(base, cagr=c / 100, sip=x)
        xlabels, xtitle = [money(x, True) for x in cols], f"SIP per {freq_name.lower()} instalment"
    z, txt = [], []
    for c in cgs:
        zr, tr = [], []
        for x in cols:
            d = simulate(mk(c, x))
            v, iv = d.value.iloc[-1], d.invested.iloc[-1]
            val = {"Final value": v, "Total gain": v - iv, "Wealth multiple": v / iv if iv else np.nan}[metric]
            zr.append(val)
            tr.append(f"{val:.1f}x" if metric == "Wealth multiple" else money(val, True))
        z.append(zr)
        txt.append(tr)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=xlabels,
            y=[f"{c}%" for c in cgs],
            text=txt,
            texttemplate="%{text}",
            colorscale="Viridis",
            hovertemplate="CAGR %{y}<br>%{x}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(template=TEMPLATE, title=f"{metric}: {axes}", height=520, xaxis_title=xtitle, yaxis_title="CAGR")
    st.plotly_chart(fig, **STRETCH)

# ───────────────────────── 6. Monte Carlo ─────────────────────────
with tabs[6]:
    st.caption(
        "Treats your CAGR as the median (geometric) return and adds random volatility. "
        "Shows the range of outcomes instead of a single line."
    )
    m1, m2, m3, m4 = st.columns(4)
    vol = m1.slider("Annual volatility (%)", 1, 50, 15)
    npaths = m2.slider("Simulated paths", 500, 5000, 2000, 500)
    seed = int(m3.number_input("Random seed", 0, 10_000, 42))
    mc_target = m4.number_input("Target value (optional)", min_value=0.0, value=0.0, step=100_000.0, format="%.0f")
    key = (astuple(base), vol, npaths, seed)
    if st.button("▶ Run simulation", type="primary"):
        with st.spinner("Simulating…"):
            pcts, finals = monte_carlo(astuple(base), vol / 100, npaths, seed)
        st.session_state["mc"] = dict(key=key, pcts=pcts, finals=finals)
    mc = st.session_state.get("mc")
    if mc is None:
        st.info("Set the volatility and press **Run simulation**.")
    else:
        if mc["key"] != key:
            st.warning("Inputs changed since this run - press Run simulation to refresh.")
        pcts, finals = mc["pcts"], mc["finals"]
        t = np.arange(pcts.shape[1]) / mc["key"][0][2]
        inv_end = df.invested.iloc[-1]
        q = st.columns(5)
        for col, lab, v in zip(q, ["5th pct (bad)", "25th", "Median", "75th", "95th pct (good)"], pcts[:, -1]):
            col.metric(lab, money(v, True))
        pr = st.columns(3)
        pr[0].metric("P(end value < invested)", f"{(finals < inv_end).mean() * 100:.1f}%")
        pr[1].metric("P(beat deterministic result)", f"{(finals >= S['value']).mean() * 100:.1f}%")
        if mc_target > 0:
            pr[2].metric("P(reach target)", f"{(finals >= mc_target).mean() * 100:.1f}%")
        fig = go.Figure()
        fig.add_scatter(x=t, y=pcts[4], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(
            x=t, y=pcts[0], fill="tonexty", fillcolor="rgba(99,110,250,0.15)", line=dict(width=0), name="5-95%"
        )
        fig.add_scatter(x=t, y=pcts[3], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(
            x=t, y=pcts[1], fill="tonexty", fillcolor="rgba(99,110,250,0.35)", line=dict(width=0), name="25-75%"
        )
        fig.add_scatter(x=t, y=pcts[2], name="Median", line=dict(width=3))
        fig.add_scatter(x=df.t, y=df.invested, name="Invested", line=dict(dash="dash", color="gray"))
        st.plotly_chart(style(fig, "Fan chart of outcomes"), **STRETCH)
        fig = go.Figure(go.Histogram(x=finals, nbinsx=60, name="Final value"))
        fig.add_vline(x=inv_end, line_dash="dash", annotation_text="Invested")
        fig.add_vline(x=float(np.median(finals)), annotation_text="Median")
        fig.update_layout(
            template=TEMPLATE,
            title="Distribution of final values",
            height=380,
            xaxis_title="Final value",
            yaxis_title="Paths",
            xaxis_tickprefix=CUR[0],
        )
        st.plotly_chart(fig, **STRETCH)

# ───────────────────────── 7. Fees, tax, inflation ─────────────────────────
with tabs[7]:
    gross = simulate(replace(base, expense=0.0)).value.iloc[-1]
    fee_cost = gross - S["value"]
    infl_cost = S["post_tax"] - S["real"]
    fig = go.Figure(
        go.Waterfall(
            x=["Gross value (no fees)", "Fee drag", "Tax", "Inflation erosion", "Real post-tax value"],
            measure=["absolute", "relative", "relative", "relative", "total"],
            y=[gross, -fee_cost, -S["tax"], -infl_cost, S["real"]],
            text=[money(v, True) for v in (gross, -fee_cost, -S["tax"], -infl_cost, S["real"])],
            textposition="outside",
        )
    )
    st.plotly_chart(style(fig, "From gross value to what it's really worth", xtitle="", h=440), **STRETCH)
    st.caption(
        f"Fees consume {fee_cost / gross * 100:.1f}% of gross end value; "
        f"inflation erodes {infl_cost / S['post_tax'] * 100:.1f}% of purchasing power."
        + ("" if tax_on else " (Tax is switched off in the sidebar.)")
    )
    ers = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    ev = [simulate(replace(base, expense=e / 100)).value.iloc[-1] for e in ers]
    ftab = pd.DataFrame(
        {
            "Expense ratio %": ers,
            "End value": ev,
            "Lost to fees vs 0%": [ev[0] - v for v in ev],
            "% of zero-fee value lost": [(ev[0] - v) / ev[0] * 100 for v in ev],
        }
    ).round(2)
    c1, c2 = st.columns([2, 3])
    c1.dataframe(ftab, hide_index=True, **STRETCH)
    fig = go.Figure(go.Bar(x=[f"{e}%" for e in ers], y=ev, text=[money(v, True) for v in ev]))
    c2.plotly_chart(style(fig, "End value by expense ratio", xtitle="Expense ratio", h=360), **STRETCH)

# ───────────────────────── 8. SWP ─────────────────────────
with tabs[8]:
    st.caption("Withdrawal phase: how long does the corpus last when you draw a monthly income?")
    s1, s2 = st.columns(2)
    src = s1.radio("Starting corpus", ["Projected end value (net of fees)", "Custom amount"])
    corpus = (
        S["value"]
        if src.startswith("Projected")
        else s2.number_input("Custom corpus", min_value=0.0, value=10_000_000.0, step=100_000.0, format="%.0f")
    )
    w1, w2, w3 = st.columns(3)
    monthly_w = w1.number_input("Monthly withdrawal", min_value=0.0, value=50_000.0, step=1000.0, format="%.0f")
    swp_ret = w2.number_input("Return during withdrawal (% p.a.)", 0.0, 30.0, 8.0, 0.5)
    swp_step = w3.number_input("Yearly withdrawal increase (%)", 0.0, 20.0, 5.0, 0.5)
    sd_, lasted, tot = swp(corpus, monthly_w, swp_ret / 100, swp_step)
    r = st.columns(4)
    r[0].metric("Initial withdrawal rate", f"{monthly_w * 12 / corpus * 100:.2f}% / yr" if corpus else "–")
    r[1].metric("Corpus lasts", f"{lasted // 12}y {lasted % 12}m" if lasted else "60+ years")
    r[2].metric("Total withdrawn", money(tot, True))
    r[3].metric("Balance at end", money(sd_.balance.iloc[-1], True))
    fig = go.Figure()
    fig.add_scatter(x=sd_.month / 12, y=sd_.balance, name="Remaining corpus", fill="tozeroy")
    fig.add_scatter(x=sd_.month / 12, y=sd_.withdrawn, name="Cumulative withdrawn")
    st.plotly_chart(style(fig, "Corpus during withdrawal"), **STRETCH)

# ───────────────────────── 9. Backtest ─────────────────────────
with tabs[9]:
    st.caption(
        "Runs your lumpsum + SIP plan (same frequency and step-up) on real prices. "
        "Pause, fees and tax are ignored here. Needs `yfinance` or an uploaded CSV (Date, Close)."
    )
    b1, b2, b3 = st.columns(3)
    ticker = b1.text_input("Yahoo Finance ticker", "^NSEI", help="e.g. ^NSEI, ^GSPC, RELIANCE.NS, VOO")
    d_start = b2.date_input("Start date", (pd.Timestamp.today() - pd.DateOffset(years=min(years, 15))).date())
    d_end = b3.date_input("End date", pd.Timestamp.today().date())
    upl = st.file_uploader("…or upload a price history CSV", type="csv")
    if st.button("▶ Run backtest", type="primary"):
        prices = None
        try:
            prices = read_price_csv(upl) if upl is not None else load_prices(ticker, str(d_start), str(d_end))
        except ImportError:
            st.error("yfinance isn't installed. Run `pip install yfinance` or upload a CSV.")
        except Exception as ex:
            st.error(f"Could not load prices: {ex}")
        if prices is None or len(prices) < 30:
            st.error("No usable price data returned.")
        else:
            st.session_state["bt"] = backtest(prices, base)
    bt = st.session_state.get("bt")
    if bt:
        bdf, bs = bt
        q = st.columns(5)
        q[0].metric("Invested", money(bs["invested"], True))
        q[1].metric("Plan final value", money(bs["final"], True))
        q[2].metric("Plan XIRR", pct(bs["xirr"]))
        q[3].metric("Index CAGR", pct(bs["index_cagr"]), f"{bs['years']:.1f} yrs", delta_color="off")
        q[4].metric("Index max drawdown", pct(bs["max_dd"]))
        fig = go.Figure()
        fig.add_scatter(x=bdf.index, y=bdf.invested, name="Invested", line=dict(dash="dash"))
        fig.add_scatter(x=bdf.index, y=bdf.plan_value, name="Your plan (lump + SIP)")
        fig.add_scatter(x=bdf.index, y=bdf.all_in_value, name="All capital as day-1 lumpsum")
        st.plotly_chart(style(fig, "Historical backtest", xtitle="Date"), **STRETCH)
        st.caption(f"Day-1 lumpsum of the same total capital ended at {money(bs['all_in'])}.")

# ───────────────────────── 10. Saved scenarios & export ─────────────────────────
with tabs[10]:
    st.session_state.setdefault("saved", [])
    n1, n2 = st.columns([3, 1])
    nm = n1.text_input("Scenario name", f"{freq_name} SIP {cagr:g}% / {years}y")
    n2.write("")
    if n2.button("💾 Save current", **STRETCH):
        st.session_state["saved"].append(
            dict(
                name=nm,
                df=df[["t", "value"]].copy(),
                row={
                    "Name": nm,
                    "Lumpsum": lump,
                    "SIP": sip,
                    "Freq": freq_name,
                    "Years": years,
                    "CAGR %": cagr,
                    "Fees %": expense,
                    "Invested": S["invested"],
                    "End value": S["value"],
                    "XIRR %": S["xirr"] * 100,
                    "Real value": S["real"],
                },
            )
        )
    saved = st.session_state["saved"]
    if saved:
        st.dataframe(pd.DataFrame([s["row"] for s in saved]).round(2), hide_index=True, **STRETCH)
        fig = go.Figure()
        for s_ in saved:
            fig.add_scatter(x=s_["df"].t, y=s_["df"].value, name=s_["name"])
        fig.add_scatter(x=df.t, y=df.value, name="Current inputs", line=dict(dash="dot", color="gray"))
        st.plotly_chart(style(fig, "Saved scenarios vs current"), **STRETCH)
        if st.button("🗑️ Clear all saved"):
            st.session_state["saved"] = []
            st.rerun()
    else:
        st.info("Save a scenario to compare it later against different inputs.")

    st.subheader("Export report")
    inputs = {
        "Currency": CUR[2],
        "Lumpsum": money(lump, plain=True),
        f"SIP ({freq_name.lower()})": money(sip, plain=True),
        "Tenure (years)": years,
        "CAGR %": cagr,
        "Expense ratio %": expense,
        "Inflation %": infl,
        "Step-up": f"{step_mode} {step_val:g}" if step_mode != "None" else "None",
        "Pause": f"{pause_len} months after month {pause_start}" if pause_len else "None",
        "Tax": f"{tax_rate:g}% above {money(exempt, plain=True)}" if tax_on else "Off",
    }
    results = {
        "Total invested": money(S["invested"], plain=True),
        "End value": money(S["value"], plain=True),
        "Gain": money(S["gain"], plain=True),
        "Absolute return": pct(S["abs_ret"]),
        "XIRR": pct(S["xirr"]),
        "Post-tax value": money(S["post_tax"], plain=True),
        "Real value (today's money)": money(S["real"], plain=True),
    }
    e1, e2, e3 = st.columns(3)
    e1.download_button("⬇️ CSV (yearly)", ytab.to_csv(index=False), "yearly_breakdown.csv", "text/csv", **STRETCH)
    try:
        xl = to_excel(
            {
                "Inputs": pd.DataFrame({"Input": list(inputs), "Value": [str(v) for v in inputs.values()]}),
                "Results": pd.DataFrame({"Metric": list(results), "Value": list(results.values())}),
                "Yearly": ytab,
                "CAGR scenarios": scen_df,
            }
        )
        e2.download_button(
            "⬇️ Excel report",
            xl,
            "sip_lumpsum_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            **STRETCH,
        )
    except ImportError:
        e2.caption("Install `openpyxl` for Excel export.")
    pdf_bytes = make_pdf("Lumpsum & SIP Report", inputs, results, ytab)
    if pdf_bytes:
        e3.download_button("⬇️ PDF report", pdf_bytes, "sip_lumpsum_report.pdf", "application/pdf", **STRETCH)
    else:
        e3.caption("Install `fpdf2` for PDF export.")

st.divider()
st.caption(
    "Illustrative projections using constant or simulated returns - not investment advice. "
    "Real markets are volatile; past performance does not guarantee future results."
)

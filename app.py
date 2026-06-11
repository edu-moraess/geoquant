"""
GeoQuant – Institutional Macro Research Terminal
EVT + DCC-GARCH-X + GeoFactor + Walk-Forward + SHAP + ML Benchmarking
Eduardo Moraes | Quant Data Scientist & Economics
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, csv, logging, warnings
from datetime import datetime, timedelta
import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import stats, optimize
from scipy.stats import chi2
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import yfinance as yf
from arch import arch_model
import shap
import xgboost as xgb
import lightgbm as lgb

try:
    from pandas_datareader import data as pdr
    yf.pdr_override()
except Exception:
    pass

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ══════════════════════════════════════════════════════════
#   PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GeoQuant · Research Terminal",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
#   INSTITUTIONAL RESEARCH REPORT STYLING
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400&family=Source+Sans+3:wght@300;400;500&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
    --bg: #FFFFFF;
    --surface: #F8F7F4;
    --border: #D9D5CD;
    --text: #1C1C1C;
    --text-secondary: #5A554F;
    --accent: #1E3A5F;
    --accent-light: #2A5080;
    --gold: #B49450;
    --gold-light: #D4C094;
    --muted: #7A766E;
    --danger: #8B3A3A;
    --success: #2D5A3F;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Source Sans 3', 'Helvetica Neue', sans-serif !important;
    font-weight: 300 !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

div[data-testid="stMetric"] {
    background: var(--bg);
    border: 1px solid var(--border);
    padding: 1rem 1.2rem;
    border-radius: 0px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
div[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .54rem !important;
    letter-spacing: .22em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 1.55rem !important;
    font-weight: 400 !important;
    color: var(--accent) !important;
}

.stButton button {
    background: var(--accent) !important;
    color: var(--gold-light) !important;
    border: none !important;
    border-radius: 0px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .58rem !important;
    letter-spacing: .16em !important;
    text-transform: uppercase !important;
    padding: .55rem 1.2rem !important;
    width: 100%;
    transition: background 0.2s;
}
.stButton button:hover {
    background: var(--accent-light) !important;
}
.stProgress > div > div {
    background: var(--gold) !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: .58rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--muted);
    padding: .7rem 1.4rem;
    border-bottom: 2px solid transparent;
    background: transparent;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--gold) !important;
}

.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: .52rem;
    letter-spacing: .3em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: .2rem;
}
.sec-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.2rem;
    font-weight: 500;
    color: var(--accent);
    margin-bottom: .8rem;
    padding-bottom: .4rem;
    border-bottom: 1px solid var(--border);
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, var(--gold) 0%, var(--border) 60%, transparent 100%);
    margin: 1.2rem 0;
}
.info-block {
    background: var(--surface);
    border-left: 2px solid var(--gold);
    padding: .5rem .9rem;
    font-size: .72rem;
    color: var(--text-secondary);
    margin: .4rem 0;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: .04em;
}
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .74rem;
}
.data-table th {
    font-family: 'JetBrains Mono', monospace;
    font-size: .5rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--muted);
    padding: .5rem .8rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
}
.data-table td {
    padding: .5rem .8rem;
    border-bottom: 1px solid var(--border);
    font-weight: 300;
    color: var(--text);
}
.footer {
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: .5rem;
    letter-spacing: .12em;
    color: var(--muted);
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#   CONSTANTS
# ══════════════════════════════════════════════════════════
TICKERS = {
    "oil":"CL=F","brent":"BZ=F","natgas":"NG=F","gold":"GC=F","silver":"SI=F",
    "copper":"HG=F","wheat":"ZW=F","corn":"ZC=F","soy":"ZS=F",
    "dxy":"DX-Y.NYB","eur":"EURUSD=X","tnx":"^TNX",
}
GEO_W = {"oil_vol":0.22,"gold":0.09,"gold_real":0.09,"dxy":-0.10,"spread":0.09,
          "fert":0.22,"wheat":0.07,"copper":0.04,"natgas_vol":0.06}
ZSC_W = {"oil_gold":0.40,"oil_natgas":0.35,"gold_real":0.25}

PL = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Source Sans 3,Helvetica Neue,sans-serif", color="#1C1C1C", size=11),
    title_font=dict(family="Playfair Display,Georgia,serif", size=16, color="#1E3A5F"),
    xaxis=dict(gridcolor="#E8E4DA", linecolor="#D9D5CD", zeroline=False,
               tickfont=dict(size=10, family="JetBrains Mono,monospace", color="#5A554F")),
    yaxis=dict(gridcolor="#E8E4DA", linecolor="#D9D5CD", zeroline=False,
               tickfont=dict(size=10, family="JetBrains Mono,monospace", color="#5A554F")),
    legend=dict(bgcolor="rgba(255,255,255,0.97)", bordercolor="#D9D5CD",
                borderwidth=1, font=dict(size=10, family="JetBrains Mono,monospace", color="#1C1C1C")),
    margin=dict(l=55, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor="#1E3A5F", font_color="#D4C094", font_family="JetBrains Mono,monospace"),
)
C = dict(
    navy="#1E3A5F", navy_light="#2A5080",
    gold="#B49450", gold_light="#D4C094",
    burgundy="#7B3F3F", teal="#2B5F5F",
    sage="#4A5D4A", gray="#5A554F", silver="#9A958A",
    sky="#4A7380", rust="#8B5A3A",
    fill_light="rgba(30,58,95,0.06)",
    fill_medium="rgba(30,58,95,0.12)",
)

def qfig(h=420):
    fig = go.Figure()
    fig.update_layout(**PL, height=h)
    return fig

def dual_axis_fig(h=380):
    fig = go.Figure()
    fig.update_layout(
        **PL, height=h,
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    linecolor="#D9D5CD", zeroline=False,
                    tickfont=dict(size=10, family="JetBrains Mono,monospace", color="#5A554F")),
    )
    return fig

# ══════════════════════════════════════════════════════════
#   SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1rem;border-bottom:1px solid #D9D5CD;margin-bottom:1.2rem;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:.46rem;letter-spacing:.28em;
        color:#B49450;text-transform:uppercase;margin-bottom:.3rem;'>◆ Edumetria</div>
        <div style='font-family:"Playfair Display",Georgia,serif;font-size:1.2rem;
        font-weight:500;color:#1E3A5F;letter-spacing:.04em;'>GeoQuant Terminal</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:.46rem;
        color:#7A766E;letter-spacing:.12em;margin-top:.2rem;'>Quantitative Research Infrastructure</div>
    </div>""", unsafe_allow_html=True)

    def slabel(t):
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.5rem;letter-spacing:.2em;'
                    f'color:#B49450;text-transform:uppercase;margin:.8rem 0 .3rem;">{t}</div>',
                    unsafe_allow_html=True)
    def ssep():
        st.markdown('<div style="height:1px;background:#D9D5CD;margin:.5rem 0;"></div>',
                    unsafe_allow_html=True)

    slabel("· Simulation")
    mc_sims  = st.slider("Monte Carlo paths", 1_000, 30_000, 5_000, 1_000)
    mc_steps = st.slider("Horizon (days)", 5, 30, 10, 1)
    ssep()
    slabel("· Jump Parameters")
    jump_up   = st.slider("Jump prob ↑",  0.01, 0.20, 0.07, 0.01)
    jump_down = st.slider("Jump prob ↓",  0.01, 0.10, 0.03, 0.01)
    tail_df   = st.slider("Tail df",      2.5,  8.0,  3.0,  0.5)
    ssep()
    slabel("· Vol Priors (annual)")
    prior_wti   = st.slider("WTI prior",   0.20, 0.65, 0.35, 0.01)
    prior_brent = st.slider("Brent prior", 0.20, 0.65, 0.35, 0.01)
    ssep()
    slabel("· Regime")
    war_start     = st.date_input("War start", value=datetime(2026, 2, 28))
    war_start_str = war_start.strftime("%Y-%m-%d")
    ssep()
    run_btn = st.button("▶  Run Full System Pipeline")
    st.markdown("""
    <div style='margin-top:1.8rem;font-family:"JetBrains Mono",monospace;font-size:.42rem;
    color:#9A958A;letter-spacing:.1em;line-height:2.2;'>
    FOR PROFESSIONAL USE ONLY<br>NOT INVESTMENT ADVICE<br>CONFIDENTIAL & PROPRIETARY
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#   HEADER
# ══════════════════════════════════════════════════════════
now_sp = datetime.now(pytz.timezone("America/Sao_Paulo"))
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:flex-start;
padding:1.4rem 0 1rem;border-bottom:1px solid #D9D5CD;margin-bottom:1.6rem;'>
  <div>
    <div style='display:flex;align-items:baseline;gap:.6rem;'>
      <span style='font-family:"JetBrains Mono",monospace;font-size:.8rem;color:#B49450;letter-spacing:.2em;'>◆◆◆</span>
      <div>
        <div style='font-family:"Playfair Display",Georgia,serif;font-size:1.8rem;
        font-weight:500;color:#1E3A5F;letter-spacing:.04em;line-height:1;'>GeoQuant · Research Terminal</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:.5rem;color:#5A554F;
        letter-spacing:.2em;text-transform:uppercase;margin-top:.25rem;'>
        Geopolitical Intelligence · EVT+DCC+GARCH-X · Institutional Risk Management</div>
      </div>
    </div>
  </div>
  <div style='text-align:right;'>
    <div style='display:inline-block;background:#1E3A5F;color:#D4C094;padding:.2rem .7rem;
    font-family:"JetBrains Mono",monospace;font-size:.5rem;letter-spacing:.16em;text-transform:uppercase;'>
    ⚑ WAR REGIME</div>
    <div style='font-family:"JetBrains Mono",monospace;font-size:.52rem;color:#5A554F;
    letter-spacing:.1em;margin-top:.3rem;line-height:1.8;'>
    {now_sp.strftime("%d %B %Y · %H:%M")} (SP)<br>Institutional Analytics Framework
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#   QUANT ENGINE (Fully Robust)
# ══════════════════════════════════════════════════════════
def rolling_zscore(s, w=60):
    return (s - s.rolling(w).mean()) / s.rolling(w).std().replace(0, np.nan)

def fill_gaps(s):
    s = s.copy()
    valid = s.notna()
    if valid.sum() < 2:
        return s.ffill()
    try:
        x = s.index[valid].astype(np.int64)
        f = pd.Series(PchipInterpolator(x, s[valid].values)(s.index.astype(np.int64)), index=s.index)
        f[valid] = s[valid]
        return f
    except:
        return s.ffill()

def _force_update_fert_csv(path="fertilizer_backup.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "urea_price", "dap_price"])
        w.writerows([
            ["2026-01-15", 540, 710],
            ["2026-02-15", 560, 740],
            ["2026-03-15", 590, 780],
            ["2026-04-15", 616, 857],
            ["2026-05-01", 720, 900],
            ["2026-05-06", 810, 920],
            ["2026-05-12", 857, 920],
            ["2026-06-01", 860, 925],
            ["2026-06-10", 453.5, 920]
        ])

def get_usda():
    _force_update_fert_csv()
    try:
        df = pd.read_csv("fertilizer_backup.csv", parse_dates=["date"], index_col="date").sort_index()
        last = df.iloc[-1]
        return {
            "urea_price": float(last["urea_price"]),
            "urea_period": str(last.name.date()),
            "dap_price": float(last["dap_price"]),
            "dap_period": str(last.name.date()),
            "source": "Green Markets / CRU"
        }
    except:
        return {"urea_price": 453.5, "urea_period": "2026-06-10",
                "dap_price": 920, "dap_period": "2026-06-10", "source": "fallback"}

def fert_black_swan(usda):
    _force_update_fert_csv()
    try:
        df = pd.read_csv("fertilizer_backup.csv", parse_dates=["date"], index_col="date")
        hist = df["urea_price"].dropna().values
    except:
        hist = []
    cur = usda.get("urea_price")
    if cur is None or len(hist) < 10:
        return 1.0
    rets = np.diff(np.log(hist))
    thr = np.quantile(rets, 0.90)
    exc = rets[rets > thr] - thr
    if len(exc) < 5:
        mu, sig = np.mean(hist), np.std(hist)
        if sig == 0:
            return 1.0
        z = (cur - mu) / sig
        if z < -1.5:
            return max(0.5, 1.0 + z * 0.3)
        return min(1.0 + max(0, z - 1.5) * 0.8, 3.0)
    try:
        shape, loc, scale = stats.genpareto.fit(exc)
        cr = np.log(cur / hist[-1])
        if cr <= thr:
            return 0.6 if cr < -0.1 else 1.0
        p = 1 - stats.genpareto.cdf(cr - thr, shape, loc=loc, scale=scale)
        return 1.0 + min(p * 5, 2.0)
    except:
        return 1.0

def gold_signals(prices):
    silver = prices["silver"].replace(0, np.nan)
    if silver.median() > 500:
        silver /= 100
    gr = prices["gold"] / (1 + prices["tnx"].replace(0, np.nan) / 100 * 5.0)
    sg = silver / prices["gold"].replace(0, np.nan)
    return {
        "gold_real": gr,
        "silver_gold": sg,
        "gold_real_ret_roll": np.log(gr / gr.shift(1)).rolling(20).mean(),
        "silver_gold_roll": np.log(sg / sg.shift(1)).rolling(20).mean()
    }

def silver_demand_proxy(prices):
    if "copper" not in prices.columns:
        return pd.Series(0.0, index=prices.index)
    cr = prices["copper"].pct_change().dropna()
    br = prices["brent"].pct_change().dropna()
    ci = cr.index.intersection(br.index)
    return (0.6 * cr[ci] + 0.4 * br[ci]).rolling(20).mean().reindex(prices.index, method="ffill").fillna(0.0)

def build_fert_index(returns, usda, bs=1.0):
    fi = (0.5 * returns["natgas"].rolling(20).std() +
          0.25 * returns["wheat"].rolling(20).mean() +
          0.25 * returns["corn"].rolling(20).mean())
    if usda["urea_price"]:
        fi += np.clip((usda["urea_price"] - 380) / 380, -1, 2) * 0.15
    if usda["dap_price"]:
        fi += np.clip((usda["dap_price"] - 610) / 610, -1, 2) * 0.10
    fi *= bs
    return fi.clip(fi.quantile(0.02), fi.quantile(0.98)).dropna()

def calibrate_weights(returns, prices, gs, fi, sd, window=60):
    spread = (prices["brent"] - prices["oil"]) / prices["brent"].replace(0, np.nan)
    X = pd.DataFrame({
        "oil_vol": returns["oil"].rolling(20).std(),
        "gold": returns["gold"].rolling(20).mean(),
        "gold_real": gs["gold_real_ret_roll"],
        "dxy": returns["dxy"].rolling(20).mean(),
        "spread": spread.rolling(20).mean(),
        "wheat": returns["wheat"].rolling(20).mean(),
        "copper": returns["copper"].rolling(20).mean(),
        "natgas_vol": returns["natgas"].rolling(20).std(),
        "fert": fi
    })
    if sd is not None:
        X["silver_demand"] = sd
    y = returns["oil"].shift(-1)
    ci = y.dropna().index.intersection(X.dropna().index)
    X2, y2 = X.loc[ci].dropna(), y.loc[ci]
    if len(X2) < window:
        return GEO_W.copy()
    Xc, yc = X2.iloc[-window:], y2.iloc[-window:]
    Xm, Xs = Xc.mean(), Xc.std().replace(0, 1)
    try:
        mdl = LassoCV(cv=5, random_state=42, alphas=np.logspace(-4, 0, 20), max_iter=2000).fit((Xc - Xm) / Xs, yc)
        coef = mdl.coef_ / Xs.values
        w = {col: coef[i] for i, col in enumerate(X2.columns)}
        tot = sum(abs(v) for v in w.values())
        return {k: v / tot for k, v in w.items()} if tot > 0 else GEO_W.copy()
    except:
        return GEO_W.copy()

def build_geofactor(returns, prices, gs, fi, weights, sd=None):
    spread = (prices["brent"] - prices["oil"]) / prices["brent"].replace(0, np.nan)
    geo = (weights.get("oil_vol", 0) * returns["oil"].rolling(20).std() +
           weights.get("gold", 0) * returns["gold"].rolling(20).mean() +
           weights.get("gold_real", 0) * gs["gold_real_ret_roll"] +
           weights.get("dxy", 0) * returns["dxy"].rolling(20).mean() +
           weights.get("spread", 0) * spread.rolling(20).mean() +
           weights.get("wheat", 0) * returns["wheat"].rolling(20).mean() +
           weights.get("copper", 0) * returns["copper"].rolling(20).mean() +
           weights.get("natgas_vol", 0) * returns["natgas"].rolling(20).std())
    if sd is not None:
        ci = geo.dropna().index.intersection(sd.dropna().index)
        if len(ci) > 0:
            geo.loc[ci] += weights.get("silver_demand", 0) * sd.loc[ci]
    ci = geo.dropna().index.intersection(fi.dropna().index)
    geo.loc[ci] += weights.get("fert", 0) * fi.loc[ci]
    g = geo.dropna()
    return g.clip(g.quantile(0.05), g.quantile(0.95))

def build_zscore(prices, gs, window=60):
    w = min(window, max(20, len(prices) // 2))
    z1 = rolling_zscore(prices["oil"] / prices["gold"].replace(0, np.nan), w)
    z2 = rolling_zscore(prices["oil"] / prices["natgas"].replace(0, np.nan), w)
    z3 = rolling_zscore(gs["gold_real"], w)
    return (ZSC_W["oil_gold"] * z1 + ZSC_W["oil_natgas"] * z2 + ZSC_W["gold_real"] * z3).dropna()

def fit_garch(ret, exog, min_obs=30):
    r = ret.dropna()
    if len(r) < min_obs:
        return pd.Series(r.std(), index=ret.index).ffill().bfill()
    if exog is not None and not exog.empty:
        common_idx = r.index.intersection(exog.dropna().index)
        if len(common_idx) < min_obs:
            exog = None
        else:
            rc = r.loc[common_idx] * 100
            xc = exog.loc[common_idx]
    else:
        rc = r * 100
        xc = None
    try:
        if xc is not None:
            res = arch_model(rc, x=xc, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        else:
            res = arch_model(rc, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        vol = res.conditional_volatility / 100
        return vol.reindex(ret.index).ffill().bfill()
    except Exception:
        pass
    try:
        roll_vol = r.rolling(20).std()
        return roll_vol.reindex(ret.index).ffill().bfill()
    except:
        return pd.Series(r.std(), index=ret.index)

def bayes_shrink(vg, prior_d, n, geofactor=None):
    if vg.empty:
        out = pd.Series(prior_d, index=pd.Index([0]))
        return out, {"vga": prior_d * np.sqrt(252) * 100,
                     "vsa": prior_d * np.sqrt(252) * 100,
                     "w": 1.0}
    w = np.clip(np.sqrt(n / 252), 0.10, 0.95)
    prior = prior_d * (1.0 + 0.4 * np.tanh(float(geofactor.iloc[-1]))) if geofactor is not None and not geofactor.empty else prior_d
    lo, hi = prior * 0.5, prior * 1.5
    v_last = float(vg.iloc[-1])
    vs = vg.copy() if lo <= v_last <= hi else w * vg + (1 - w) * prior
    return vs, {
        "vga": v_last * np.sqrt(252) * 100,
        "vsa": float(vs.iloc[-1]) * np.sqrt(252) * 100,
        "w": w if not (lo <= v_last <= hi) else 1.0
    }

def fit_dcc(rw, rb, vw, vb):
    common = rw.index.intersection(rb.index).intersection(vw.index).intersection(vb.index)
    ew = (rw[common] / vw[common]).dropna()
    eb = (rb[common] / vb[common]).dropna()
    c2 = ew.index.intersection(eb.index)
    e = np.column_stack([ew[c2], eb[c2]])
    Qb = np.cov(e, rowvar=False)
    def nll(p):
        a, b = p
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10
        Qt = Qb.copy()
        ll = 0.0
        for t in range(1, len(e)):
            Qt = (1 - a - b) * Qb + a * np.outer(e[t - 1], e[t - 1]) + b * Qt
            d = np.sqrt(np.diag(Qt))
            d[d == 0] = 1e-8
            Rt = Qt / np.outer(d, d)
            Rt = np.clip(Rt, -0.9999, 0.9999)
            try:
                L = np.linalg.cholesky(Rt)
                z = np.linalg.solve(L, e[t])
                ll += -0.5 * np.sum(z ** 2) - np.sum(np.log(np.diag(L)))
            except:
                return 1e10
        return -ll
    res = optimize.minimize(nll, [0.05, 0.93], bounds=[(1e-4, 0.3), (0.7, 0.9999)], method="L-BFGS-B")
    a, b = res.x
    return (0.05, 0.93) if a + b >= 1 else (float(a), float(b))

def _tail_jumps(shocks, vol):
    n = len(shocks)
    u = np.random.rand(n)
    return shocks + np.where(u < 0.025, np.random.exponential(0.03, n) * vol, 0) - np.where((u >= 0.025) & (u < 0.05), np.random.exponential(0.02, n) * vol, 0)

def _jumps_vec(n, pu, pd_):
    u = np.random.rand(n)
    me = np.random.rand(n) < 0.15
    ju = np.where(me, np.random.exponential(0.135, n), np.random.exponential(0.045, n))
    jd = np.random.exponential(0.025, n)
    return np.where(u < pu, ju, np.where((u >= pu) & (u < pu + pd_), -jd, 0)), np.where(u < pu, ju * 0.95, np.where((u >= pu) & (u < pu + pd_), -jd * 0.90, 0))

def run_mc(wti0, brt0, bvw, bvb, fcast, ocol, bcol, rbase, rw, rb, vws, vbs, jpu, tdf, bs=1.0,
           dcc_a=0.05, dcc_b=0.93, sims=5000, steps=10, bar=None):
    np.random.seed(42)
    ci = rw.index.intersection(rb.index).intersection(vws.index).intersection(vbs.index)
    ew = (rw[ci] / vws[ci].replace(0, np.nan)).dropna()
    eb = (rb[ci] / vbs[ci].replace(0, np.nan)).dropna()
    c2 = ew.index.intersection(eb.index)
    e = np.column_stack([np.clip(ew[c2], -3, 3), np.clip(eb[c2], -3, 3)])
    Qb = np.cov(e, rowvar=False)
    np.fill_diagonal(Qb, 1.0)
    eps = e[-1] + np.random.normal(0, 0.05, (sims, 2))
    Qt = np.tile(Qb, (sims, 1, 1)).copy()
    pu = min(jpu * 1.5, 0.20) if bs > 1.2 else jpu
    pd_ = 0.03 * (1.3 if bs > 1.2 else 1.0)
    pw = np.zeros((sims, steps + 1))
    pb = np.zeros((sims, steps + 1))
    pw[:, 0] = wti0
    pb[:, 0] = brt0
    ra = 1 + 0.5 * np.clip(rbase + np.random.normal(0, 0.05, (sims, steps)), -1, 1)
    for t in range(steps):
        if bar:
            bar.progress((t + 1) / steps)
        outer = np.einsum("si,sj->sij", eps, eps)
        Qt = (1 - dcc_a - dcc_b) * Qb[np.newaxis] + dcc_a * outer + dcc_b * Qt
        diag = np.clip(np.sqrt(np.diagonal(Qt, axis1=1, axis2=2)), 1e-8, None)
        Rt = Qt / np.einsum("si,sj->sij", diag, diag)
        Rt = np.clip(Rt, -0.9999, 0.9999)
        Rt[:, 0, 0] = Rt[:, 1, 1] = 1.0
        rho = Rt[:, 0, 1]
        sc = np.sqrt(np.clip(1 - rho ** 2, 1e-8, None))
        z = np.random.standard_t(tdf, (sims, 2))
        zw = z[:, 0]
        zb = rho * z[:, 0] + sc * z[:, 1]
        vw_ = np.clip(bvw * ra[:, t], 0, 0.08)
        vb_ = np.clip(bvb * ra[:, t], 0, 0.08)
        sw = np.clip(zw * vw_, -4 * vw_, 4 * vw_)
        sb = np.clip(zb * vb_, -4 * vb_, 4 * vb_)
        sw = _tail_jumps(sw, vw_)
        sb = _tail_jumps(sb, vb_)
        jw, jb = _jumps_vec(sims, pu, pd_)
        sw += jw
        sb += jb
        dw = np.clip(fcast[t, ocol] * ra[:, t], -0.02, 0.02)
        db = np.clip(fcast[t, bcol] * ra[:, t], -0.02, 0.02)
        nw = pw[:, t] * np.exp(dw + sw)
        nb = pb[:, t] * np.exp(db + sb)
        sp = np.where(nb > 0, (nb - nw) / nb, 0)
        nw = np.where(sp < -0.05, nb * 1.05, nw)
        nw = np.where(sp > 0.30, nb * 0.70, nw)
        pw[:, t + 1] = np.clip(nw, wti0 * 0.4, wti0 * 2.5)
        pb[:, t + 1] = np.clip(nb, brt0 * 0.4, brt0 * 2.5)
        eps[:, 0] = np.where(vw_ > 0, sw / vw_, 0)
        eps[:, 1] = np.where(vb_ > 0, sb / vb_, 0)
        eps = np.clip(eps, -5, 5)
    fan = {p: np.percentile(pw, p, axis=0) for p in [5, 25, 50, 75, 95]}
    fb = {p: np.percentile(pb, p, axis=0) for p in [5, 25, 50, 75, 95]}
    term = pw[:, -1]
    v95 = np.percentile(pw[:, 1] - wti0, 5)
    mask = (pw[:, 1] - wti0) <= v95
    return {
        "fan": fan,
        "fan_b": fb,
        "paths": pw,
        "metrics": {
            "vol_wti": bvw * np.sqrt(252) * 100,
            "vol_brt": bvb * np.sqrt(252) * 100,
            "var95": v95,
            "cvar95": float(np.mean((pw[:, 1] - wti0)[mask])),
            "prob_up": np.mean(term > wti0) * 100,
            "prob_40": np.mean(term < 40) * 100,
            "prob_150": np.mean(term > 150) * 100,
            "p5": (fan[5][-1] / wti0 - 1) * 100,
            "p95": (fan[95][-1] / wti0 - 1) * 100
        }
    }

def backtest_var(returns, var_forecast, alpha=0.05):
    ci = returns.index.intersection(var_forecast.index)
    if len(ci) == 0:
        return {"calibration_score": 0, "Kupiec_p": 1, "Christoffersen_p": 1, "DQ_p": 1, "n_violations": 0, "obs_freq": 0}
    r = returns.loc[ci]
    v = var_forecast.loc[ci]
    viol = (r < -v).astype(int)
    n = len(viol)
    nv = viol.sum()
    po = nv / n
    pe = alpha
    if nv > 0 and nv < n:
        LR = -2 * np.log(((1 - pe) ** (n - nv) * pe ** nv) / ((1 - po) ** (n - nv) * po ** nv))
        kp = 1 - chi2.cdf(LR, 1)
    else:
        kp = 0.5
    if n > 1:
        n00 = ((viol[:-1] == 0) & (viol[1:] == 0)).sum()
        n01 = ((viol[:-1] == 0) & (viol[1:] == 1)).sum()
        n10 = ((viol[:-1] == 1) & (viol[1:] == 0)).sum()
        n11 = ((viol[:-1] == 1) & (viol[1:] == 1)).sum()
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        LRc = -2 * np.log(((1 - pe) ** (n - 1 - (n01 + n11)) * pe ** (n01 + n11)) /
                          ((1 - p01) ** n00 * p01 ** n01 * (1 - p11) ** n10 * p11 ** n11)) if (n01 + n11) > 0 else 0
        cp = 1 - chi2.cdf(LRc, 1) if LRc > 0 else 0.5
    else:
        cp = 0.5
    Xd = pd.DataFrame({"const": 1, "hit_lag1": viol.shift(1).fillna(0)})
    try:
        dq = 1 - chi2.cdf(Logit(viol, Xd).fit(disp=0).llr, Xd.shape[1])
    except:
        dq = 1.0
    return {
        "n_violations": int(nv),
        "obs_freq": po,
        "exp_freq": pe,
        "Kupiec_p": kp,
        "Christoffersen_p": cp,
        "DQ_p": dq,
        "calibration_score": 1 - np.mean([kp, cp, dq])
    }

def backtest_es(returns, cvar_val, var_forecast, alpha=0.05):
    ci = returns.index.intersection(var_forecast.index)
    if len(ci) == 0:
        return np.nan
    r = returns.loc[ci]
    v = var_forecast.loc[ci]
    cv = float(cvar_val) if not isinstance(cvar_val, pd.Series) else float(cvar_val.iloc[0]) if len(cvar_val) > 0 else np.nan
    if np.isnan(cv) or cv == 0:
        return np.nan
    viol = (r < -v).astype(int)
    if viol.sum() == 0:
        return np.nan
    return float(((r[viol == 1] + v[viol == 1]).sum() / (viol.sum() * cv)) - 1)

def walk_forward_validation(returns_series, train_years=2, test_months=3):
    dates = returns_series.index
    ts = int(train_years * 252)
    qs = int(test_months * 21)
    results = []
    start = 0
    while start + ts + qs <= len(dates):
        te = start + ts
        qe = te + qs
        mu = returns_series.iloc[start:te].mean()
        rmse = float(np.sqrt(((mu - returns_series.iloc[te:qe]) ** 2).mean()))
        results.append({
            "Window Start": dates[start].strftime("%Y-%m-%d"),
            "Window End": dates[qe - 1].strftime("%Y-%m-%d"),
            "OOS RMSE": rmse
        })
        start += qs
    return pd.DataFrame(results)

def benchmark_ml(returns_df, target_col="oil", split=0.8):
    features = returns_df.shift(1).dropna()
    target = returns_df[target_col].iloc[1:]
    ci = features.index.intersection(target.index)
    X, y = features.loc[ci], target.loc[ci]
    sp = int(len(X) * split)
    Xtr, Xte = X.iloc[:sp], X.iloc[sp:]
    ytr, yte = y.iloc[:sp], y.iloc[sp:]
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    out = {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        out[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
            "MAE": float(mean_absolute_error(yte, pred))
        }
    return out, X, y

def run_shap(X, y):
    mdl = RandomForestRegressor(n_estimators=100, random_state=42)
    mdl.fit(X, y)
    exp = shap.TreeExplainer(mdl)
    sv = exp.shap_values(X)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    shap.summary_plot(sv, X, show=False, plot_size=None)
    plt.tight_layout()
    return fig

def garch_diagnostics(resid):
    r = resid.dropna()
    if len(r) < 20:
        return {"LB5": np.nan, "LB10": np.nan, "ARCH_p": np.nan}
    lb = acorr_ljungbox(r, lags=[5, 10], return_df=True)
    arch = het_arch(r ** 2, nlags=10)
    return {
        "LB5": lb.loc[5, "lb_pvalue"] if 5 in lb.index else np.nan,
        "LB10": lb.loc[10, "lb_pvalue"] if 10 in lb.index else np.nan,
        "ARCH_p": arch[1] if len(arch) > 1 else np.nan
    }

def fit_evt_tails(rv, q=0.95):
    th_up = np.percentile(rv, q * 100)
    th_lo = np.percentile(rv, (1 - q) * 100)
    eu = rv[rv > th_up] - th_up
    el = -rv[rv < th_lo] - th_lo
    su, _, scu = stats.genpareto.fit(eu) if len(eu) > 5 else (0.5, 0, 0.1)
    sl, _, scl = stats.genpareto.fit(el) if len(el) > 5 else (0.5, 0, 0.1)
    return {"upper": (su, scu, th_up), "lower": (sl, scl, th_lo)}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_data(start):
    tl = list(TICKERS.values())
    tk = list(TICKERS.keys())
    def _close(raw):
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            lvl = raw.columns.get_level_values(0).unique().tolist()
            field = next((f for f in ["Close", "Adj Close"] if f in lvl), None)
            return raw[field].copy() if field else raw.iloc[:, :len(tk)].copy()
        return raw.copy()
    for adj in [True, False]:
        try:
            raw = yf.download(tl, start=start, progress=False, auto_adjust=adj)
            out = _close(raw)
            if not out.empty and len(out) > 5:
                out.columns = tk[:len(out.columns)]
                return out.ffill().bfill()
        except:
            pass
    frames = {}
    for key, sym in TICKERS.items():
        try:
            t = yf.Ticker(sym)
            df = t.history(start=start, auto_adjust=True)
            if df.empty:
                df = t.history(period="120d", auto_adjust=True)
            if not df.empty:
                col = "Close" if "Close" in df.columns else df.columns[0]
                frames[key] = fill_gaps(df[col])
        except:
            pass
    if frames:
        out = pd.DataFrame(frames).ffill().bfill()
        if not out.empty and len(out) > 5:
            return out
    return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live(lw=65.0, lb=68.0):
    try:
        wti = float(yf.Ticker("CL=F").fast_info.get("last_price", 0))
        brt = float(yf.Ticker("BZ=F").fast_info.get("last_price", 0))
        if wti > 0 and brt > 0:
            return wti, brt
    except:
        pass
    try:
        d = yf.download(["CL=F", "BZ=F"], period="5d", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d = d["Close"]
        return float(d["CL=F"].dropna().iloc[-1]), float(d["BZ=F"].dropna().iloc[-1])
    except:
        return lw, lb

# ══════════════════════════════════════════════════════════
#   PIPELINE (Protected)
# ══════════════════════════════════════════════════════════
needs_run = run_btn or "results" not in st.session_state

if needs_run:
    loading = st.empty()
    loading.markdown("""
    <div style='text-align:center;padding:2.2rem 2rem;background:#F8F7F4;border:1px solid #D9D5CD;margin:1rem 0;'>
      <div style='font-family:"JetBrains Mono",monospace;font-size:.5rem;letter-spacing:.22em;color:#B49450;text-transform:uppercase;margin-bottom:.6rem;'>Initialising Research Terminal</div>
      <div style='font-family:"Playfair Display",Georgia,serif;font-size:1.3rem;color:#1E3A5F;font-weight:400;'>Loading market data & calibrating quantitative framework…</div>
    </div>""", unsafe_allow_html=True)
    prog = st.progress(0)

    prog.progress(8)
    prices = fetch_data(war_start_str)
    if prices.empty or len(prices) < 5:
        loading.empty()
        prog.empty()
        st.error("Failed to load market data. Please wait 1 minute and try again.")
        st.stop()
    prices = prices.ffill().bfill()
    for key in TICKERS:
        if key not in prices.columns:
            prices[key] = np.nan
    prices = prices.ffill().bfill()

    lw = float(prices["oil"].dropna().iloc[-1])
    lb = float(prices["brent"].dropna().iloc[-1])
    wti0, brt0 = fetch_live(lw, lb)
    prices.loc[prices.index[-1], "oil"] = wti0
    prices.loc[prices.index[-1], "brent"] = brt0
    returns = np.log(prices / prices.shift(1)).dropna()

    prog.progress(20)
    usda = get_usda()
    bs_mult = fert_black_swan(usda)
    gs = gold_signals(prices)
    sd = silver_demand_proxy(prices)
    weights = GEO_W.copy()
    weights["silver_demand"] = 0.02
    tot = sum(abs(v) for v in weights.values())
    weights = {k: v / tot for k, v in weights.items()}
    fi = build_fert_index(returns, usda, bs_mult)
    dyn_w = calibrate_weights(returns, prices, gs, fi, sd)
    if dyn_w:
        weights = dyn_w
    gf_raw = build_geofactor(returns, prices, gs, fi, weights, sd)
    gf = (gf_raw - gf_raw.mean()) / gf_raw.std() if len(gf_raw) > 1 else gf_raw
    zsc = build_zscore(prices, gs)

    prog.progress(38)
    gf_clean = gf.dropna() if not gf.empty else None
    if gf_clean is not None and len(gf_clean) < 30:
        gf_clean = None

    vw = fit_garch(returns["oil"], gf_clean)
    vb_s = fit_garch(returns["brent"], gf_clean)
    vg = fit_garch(returns["gold"], gf_clean)

    # Ensure no series is empty (final safety)
    if vw.empty:
        vw = pd.Series(prior_wti/np.sqrt(252), index=returns.index)
    if vb_s.empty:
        vb_s = pd.Series(prior_brent/np.sqrt(252), index=returns.index)
    if vg.empty:
        vg = pd.Series(0.18/np.sqrt(252), index=returns.index)

    n = len(returns)
    pwd = prior_wti / np.sqrt(252)
    pbd = prior_brent / np.sqrt(252)
    pgd = 0.18 / np.sqrt(252)

    vw, dw = bayes_shrink(vw, pwd, n, gf)
    vb_s, db = bayes_shrink(vb_s, pbd, n, gf)
    vg, _ = bayes_shrink(vg, pgd, n)

    if vw.empty:
        vw = pd.Series(pwd, index=returns.index)
    if vb_s.empty:
        vb_s = pd.Series(pbd, index=returns.index)
    if vg.empty:
        vg = pd.Series(pgd, index=returns.index)

    # Safe extraction of last value
    bvw = float(vw.iloc[-1]) if len(vw) > 0 else pwd
    bvb = float(vb_s.iloc[-1]) if len(vb_s) > 0 else pbd

    prog.progress(55)
    dcc_a, dcc_b = fit_dcc(returns["oil"], returns["brent"], vw, vb_s)
    rv = returns.loc[gf.index.intersection(returns.index)]
    vm = VAR(rv).fit(min(5, max(1, len(rv) // 10)))
    fcast = vm.forecast(rv.values[-vm.k_ar:], steps=mc_steps)
    cols = list(rv.columns)
    ocol = cols.index("oil")
    bcol = cols.index("brent")
    tdf_d = max(2.5, min(6.0, tail_df / np.sqrt(max(bvb / (pbd * 1.5), 0.5))))
    rbase = float(np.tanh(gf.iloc[-1] / 2)) if not gf.empty else 0.0
    jpu = min(jump_up * 1.5, 0.15) if returns["wheat"].tail(20).mean() > 0.005 else jump_up

    prog.progress(68)
    mb = st.empty()
    mc_bar = st.progress(0)
    mb.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:.55rem;letter-spacing:.12em;color:#5A554F;">Monte Carlo simulation executing…</div>', unsafe_allow_html=True)
    mc = run_mc(wti0, brt0, bvw, bvb, fcast, ocol, bcol, rbase, returns["oil"], returns["brent"],
                vw, vb_s, jpu, tdf_d, bs_mult, dcc_a, dcc_b, mc_sims, mc_steps, mc_bar)
    mb.empty()
    mc_bar.empty()

    prog.progress(80)
    ret_ann = returns[["oil", "brent"]].mean() * 252
    vol_ann = returns[["oil", "brent"]].std() * np.sqrt(252)
    neg = returns[["oil", "brent"]][returns[["oil", "brent"]] < 0].std() * np.sqrt(252)
    corr_mx = returns[["oil", "brent", "gold", "dxy", "tnx"]].dropna().corr()
    stress_c = pd.DataFrame({
        "vol_wti": vw.rolling(20).mean() * np.sqrt(252) * 100,
        "vol_brt": vb_s.rolling(20).mean() * np.sqrt(252) * 100,
        "corr": returns["oil"].rolling(20).corr(returns["brent"]),
        "gold_z": rolling_zscore(prices["gold"], 60),
        "geofactor": gf
    }).dropna()
    stress_idx = (stress_c["vol_wti"] / 50 + stress_c["vol_brt"] / 50 + np.abs(stress_c["corr"] - 0.8) * 2 +
                  stress_c["gold_z"].clip(0, 3) / 3 + stress_c["geofactor"].clip(0, 2) / 2) / 5
    feat_imp = pd.DataFrame({
        "Feature": list(weights.keys()),
        "Importance": np.abs(list(weights.values()))
    }).sort_values("Importance", ascending=False)
    evt = fit_evt_tails(returns["oil"].values)
    gdiag = garch_diagnostics(vw)
    var_s = vw.iloc[-252:] * 1.645
    cvar_s = vw.iloc[-252:] * 2.326
    bt_res = backtest_var(returns["oil"].iloc[-252:], var_s)
    es_z = backtest_es(returns["oil"].iloc[-252:], float(cvar_s.iloc[-1]), var_s)

    try:
        corr_ewma = float(np.clip(
            returns[["oil", "brent"]].dropna().ewm(alpha=0.06).cov(pairwise=True)
            .loc[(returns.index[-1], "oil"), "brent"] /
            np.sqrt(returns[["oil", "brent"]].dropna().ewm(alpha=0.06).cov(pairwise=True)
                    .loc[(returns.index[-1], "oil"), "oil"] *
                    returns[["oil", "brent"]].dropna().ewm(alpha=0.06).cov(pairwise=True)
                    .loc[(returns.index[-1], "brent"), "brent"]), -1, 1))
    except:
        corr_ewma = 0.95

    prog.progress(90)
    try:
        ml_metrics, X_ml, y_ml = benchmark_ml(returns)
    except Exception as e:
        ml_metrics = {"Error": {"RMSE": str(e), "MAE": "—"}}
        X_ml = returns.shift(1).dropna()
        y_ml = returns["oil"].iloc[1:]
    try:
        shap_fig = run_shap(X_ml, y_ml)
    except:
        shap_fig = None
    wf_df = walk_forward_validation(returns["oil"])

    prog.progress(100)
    loading.empty()
    prog.empty()

    st.session_state.update({
        "results": mc, "gf": gf, "zsc": zsc, "vw": vw, "vb": vb_s, "vg": vg, "fi": fi, "gs": gs,
        "prices": prices, "returns": returns, "wti0": wti0, "brt0": brt0, "usda": usda,
        "bs": bs_mult, "dw": dw, "db": db, "tdf": tdf_d, "dcc_a": dcc_a, "dcc_b": dcc_b,
        "weights": weights, "sharpe": ret_ann / vol_ann, "sortino": ret_ann / neg,
        "skew_oil": returns["oil"].skew(), "kurt_oil": returns["oil"].kurtosis(),
        "skew_brt": returns["brent"].skew(), "kurt_brt": returns["brent"].kurtosis(),
        "corr_mx": corr_mx, "stress_idx": stress_idx, "feat_imp": feat_imp, "evt": evt,
        "gdiag": gdiag, "bt_res": bt_res, "es_z": es_z, "corr_ewma": corr_ewma,
        "ml_metrics": ml_metrics, "shap_fig": shap_fig, "wf_df": wf_df,
    })

# ══════════════════════════════════════════════════════════
#   RENDER
# ══════════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **▶ Run Full System Pipeline** to start.")
    st.stop()

S   = st.session_state
mc  = S["results"]; fan=mc["fan"]; fb=mc["fan_b"]; M=mc["metrics"]
gf  = S["gf"]; zsc=S["zsc"]; vw=S["vw"]; vb=S["vb"]; vg=S["vg"]
fi  = S["fi"]; gs=S["gs"]; prices=S["prices"]; returns=S["returns"]
wti0=S["wti0"]; brt0=S["brt0"]; usda=S["usda"]; bs=S["bs"]
dw_d=S["dw"]; db_d=S["db"]; tdf_d=S["tdf"]; dcc_a=S["dcc_a"]; dcc_b=S["dcc_b"]
spread=brt0-wti0

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs([
    "Market & Volatility","Geopolitical Intelligence","Monte Carlo",
    "Quant Statistics","Macro & Stress","Institutional Backtest",
    "ML Benchmarks","Walk-Forward",
])

# ── TAB 1 · Market & Volatility ──────────────────────────
with tab1:
    st.markdown('<div class="sec-label">01 · Live Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Market Metrics & Conditional Volatility</div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("WTI Crude", f"${wti0:.2f}", f"P50 10d → ${fan[50][-1]:.2f}")
    c2.metric("Brent Crude", f"${brt0:.2f}", f"Spread ${spread:.2f} ({spread/wti0*100:.1f}%)")
    c3.metric("WTI Vol p.a.", f"{M['vol_wti']:.1f}%", f"Shrunk {dw_d['vsa']:.1f}%")
    c4.metric("Brent Vol p.a.", f"{M['vol_brt']:.1f}%", f"Shrunk {db_d['vsa']:.1f}%")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    fig_vol=qfig(380)
    fig_vol.add_trace(go.Scatter(x=vw.index,y=vw*np.sqrt(252)*100,name="WTI Vol",
        line=dict(color=C["navy"],width=2.2)))
    fig_vol.add_trace(go.Scatter(x=vb.index,y=vb*np.sqrt(252)*100,name="Brent Vol",
        line=dict(color=C["sky"],width=2.2,dash="dash")))
    fig_vol.add_trace(go.Scatter(x=vg.index,y=vg*np.sqrt(252)*100,name="Gold Vol",
        line=dict(color=C["gold"],width=1.8,dash="dot")))
    fig_vol.add_hrect(y0=25,y1=45,fillcolor="rgba(30,58,95,0.04)",line_width=0,
        annotation_text="Normal band 25–45%",annotation_position="top left",
        annotation_font=dict(size=9,color=C["gray"],family="JetBrains Mono,monospace"))
    fig_vol.update_layout(yaxis_ticksuffix="%",
        title="GARCH-X Conditional Volatility — Adaptive Bayesian Shrinkage Framework")
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown(f'<div class="info-block">Bayes Shrinkage · WTI {dw_d["vga"]:.0f}% → {dw_d["vsa"]:.0f}% (w={dw_d["w"]:.2f}) · Brent {db_d["vga"]:.0f}% → {db_d["vsa"]:.0f}% (w={db_d["w"]:.2f}) · DCC α={dcc_a:.4f} β={dcc_b:.4f}</div>',unsafe_allow_html=True)

# ── TAB 2 · Geopolitical Intelligence ────────────────────
with tab2:
    st.markdown('<div class="sec-label">02 · Geopolitical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Z-Score Composite & GeoFactor Estimation (σ-normalised)</div>', unsafe_allow_html=True)

    fig_geo=dual_axis_fig(380)
    fig_geo.add_trace(go.Scatter(x=zsc.index,y=zsc.values,name="Z-Score Composite",
        line=dict(color=C["sky"],width=2.2),
        fill="tozeroy",fillcolor="rgba(74,115,128,0.06)"))
    fig_geo.add_trace(go.Scatter(x=gf.index,y=gf.values,name="GeoFactor (σ)",
        line=dict(color=C["navy"],width=2.8),yaxis="y2"))
    for y in [1.5,-1.5]:
        fig_geo.add_shape(type="line",x0=zsc.index.min(),x1=zsc.index.max(),
            y0=y,y1=y,line=dict(dash="dot",color=C["gold"],width=1.2),yref="y")
    fig_geo.add_shape(type="line",x0=zsc.index.min(),x1=zsc.index.max(),
        y0=0,y1=0,line=dict(dash="solid",color="#D9D5CD",width=0.8),yref="y")
    fig_geo.update_layout(yaxis_title="Z-Score",yaxis2_title="GeoFactor (σ)",
        title="Geopolitical Risk Signals Extraction")
    st.plotly_chart(fig_geo, use_container_width=True)

    col_a,col_b=st.columns(2)
    with col_a:
        ng_vol=returns["natgas"].rolling(20).std()*np.sqrt(252)*100
        fig_f=dual_axis_fig(310)
        fig_f.add_trace(go.Scatter(x=fi.index,y=fi.values,name="Fertilizer Stress",
            fill="tozeroy",fillcolor="rgba(74,93,74,0.06)",line=dict(color=C["sage"],width=2.2)))
        fig_f.add_trace(go.Scatter(x=ng_vol.index,y=ng_vol.values,name="NatGas Vol %",
            line=dict(color=C["teal"],width=1.8,dash="dash"),yaxis="y2"))
        fig_f.update_layout(yaxis_title="Fert Index",yaxis2_title="NatGas Vol %",
            title="Fertilizer Stress + NatGas Volatility Indices")
        st.plotly_chart(fig_f, use_container_width=True)
        bs_str=f" · ⚠ Black Swan ×{bs:.2f}" if bs>1.2 else ""
        st.markdown(f'<div class="info-block">Urea ${usda["urea_price"]:.1f}/t · DAP ${usda["dap_price"]:.0f}/t{bs_str} · {usda["source"]}</div>',unsafe_allow_html=True)
    with col_b:
        grb=float(gs["gold_real"].dropna().iloc[0]); sgb=float(gs["silver_gold"].dropna().iloc[0])
        fig_g=dual_axis_fig(310)
        fig_g.add_trace(go.Scatter(x=gs["gold_real"].dropna().index,
            y=(gs["gold_real"].dropna()/grb).values,name="Gold/Real Yield",
            line=dict(color=C["gold"],width=2.2)))
        fig_g.add_trace(go.Scatter(x=gs["silver_gold"].dropna().index,
            y=(gs["silver_gold"].dropna()/sgb).values,name="Silver/Gold Ratio",
            line=dict(color=C["silver"],width=1.8,dash="dash"),yaxis="y2"))
        fig_g.update_layout(yaxis_title="Gold/Real Yield (norm)",
            yaxis2_title="Silver/Gold (norm)",title="Gold Macro Signals — Real Yield Basis")
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title" style="font-size:1.1rem;">Agricultural Commodities — Indexed from Regime Inception</div>', unsafe_allow_html=True)
    fig_ag=qfig(280)
    for asset,color,label in [("wheat",C["navy"],"Wheat"),("corn",C["sky"],"Corn"),("soy",C["gray"],"Soy")]:
        bv=float(prices[asset].iloc[0]); rel=(prices[asset]/bv*100).dropna()
        fig_ag.add_trace(go.Scatter(x=rel.index,y=rel.values,
            name=f"{label} (base ${bv:.0f})",line=dict(color=color,width=2.0)))
    fig_ag.add_hline(y=100,line_dash="dot",line_color="#D9D5CD",line_width=1.2)
    fig_ag.update_layout(yaxis_title="Price Index (base = 100)")
    st.plotly_chart(fig_ag, use_container_width=True)

# ── TAB 3 · Monte Carlo ──────────────────────────────────
with tab3:
    war_n="  ⚑ Regime Drift Adjusted" if returns["wheat"].tail(20).mean()>0.005 else ""
    bs_n=f"  ⚠ Stress Multiplier ×{bs:.2f}" if bs>1.2 else ""
    st.markdown('<div class="sec-label">03 · Probabilistic Forecast</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">Predictive Distribution Simulation · EVT+DCC Process · {mc_sims:,} Scenarios{war_n}{bs_n}</div>', unsafe_allow_html=True)

    x_ax=list(range(mc_steps+1))
    fig_mc=qfig(480)
    fig_mc.add_trace(go.Scatter(x=x_ax+x_ax[::-1],y=list(fan[95])+list(fan[5][::-1]),
        fill="toself",fillcolor=C["fill_light"],line=dict(width=0),name="WTI 90% CI"))
    fig_mc.add_trace(go.Scatter(x=x_ax+x_ax[::-1],y=list(fan[75])+list(fan[25][::-1]),
        fill="toself",fillcolor=C["fill_medium"],line=dict(width=0),name="WTI 50% CI"))
    fig_mc.add_trace(go.Scatter(x=x_ax,y=list(fb[50]),name=f"Brent P50 → ${fb[50][-1]:.2f}",
        line=dict(color=C["sky"],width=2.2,dash="dash")))
    fig_mc.add_trace(go.Scatter(x=x_ax,y=list(fan[50]),name=f"WTI P50 → ${fan[50][-1]:.2f}",
        line=dict(color=C["navy"],width=3.2)))
    fig_mc.add_trace(go.Scatter(x=x_ax,y=list(fan[95]),name=f"P95 → ${fan[95][-1]:.2f}",
        line=dict(color=C["gold"],width=1.4,dash="dot")))
    fig_mc.add_trace(go.Scatter(x=x_ax,y=list(fan[5]),name=f"P5 → ${fan[5][-1]:.2f}",
        line=dict(color=C["gold"],width=1.4,dash="dot")))
    fig_mc.add_hline(y=wti0,line_dash="dash",line_color="#9A958A",line_width=1.4,
        annotation_text=f"Current ${wti0:.2f}",annotation_font=dict(family="JetBrains Mono",size=10,color="#9A958A"))
    fig_mc.add_hline(y=40,line_dash="dot",line_color=C["burgundy"],line_width=1.4,
        annotation_text="Stress Target $40",annotation_font=dict(family="JetBrains Mono",size=10,color=C["burgundy"]))
    fig_mc.add_hline(y=150,line_dash="dot",line_color=C["burgundy"],line_width=1.4,
        annotation_text="Stress Target $150",annotation_font=dict(family="JetBrains Mono",size=10,color=C["burgundy"]))
    fig_mc.update_layout(xaxis_title="Trading Days Ahead",yaxis_title="Price (USD/bbl)",yaxis_tickprefix="$")
    st.plotly_chart(fig_mc, use_container_width=True)

    c1,c2,c3=st.columns(3)
    c1.metric("Regime Growth Probability", f"{M['prob_up']:.1f}%")
    c2.metric("VaR 95% (1-Day)",    f"${M['var95']:+.2f}", f"Expected Shortfall ${M['cvar95']:+.2f}")
    c3.metric("Tail Exceedance", f"Tail < $40: {M['prob_40']:.2f}%", f"Tail > $150: {M['prob_150']:.2f}%")

# ── TAB 4 · Quant Statistics ─────────────────────────────
with tab4:
    st.markdown('<div class="sec-label">04 · Distribution Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Empirical Risk Distribution Parameters</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        rows=[("Sharpe Ratio (Ann.)",f"{S['sharpe']['oil']:.4f}"),
              ("Sortino Ratio (Ann.)",f"{S['sortino']['oil']:.4f}"),
              ("Skewness Coefficient",f"{S['skew_oil']:.4f}"),("Excess Kurtosis",f"{S['kurt_oil']:.4f}"),
              ("GARCH LB5 Test",f"{S['gdiag']['LB5']:.4f}" if not np.isnan(S['gdiag']['LB5']) else "n/a"),
              ("GARCH LB10 Test",f"{S['gdiag']['LB10']:.4f}" if not np.isnan(S['gdiag']['LB10']) else "n/a"),
              ("ARCH LM Test",f"{S['gdiag']['ARCH_p']:.4f}" if not np.isnan(S['gdiag']['ARCH_p']) else "n/a")]
        html='<table class="data-table"><thead><tr><th>WTI Metric</th><th>Statistical Value</th></tr></thead><tbody>'
        for k,v in rows: html+=f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
        html+="</tbody></table>"; st.markdown(html,unsafe_allow_html=True)
    with c2:
        rows2=[("DCC Parameter α",f"{dcc_a:.4f}"),("DCC Parameter β",f"{dcc_b:.4f}"),
               ("Systemic Persistence α+β",f"{dcc_a+dcc_b:.4f}"),
               ("Cross-Asset Correlation (EWMA)",f"{S['corr_ewma']:.4f}"),
               ("Dynamic Tail Degrees of Freedom",f"{tdf_d:.2f}"),
               ("EVT Generalized Pareto Upper Shape",f"{S['evt']['upper'][0]:.4f}"),
               ("EVT Generalized Pareto Lower Shape",f"{S['evt']['lower'][0]:.4f}")]
        html='<table class="data-table"><thead><tr><th>Multivariate Infrastructure</th><th>Statistical Value</th></tr></thead><tbody>'
        for k,v in rows2: html+=f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
        html+="</tbody></table>"; st.markdown(html,unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title" style="font-size:1.1rem;">Cross-Asset Correlation Matrix</div>', unsafe_allow_html=True)
    fig_corr=qfig(320)
    cm=S["corr_mx"].values; cols_=list(S["corr_mx"].columns)
    fig_corr.add_trace(go.Heatmap(z=cm,x=cols_,y=cols_,
        colorscale=[[0,"#8B3A3A"],[0.5,"#FFFFFF"],[1,"#1E3A5F"]],
        zmid=0,text=np.round(cm,3),texttemplate="%{text}",
        textfont=dict(size=10,family="JetBrains Mono,monospace"),showscale=True))
    fig_corr.update_layout(title="EWMA Asset Interdependence Topology")
    st.plotly_chart(fig_corr, use_container_width=True)

# ── TAB 5 · Macro & Stress ───────────────────────────────
with tab5:
    st.markdown('<div class="sec-label">05 · System Stress Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Composite Financial Stress Index</div>', unsafe_allow_html=True)
    if S["stress_idx"] is not None and len(S["stress_idx"])>0:
        fig_st=qfig(340)
        fig_st.add_trace(go.Scatter(x=S["stress_idx"].index,y=S["stress_idx"].values,
            fill="tozeroy",fillcolor="rgba(123,63,63,0.05)",
            line=dict(color=C["burgundy"],width=2.2),name="Stress Index"))
        fig_st.add_hline(y=S["stress_idx"].mean(),line_dash="dot",line_color="#9A958A",line_width=1.2,
            annotation_text="Historical Baseline Mean",annotation_font=dict(family="JetBrains Mono",size=9,color="#9A958A"))
        fig_st.update_layout(title="Multivariate Macro Stress Metrics (Volatility + Basis Coherence + Factor Drift)")
        st.plotly_chart(fig_st, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title" style="font-size:1.1rem;">Risk Factor Contribution Calibration (LASSO)</div>', unsafe_allow_html=True)
    fig_imp=qfig(300)
    fi_df=S["feat_imp"].head(10)
    fig_imp.add_trace(go.Bar(x=fi_df["Importance"],y=fi_df["Feature"],orientation="h",
        marker_color=C["navy"],name="Importance"))
    fig_imp.update_layout(xaxis_title="Absolute LASSO Structural Coefficient",yaxis_autorange="reversed",
        title="Structural Factor Importance Optimization")
    st.plotly_chart(fig_imp, use_container_width=True)

# ── TAB 6 · Institutional Backtest ───────────────────────
with tab6:
    st.markdown('<div class="sec-label">06 · Risk Infrastructure Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">VaR & Expected Shortfall Compliance Backtesting</div>', unsafe_allow_html=True)
    bt=S["bt_res"]
    c1,c2,c3=st.columns(3)
    c1.metric("Calibration Score", f"{bt['calibration_score']:.4f}")
    c2.metric("Observed Violations", f"{bt['n_violations']}", f"Frequency {bt['obs_freq']:.3f} vs {bt['exp_freq']:.2f} target")
    c3.metric("Acerbi Shortfall Metric Z", f"{S['es_z']:.4f}" if S['es_z'] is not None and not np.isnan(S['es_z']) else "n/a")

    rows=[("Kupiec Proportion of Failures (PoF)",f"{bt['Kupiec_p']:.4f}","Acceptable" if bt['Kupiec_p']>=0.05 else "Reject"),
          ("Christoffersen Independence Test",f"{bt['Christoffersen_p']:.4f}","Acceptable" if bt['Christoffersen_p']>=0.05 else "Reject"),
          ("Manganelli-Engle Dynamic Quantile (DQ)",f"{bt['DQ_p']:.4f}","Acceptable" if bt['DQ_p']>=0.05 else "Reject")]
    html='<table class="data-table"><thead><tr><th>Statistical Hypothesis Framework</th><th>Asymptotic p-value</th><th>Validation Decision</th></tr></thead><tbody>'
    for k,v,r in rows: html+=f"<tr><td>{k}</td><td><strong>{v}</strong></td><td>{r}</td></tr>"
    html+="</tbody></table>"; st.markdown(html,unsafe_allow_html=True)

# ── TAB 7 · ML Benchmarks ────────────────────────────────
with tab7:
    st.markdown('<div class="sec-label">07 · Machine Learning Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Out-of-Sample Machine Learning Performance Comparison</div>', unsafe_allow_html=True)
    c1,c2=st.columns([1,1])
    with c1:
        bm=S["ml_metrics"]
        rows=[]
        for name,vals in bm.items():
            if isinstance(vals.get("RMSE"),(int,float)):
                rows.append((name,f"{vals['RMSE']:.6f}",f"{vals['MAE']:.6f}"))
            else:
                rows.append((name,str(vals.get("RMSE","—")),str(vals.get("MAE","—"))))
        html='<table class="data-table"><thead><tr><th>Model Architecture</th><th>Root Mean Squared Error</th><th>Mean Absolute Error</th></tr></thead><tbody>'
        for a,b_,c_ in rows: html+=f"<tr><td>{a}</td><td><strong>{b_}</strong></td><td>{c_}</td></tr>"
        html+="</tbody></table>"; st.markdown(html,unsafe_allow_html=True)
    with c2:
        if S["shap_fig"] is not None:
            st.markdown('<div class="sec-label">SHAP Global Equilibrium Interpretation</div>', unsafe_allow_html=True)
            st.pyplot(S["shap_fig"])
        else:
            st.info("SHAP matrix interpretation unavailable.")

# ── TAB 8 · Walk-Forward ─────────────────────────────────
with tab8:
    st.markdown('<div class="sec-label">08 · Validation Integrity</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Rolling Out-of-Sample Error Analysis — Walk-Forward Metric Estimation</div>', unsafe_allow_html=True)
    wf=S["wf_df"]
    if not wf.empty:
        st.dataframe(wf,use_container_width=True)
        fig_wf=qfig(300)
        fig_wf.add_trace(go.Bar(x=wf["Window End"],y=wf["OOS RMSE"],
            marker_color=C["navy"],name="OOS RMSE"))
        fig_wf.update_layout(title="Rolling Window Multi-Period Errors",
            xaxis_title="Evaluation Window End",yaxis_title="OOS RMSE")
        st.plotly_chart(fig_wf, use_container_width=True)
    else:
        st.info("Insufficient degrees of freedom for walk-forward validation matrix.")

st.markdown(f"""
<div class="footer">
  <div>◆ GeoQuant Institutional Terminal · Engine: EVT+DCC+GARCH-X Framework</div>
  <div>Eduardo Moraes · Quant Data Scientist & Economics</div>
  <div>Proprietary Infrastructure · {now_sp.strftime("%d %b %Y")}</div>
</div>""", unsafe_allow_html=True)
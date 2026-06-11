"""
╔══════════════════════════════════════════════════════════════════╗
║   MACRO GEOPOLITICAL QUANT MODEL  v4.0-beta — Streamlit Edition  ║
║   Eduardo Moraes | Quant Data Scientist & Economics              ║
║   Swiss Private Bank Aesthetic · Navy · Gold · Precision         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import os
import csv
import logging
import warnings
from datetime import datetime, timedelta
import pytz
from scipy.interpolate import PchipInterpolator
from scipy import stats, optimize
from sklearn.linear_model import LassoCV
from statsmodels.tsa.vector_ar.var_model import VAR
import yfinance as yf
from arch import arch_model

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
#   SWISS BANK THEME — Page Config & CSS
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GeoQuant · Private Research",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

SWISS_CSS = """
<style>
/* ── Swiss Bank Typography & Base ─────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --navy:      #0B1628;
    --navy-mid:  #142038;
    --navy-light:#1E3050;
    --gold:      #C8A96E;
    --gold-dim:  #9E8050;
    --cream:     #F5F1EB;
    --warm-white:#FDFBF8;
    --gray-10:   #EAE6DF;
    --gray-30:   #C4BDAF;
    --gray-50:   #8C8377;
    --gray-70:   #4A4540;
    --text:      #1A1814;
    --success:   #3D6B4F;
    --danger:    #8B3030;
    --border:    rgba(200,169,110,0.18);
}

/* ── Global Reset ─── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--warm-white) !important;
    font-family: 'DM Sans', 'Helvetica Neue', sans-serif;
    font-weight: 300;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--cream) !important;
}

[data-testid="stSidebar"] .stSlider > div > div {
    background: var(--gold-dim) !important;
}

/* ── Header ─── */
.site-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}

.brand-mark {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
}

.brand-gem {
    font-size: 1rem;
    color: var(--gold);
    letter-spacing: 0.2em;
}

.brand-name {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.9rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    color: var(--navy);
    line-height: 1;
}

.brand-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    color: var(--gray-50);
    text-transform: uppercase;
    margin-top: 0.2rem;
}

.header-meta {
    text-align: right;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--gray-50);
    letter-spacing: 0.1em;
    line-height: 1.7;
}

.regime-pill {
    display: inline-block;
    background: var(--navy);
    color: var(--gold) !important;
    padding: 0.2rem 0.7rem;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
}

/* ── Section Labels ─── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--gold-dim);
    margin-bottom: 0.3rem;
}

.section-title {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.35rem;
    font-weight: 400;
    color: var(--navy);
    letter-spacing: 0.02em;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--gray-10);
}

/* ── Metric Cards ─── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1px;
    background: var(--gray-10);
    border: 1px solid var(--gray-10);
    margin-bottom: 2.5rem;
}

.metric-card {
    background: var(--warm-white);
    padding: 1.4rem 1.2rem;
}

.metric-card-dark {
    background: var(--navy);
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gray-50);
    margin-bottom: 0.5rem;
}

.metric-card-dark .metric-label {
    color: rgba(200,169,110,0.6);
}

.metric-value {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.8rem;
    font-weight: 300;
    color: var(--navy);
    line-height: 1;
    letter-spacing: -0.01em;
}

.metric-card-dark .metric-value {
    color: var(--gold);
}

.metric-change {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--gray-50);
    margin-top: 0.3rem;
}

.metric-up   { color: var(--success); }
.metric-down { color: var(--danger); }

/* ── Risk Badge ─── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(139,48,48,0.08);
    border: 1px solid rgba(139,48,48,0.25);
    color: var(--danger);
    padding: 0.25rem 0.7rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.risk-badge-dot {
    width: 5px; height: 5px;
    background: var(--danger);
    border-radius: 50%;
    animation: pulse-dot 1.6s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Info Blocks ─── */
.info-block {
    background: var(--cream);
    border-left: 2px solid var(--gold);
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.78rem;
    color: var(--gray-70);
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
}

.data-table th {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gray-50);
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--gray-10);
    text-align: left;
    background: var(--cream);
}

.data-table td {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid var(--gray-10);
    color: var(--text);
    font-weight: 300;
}

.data-table tr:hover td { background: var(--cream); }

/* ── Streamlit overrides ─── */
div[data-testid="stMetric"] {
    background: var(--warm-white);
    border: 1px solid var(--gray-10);
    padding: 1rem;
}

div[data-testid="stMetric"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--gray-50) !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 1.7rem !important;
    font-weight: 300 !important;
    color: var(--navy) !important;
}

.stButton button {
    background: var(--navy) !important;
    color: var(--gold) !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    padding: 0.5rem 1.5rem !important;
}

.stButton button:hover {
    background: var(--navy-light) !important;
    border: 1px solid var(--gold-dim) !important;
}

.stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--gray-10) !important;
    border-radius: 0 !important;
}

.stProgress > div > div {
    background: var(--gold) !important;
}

/* ── Divider ─── */
.swiss-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--gold) 0%, var(--gray-10) 60%, transparent 100%);
    margin: 2rem 0;
}

.footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--gray-10);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    color: var(--gray-30);
    text-transform: uppercase;
}
</style>
"""
st.markdown(SWISS_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#   CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════
CONFIG = {
    "eia_api_key":      "kVSuPa0tfnUmHzQ2VVSCPC6owKhPQQY2PbEc9hA1",
    "fred_api_key":     "876c9f95b965eb9d423ef2c7b68ae51b",
    "oilprice_api_key": "e241c0914287d05fcbbeb18669c23d86e9cdf36c63193a95d42854eb53ed354d",
    "tickers": {
        "oil": "CL=F", "brent": "BZ=F", "natgas": "NG=F",
        "gold": "GC=F", "silver": "SI=F", "copper": "HG=F",
        "wheat": "ZW=F", "corn": "ZC=F", "soy": "ZS=F",
        "dxy": "DX-Y.NYB", "eur": "EURUSD=X", "tnx": "^TNX",
    },
    "mc_steps":   10,
    "mc_sims":    10_000,
    "mc_seed":    42,
    "max_daily_vol":   0.08,
    "max_drift":       0.02,
    "tail_df_base":    3.0,
    "tail_df_min":     2.5,
    "tail_df_max":     6.0,
    "wti_min":    40,
    "wti_max":    200,
    "spread_min_pct":  -0.05,
    "spread_max_pct":   0.30,
    "guerra_start":    "2026-02-28",
    "jump_prob_up":    0.07,
    "jump_prob_down":  0.03,
    "jump_skew_up_normal":  0.045,
    "jump_skew_up_extreme": 0.135,
    "jump_prob_extreme":    0.15,
    "jump_skew_down":       0.025,
    "regime_noise_std":     0.05,
    "enable_pchip_fill":    True,
    "vol_prior_wti_annual":   0.35,
    "vol_prior_brent_annual": 0.35,
    "vol_prior_gold_annual":  0.18,
    "vol_shrink_n_full": 252,
    "geo_weights": {
        "oil_vol":0.22,"gold":0.09,"gold_real":0.09,
        "dxy":-0.10,"spread":0.09,"fert":0.22,
        "wheat":0.07,"copper":0.04,"natgas_vol":0.06,
    },
    "zscore_weights": {"oil_gold":0.40,"oil_natgas":0.35,"gold_real":0.25},
    "fert_black_swan_z_threshold": 1.5,
    "fert_evt_threshold_q":        0.90,
}

# Swiss Plotly theme
SWISS = dict(
    paper_bgcolor = "#FDFBF8",
    plot_bgcolor  = "#F5F1EB",
    font          = dict(family="DM Sans, Helvetica Neue, sans-serif", color="#1A1814", size=12),
    title_font    = dict(family="Cormorant Garamond, Georgia, serif", size=18, color="#0B1628"),
    xaxis         = dict(gridcolor="#DDD9D1", linecolor="#C4BDAF", tickfont=dict(size=10, family="DM Mono, monospace"), zeroline=False),
    yaxis         = dict(gridcolor="#DDD9D1", linecolor="#C4BDAF", tickfont=dict(size=10, family="DM Mono, monospace"), zeroline=False),
    legend        = dict(bgcolor="rgba(253,251,248,0.9)", bordercolor="#C4BDAF", borderwidth=1, font=dict(size=11)),
    margin        = dict(l=50, r=30, t=60, b=40),
    hoverlabel    = dict(bgcolor="#0B1628", font_color="#C8A96E", font_family="DM Mono, monospace"),
)

C = dict(
    navy    = "#0B1628",
    gold    = "#C8A96E",
    gold_dim= "#9E8050",
    blue    = "#3A5F8A",
    teal    = "#2D6B6B",
    sage    = "#5F6B47",
    rust    = "#7A3F30",
    cream   = "#F5F1EB",
    gray    = "#8C8377",
    fan90   = "#C4BDAF",
    fan50   = "#9E9488",
)

def swiss_fig(**kwargs):
    fig = go.Figure(**kwargs)
    fig.update_layout(**SWISS)
    return fig

def apply_swiss(fig):
    fig.update_layout(**SWISS)
    return fig

# ══════════════════════════════════════════════════════════
#   SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem; border-bottom:1px solid rgba(200,169,110,0.2); margin-bottom:1.5rem;'>
        <div style='font-family:"DM Mono",monospace; font-size:0.55rem; letter-spacing:0.25em; color:#C8A96E; text-transform:uppercase; margin-bottom:0.5rem;'>◆ GeoQuant</div>
        <div style='font-family:"Cormorant Garamond",Georgia,serif; font-size:1.3rem; font-weight:300; color:#F5F1EB; letter-spacing:0.06em;'>Research Terminal</div>
        <div style='font-family:"DM Mono",monospace; font-size:0.55rem; color:rgba(245,241,235,0.4); letter-spacing:0.15em; margin-top:0.3rem;'>v4.0-beta · Private</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'DM Mono\',monospace; font-size:0.58rem; letter-spacing:0.2em; color:#C8A96E; text-transform:uppercase; margin-bottom:0.8rem;">· Simulation Parameters</div>', unsafe_allow_html=True)

    CONFIG["mc_sims"]  = st.slider("Monte Carlo Simulations", 1_000, 50_000, 10_000, 1_000)
    CONFIG["mc_steps"] = st.slider("Horizon (Days)", 5, 30, 10, 1)
    CONFIG["jump_prob_up"]   = st.slider("Jump Probability ↑", 0.01, 0.20, 0.07, 0.01)
    CONFIG["jump_prob_down"] = st.slider("Jump Probability ↓", 0.01, 0.10, 0.03, 0.01)
    CONFIG["tail_df_base"]   = st.slider("Tail Degrees of Freedom", 2.5, 8.0, 3.0, 0.5)

    st.markdown('<div style="height:1px; background:rgba(200,169,110,0.15); margin:1.2rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'DM Mono\',monospace; font-size:0.58rem; letter-spacing:0.2em; color:#C8A96E; text-transform:uppercase; margin-bottom:0.8rem;">· Volatility Priors</div>', unsafe_allow_html=True)

    CONFIG["vol_prior_wti_annual"]   = st.slider("WTI Vol Prior (annual)", 0.20, 0.60, 0.35, 0.01)
    CONFIG["vol_prior_brent_annual"] = st.slider("Brent Vol Prior (annual)", 0.20, 0.60, 0.35, 0.01)

    st.markdown('<div style="height:1px; background:rgba(200,169,110,0.15); margin:1.2rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'DM Mono\',monospace; font-size:0.58rem; letter-spacing:0.2em; color:#C8A96E; text-transform:uppercase; margin-bottom:0.8rem;">· War Regime</div>', unsafe_allow_html=True)
    CONFIG["guerra_start"] = st.date_input("War Scenario Start", value=datetime(2026, 2, 28)).strftime("%Y-%m-%d")

    st.markdown('<div style="height:1px; background:rgba(200,169,110,0.15); margin:1.2rem 0;"></div>', unsafe_allow_html=True)

    run_btn = st.button("▶  Run Full Analysis", use_container_width=True)

    st.markdown("""
    <div style='margin-top:2rem; font-family:"DM Mono",monospace; font-size:0.52rem; color:rgba(245,241,235,0.3); letter-spacing:0.1em; line-height:2;'>
    FOR PROFESSIONAL USE ONLY<br>
    NOT INVESTMENT ADVICE<br>
    CONFIDENTIAL & PROPRIETARY
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#   HEADER
# ══════════════════════════════════════════════════════════
now_sp = datetime.now(pytz.timezone("America/Sao_Paulo"))
st.markdown(f"""
<div class="site-header">
    <div>
        <div class="brand-mark">
            <span class="brand-gem">◆◆◆</span>
            <div>
                <div class="brand-name">GeoQuant · Macro Research</div>
                <div class="brand-sub">Geopolitical Intelligence · Commodity Markets · Private</div>
            </div>
        </div>
    </div>
    <div class="header-meta">
        <div class="regime-pill">⚑ WAR REGIME ACTIVE</div><br>
        <div>{now_sp.strftime("%d %B %Y · %H:%M")} (São Paulo)</div>
        <div>Model v4.0-beta · EVT+DCC+GARCH-X</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#   QUANT FUNCTIONS (unchanged logic, Swiss-adapted outputs)
# ══════════════════════════════════════════════════════════
def rolling_zscore(series, window=60):
    mu  = series.rolling(window).mean()
    sig = series.rolling(window).std()
    return (series - mu) / sig.replace(0, np.nan)

def fill_intraday_gaps(series, max_gap_hours=2):
    series = series.copy()
    if not isinstance(series.index, pd.DatetimeIndex): return series.ffill()
    valid = series.notna()
    if valid.sum() < 2: return series.ffill()
    x = series.index[valid].astype(np.int64); y = series[valid].values
    try:
        filled = pd.Series(PchipInterpolator(x, y)(series.index.astype(np.int64)), index=series.index)
        filled[valid] = series[valid]; return filled
    except: return series.ffill()

def sample_geopolitical_jump(direction='up'):
    if direction == 'up':
        if np.random.rand() < CONFIG["jump_prob_extreme"]:
            return np.random.exponential(CONFIG["jump_skew_up_extreme"])
        return np.random.exponential(CONFIG["jump_skew_up_normal"])
    return np.random.exponential(CONFIG["jump_skew_down"])

def generate_default_fertilizer_csv(csv_path="fertilizer_backup.csv"):
    if os.path.exists(csv_path): return
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date","urea_price","dap_price"])
        w.writerows([["2026-01-15",540,710],["2026-02-15",560,740],["2026-03-15",590,780],
                     ["2026-04-15",616,857],["2026-05-01",720,900],["2026-05-06",810,920],
                     ["2026-05-12",857,920]])

def load_fertilizer_backup(csv_path="fertilizer_backup.csv"):
    generate_default_fertilizer_csv(csv_path)
    try:
        df   = pd.read_csv(csv_path, parse_dates=["date"], index_col="date").sort_index()
        last = df.iloc[-1]
        return {"urea_price":float(last["urea_price"]), "urea_period":str(last.name.date()),
                "dap_price":float(last["dap_price"]),   "dap_period":str(last.name.date()),
                "source":"local CSV backup"}
    except: return None

def get_live_urea_price():
    backup = load_fertilizer_backup()
    if backup: return backup
    return {"urea_price":857,"urea_period":"2026-05-12","dap_price":920,
            "dap_period":"2026-05-12","source":"hardcoded benchmark"}

def compute_fertilizer_black_swan(usda_data):
    urea_hist = []
    if os.path.exists("fertilizer_backup.csv"):
        try:
            df = pd.read_csv("fertilizer_backup.csv", parse_dates=["date"], index_col="date")
            urea_hist = df["urea_price"].dropna().values
        except: pass
    current_urea = usda_data.get("urea_price")
    if current_urea is None or len(urea_hist)<10: return 1.0, 0.0
    rets = np.diff(np.log(urea_hist))
    q = CONFIG["fert_evt_threshold_q"]
    threshold = np.quantile(rets, q)
    exceed    = rets[rets>threshold] - threshold
    if len(exceed)<5:
        mu  = np.mean(urea_hist); sig = np.std(urea_hist)
        if sig==0: return 1.0, 0.0
        z    = (current_urea - mu)/sig
        mult = 1.0 + max(0, z - CONFIG["fert_black_swan_z_threshold"])*0.8
        return min(mult, 3.0), z
    try:
        shape, loc, scale = stats.genpareto.fit(exceed)
        cur_ret = np.log(current_urea/urea_hist[-1])
        if cur_ret <= threshold: return 1.0, 0.0
        p_exceed = 1 - stats.genpareto.cdf(cur_ret-threshold, shape, loc=loc, scale=scale)
        return 1.0 + min(p_exceed*5, 2.0), cur_ret
    except: return 1.0, 0.0

def build_gold_signals(prices):
    silver   = prices["silver"].replace(0, np.nan)
    if silver.median()>500: silver = silver/100
    gold_real    = prices["gold"] / (1 + prices["tnx"].replace(0, np.nan)/100*5.0)
    silver_gold  = silver / prices["gold"].replace(0, np.nan)
    return {"gold_real": gold_real, "silver_gold": silver_gold,
            "gold_real_ret_roll":  np.log(gold_real/gold_real.shift(1)).rolling(20).mean(),
            "silver_gold_roll":    np.log(silver_gold/silver_gold.shift(1)).rolling(20).mean()}

def build_silver_demand_proxy(prices):
    if "copper" not in prices.columns or "brent" not in prices.columns:
        return pd.Series(index=prices.index, data=0.0)
    cr = prices["copper"].pct_change().dropna()
    br = prices["brent"].pct_change().dropna()
    common = cr.index.intersection(br.index)
    return (0.6*cr[common]+0.4*br[common]).rolling(20).mean().reindex(prices.index, method="ffill").fillna(0.0)

def build_geofactor(returns, prices, gold_signals, fert_index, weights, silver_demand=None):
    spread = (prices["brent"]-prices["oil"])/prices["brent"].replace(0,np.nan)
    geo = (weights.get("oil_vol",0) * returns["oil"].rolling(20).std() +
           weights.get("gold",0)    * returns["gold"].rolling(20).mean() +
           weights.get("gold_real",0)* gold_signals["gold_real_ret_roll"] +
           weights.get("dxy",0)     * returns["dxy"].rolling(20).mean() +
           weights.get("spread",0)  * spread.rolling(20).mean() +
           weights.get("wheat",0)   * returns["wheat"].rolling(20).mean() +
           weights.get("copper",0)  * returns["copper"].rolling(20).mean() +
           weights.get("natgas_vol",0)*returns["natgas"].rolling(20).std())
    if silver_demand is not None:
        common = geo.dropna().index.intersection(silver_demand.dropna().index)
        if len(common)>0: geo.loc[common] += weights.get("silver_demand",0)*silver_demand.loc[common]
    common = geo.dropna().index.intersection(fert_index.dropna().index)
    geo.loc[common] += weights.get("fert",0)*fert_index.loc[common]
    return geo.dropna().clip(geo.dropna().quantile(0.05), geo.dropna().quantile(0.95))

def calibrate_geo_weights(returns, prices, gold_signals, fert_index, silver_demand=None, window=60):
    spread = (prices["brent"]-prices["oil"])/prices["brent"].replace(0,np.nan)
    X = pd.DataFrame({
        "oil_vol": returns["oil"].rolling(20).std(),
        "gold":    returns["gold"].rolling(20).mean(),
        "gold_real":gold_signals["gold_real_ret_roll"],
        "dxy":     returns["dxy"].rolling(20).mean(),
        "spread":  spread.rolling(20).mean(),
        "wheat":   returns["wheat"].rolling(20).mean(),
        "copper":  returns["copper"].rolling(20).mean(),
        "natgas_vol":returns["natgas"].rolling(20).std(),
        "fert":    fert_index,
    })
    if silver_demand is not None: X["silver_demand"] = silver_demand
    y = returns["oil"].shift(-1)
    common = y.dropna().index.intersection(X.dropna().index)
    X, y = X.loc[common].dropna(), y.loc[common]
    if len(X)<window: return CONFIG["geo_weights"]
    X_cal, y_cal = X.iloc[-window:], y.iloc[-window:]
    X_mean, X_std = X_cal.mean(), X_cal.std().replace(0,1)
    Xs = (X_cal-X_mean)/X_std
    model = LassoCV(cv=5, random_state=42, alphas=np.logspace(-4,0,20), max_iter=2000).fit(Xs, y_cal)
    coef  = model.coef_/X_std.values
    new_w = {col:coef[i] for i,col in enumerate(X.columns)}
    total = sum(abs(v) for v in new_w.values())
    if total>0: new_w = {k:v/total for k,v in new_w.items()}
    else: return CONFIG["geo_weights"]
    return new_w

def build_composite_zscore(prices, gold_signals, window=60):
    w  = min(window, max(20, len(prices)//2))
    z1 = rolling_zscore(prices["oil"]/prices["gold"].replace(0,np.nan), w)
    z2 = rolling_zscore(prices["oil"]/prices["natgas"].replace(0,np.nan), w)
    z3 = rolling_zscore(gold_signals["gold_real"], w)
    return (CONFIG["zscore_weights"]["oil_gold"]*z1 +
            CONFIG["zscore_weights"]["oil_natgas"]*z2 +
            CONFIG["zscore_weights"]["gold_real"]*z3).dropna()

def build_fertilizer_stress_index(returns, usda_data, black_swan_mult=1.0):
    fert = (0.5*returns["natgas"].rolling(20).std() +
            0.25*returns["wheat"].rolling(20).mean() +
            0.25*returns["corn"].rolling(20).mean())
    if usda_data["urea_price"]: fert += np.clip((usda_data["urea_price"]-380)/380,-1,2)*0.15
    if usda_data["dap_price"]:  fert += np.clip((usda_data["dap_price"]-610)/610,-1,2)*0.10
    fert *= black_swan_mult
    return fert.clip(fert.quantile(0.02), fert.quantile(0.98)).dropna()

def fit_garch_x(ret, exog):
    ret_c  = ret.loc[ret.index.intersection(exog.index)]*100
    exog_c = exog.loc[ret_c.index]
    try: res = arch_model(ret_c, x=exog_c, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt").fit(disp="off")
    except: res = arch_model(ret_c, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt").fit(disp="off")
    return res.conditional_volatility/100

def bayesian_vol_shrinkage(vol_garch, vol_prior_daily, n_obs, n_full=252, label="", geofactor=None):
    w = np.clip(np.sqrt(n_obs/n_full), 0.10, 0.95)
    v_last    = float(vol_garch.iloc[-1])
    prior_adj = vol_prior_daily
    if geofactor is not None and not geofactor.empty:
        prior_adj *= (1.0+0.4*np.tanh(float(geofactor.iloc[-1])))
    lo, hi = prior_adj*0.50, prior_adj*1.50
    if lo <= v_last <= hi:
        vs = vol_garch.copy(); w_eff = 1.0
    else:
        vs = w*vol_garch + (1-w)*prior_adj; w_eff = w
    vga = v_last*np.sqrt(252)*100; vsa = float(vs.iloc[-1])*np.sqrt(252)*100
    return vs, {"label":label,"weight_data":w_eff,"vol_garch_aa":vga,"vol_final_aa":vsa}

def estimate_dcc_params(ret_wti, ret_brent, vol_wti, vol_brent):
    common = ret_wti.index.intersection(ret_brent.index).intersection(vol_wti.index).intersection(vol_brent.index)
    ew = (ret_wti[common]/vol_wti[common]).dropna()
    eb = (ret_brent[common]/vol_brent[common]).dropna()
    c2 = ew.index.intersection(eb.index)
    e  = np.column_stack([ew[c2], eb[c2]])
    def nll(params):
        a,b = params
        if a<=0 or b<=0 or a+b>=1: return 1e10
        Q_bar = np.cov(e, rowvar=False); Q = Q_bar.copy(); ll=0
        for t in range(1,len(e)):
            Qt = (1-a-b)*Q_bar + a*np.outer(e[t-1],e[t-1]) + b*Q
            d  = np.sqrt(np.diag(Qt)); d[d==0]=1e-8
            R  = Qt/np.outer(d,d); R = np.clip(R,-0.9999,0.9999)
            try:
                L    = np.linalg.cholesky(R)
                invL = np.linalg.inv(L)
                z    = invL @ e[t]
                ll  += -0.5*np.sum(z**2) - np.sum(np.log(np.diag(L)))
                Q    = Qt
            except: return 1e10
        return -ll
    res = optimize.minimize(nll, [0.05,0.93], bounds=[(1e-4,0.3),(0.7,0.9999)], method="L-BFGS-B")
    a,b = res.x
    if a+b>=1: a,b=0.05,0.93
    return a,b

def _add_asymmetric_tail_jumps(shocks, vol):
    n=len(shocks); u=np.random.rand(n)
    mu=u<0.025; md=(u>=0.025)&(u<0.05)
    ju=np.random.exponential(0.03,n)*vol; jd=np.random.exponential(0.02,n)*vol
    return shocks + np.where(mu,ju,0) - np.where(md,jd,0)

def _sample_jumps_vec(n, pu, pd_):
    u  = np.random.rand(n); mu=u<pu; md=(u>=pu)&(u<pu+pd_)
    me = np.random.rand(n)<CONFIG["jump_prob_extreme"]
    ju = np.where(me, np.random.exponential(CONFIG["jump_skew_up_extreme"],n),
                      np.random.exponential(CONFIG["jump_skew_up_normal"],n))
    jd = np.random.exponential(CONFIG["jump_skew_down"],n)
    return np.where(mu,ju,np.where(md,-jd,0)), np.where(mu,ju*0.95,np.where(md,-jd*0.90,0))

def run_monte_carlo(wti_last, brent_last, base_vol, base_vol_brent, forecast,
                    oil_col, brent_col, regime_base, returns_wti, returns_brent,
                    vol_oil_series, vol_brent_series, jump_prob_up_eff, tail_df_dynamic,
                    black_swan_mult=1.0, dcc_a=0.05, dcc_b=0.93, progress_bar=None):
    sims  = CONFIG["mc_sims"]
    steps = CONFIG["mc_steps"]
    np.random.seed(CONFIG["mc_seed"])
    common = (returns_wti.index.intersection(returns_brent.index)
              .intersection(vol_oil_series.index).intersection(vol_brent_series.index))
    ew = (returns_wti[common]/vol_oil_series[common].replace(0,np.nan)).dropna()
    eb = (returns_brent[common]/vol_brent_series[common].replace(0,np.nan)).dropna()
    c2 = ew.index.intersection(eb.index)
    e  = np.column_stack([np.clip(ew[c2],-3,3), np.clip(eb[c2],-3,3)])
    Q_bar    = np.cov(e, rowvar=False); np.fill_diagonal(Q_bar,1.0)
    eps_prev = e[-1] + np.random.normal(0,0.05,(sims,2))
    Q_t      = np.tile(Q_bar,(sims,1,1)).copy()
    pu  = min(jump_prob_up_eff*1.5,0.20) if black_swan_mult>1.2 else jump_prob_up_eff
    pd_ = CONFIG["jump_prob_down"]*(1.3 if black_swan_mult>1.2 else 1.0)
    paths_wti   = np.zeros((sims,steps+1)); paths_brent = np.zeros((sims,steps+1))
    paths_wti[:,0] = wti_last; paths_brent[:,0] = brent_last
    regime_noise = np.random.normal(0,CONFIG["regime_noise_std"],(sims,steps))
    regime_all   = np.clip(regime_base+regime_noise,-1,1); ra = 1+0.5*regime_all
    for t in range(steps):
        if progress_bar: progress_bar.progress((t+1)/steps)
        outer = np.einsum("si,sj->sij",eps_prev,eps_prev)
        Q_t   = (1-dcc_a-dcc_b)*Q_bar[np.newaxis] + dcc_a*outer + dcc_b*Q_t
        diag  = np.clip(np.sqrt(np.diagonal(Q_t,axis1=1,axis2=2)),1e-8,None)
        R_t   = Q_t / np.einsum("si,sj->sij",diag,diag)
        R_t   = np.clip(R_t,-0.9999,0.9999); R_t[:,0,0]=R_t[:,1,1]=1.0
        rho   = R_t[:,0,1]; sc = np.sqrt(np.clip(1-rho**2,1e-8,None))
        z     = np.random.standard_t(tail_df_dynamic,(sims,2))
        zc1   = z[:,0]; zc2 = rho*z[:,0]+sc*z[:,1]
        vw = np.clip(base_vol*ra[:,t],0,CONFIG["max_daily_vol"])
        vb = np.clip(base_vol_brent*ra[:,t],0,CONFIG["max_daily_vol"])
        sw = np.clip(zc1*vw,-4*vw,4*vw); sb = np.clip(zc2*vb,-4*vb,4*vb)
        sw = _add_asymmetric_tail_jumps(sw,vw); sb = _add_asymmetric_tail_jumps(sb,vb)
        jw,jb = _sample_jumps_vec(sims,pu,pd_); sw+=jw; sb+=jb
        dw = np.clip(forecast[t,oil_col]*ra[:,t],-CONFIG["max_drift"],CONFIG["max_drift"])
        db = np.clip(forecast[t,brent_col]*ra[:,t],-CONFIG["max_drift"],CONFIG["max_drift"])
        pw = paths_wti[:,t]*np.exp(dw+sw); pb = paths_brent[:,t]*np.exp(db+sb)
        spread = np.where(pb>0,(pb-pw)/pb,0)
        pw = np.where(spread<CONFIG["spread_min_pct"], pb*(1+abs(CONFIG["spread_min_pct"])),pw)
        pw = np.where(spread>CONFIG["spread_max_pct"], pb*(1-CONFIG["spread_max_pct"]),pw)
        pw = np.clip(pw,wti_last*0.4,wti_last*2.5); pb = np.clip(pb,brent_last*0.4,brent_last*2.5)
        paths_wti[:,t+1]=pw; paths_brent[:,t+1]=pb
        eps_prev[:,0]=np.where(vw>0,sw/vw,0); eps_prev[:,1]=np.where(vb>0,sb/vb,0)
        eps_prev = np.clip(eps_prev,-5,5)
    fan       = {p:np.percentile(paths_wti,p,axis=0)   for p in [5,25,50,75,95]}
    fan_brent = {p:np.percentile(paths_brent,p,axis=0) for p in [5,25,50,75,95]}
    term      = paths_wti[:,-1]
    var95     = np.percentile(paths_wti[:,1]-wti_last,5)
    cvar_mask = (paths_wti[:,1]-wti_last)<=var95
    metrics = {
        "p95_chg": (fan[95][-1]/wti_last-1)*100,
        "p5_chg":  (fan[5][-1]/wti_last-1)*100,
        "vol_wti_aa":   base_vol*np.sqrt(252)*100,
        "vol_brent_aa": base_vol_brent*np.sqrt(252)*100,
        "var_95_1d":    var95,
        "cvar_95_1d":   float(np.mean((paths_wti[:,1]-wti_last)[cvar_mask])),
        "prob_up_10d":  np.mean(term>wti_last)*100,
        "uncertainty_pct": (fan[95][-1]-fan[5][-1])/fan[50][-1]*100,
        "prob_wti_below_40":  np.mean(term<40)*100,
        "prob_wti_above_150": np.mean(term>150)*100,
    }
    return {"paths_wti":paths_wti,"paths_brent":paths_brent,
            "fan":fan,"fan_brent":fan_brent,"metrics":metrics}

# ══════════════════════════════════════════════════════════
#   DATA FETCH (cached)
# ══════════════════════════════════════════════════════════
@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(start, tickers):
    prices = yf.download(list(tickers.values()), start=start, progress=False)["Close"]
    prices.columns = list(tickers.keys())
    for col in prices.columns: prices[col] = fill_intraday_gaps(prices[col])
    return prices.ffill().dropna()

@st.cache_data(ttl=60, show_spinner=False)
def get_live_prices():
    wti   = float(yf.Ticker("CL=F").fast_info["last_price"])
    brent = float(yf.Ticker("BZ=F").fast_info["last_price"])
    return wti, brent

# ══════════════════════════════════════════════════════════
#   MAIN — Auto-run or button
# ══════════════════════════════════════════════════════════
if run_btn or "mc_results" not in st.session_state:
    # ── Loading Screen ──────────────────────────────────
    status_box = st.empty()
    with status_box.container():
        st.markdown("""
        <div style='text-align:center; padding:3rem; background:#F5F1EB; border:1px solid #C4BDAF;'>
            <div style='font-family:"DM Mono",monospace; font-size:0.6rem; letter-spacing:0.2em; color:#9E8050; text-transform:uppercase; margin-bottom:1rem;'>Initialising Research Terminal</div>
            <div style='font-family:"Cormorant Garamond",Georgia,serif; font-size:1.4rem; color:#0B1628;'>Loading market data & calibrating model…</div>
        </div>
        """, unsafe_allow_html=True)

    prog = st.progress(0)

    # 1 · Market data
    prog.progress(10)
    prices = load_market_data(CONFIG["guerra_start"], CONFIG["tickers"])
    wti_last, brent_last = get_live_prices()
    prices.loc[prices.index[-1], "oil"]   = wti_last
    prices.loc[prices.index[-1], "brent"] = brent_last
    returns = np.log(prices/prices.shift(1)).dropna()

    # 2 · Fertilizer
    prog.progress(20)
    usda_data = get_live_urea_price()
    black_swan_mult, _ = compute_fertilizer_black_swan(usda_data)

    # 3 · Signals
    prog.progress(30)
    gold_signals   = build_gold_signals(prices)
    silver_demand  = build_silver_demand_proxy(prices)
    if "silver_demand" not in CONFIG["geo_weights"]:
        CONFIG["geo_weights"]["silver_demand"] = 0.02
        total = sum(abs(v) for v in CONFIG["geo_weights"].values())
        for k in CONFIG["geo_weights"]: CONFIG["geo_weights"][k] /= total
    fert_index  = build_fertilizer_stress_index(returns, usda_data, black_swan_mult)
    dyn_weights = calibrate_geo_weights(returns, prices, gold_signals, fert_index, silver_demand, window=60)
    if dyn_weights: CONFIG["geo_weights"] = dyn_weights
    geofactor   = build_geofactor(returns, prices, gold_signals, fert_index, CONFIG["geo_weights"], silver_demand)
    z_composite = build_composite_zscore(prices, gold_signals, window=60)

    # 4 · GARCH
    prog.progress(50)
    vol_oil   = fit_garch_x(returns["oil"],   geofactor)
    vol_brent = fit_garch_x(returns["brent"], geofactor)
    vol_gold  = fit_garch_x(returns["gold"],  geofactor)
    n_calib   = len(returns)
    prior_wti   = CONFIG["vol_prior_wti_annual"]/np.sqrt(252)
    prior_brent = CONFIG["vol_prior_brent_annual"]/np.sqrt(252)
    prior_gold  = CONFIG["vol_prior_gold_annual"]/np.sqrt(252)
    vol_oil,   diag_wti   = bayesian_vol_shrinkage(vol_oil,   prior_wti,   n_calib, label="WTI",  geofactor=geofactor)
    vol_brent, diag_brent = bayesian_vol_shrinkage(vol_brent, prior_brent, n_calib, label="BRT",  geofactor=geofactor)
    vol_gold,  _          = bayesian_vol_shrinkage(vol_gold,  prior_gold,  n_calib, label="GLD")
    base_vol       = float(vol_oil.iloc[-1])
    base_vol_brent = float(vol_brent.iloc[-1])

    # 5 · DCC + VAR
    prog.progress(65)
    dcc_a, dcc_b = estimate_dcc_params(returns["oil"], returns["brent"], vol_oil, vol_brent)
    ret_var   = returns.loc[geofactor.index.intersection(returns.index)]
    lags_var  = min(5, max(1, len(ret_var)//10))
    var_model = VAR(ret_var).fit(lags_var)
    forecast  = var_model.forecast(ret_var.values[-var_model.k_ar:], steps=CONFIG["mc_steps"])
    col_names = list(ret_var.columns)
    oil_col   = col_names.index("oil"); brent_col = col_names.index("brent")
    vol_ratio       = base_vol_brent/(prior_brent*1.5)
    tail_df_dynamic = max(CONFIG["tail_df_min"], min(CONFIG["tail_df_max"], CONFIG["tail_df_base"]/np.sqrt(max(vol_ratio,0.5))))
    regime_base     = float(np.tanh(geofactor.iloc[-1]/2)) if not geofactor.empty else 0.0

    # 6 · War signal
    ws = (returns["wheat"].tail(20).mean() + returns["natgas"].tail(20).mean())/2
    war_trigger     = ws>0.005
    jump_prob_up_eff = min(CONFIG["jump_prob_up"]*1.5,0.15) if war_trigger else CONFIG["jump_prob_up"]

    # 7 · Monte Carlo
    prog.progress(75)
    mc_prog = st.empty()
    with mc_prog.container():
        st.markdown('<div style="font-family:\'DM Mono\',monospace; font-size:0.6rem; letter-spacing:0.15em; color:#9E8050;">MONTE CARLO SIMULATION IN PROGRESS…</div>', unsafe_allow_html=True)
        bar = st.progress(0)
    mc = run_monte_carlo(
        wti_last, brent_last, base_vol, base_vol_brent, forecast,
        oil_col, brent_col, regime_base, returns["oil"], returns["brent"],
        vol_oil, vol_brent, jump_prob_up_eff, tail_df_dynamic,
        black_swan_mult=black_swan_mult, dcc_a=dcc_a, dcc_b=dcc_b, progress_bar=bar
    )
    mc_prog.empty()
    prog.progress(100)

    # Correlation
    try:
        ret_joint = pd.concat([returns["oil"],returns["brent"]],axis=1).dropna()
        ewma_cov  = ret_joint.ewm(alpha=0.06).cov(pairwise=True)
        lc        = ewma_cov.loc[ewma_cov.index.get_level_values(0)[-1]]
        corr_wti_brent = np.clip(lc.loc["oil","brent"]/np.sqrt(lc.loc["oil","oil"]*lc.loc["brent","brent"]),-1,1)
    except: corr_wti_brent = 0.95

    # Save to session
    st.session_state["mc_results"]     = mc
    st.session_state["geofactor"]      = geofactor
    st.session_state["z_composite"]    = z_composite
    st.session_state["vol_oil"]        = vol_oil
    st.session_state["vol_brent"]      = vol_brent
    st.session_state["vol_gold"]       = vol_gold
    st.session_state["fert_index"]     = fert_index
    st.session_state["gold_signals"]   = gold_signals
    st.session_state["prices"]         = prices
    st.session_state["returns"]        = returns
    st.session_state["wti_last"]       = wti_last
    st.session_state["brent_last"]     = brent_last
    st.session_state["usda_data"]      = usda_data
    st.session_state["black_swan_mult"]= black_swan_mult
    st.session_state["diag_wti"]       = diag_wti
    st.session_state["diag_brent"]     = diag_brent
    st.session_state["tail_df"]        = tail_df_dynamic
    st.session_state["corr"]           = corr_wti_brent
    st.session_state["regime_base"]    = regime_base
    st.session_state["war_trigger"]    = war_trigger
    st.session_state["war_signal"]     = float(ws)
    st.session_state["jump_prob_up_eff"]= jump_prob_up_eff
    st.session_state["dcc_a"]          = dcc_a
    st.session_state["dcc_b"]          = dcc_b

    status_box.empty(); prog.empty()

# ══════════════════════════════════════════════════════════
#   RENDER FROM SESSION STATE
# ══════════════════════════════════════════════════════════
mc         = st.session_state["mc_results"]
geofactor  = st.session_state["geofactor"]
z_comp     = st.session_state["z_composite"]
vol_oil    = st.session_state["vol_oil"]
vol_brent  = st.session_state["vol_brent"]
vol_gold   = st.session_state["vol_gold"]
fert_index = st.session_state["fert_index"]
gold_sigs  = st.session_state["gold_signals"]
prices     = st.session_state["prices"]
returns    = st.session_state["returns"]
wti_last   = st.session_state["wti_last"]
brent_last = st.session_state["brent_last"]
usda       = st.session_state["usda_data"]
bs_mult    = st.session_state["black_swan_mult"]
d_wti      = st.session_state["diag_wti"]
d_brt      = st.session_state["diag_brent"]
tail_df    = st.session_state["tail_df"]
corr       = st.session_state["corr"]
rbase      = st.session_state["regime_base"]
war_trig   = st.session_state["war_trigger"]
war_sig    = st.session_state["war_signal"]
jump_up    = st.session_state["jump_prob_up_eff"]
fan        = mc["fan"]; fan_brent = mc["fan_brent"]; mets = mc["metrics"]

# ── § 1 · KEY METRICS ─────────────────────────────────────
st.markdown('<div class="section-label">01 · Market Snapshot</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Commodity & Risk Metrics</div>', unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
spread = brent_last - wti_last
with c1: st.metric("WTI Crude", f"${wti_last:.2f}", f"{mets['p50_chg']:.1f}% P50 10d" if 'p50_chg' in mets else "")
with c2: st.metric("Brent Crude", f"${brent_last:.2f}", f"${spread:.2f} spread")
with c3: st.metric("WTI Vol p.a.", f"{mets['vol_wti_aa']:.1f}%", f"Shrink {d_wti['vol_garch_aa']:.0f}%→{d_wti['vol_final_aa']:.0f}%")
with c4: st.metric("GeoFactor", f"{float(geofactor.iloc[-1]):.4f}", f"Regime {rbase:+.2f}")
with c5: st.metric("VaR 95% 1d", f"${mets['var_95_1d']:+.2f}", f"CVaR ${mets['cvar_95_1d']:+.2f}")
with c6: st.metric("Z-Composite", f"{float(z_comp.iloc[-1]):+.3f}", "War ACTIVE" if war_trig else "Subdued")

st.markdown('<div class="swiss-divider"></div>', unsafe_allow_html=True)

# ── § 2 · GEOPOLITICAL SIGNALS ────────────────────────────
st.markdown('<div class="section-label">02 · Geopolitical Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Risk Signals — Z-Score & GeoFactor v4.0</div>', unsafe_allow_html=True)

fig_geo = make_subplots(specs=[[{"secondary_y":True}]])
fig_geo.add_trace(go.Scatter(
    x=z_comp.index, y=z_comp.values, name="Z-Score Composite",
    line=dict(color=C["blue"], width=2.5),
    fill="tozeroy", fillcolor=f"rgba(58,95,138,0.07)",
), secondary_y=False)
fig_geo.add_trace(go.Scatter(
    x=geofactor.index, y=geofactor.values, name="GeoFactor v4.0",
    line=dict(color=C["navy"], width=3, dash="solid"),
), secondary_y=True)
fig_geo.add_hline(y=1.5,  line_dash="dot", line_color="#9E8050", line_width=1.5, secondary_y=False)
fig_geo.add_hline(y=-1.5, line_dash="dot", line_color="#9E8050", line_width=1.5, secondary_y=False)
fig_geo.add_hline(y=0,    line_dash="solid", line_color="#C4BDAF", line_width=1, secondary_y=False)
fig_geo.update_layout(**SWISS, title="", height=340,
    yaxis=dict(title="Z-Score", **SWISS["yaxis"]),
    yaxis2=dict(title="GeoFactor", **SWISS["yaxis"]))
st.plotly_chart(fig_geo, use_container_width=True)

st.markdown('<div class="swiss-divider"></div>', unsafe_allow_html=True)

# ── § 3 · VOLATILITY ──────────────────────────────────────
st.markdown('<div class="section-label">03 · Volatility Surface</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">GARCH-X Conditional Volatility — Adaptive Bayesian Shrinkage</div>', unsafe_allow_html=True)

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=vol_oil.index,   y=vol_oil*np.sqrt(252)*100,   name="WTI",   line=dict(color=C["navy"],  width=2.5)))
fig_vol.add_trace(go.Scatter(x=vol_brent.index, y=vol_brent*np.sqrt(252)*100, name="Brent", line=dict(color=C["blue"],  width=2.5, dash="dash")))
fig_vol.add_trace(go.Scatter(x=vol_gold.index,  y=vol_gold*np.sqrt(252)*100,  name="Gold",  line=dict(color=C["gold"],  width=2.5, dash="dot")))
fig_vol.add_hrect(y0=25, y1=45, fillcolor=f"rgba(61,107,79,0.06)", line_width=0, annotation_text="Normal range", annotation_position="top left")
fig_vol.update_layout(**SWISS, title="", height=320,
    yaxis=dict(title="Annualised Volatility (%)", ticksuffix="%", **SWISS["yaxis"]))
st.plotly_chart(fig_vol, use_container_width=True)

st.markdown('<div class="swiss-divider"></div>', unsafe_allow_html=True)

# ── § 4 · STRESS INDICES ──────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-label">04a · Fertilizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1.1rem;">Stress Index + NatGas Vol</div>', unsafe_allow_html=True)
    ng_vol_s = returns["natgas"].rolling(20).std()*np.sqrt(252)*100
    fig_f = make_subplots(specs=[[{"secondary_y":True}]])
    fig_f.add_trace(go.Scatter(
        x=fert_index.index, y=fert_index.values, name="Fert Stress",
        fill="tozeroy", fillcolor=f"rgba(95,107,71,0.12)",
        line=dict(color=C["sage"], width=2.5)
    ), secondary_y=False)
    fig_f.add_trace(go.Scatter(
        x=ng_vol_s.index, y=ng_vol_s.values, name="NatGas Vol",
        line=dict(color=C["teal"], width=2, dash="dash")
    ), secondary_y=True)
    fig_f.update_layout(**SWISS, height=300,
        yaxis=dict(title="Fert Index",    **SWISS["yaxis"]),
        yaxis2=dict(title="NatGas Vol %", **SWISS["yaxis"]))
    st.plotly_chart(fig_f, use_container_width=True)
    urea_str = f"${usda['urea_price']:.0f}/t" if usda["urea_price"] else "N/A"
    dap_str  = f"${usda['dap_price']:.0f}/t"  if usda["dap_price"]  else "N/A"
    bs_warn  = f" · ⚠ Black Swan ×{bs_mult:.2f}" if bs_mult>1.2 else ""
    st.markdown(f'<div class="info-block">Urea {urea_str} · DAP {dap_str}{bs_warn}<br><span style="font-family:\'DM Mono\',monospace; font-size:0.6rem; color:#8C8377;">{usda["source"]}</span></div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="section-label">04b · Gold</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1.1rem;">Real Yield Signal + Silver/Gold Ratio</div>', unsafe_allow_html=True)
    gr_base = float(gold_sigs["gold_real"].dropna().iloc[0])
    sg_base = float(gold_sigs["silver_gold"].dropna().iloc[0])
    gr_norm = gold_sigs["gold_real"] / gr_base
    sg_norm = gold_sigs["silver_gold"] / sg_base
    fig_g = make_subplots(specs=[[{"secondary_y":True}]])
    fig_g.add_trace(go.Scatter(
        x=gr_norm.dropna().index, y=gr_norm.dropna().values, name="Gold/Real Yield",
        line=dict(color=C["gold"], width=2.5)
    ), secondary_y=False)
    fig_g.add_trace(go.Scatter(
        x=sg_norm.dropna().index, y=sg_norm.dropna().values, name="Silver/Gold",
        line=dict(color=C["gray"], width=2, dash="dash")
    ), secondary_y=True)
    fig_g.add_hline(y=1.0, line_dash="dot", line_color="#C4BDAF", line_width=1.5, secondary_y=False)
    fig_g.update_layout(**SWISS, height=300,
        yaxis=dict(title="Gold/Real-Yield (norm)",   **SWISS["yaxis"]),
        yaxis2=dict(title="Silver/Gold (norm)",       **SWISS["yaxis"]))
    st.plotly_chart(fig_g, use_container_width=True)

st.markdown('<div class="swiss-divider"></div>', unsafe_allow_html=True)

# ── § 5 · AGRICULTURAL ────────────────────────────────────
st.markdown('<div class="section-label">05 · Agricultural Commodities</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Wheat · Corn · Soy — Indexed from War Start</div>', unsafe_allow_html=True)

fig_ag = go.Figure()
ag_assets = [("wheat",C["navy"],"Wheat"),("corn",C["blue"],"Corn"),("soy",C["gray"],"Soy")]
for asset, color, label in ag_assets:
    base_val = float(prices[asset].iloc[0])
    rel      = (prices[asset]/base_val*100).dropna()
    fig_ag.add_trace(go.Scatter(
        x=rel.index, y=rel.values, name=f"{label} (base ${base_val:.0f})",
        line=dict(color=color, width=2.5)
    ))
fig_ag.add_hline(y=100, line_dash="dot", line_color="#C4BDAF", line_width=1.5)
fig_ag.update_layout(**SWISS, height=300, yaxis=dict(title="Price Index (base=100)", **SWISS["yaxis"]))
st.plotly_chart(fig_ag, use_container_width=True)

st.markdown('<div class="swiss-divider"></div>', unsafe_allow_html=True)

# ── § 6 · MONTE CARLO FAN ─────────────────────────────────
war_note  = " · ⚑ War Boost Active" if war_trig else ""
bs_note   = f" · ⚠ Fert Black Swan ×{bs_mult:.2f}" if bs_mult>1.2 else ""
st.markdown('<div class="section-label">06 · Probabilistic Forecast</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">Monte Carlo · EVT+DCC · {CONFIG["mc_sims"]:,} paths × {CONFIG["mc_steps"]}d{war_note}{bs_note}</div>', unsafe_allow_html=True)

x_ax = list(range(CONFIG["mc_steps"]+1))
fig_mc = go.Figure()
fig_mc.add_trace(go.Scatter(
    x=x_ax+x_ax[::-1], y=list(fan[95])+list(fan[5][::-1]),
    fill="toself", fillcolor="rgba(196,189,175,0.22)", line=dict(width=0),
    name="WTI 90% CI", showlegend=True,
))
fig_mc.add_trace(go.Scatter(
    x=x_ax+x_ax[::-1], y=list(fan[75])+list(fan[25][::-1]),
    fill="toself", fillcolor="rgba(158,148,136,0.32)", line=dict(width=0),
    name="WTI 50% CI", showlegend=True,
))
fig_mc.add_trace(go.Scatter(
    x=x_ax, y=list(fan_brent[50]),
    name=f"Brent P50 → ${fan_brent[50][-1]:.2f}",
    line=dict(color=C["blue"], width=2.5, dash="dash"),
))
fig_mc.add_trace(go.Scatter(
    x=x_ax, y=list(fan[50]),
    name=f"WTI P50 → ${fan[50][-1]:.2f}",
    line=dict(color=C["navy"], width=3.5),
))
fig_mc.add_trace(go.Scatter(
    x=x_ax, y=list(fan[95]),
    name=f"P95 → ${fan[95][-1]:.2f}",
    line=dict(color=C["gold_dim"], width=1.5, dash="dot"),
))
fig_mc.add_trace(go.Scatter(
    x=x_ax, y=list(fan[5]),
    name=f"P5  → ${fan[5][-1]:.2f}",
    line=dict(color=C["gold_dim"], width=1.5, dash="dot"),
))
fig_mc.add_hline(y=wti_last, line_dash="dash", line_color="#8C8377", line_width=1.5,
    annotation_text=f"Current ${wti_last:.2f}", annotation_position="right",
    annotation_font=dict(family="DM Mono, monospace", size=10, color="#8C8377"))
fig_mc.add_hline(y=40,  line_dash="dot", line_color=C["rust"], line_width=1.5,
    annotation_text="Stress $40", annotation_position="right",
    annotation_font=dict(family="DM Mono, monospace", size=10, color=C["rust"]))
fig_mc.add_hline(y=150, line_dash="dot", line_color=C["rust"], line_width=1.5,
    annotation_text="Stress $150", annotation_position="right",
    annotation_font=dict(family="DM Mono, monospace", size=10, color=C["rust"]))
fig_mc.update_layout(
    **SWISS, height=480,
    xaxis=dict(title="Trading Days Ahead", **SWISS["xaxis"]),
    yaxis=dict(title="Price (USD/bbl)", tickprefix="$", **SWISS["yaxis"]),
)
st.plotly_chart(fig_mc, use_container_width=True)

# ── § 7 · EXECUTIVE SUMMARY TABLE ─────────────────────────
st.markdown('<div class="swiss-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">07 · Executive Summary</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Risk Metrics & Model Diagnostics</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

left_rows = [
    ("WTI Crude",          f"${wti_last:.2f}"),
    ("Brent Crude",        f"${brent_last:.2f}"),
    ("WTI-Brent Spread",   f"${spread:.2f} ({(spread/wti_last)*100:.1f}%)"),
    ("GeoFactor v4.0",     f"{float(geofactor.iloc[-1]):.5f}"),
    ("Risk Regime",        f"WAR ({rbase:+.3f})"),
    ("War Signal",         f"{war_sig:.5f} · {'ACTIVE' if war_trig else 'subdued'}"),
    ("DCC Parameters",     f"α={st.session_state['dcc_a']:.4f}  β={st.session_state['dcc_b']:.4f}"),
]
right_rows = [
    ("WTI Vol p.a.",       f"{mets['vol_wti_aa']:.1f}%"),
    ("Brent Vol p.a.",     f"{mets['vol_brent_aa']:.1f}%"),
    ("WTI–Brent ρ (EWMA)", f"{corr:.4f}"),
    ("Dynamic df (tail)",  f"{tail_df:.2f}"),
    ("Prob. Up 10d",       f"{mets['prob_up_10d']:.1f}%"),
    ("VaR 95% 1d",         f"${mets['var_95_1d']:+.2f}"),
    ("CVaR 95% 1d",        f"${mets['cvar_95_1d']:+.2f}"),
    ("Z-Composite",        f"{float(z_comp.iloc[-1]):+.4f}"),
    ("Prob WTI < $40",     f"{mets['prob_wti_below_40']:.2f}%"),
    ("Prob WTI > $150",    f"{mets['prob_wti_above_150']:.2f}%"),
]

def render_table(rows, container):
    with container:
        html = '<table class="data-table"><thead><tr><th>Indicator</th><th>Value</th></tr></thead><tbody>'
        for lbl, val in rows:
            html += f"<tr><td>{lbl}</td><td><strong>{val}</strong></td></tr>"
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)

render_table(left_rows, col_l)
render_table(right_rows, col_r)

# ── Bayesian shrinkage note
st.markdown(f"""
<div style='margin-top:1.2rem; padding:0.9rem 1.2rem; background:#F5F1EB; border-left:2px solid #C8A96E;
     font-family:"DM Mono",monospace; font-size:0.65rem; letter-spacing:0.08em; color:#4A4540;'>
<strong>Bayesian Vol Shrinkage</strong> · WTI {d_wti['vol_garch_aa']:.0f}% → {d_wti['vol_final_aa']:.0f}% (w_data={d_wti['weight_data']:.2f}) &nbsp;·&nbsp;
Brent {d_brt['vol_garch_aa']:.0f}% → {d_brt['vol_final_aa']:.0f}% (w_data={d_brt['weight_data']:.2f}) &nbsp;·&nbsp;
Fert: Urea ${usda['urea_price']:.0f}/t · DAP ${usda['dap_price']:.0f}/t · {usda['source']}
{f"&nbsp;·&nbsp;⚠ Black Swan ×{bs_mult:.2f}" if bs_mult>1.2 else ""}
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    <div>◆ GeoQuant v4.0-beta · EVT + DCC + GARCH-X · {CONFIG['mc_sims']:,} Monte Carlo Paths</div>
    <div>Eduardo Moraes · Quant Data Scientist & Economics</div>
    <div>For professional use only · {now_sp.strftime("%d %b %Y")}</div>
</div>
""", unsafe_allow_html=True)

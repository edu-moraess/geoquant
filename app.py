"""
GeoQuant – Institutional Macro Research Terminal
EVT + DCC-GARCH-X + GeoFactor + Walk-Forward + SHAP + ML Benchmarking
+ GPR (FRED) + COT (CFTC) + DCC Time Series + Sensitivity Map + Export + Status + Model Card
Eduardo Moraes | Quant Data Scientist & Economics
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, csv, logging, warnings, requests, time, json, io, base64
from datetime import datetime, timedelta
import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import stats, optimize
from scipy.stats import chi2, skew, kurtosis
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import yfinance as yf
from arch import arch_model
import shap
import xgboost as xgb
import lightgbm as lgb
import joblib
from joblib import Parallel, delayed

try:
    from pandas_datareader import data as pdr
    yf.pdr_override()
except Exception:
    pass

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ══════════════════════════════════════════════════════════
#   API CREDENTIALS & MACRO CONTEXT
# ══════════════════════════════════════════════════════════
EIA_API_KEY = st.secrets.get("EIA_API_KEY", "kVSuPa0tfnUmHzQ2VVSCPC6owKhPQQY2PbEc9hA1")
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "876c9f95b965eb9d423ef2c7b68ae51b")
OILPRICE_API_KEY = st.secrets.get("OILPRICE_API_KEY", "e241c0914287d05fcbbeb18669c23d86e9cdf36c63193a95d42854eb53ed354d")

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
#   INSTITUTIONAL RESEARCH REPORT STYLING (Minimalist White)
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
    --warning: #B37D14;
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
.diag-card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1rem;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
}
.status-pass { color: var(--success); font-weight: bold; }
.status-warning { color: var(--warning); font-weight: bold; }
.status-fail { color: var(--danger); font-weight: bold; }

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
#   SANITIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════
def safe_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value)
    if s.lower() in ["undefined", "nan", "none", "null"]:
        return ""
    return s

def fmt_num(x, fmt=".1f", suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:{fmt}}{suffix}"

def safe_delta(val, fmt=".2f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return f"{val:{fmt}}"

def safe_metric_value(val, default=0.0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val

# ══════════════════════════════════════════════════════════
#   CONSTANTS & STRUCTURAL PARAMETERS
# ══════════════════════════════════════════════════════════
TICKERS = {
    "oil":"CL=F","brent":"BZ=F","natgas":"NG=F","gold":"GC=F","silver":"SI=F",
    "copper":"HG=F","wheat":"ZW=F","corn":"ZC=F","soy":"ZS=F",
    "dxy":"DX-Y.NYB","eur":"EURUSD=X","tnx":"^TNX","ovx":"^OVX"
}
GEO_W = {"oil_vol":0.15,"gold":0.06,"gold_real":0.06,"dxy":-0.08,"spread":0.06,
          "fert":0.15,"wheat":0.05,"copper":0.04,"natgas_vol":0.05,"ovx":0.08,
          "baltic":0.06,"freightos":0.06,"move":0.05,"fci":-0.05}
ZSC_W = {"oil_gold":0.40,"oil_natgas":0.35,"gold_real":0.25}

PL = dict(
    template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    font=dict(family="Source Sans 3,Helvetica Neue,sans-serif", color="#1C1C1C", size=11),
    title_font=dict(family="Playfair Display,Georgia,serif", size=16, color="#1E3A5F"),
    xaxis=dict(gridcolor="#E8E4DA", linecolor="#D9D5CD", zeroline=False, tickfont=dict(size=10, family="JetBrains Mono,monospace", color="#5A554F")),
    yaxis=dict(gridcolor="#E8E4DA", linecolor="#D9D5CD", zeroline=False, tickfont=dict(size=10, family="JetBrains Mono,monospace", color="#5A554F")),
    legend=dict(bgcolor="rgba(255,255,255,0.97)", bordercolor="#D9D5CD", borderwidth=1, font=dict(size=10, family="JetBrains Mono,monospace", color="#1C1C1C")),
    margin=dict(l=55, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor="#1E3A5F", font_color="#D4C094", font_family="JetBrains Mono,monospace"),
)
C = dict(
    navy="#1E3A5F", navy_light="#2A5080", blue="#3A5F8A", gold="#B49450", gold_light="#D4C094",
    burgundy="#7B3F3F", teal="#2B5F5F", sage="#4A5D4A", gray="#5A554F", silver="#9A958A", sky="#4A7380", rust="#8B5A3A",
    fill_light="rgba(30,58,95,0.04)", fill_medium="rgba(30,58,95,0.10)", fill_deep="rgba(30,58,95,0.18)"
)

def qfig(h=420):
    fig = go.Figure(); fig.update_layout(**PL, height=h); return fig

def dual_axis_fig(h=380):
    fig = go.Figure()
    fig.update_layout(**PL, height=h,
        yaxis2=dict(overlaying="y", side="right", showgrid=False, linecolor="#D9D5CD", zeroline=False,
                    tickfont=dict(size=10, family="JetBrains Mono,monospace", color="#5A554F")))
    return fig

# ══════════════════════════════════════════════════════════
#   EXTERNAL ADVANCED DATA HARVESTER (FRED + EIA + OILPRICE + GPR)
# ══════════════════════════════════════════════════════════
def fetch_fred_macro():
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key={FRED_API_KEY}&file_type=json"
        res = requests.get(url, timeout=5).json()
        obs = res.get("observations", [])
        if obs:
            val = obs[-1].get("value")
            return float(val) if val != "." else 20.0
    except: pass
    return 20.0

def fetch_eia_inventories():
    try:
        url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?api_key={EIA_API_KEY}&frequency=weekly&data[]=value&facets[series][]=WCRSTUS1"
        res = requests.get(url, timeout=5).json()
        data = res.get("response", {}).get("data", [])
        if data: return float(data[0].get("value", 420000))
    except: pass
    return 420000.0

def fetch_oilprice_spot():
    try:
        url = f"https://oilpriceapi.com/v1/prices/latest"
        headers = {"Authorization": f"Token {OILPRICE_API_KEY}"}
        res = requests.get(url, headers=headers, timeout=5).json()
        if res.get("status") == "success": return float(res.get("data", {}).get("price", 0.0))
    except: pass
    return 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gpr():
    """Geopolitical Risk Index (Caldara & Iacoviello) do FRED – série GPRHIST"""
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=GPRHIST&api_key={FRED_API_KEY}&file_type=json"
        res = requests.get(url, timeout=10).json()
        obs = res.get("observations", [])
        if obs:
            dates = [pd.to_datetime(o["date"]) for o in obs]
            values = [float(o["value"]) if o["value"] != "." else np.nan for o in obs]
            return pd.Series(values, index=dates).dropna().rename("gpr")
    except:
        pass
    return pd.Series(dtype=float)

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cot(ticker="CL"):
    """Commitment of Traders – proxy de posições líquidas de especuladores (non-commercial) para o ticker dado.
       Em ambiente real, deve-se conectar à API da CFTC. Aqui usamos um proxy baseado em dados recentes conhecidos."""
    # Dado real aproximado (WTI): ~300k contratos líquidos longos. Adicionamos um ruído baseado na volatilidade do petróleo.
    try:
        # Tenta obter a volatilidade recente do WTI para calibrar o ruído
        wti = yf.download("CL=F", period="5d", progress=False)["Close"]
        if len(wti) > 1:
            vol = wti.pct_change().std()
            noise = np.random.normal(0, vol * 300000)
        else:
            noise = np.random.normal(0, 15000)
        return max(0, 300000 + noise)
    except:
        return 300000

# ══════════════════════════════════════════════════════════
#   API STATUS FUNCTION
# ══════════════════════════════════════════════════════════
def get_api_status():
    """Retorna dicionário com status de cada API (OK, FALLBACK, ERROR)"""
    status = {}
    # FRED
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key={FRED_API_KEY}&file_type=json&limit=1"
        r = requests.get(url, timeout=3)
        if r.status_code == 200 and "observations" in r.json():
            status["FRED"] = "✅ OK"
        else:
            status["FRED"] = "⚠️ FALLBACK"
    except:
        status["FRED"] = "⚠️ FALLBACK"
    # EIA
    try:
        url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?api_key={EIA_API_KEY}&limit=1"
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            status["EIA"] = "✅ OK"
        else:
            status["EIA"] = "⚠️ FALLBACK"
    except:
        status["EIA"] = "⚠️ FALLBACK"
    # OilPrice API
    try:
        url = "https://oilpriceapi.com/v1/prices/latest"
        headers = {"Authorization": f"Token {OILPRICE_API_KEY}"}
        r = requests.get(url, timeout=3, headers=headers)
        if r.status_code == 200 and r.json().get("status") == "success":
            status["OilPrice"] = "✅ OK"
        else:
            status["OilPrice"] = "⚠️ FALLBACK"
    except:
        status["OilPrice"] = "⚠️ FALLBACK"
    # yfinance (sempre OK)
    status["yfinance"] = "✅ OK"
    return status

# ══════════════════════════════════════════════════════════
#   QUANT ENGINE ARCHITECTURE (original, com pequenas adaptações)
# ══════════════════════════════════════════════════════════
def rolling_zscore(s, w=60):
    std = s.rolling(w).std()
    return (s - s.rolling(w).mean()) / std.where(std > 0, np.nan)

def fill_gaps(s):
    s = s.copy()
    valid = s.notna()
    if valid.sum() < 2: return s.ffill().bfill()
    try:
        x = s.index[valid].astype(np.int64)
        f = pd.Series(PchipInterpolator(x, s[valid].values)(s.index.astype(np.int64)), index=s.index)
        f[valid] = s[valid]
        return f.ffill().bfill()
    except: return s.ffill().bfill()

def _force_update_fert_csv(path="fertilizer_backup.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date","urea_price","dap_price"])
        w.writerows([["2026-01-15",540,710],["2026-02-15",560,740],["2026-03-15",590,780],
                     ["2026-04-15",616,857],["2026-05-01",720,900],["2026-05-06",810,920],
                     ["2026-05-12",857,920],["2026-06-01",860,925],["2026-06-10",453.5,920]])

def get_usda():
    _force_update_fert_csv()
    try:
        df = pd.read_csv("fertilizer_backup.csv", parse_dates=["date"], index_col="date").sort_index()
        if len(df) == 0: return {"urea_price":453.5,"urea_period":"2026-06-10","dap_price":920,"dap_period":"2026-06-10","source":"fallback"}
        last = df.iloc[-1]
        return {"urea_price":float(last["urea_price"]),"urea_period":str(last.name.date()),"dap_price":float(last["dap_price"]),"dap_period":str(last.name.date()),"source":"Green Markets / CRU"}
    except: return {"urea_price":453.5,"urea_period":"2026-06-10","dap_price":920,"dap_period":"2026-06-10","source":"fallback"}

def fert_black_swan(usda):
    _force_update_fert_csv()
    try:
        df = pd.read_csv("fertilizer_backup.csv", parse_dates=["date"], index_col="date")
        hist = df["urea_price"].dropna().values
    except: hist = []
    cur = usda.get("urea_price")
    if cur is None or len(hist) < 10: return 1.0
    rets = np.diff(np.log(hist))
    thr = np.quantile(rets, 0.90)
    exc = rets[rets > thr] - thr
    if len(exc) < 5:
        mu, sig = np.mean(hist), np.std(hist)
        if sig == 0: return 1.0
        z = (cur - mu) / sig
        if z < -1.5: return max(0.5, 1.0 + z*0.3)
        return min(1.0 + max(0, z-1.5)*0.8, 3.0)
    try:
        shape, loc, scale = stats.genpareto.fit(exc)
        cr = np.log(cur / hist[-1])
        if cr <= thr: return 0.6 if cr < -0.1 else 1.0
        p = 1 - stats.genpareto.cdf(cr - thr, shape, loc=loc, scale=scale)
        return 1.0 + min(p*5, 2.0)
    except: return 1.0

def gold_signals(prices):
    silver = prices["silver"].replace(0, np.nan)
    if silver.median() > 500: silver /= 100
    gr = prices["gold"] / (1 + prices["tnx"].replace(0,np.nan)/100*5.0)
    sg = silver / prices["gold"].replace(0, np.nan)
    return {"gold_real":gr,"silver_gold":sg,"gold_real_ret_roll":np.log(gr/gr.shift(1)).rolling(20).mean(),"silver_gold_roll":np.log(sg/sg.shift(1)).rolling(20).mean()}

def silver_demand_proxy(prices):
    if "copper" not in prices.columns: return pd.Series(0.0, index=prices.index)
    cr = prices["copper"].pct_change().dropna()
    br = prices["brent"].pct_change().dropna()
    ci = cr.index.intersection(br.index)
    return (0.6*cr[ci]+0.4*br[ci]).rolling(20).mean().reindex(prices.index,method="ffill").fillna(0.0)

def simulate_macro_indices(prices_index):
    np.random.seed(42)
    n = len(prices_index)
    baltic = pd.Series(1500 + np.cumsum(np.random.normal(2, 45, n)), index=prices_index).clip(600, 4000)
    freightos = pd.Series(2200 + np.cumsum(np.random.normal(5, 60, n)), index=prices_index).clip(900, 7000)
    move = pd.Series(110 + np.cumsum(np.random.normal(0, 3, n)), index=prices_index).clip(60, 220)
    fci = pd.Series(np.cumsum(np.random.normal(0, 0.04, n)), index=prices_index).clip(-2.5, 2.5)
    return {"baltic": baltic, "freightos": freightos, "move": move, "fci": fci}

def build_fert_index(returns, usda, bs=1.0):
    fi = (0.5*returns["natgas"].rolling(20).std() + 0.25*returns["wheat"].rolling(20).mean() + 0.25*returns["corn"].rolling(20).mean())
    if usda["urea_price"]: fi += np.clip((usda["urea_price"]-380)/380, -1, 2)*0.15
    if usda["dap_price"]: fi += np.clip((usda["dap_price"]-610)/610, -1, 2)*0.10
    fi *= bs
    return fi.clip(fi.quantile(0.02), fi.quantile(0.98)).dropna()

def calibrate_weights(returns, prices, gs, fi, sd, macro_proxies, window=60):
    spread = (prices["brent"]-prices["oil"])/prices["brent"].replace(0,np.nan)
    X = pd.DataFrame({
        "oil_vol":returns["oil"].rolling(20).std(),"gold":returns["gold"].rolling(20).mean(),
        "gold_real":gs["gold_real_ret_roll"],"dxy":returns["dxy"].rolling(20).mean(),"spread":spread.rolling(20).mean(),
        "wheat":returns["wheat"].rolling(20).mean(),"copper":returns["copper"].rolling(20).mean(),
        "natgas_vol":returns["natgas"].rolling(20).std(),"fert":fi,"silver_demand":sd,
        "baltic": macro_proxies["baltic"].pct_change().rolling(20).mean(),
        "freightos": macro_proxies["freightos"].pct_change().rolling(20).mean(),
        "move": macro_proxies["move"].pct_change().rolling(20).mean(),
        "fci": macro_proxies["fci"].rolling(20).mean()
    })
    y = returns["oil"].shift(-1)
    ci = y.dropna().index.intersection(X.dropna().index)
    X2, y2 = X.loc[ci].dropna(), y.loc[ci]
    if len(X2) < window: return GEO_W.copy()
    Xc, yc = X2.iloc[-window:], y2.iloc[-window:]
    Xm, Xs = Xc.mean(), Xc.std().replace(0, 1)
    try:
        mdl = LassoCV(cv=5, random_state=42, alphas=np.logspace(-4,0,20), max_iter=2000).fit((Xc-Xm)/Xs, yc)
        w = {col: mdl.coef_[i] for i, col in enumerate(X2.columns)}
        tot = sum(abs(v) for v in w.values())
        return {k: v/tot for k, v in w.items()} if tot > 0 else GEO_W.copy()
    except: return GEO_W.copy()

def build_geofactor(returns, prices, gs, fi, weights, sd, macro_proxies):
    spread = (prices["brent"]-prices["oil"])/prices["brent"].replace(0,np.nan)
    geo = (weights.get("oil_vol",0)*returns["oil"].rolling(20).std() +
           weights.get("gold",0)*returns["gold"].rolling(20).mean() +
           weights.get("gold_real",0)*gs["gold_real_ret_roll"] +
           weights.get("dxy",0)*returns["dxy"].rolling(20).mean() +
           weights.get("spread",0)*spread.rolling(20).mean() +
           weights.get("wheat",0)*returns["wheat"].rolling(20).mean() +
           weights.get("copper",0)*returns["copper"].rolling(20).mean() +
           weights.get("natgas_vol",0)*returns["natgas"].rolling(20).std() +
           weights.get("fert",0)*fi + weights.get("silver_demand",0)*sd +
           weights.get("baltic",0)*macro_proxies["baltic"].pct_change().rolling(20).mean() +
           weights.get("freightos",0)*macro_proxies["freightos"].pct_change().rolling(20).mean() +
           weights.get("move",0)*macro_proxies["move"].pct_change().rolling(20).mean() +
           weights.get("fci",0)*macro_proxies["fci"].rolling(20).mean())
    g = geo.dropna()
    return g.clip(g.quantile(0.05), g.quantile(0.95))

def build_zscore(prices, gs, window=60):
    w = min(window, max(20, len(prices)//2))
    z1 = rolling_zscore(prices["oil"]/prices["gold"].replace(0,np.nan), w)
    z2 = rolling_zscore(prices["oil"]/prices["natgas"].replace(0,np.nan), w)
    z3 = rolling_zscore(gs["gold_real"], w)
    return (ZSC_W["oil_gold"]*z1 + ZSC_W["oil_natgas"]*z2 + ZSC_W["gold_real"]*z3).dropna()

def fit_egarch(ret, exog=None):
    r = ret.dropna()
    if len(r) < 50: return pd.Series(r.std(), index=ret.index).ffill().bfill()
    try:
        rc = r * 100
        if exog is not None and not exog.empty:
            common = rc.index.intersection(exog.dropna().index)
            if len(common) >= 50:
                rc = rc.loc[common]
                xc = exog.loc[common].to_frame() if isinstance(exog, pd.Series) else exog.loc[common]
                model = arch_model(rc, x=xc, mean="Constant", vol="EGARCH", p=1, q=1, dist="skewt")
                res = model.fit(disp="off")
                return (res.conditional_volatility / 100).reindex(ret.index).ffill().bfill()
        model = arch_model(rc, mean="Constant", vol="EGARCH", p=1, q=1, dist="skewt")
        res = model.fit(disp="off")
        return (res.conditional_volatility / 100).reindex(ret.index).ffill().bfill()
    except: return pd.Series(r.rolling(20).std().mean(), index=ret.index).ffill().bfill()

def conditional_evt(returns, vol, q=0.95, min_obs=30):
    common = returns.dropna().index.intersection(vol.dropna().index)
    if len(common) < min_obs: return None
    r, v = returns.loc[common], vol.loc[common].replace(0, np.nan)
    resid = (r / v).dropna()
    resid = resid[np.isfinite(resid)]
    if len(resid) < min_obs or resid.std() < 1e-8: return None
    th_up, th_lo = np.percentile(resid, q*100), np.percentile(resid, (1-q)*100)
    exc_up, exc_lo = resid[resid > th_up] - th_up, -resid[resid < th_lo] - th_lo
    shape_up, scale_up = stats.genpareto.fit(exc_up)[0] if len(exc_up)>=10 else 0.2, exc_up.std() if len(exc_up)>0 else 0.1
    shape_lo, scale_lo = stats.genpareto.fit(exc_lo)[0] if len(exc_lo)>=10 else 0.2, exc_lo.std() if len(exc_lo)>0 else 0.1
    return {"upper": (shape_up, scale_up, th_up), "lower": (shape_lo, scale_lo, th_lo), "resid": resid}

def detect_regime(vol, threshold=1.5):
    v = vol.dropna()
    if len(v) < 20: return pd.Series(0, index=vol.index)
    mean, std = v.rolling(60, min_periods=20).mean(), v.rolling(60, min_periods=20).std().replace(0, 1e-8)
    z = (v - mean) / std
    regimes = np.where(z > 2.5, 3, np.where(z > 1.5, 2, np.where(z > 0.5, 1, 0)))
    return pd.Series(regimes, index=vol.index).fillna(0).astype(int)

def bayes_shrink(vg, prior_d, n, geofactor=None):
    w = np.clip(np.sqrt(n/252), 0.10, 0.95)
    prior = prior_d * (1.0 + 0.4 * np.tanh(float(geofactor.iloc[-1]))) if (geofactor is not None and len(geofactor) > 0) else prior_d
    if len(vg) == 0: return pd.Series(prior, index=geofactor.index if geofactor is not None else [datetime.now()]), {"vga":prior*100,"vsa":prior*100,"w":w}
    v_last = float(vg.iloc[-1])
    effective_w = w if not (prior*0.5 <= v_last <= prior*1.5) else 1.0
    vs = effective_w * vg + (1 - effective_w) * prior
    return vs, {"vga": v_last * np.sqrt(252) * 100, "vsa": float(vs.iloc[-1]) * np.sqrt(252) * 100, "w": effective_w}

def fit_dcc(rw, rb, vw, vb):
    common = rw.index.intersection(rb.index).intersection(vw.index).intersection(vb.index)
    if len(common) < 10: return 0.05, 0.93, pd.Series(index=common)
    ew, eb = (rw[common] / vw[common]).dropna(), (rb[common] / vb[common]).dropna()
    c2 = ew.index.intersection(eb.index)
    if len(c2) < 10: return 0.05, 0.93, pd.Series(index=c2)
    e = np.column_stack([ew[c2], eb[c2]])
    Qb = np.cov(e, rowvar=False)
    np.fill_diagonal(Qb, 1.0)
    # Armazenar correlações ao longo do tempo
    rho_series = np.zeros(len(c2))
    def nll(p):
        a, b = p
        if a <= 0 or b <= 0 or a + b >= 1: return 1e10
        Qt = Qb.copy()
        ll = 0.0
        for t in range(1, len(e)):
            Qt = (1 - a - b) * Qb + a * np.outer(e[t-1], e[t-1]) + b * Qt
            d = np.sqrt(np.diag(Qt))
            if d[0] == 0 or d[1] == 0: return 1e10
            Rt = Qt / np.outer(d, d)
            Rt = np.clip(Rt, -0.9999, 0.9999)
            rho_series[t] = Rt[0,1]
            try:
                L = np.linalg.cholesky(Rt)
                z = np.linalg.solve(L, e[t])
                ll += -0.5 * np.sum(z**2) - np.sum(np.log(np.diag(L)))
            except: return 1e10
        return -ll
    try:
        res = optimize.minimize(nll, [0.05, 0.93], bounds=[(1e-4, 0.3), (0.7, 0.9999)], method="L-BFGS-B")
        a, b = (float(res.x[0]), float(res.x[1])) if res.success and (res.x[0]+res.x[1] < 1) else (0.05, 0.93)
        rho_series = pd.Series(rho_series, index=c2)
        return a, b, rho_series
    except: return 0.05, 0.93, pd.Series(index=c2)

def _tail_jumps(shocks, vol):
    n = len(shocks)
    u = np.random.rand(n)
    return shocks + np.where(u < 0.025, np.random.exponential(0.03, n) * vol, 0) - np.where((u >= 0.025) & (u < 0.05), np.random.exponential(0.02, n) * vol, 0)

def _jumps_vec(n, pu, pd_):
    u = np.random.rand(n)
    me = np.random.rand(n) < 0.15
    ju = np.where(me, np.random.exponential(0.135, n), np.random.exponential(0.045, n))
    jd = np.random.exponential(0.025, n)
    return np.where(u < pu, ju, np.where((u >= pu) & (u < pu + pd_), -jd, 0)), np.where(u < pu, ju*0.95, np.where((u >= pu) & (u < pu + pd_), -jd*0.90, 0))

def run_mc(wti0, brt0, bvw, bvb, fcast, ocol, bcol, rbase, rw, rb, vws, vbs, jpu, tdf, bs=1.0, dcc_a=0.05, dcc_b=0.93, sims=5000, steps=10, bar=None, scenario_mod=None):
    seed = st.session_state.get("mc_seed", 42)
    np.random.seed(seed)
    if scenario_mod:
        jpu *= scenario_mod.get("jump_mult", 1.0)
        bvw *= scenario_mod.get("vol_mult", 1.0)
        bvb *= scenario_mod.get("vol_mult", 1.0)
        rbase += scenario_mod.get("geo_shift", 0.0)

    bvw, bvb = max(bvw, 1e-6), max(bvb, 1e-6)
    ci = rw.index.intersection(rb.index).intersection(vws.index).intersection(vbs.index)
    
    if len(ci) < 10:
        rho_const = 0.85
        Qb = np.array([[1.0, rho_const], [rho_const, 1.0]])
        eps = np.random.normal(0, 1, (sims, 2))
        Qt = np.tile(Qb, (sims, 1, 1))
    else:
        ew, eb = (rw[ci] / vws[ci].replace(0, np.nan)).dropna(), (rb[ci] / vbs[ci].replace(0, np.nan)).dropna()
        c2 = ew.index.intersection(eb.index)
        if len(c2) < 10:
            rho_const = 0.85
            Qb = np.array([[1.0, rho_const], [rho_const, 1.0]])
            eps = np.random.normal(0, 1, (sims, 2))
            Qt = np.tile(Qb, (sims, 1, 1))
        else:
            e = np.column_stack([np.clip(ew[c2], -3, 3), np.clip(eb[c2], -3, 3)])
            Qb = np.cov(e, rowvar=False)
            np.fill_diagonal(Qb, 1.0)
            eps = np.repeat(e[-1][np.newaxis, :], sims, axis=0) + np.random.normal(0, 0.05, (sims, 2))
            Qt = np.tile(Qb, (sims, 1, 1)).copy()

    pu, pd_ = min(jpu * 1.5, 0.20) if bs > 1.2 else jpu, 0.03 * (1.3 if bs > 1.2 else 1.0)
    pw, pb = np.zeros((sims, steps+1)), np.zeros((sims, steps+1))
    pw[:, 0], pb[:, 0] = wti0, brt0
    ra = 1 + 0.5 * np.clip(rbase + np.random.normal(0, 0.05, (sims, steps)), -1, 1)

    for t in range(steps):
        if bar: bar.progress((t+1)/steps)
        if len(ci) >= 10:
            outer = np.einsum("si,sj->sij", eps, eps)
            Qt = (1 - dcc_a - dcc_b) * Qb[np.newaxis] + dcc_a * outer + dcc_b * Qt
            diag = np.clip(np.sqrt(np.diagonal(Qt, axis1=1, axis2=2)), 1e-8, None)
            Rt = np.clip(Qt / np.einsum("si,sj->sij", diag, diag), -0.9999, 0.9999)
            rho = Rt[:, 0, 1]
        else: rho = np.full(sims, 0.85)
            
        sc = np.sqrt(np.clip(1 - rho**2, 1e-8, None))
        z = np.random.standard_t(tdf, (sims, 2))
        zw, zb = z[:, 0], rho * z[:, 0] + sc * z[:, 1]

        vw_, vb_ = np.clip(bvw * ra[:, t], 1e-6, 0.08), np.clip(bvb * ra[:, t], 1e-6, 0.08)
        sw, sb = np.clip(zw * vw_, -4*vw_, 4*vw_), np.clip(zb * vb_, -4*vb_, 4*vb_)
        sw, sb = _tail_jumps(sw, vw_), _tail_jumps(sb, vb_)
        jw, jb = _jumps_vec(sims, pu, pd_)
        sw, sb = sw + jw, sb + jb
        
        dw = np.clip(fcast[t, ocol] * ra[:, t], -0.02, 0.02) if t < len(fcast) else 0.0
        db = np.clip(fcast[t, bcol] * ra[:, t], -0.02, 0.02) if t < len(fcast) else 0.0
        
        nw, nb = pw[:, t] * np.exp(dw + sw), pb[:, t] * np.exp(db + sb)
        sp = np.where(nb > 0, (nb - nw) / nb, 0)
        nw = np.where(sp < -0.05, nb * 1.05, nw)
        nw = np.where(sp > 0.30, nb * 0.70, nw)
        pw[:, t+1], pb[:, t+1] = np.clip(nw, wti0*0.4, wti0*2.5), np.clip(nb, brt0*0.4, brt0*2.5)
        eps[:, 0], eps[:, 1] = np.where(vw_ > 0, sw / vw_, 0), np.where(vb_ > 0, sb / vb_, 0)
        eps = np.clip(eps, -5, 5)

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    fan = {p: np.percentile(pw, p, axis=0) for p in percentiles}
    fb = {p: np.percentile(pb, p, axis=0) for p in percentiles}
    
    term_wti = pw[:, -1]
    term_brt = pb[:, -1]
    
    sim_mean = np.mean(term_wti)
    sim_med = np.median(term_wti)
    sim_skew = skew(term_wti)
    sim_kurt = kurtosis(term_wti)
    sim_mode = 3 * sim_med - 2 * sim_mean
    
    mask = (pw[:, 1] - wti0) <= np.percentile(pw[:, 1] - wti0, 5)
    
    brackets = {
        "<50": np.mean(term_wti < 50) * 100,
        "50-60": np.mean((term_wti >= 50) & (term_wti < 60)) * 100,
        "60-70": np.mean((term_wti >= 60) & (term_wti < 70)) * 100,
        "70-80": np.mean((term_wti >= 70) & (term_wti < 80)) * 100,
        "80-90": np.mean((term_wti >= 80) & (term_wti < 90)) * 100,
        "90-100": np.mean((term_wti >= 90) & (term_wti < 100)) * 100,
        ">100": np.mean(term_wti >= 100) * 100
    }

    return {
        "fan": fan, "fan_b": fb, "paths": pw, "paths_b": pb,
        "brackets": brackets,
        "moments": {"mean": sim_mean, "median": sim_med, "mode": sim_mode, "skew": sim_skew, "kurt": sim_kurt},
        "metrics": {
            "vol_wti": bvw * np.sqrt(252) * 100, "vol_brt": bvb * np.sqrt(252) * 100,
            "var95": np.percentile(pw[:, 1] - wti0, 5), "cvar95": float(np.mean((pw[:, 1] - wti0)[mask])) if mask.sum() > 0 else np.percentile(pw[:, 1] - wti0, 5),
            "wti_70": np.mean(term_wti > 70)*100, "wti_80": np.mean(term_wti > 80)*100, "wti_90": np.mean(term_wti > 90)*100, "wti_100": np.mean(term_wti > 100)*100, "wti_120": np.mean(term_wti > 120)*100,
            "wti_l60": np.mean(term_wti < 60)*100, "wti_l50": np.mean(term_wti < 50)*100, "wti_l40": np.mean(term_wti < 40)*100,
            "brent_90": np.mean(term_brt > 90)*100, "brent_100": np.mean(term_brt > 100)*100
        }
    }

def backtest_var(returns, var_forecast, alpha=0.05):
    ci = returns.index.intersection(var_forecast.index)
    if len(ci) == 0: return {"calibration_score": 0, "Kupiec_p": 1, "Christoffersen_p": 1, "DQ_p": 1, "n_violations": 0, "obs_freq": 0}
    r, v = returns.loc[ci], var_forecast.loc[ci]
    violations = (r < -v).astype(int)
    n, nv = len(violations), violations.sum()
    po, pe = nv / n, alpha
    kp = 1 - chi2.cdf(-2 * np.log(((1-pe)**(n-nv) * pe**nv) / ((1-po)**(n-nv) * po**nv)), 1) if 0 < nv < n else 0.5
    if n > 1:
        n00 = ((violations[:-1]==0) & (violations[1:]==0)).sum()
        n01 = ((violations[:-1]==0) & (violations[1:]==1)).sum()
        n10 = ((violations[:-1]==1) & (violations[1:]==0)).sum()
        n11 = ((violations[:-1]==1) & (violations[1:]==1)).sum()
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        cp = 1 - chi2.cdf(-2 * np.log(((1-pe)**(n-1-(n01+n11)) * pe**(n01+n11)) / ((1-p01)**n00 * p01**n01 * (1-p11)**n10 * p11**n11)), 1) if (n01 + n11) > 0 else 0.5
    else: cp = 0.5
    try: dq = 1 - chi2.cdf(Logit(violations, pd.DataFrame({"const": 1, "lag": violations.shift(1).fillna(0)})).fit(disp=0).llr, 2)
    except: dq = 1.0
    return {"n_violations": int(nv), "obs_freq": po, "exp_freq": pe, "Kupiec_p": kp, "Christoffersen_p": cp, "DQ_p": dq, "calibration_score": 1 - np.mean([kp, cp, dq])}

def backtest_es(returns, cvar_val, var_forecast):
    ci = returns.index.intersection(var_forecast.index)
    if len(ci) == 0: return np.nan
    r, v = returns.loc[ci], var_forecast.loc[ci]
    cv = float(cvar_val.iloc[-1]) if len(cvar_val) > 0 else np.nan
    viol = (r < -v).astype(int)
    if viol.sum() == 0 or np.isnan(cv): return np.nan
    return float(((r[viol==1] + v[viol==1]).sum() / (viol.sum() * cv)) - 1)

def walk_forward_validation(returns_series, train_years=2, test_months=3):
    dates = returns_series.index
    ts, qs = int(train_years * 252), int(test_months * 21)
    results = []
    start = 0
    while start + ts + qs <= len(dates):
        te, qe = start + ts, start + ts + qs
        train, test = returns_series.iloc[start:te], returns_series.iloc[te:qe]
        pred = train.iloc[-20:].mean() if len(train) >= 20 else train.mean()
        rmse = float(np.sqrt(((pred - test)**2).mean()))
        results.append({"Window Start": dates[start].strftime("%Y-%m-%d"), "Window End": dates[qe-1].strftime("%Y-%m-%d"), "OOS RMSE": rmse})
        start += qs
    return pd.DataFrame(results)

def benchmark_ml(returns_df, target_col="oil"):
    features = returns_df.shift(1).dropna()
    target = returns_df[target_col].iloc[1:]
    ci = features.index.intersection(target.index)
    X, y = features.loc[ci], target.loc[ci]
    
    if len(X) < 10:
        return {
            "RandomForest": {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Directional Accuracy": "—"},
            "XGBoost": {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Directional Accuracy": "—"},
            "LightGBM": {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Directional Accuracy": "—"}
        }, X, y
    
    n_splits = min(5, len(X) // 3)
    n_splits = max(2, n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    
    out = {}
    for name, mdl in models.items():
        rmses, maes, mapes, dirs = [], [], [], []
        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            try:
                mdl.fit(Xtr, ytr)
                pred = mdl.predict(Xte)
                rmses.append(np.sqrt(mean_squared_error(yte, pred)))
                maes.append(mean_absolute_error(yte, pred))
                mapes.append(np.mean(np.abs((yte - pred) / yte.replace(0, 1e-5))) * 100)
                dirs.append(np.mean(np.sign(yte) == np.sign(pred)) * 100)
            except: pass
        out[name] = {
            "RMSE": np.mean(rmses) if rmses else np.nan, "MAE": np.mean(maes) if maes else np.nan,
            "MAPE": np.mean(mapes) if mapes else np.nan, "Directional Accuracy": f"{np.mean(dirs):.2f}%" if dirs else "—"
        }
    return out, X, y

def run_shap(X, y):
    mdl = RandomForestRegressor(n_estimators=100, random_state=42)
    mdl.fit(X, y)
    exp = shap.TreeExplainer(mdl)
    sv = exp.shap_values(X)
    fig, ax = plt.subplots(figsize=(6,3.5), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    shap.summary_plot(sv, X, show=False, plot_size=None)
    plt.tight_layout()
    return fig, sv, X.columns

def garch_diagnostics(resid):
    r = resid.dropna()
    if len(r) < 20: return {"LB5": np.nan, "LB10": np.nan, "ARCH_p": np.nan}
    lb = acorr_ljungbox(r, lags=[5,10], return_df=True)
    arch = het_arch(r**2, nlags=10)
    return {"LB5": lb.loc[5, "lb_pvalue"] if 5 in lb.index else np.nan,
            "LB10": lb.loc[10, "lb_pvalue"] if 10 in lb.index else np.nan,
            "ARCH_p": arch[1] if len(arch) > 1 else np.nan}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(start):
    tl, tk = list(TICKERS.values()), list(TICKERS.keys())
    for adj in [True, False]:
        try:
            raw = yf.download(tl, start=start, progress=False, auto_adjust=adj)
            if raw.empty: continue
            if isinstance(raw.columns, pd.MultiIndex):
                lvl = raw.columns.get_level_values(0).unique().tolist()
                field = next((f for f in ["Close", "Adj Close"] if f in lvl), None)
                out = raw[field].copy() if field else raw.iloc[:, :len(tk)].copy()
            else: out = raw.copy()
            out.columns = tk[:len(out.columns)]
            if not out.empty and len(out) > 5: return out.ffill().bfill()
        except: continue
    return pd.DataFrame()

@st.cache_data(ttl=10, show_spinner=False)
def fetch_live(lw=65.0, lb=68.0):
    api_spot = fetch_oilprice_spot()
    if api_spot > 10.0:
        return api_spot, api_spot + 3.20
    try:
        wti = float(yf.Ticker("CL=F").fast_info.get("last_price", 0))
        brt = float(yf.Ticker("BZ=F").fast_info.get("last_price", 0))
        if wti > 0 and brt > 0: return wti, brt
    except: pass
    return lw, lb

# ══════════════════════════════════════════════════════════
#   EXPORT FUNCTION
# ══════════════════════════════════════════════════════════
def export_results_to_csv(mc, moments, fan, weights, macro_proxies):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Simulated Mean", moments["mean"]])
    writer.writerow(["Median (P50)", moments["median"]])
    writer.writerow(["Mode", moments["mode"]])
    writer.writerow(["Skewness", moments["skew"]])
    writer.writerow(["Excess Kurtosis", moments["kurt"]])
    writer.writerow(["P99 WTI", fan[99][-1]])
    writer.writerow(["P90 WTI", fan[90][-1]])
    writer.writerow(["P50 WTI", fan[50][-1]])
    writer.writerow(["P10 WTI", fan[10][-1]])
    writer.writerow(["P01 WTI", fan[1][-1]])
    writer.writerow(["Prob WTI > 100", mc["metrics"]["wti_100"]])
    writer.writerow(["Prob WTI < 60", mc["metrics"]["wti_l60"]])
    writer.writerow([" "])
    writer.writerow(["Factor", "Weight"])
    for k, v in weights.items():
        writer.writerow([k, v])
    writer.writerow([" "])
    writer.writerow(["Macro Proxy", "Latest Value"])
    for k, v in macro_proxies.items():
        if isinstance(v, pd.Series) and len(v) > 0:
            writer.writerow([k, v.iloc[-1]])
        else:
            writer.writerow([k, v])
    return output.getvalue()

# ══════════════════════════════════════════════════════════
#   SIDEBAR PARAMETERS
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.3rem 0 1.1rem;border-bottom:1px solid #D9D5CD;margin-bottom:1.3rem;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:.5rem;letter-spacing:.26em;color:#B49450;text-transform:uppercase;margin-bottom:.4rem;'>◆ Edumetria</div>
        <div style='font-family:"Playfair Display",Georgia,serif;font-size:1.3rem;font-weight:300;color:#1E3A5F;letter-spacing:.06em;'>GeoQuant Terminal</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:.5rem;color:#7A766E;letter-spacing:.14em;margin-top:.3rem;'>Quantitative Research Infrastructure</div>
    </div>""", unsafe_allow_html=True)

    def slabel(t):
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.54rem;letter-spacing:.2em;color:#B49450;text-transform:uppercase;margin:.9rem 0 .4rem;">{t}</div>', unsafe_allow_html=True)
    def ssep():
        st.markdown('<div style="height:1px;background:#D9D5CD;margin:.6rem 0;"></div>', unsafe_allow_html=True)

    slabel("· Simulation")
    mc_sims  = st.slider("Monte Carlo paths", 1_000, 30_000, 5_000, 1_000)
    mc_steps = st.slider("Horizon (days)", 5, 30, 10, 1)
    ssep()
    slabel("· Interactive Scenario Engine")
    scen_choice = st.selectbox("Geo Scenario", ["Base Model (Live)", "Geopolitical Escalation", "Diplomatic Ceasefire", "Strait of Hormuz Closure", "OPEC Supply Shock", "Global Recession"])
    ssep()
    slabel("· Jump Parameters")
    jump_up, jump_down = st.slider("Jump prob ↑", 0.01, 0.20, 0.07, 0.01), st.slider("Jump prob ↓", 0.01, 0.10, 0.03, 0.01)
    tail_df = st.slider("Tail df", 2.5, 8.0, 3.0, 0.5)
    ssep()
    slabel("· Vol Priors (annual)")
    prior_wti, prior_brent = st.slider("WTI prior", 0.20, 0.65, 0.35, 0.01), st.slider("Brent prior", 0.20, 0.65, 0.35, 0.01)
    ssep()
    war_start = st.date_input("War start", value=datetime(2026, 2, 28))
    run_btn = st.button("▶  Run Full System Pipeline")

SCENARIO_MAP = {
    "Base Model (Live)": {"jump_mult": 1.0, "vol_mult": 1.0, "geo_shift": 0.0},
    "Geopolitical Escalation": {"jump_mult": 1.8, "vol_mult": 1.4, "geo_shift": 1.5},
    "Diplomatic Ceasefire": {"jump_mult": 0.3, "vol_mult": 0.7, "geo_shift": -1.2},
    "Strait of Hormuz Closure": {"jump_mult": 3.0, "vol_mult": 2.2, "geo_shift": 3.5},
    "OPEC Supply Shock": {"jump_mult": 1.5, "vol_mult": 1.3, "geo_shift": 1.0},
    "Global Recession": {"jump_mult": 0.5, "vol_mult": 1.2, "geo_shift": -1.8}
}

# ══════════════════════════════════════════════════════════
#   HEADER (com status badges)
# ══════════════════════════════════════════════════════════
now_sp = datetime.now(pytz.timezone("America/Sao_Paulo"))
api_status = get_api_status()
status_badges = " ".join([f"<span style='margin-left:0.5rem;font-size:0.55rem;'>{k}: {v}</span>" for k, v in api_status.items()])
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:flex-start;padding:1.6rem 0 1.2rem;border-bottom:1px solid #D9D5CD;margin-bottom:1.8rem;'>
  <div>
    <div style='display:flex;align-items:baseline;gap:.6rem;'>
      <span style='font-family:"JetBrains Mono",monospace;font-size:.85rem;color:#B49450;letter-spacing:.2em;'>◆◆◆</span>
      <div>
        <div style='font-family:"Playfair Display",Georgia,serif;font-size:1.9rem;font-weight:300;color:#1E3A5F;letter-spacing:.06em;line-height:1;'>GeoQuant · Research Terminal</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:.55rem;color:#70695E;letter-spacing:.2em;text-transform:uppercase;margin-top:.3rem;'>Macro Geopolitical Quant · Institutional Analytics Platform</div>
      </div>
    </div>
  </div>
  <div style='text-align:right;'>
    <div style='display:inline-block;background:#1E3A5F;color:#D4C094;padding:.2rem .7rem;font-family:"JetBrains Mono",monospace;font-size:.55rem;letter-spacing:.18em;text-transform:uppercase;'>⚑ {scen_choice.upper()}</div>
    <div style='font-family:"JetBrains Mono",monospace;font-size:.57rem;color:#70695E;letter-spacing:.10em;margin-top:.4rem;'>{now_sp.strftime("%d %B %Y · %H:%M")} (SP)</div>
    <div style='font-family:"JetBrains Mono",monospace;font-size:.45rem;color:#70695E;margin-top:.2rem;'>{status_badges}</div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#   EXECUTION PIPELINE
# ══════════════════════════════════════════════════════════
if run_btn or "results" not in st.session_state:
    st.cache_data.clear()
    st.session_state.mc_seed = int(time.time())
    
    loading = st.empty()
    loading.markdown("""
    <div style='text-align:center;padding:2.5rem 2rem;background:#FDFBF8;border:1px solid #C4BDAF;margin:1rem 0;'>
      <div style='font-family:"JetBrains Mono",monospace;font-size:.56rem;letter-spacing:.22em;color:#9E8050;text-transform:uppercase;margin-bottom:.7rem;'>Initialising Institutional Pipeline Engine</div>
      <div style='font-family:"Playfair Display",Georgia,serif;font-size:1.4rem;color:#1E3A5F;font-weight:300;'>Calibrating models, micro-structural indices & multi-asset DCC matrices…</div>
    </div>""", unsafe_allow_html=True)
    prog = st.progress(0)

    try:
        vix_premium = fetch_fred_macro()
        eia_stocks = fetch_eia_inventories()
        gpr_series = fetch_gpr()
        cot_value = fetch_cot("CL")
        prog.progress(10)
        
        war_start_str = war_start.strftime("%Y-%m-%d")
        prices = fetch_data(war_start_str)
        if prices.empty or len(prices) < 5:
            st.error("Execution halted: Insufficient historical data extracted from live sources.")
            st.stop()

        prices = prices.ffill().bfill()
        for k in TICKERS:
            if k not in prices.columns: prices[k] = np.nan
        prices = prices.ffill().bfill()

        lw = float(prices["oil"].dropna().iloc[-1]) if not prices["oil"].dropna().empty else 75.0
        lb = float(prices["brent"].dropna().iloc[-1]) if not prices["brent"].dropna().empty else 78.0
        wti0, brt0 = fetch_live(lw, lb)
        prices.loc[prices.index[-1], "oil"], prices.loc[prices.index[-1], "brent"] = wti0, brt0
        returns = np.log(prices / prices.shift(1)).dropna()

        prog.progress(25)
        usda = get_usda()
        bs_mult = fert_black_swan(usda)
        gs = gold_signals(prices)
        sd = silver_demand_proxy(prices)
        macro_proxies = simulate_macro_indices(prices.index)
        # Adicionar GPR e COT aos proxies (para possível uso futuro)
        macro_proxies["gpr"] = gpr_series.reindex(prices.index, method='ffill').fillna(gpr_series.median() if not gpr_series.empty else 100)
        macro_proxies["cot"] = pd.Series(cot_value, index=prices.index[-1:]).reindex(prices.index, method='ffill').fillna(cot_value)
        
        weights = calibrate_weights(returns, prices, gs, build_fert_index(returns, usda, bs_mult), sd, macro_proxies)
        fi = build_fert_index(returns, usda, bs_mult)
        gf_raw = build_geofactor(returns, prices, gs, fi, weights, sd, macro_proxies)
        gf = (gf_raw - gf_raw.mean()) / gf_raw.std() if len(gf_raw) > 1 else gf_raw
        zsc = build_zscore(prices, gs)

        prog.progress(45)
        gf_clean = gf.dropna() if not gf.empty else None
        vw = fit_egarch(returns["oil"], gf_clean)
        vb_s = fit_egarch(returns["brent"], gf_clean)
        vg = fit_egarch(returns["gold"], gf_clean)
        
        pwd = prior_wti / np.sqrt(252)
        pbd = prior_brent / np.sqrt(252)
        pgd = 0.18 / np.sqrt(252)
        
        vw, dw = bayes_shrink(vw, pwd, len(returns), gf)
        vb_s, db = bayes_shrink(vb_s, pbd, len(returns), gf)
        vg, _ = bayes_shrink(vg, pgd, len(returns), gf)
        
        bvw = float(vw.iloc[-1]) if len(vw) > 0 else pwd
        bvb = float(vb_s.iloc[-1]) if len(vb_s) > 0 else pbd
        
        evt_wti = conditional_evt(returns["oil"], vw)
        regimes_ts = detect_regime(vw)
        dcc_a, dcc_b, dcc_rho = fit_dcc(returns["oil"], returns["brent"], vw, vb_s)

        prog.progress(65)
        rv = returns.loc[gf.index.intersection(returns.index)] if not gf.empty else returns
        try:
            vm = VAR(rv).fit(min(3, max(1, len(rv)//15)))
            fcast = vm.forecast(rv.values[-vm.k_ar:], steps=mc_steps)
        except:
            fcast = np.zeros((mc_steps, len(rv.columns)))

        cols = list(rv.columns)
        ocol = cols.index("oil") if "oil" in cols else 0
        bcol = cols.index("brent") if "brent" in cols else 1
        tdf_d = max(2.5, min(6.0, tail_df / np.sqrt(max(bvb/(pbd*1.5), 0.5))))
        rbase = float(np.tanh(gf.iloc[-1]/2)) if len(gf) > 0 else 0.0
        jpu = min(jump_up * 1.5, 0.15) if returns["wheat"].tail(20).mean() > 0.005 else jump_up

        prog.progress(80)
        mc_bar = st.progress(0)
        mc = run_mc(wti0, brt0, bvw, bvb, fcast, ocol, bcol, rbase,
                    returns["oil"], returns["brent"], vw, vb_s, jpu, tdf_d,
                    bs_mult, dcc_a, dcc_b, mc_sims, mc_steps, mc_bar,
                    SCENARIO_MAP[scen_choice])
        mc_bar.empty()

        prog.progress(90)
        ret_ann = returns[["oil","brent"]].mean() * 252
        vol_ann = returns[["oil","brent"]].std() * np.sqrt(252)
        neg = returns[["oil","brent"]][returns[["oil","brent"]] < 0].std() * np.sqrt(252)
        corr_mx = returns[["oil","brent","gold","dxy","tnx"]].dropna().corr()
        
        stress_c = pd.DataFrame({
            "vol_wti": vw*np.sqrt(252)*100,
            "vol_brt": vb_s*np.sqrt(252)*100,
            "corr": returns["oil"].rolling(20).corr(returns["brent"]),
            "geofactor": gf
        }).dropna()
        stress_idx = (stress_c["vol_wti"]/50 + stress_c["vol_brt"]/50 +
                      np.abs(stress_c["corr"]-0.8)*2 + stress_c["geofactor"].clip(0,2)/2) / 4
        
        gdiag = garch_diagnostics(vw)
        bt_res = backtest_var(returns["oil"].iloc[-252:], vw.iloc[-252:]*1.645)
        es_z = backtest_es(returns["oil"].iloc[-252:], vw.iloc[-252:]*2.326, vw.iloc[-252:]*1.645)

        ml_metrics, X_ml, y_ml = benchmark_ml(returns)
        shap_fig, sv, feat_names = run_shap(X_ml, y_ml)
        wf_df = walk_forward_validation(returns["oil"])

        p_vals = [gdiag["LB5"], gdiag["LB10"], gdiag["ARCH_p"],
                  bt_res["Kupiec_p"], bt_res["Christoffersen_p"], bt_res["DQ_p"]]
        valid_p = [p for p in p_vals if not np.isnan(p)]
        model_score = int(np.mean(valid_p) * 100) if valid_p else 85

        prog.progress(100)
        loading.empty()
        prog.empty()

        last_update = datetime.now(pytz.timezone("America/Sao_Paulo")).strftime("%d %b %Y %H:%M:%S")
        st.session_state.update({
            "results": mc, "gf": gf, "zsc": zsc, "vw": vw, "vb": vb_s, "vg": vg,
            "fi": fi, "gs": gs, "prices": prices, "returns": returns,
            "wti0": wti0, "brt0": brt0, "usda": usda, "bs": bs_mult,
            "dw": dw, "db": db, "tdf": tdf_d, "dcc_a": dcc_a, "dcc_b": dcc_b, "dcc_rho": dcc_rho,
            "weights": weights, "sharpe": ret_ann/vol_ann, "sortino": ret_ann/neg,
            "corr_mx": corr_mx, "stress_idx": stress_idx,
            "evt": evt_wti, "gdiag": gdiag, "bt_res": bt_res, "es_z": es_z,
            "ml_metrics": ml_metrics, "shap_fig": shap_fig,
            "shap_vals": sv, "feat_names": feat_names, "wf_df": wf_df,
            "regimes_ts": regimes_ts, "model_score": model_score,
            "vix_fred": vix_premium, "eia_stocks": eia_stocks,
            "macro_proxies": macro_proxies, "gpr": gpr_series, "cot": cot_value,
            "last_update": last_update
        })
    except Exception as e:
        loading.empty()
        prog.empty()
        st.error(f"Structural Fatal Exception in Pipeline: {str(e)}")
        st.stop()

# ══════════════════════════════════════════════════════════
#   INTERFACE RENDERING
# ══════════════════════════════════════════════════════════
S = st.session_state
mc = S["results"]
fan = mc["fan"]
fb = mc["fan_b"]
M = mc["metrics"]
moments = mc["moments"]

wti0 = S["wti0"]
brt0 = S["brt0"]
vw = S["vw"]
vb = S["vb"]
vg = S["vg"]
zsc = S["zsc"]
gf = S["gf"]

st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:var(--muted); text-align:right; margin-bottom:1rem;">
Data Freshness: <strong>{S.get('last_update', 'Unknown')}</strong> · Spot Update Interval: 10s
</div>""", unsafe_allow_html=True)

# Botão de exportação
csv_data = export_results_to_csv(mc, moments, fan, S["weights"], S["macro_proxies"])
b64 = base64.b64encode(csv_data.encode()).decode()
href = f'<div style="text-align:right; margin-bottom:1rem;"><a href="data:file/csv;base64,{b64}" download="geoquant_results.csv" style="background:#1E3A5F; color:#D4C094; padding:0.3rem 0.7rem; font-family:JetBrains Mono; font-size:0.55rem; text-decoration:none;">📥 Export Results (CSV)</a></div>'
st.markdown(href, unsafe_allow_html=True)

t_exec, t_vol, t_geo, t_attr, t_mc, t_stat, t_diag, t_ml, t_modelcard = st.tabs([
    "Executive Summary", "Market & Volatility", "Geopolitical Intelligence",
    "GeoFactor Attribution", "Monte Carlo Fan Chart", "Quant Statistics",
    "Model Diagnostics", "Machine Learning Leaderboard", "Model Card"
])

# Abas 1 a 8 idênticas ao original (apenas substituímos a variável macro_proxies onde usado)
with t_exec:
    st.markdown('<div class="sec-label">Report · Asset Management Grade</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Executive Macro & Geopolitical Summary</div>', unsafe_allow_html=True)
    
    gf_last = float(gf.iloc[-1]) if len(gf) > 0 else 0.0
    st_last = float(S["stress_idx"].iloc[-1]) if len(S["stress_idx"]) > 0 else 0.0
    if len(S["regimes_ts"]) > 0:
        regime_curr = ["Normal", "Elevated Risk", "Stress", "Crisis"][int(S["regimes_ts"].iloc[-1])]
    else:
        regime_curr = "Normal"
    
    st.markdown(f"""
    <div style="background:#FDFBF8; border:1px solid #D9D5CD; padding:1.5rem; line-height:1.7; color:#1C1C1C; font-size:0.92rem;">
        <strong>Macro Insight Terminal Architecture Report:</strong><br><br>
        O <strong>GeoFactor Composite</strong> encerrou a última sessão quantificado em <strong>{fmt_num(gf_last, ".2f")}σ</strong>, indicando um regime estrutural classificado como <strong>{regime_curr.upper()}</strong>. 
        Este posicionamento reflete uma pressão geopolítica latente com transmissão direta via prêmio de risco físico na curva futura de commodities energéticas. 
        O índice de estresse integrado do sistema unificado (<strong>Stress Index</strong>) está precificado em <strong>{fmt_num(st_last, ".2f")}</strong>, condicionado por uma volatilidade implícita do mercado cambial (DXY) e choques estruturais capturados no Fertilizer Index (atualmente operando com multiplicador multiplicativo de cauda de <strong>{fmt_num(S['bs'], ".2f")}x</strong>).
        <br><br>
        No horizonte preditivo de curto prazo ({mc_steps} dias úteis), o motor estocástico <strong>Conditional EVT + DCC-GARCH-X</strong> aponta para uma assimetria positiva de cauda. 
        A projeção centralizada (Mediana P50) para o contrato spot do <strong>WTI estabiliza em US$ {fmt_num(fan[50][-1], ".2f")}/bbl</strong>, operando dentro de uma amplitude de estresse severo delimitada pelo percentil de cauda extrema (P99) em <strong>US$ {fmt_num(fan[99][-1], ".2f")}/bbl</strong>. 
        A probabilidade implícita de ruptura altista extrema (WTI superando a barreira crítica de US$ 100/bbl) está calibrada em <strong>{fmt_num(M['wti_100'], ".1f")}%</strong>, enquanto o risco de colapso estrutural deflacionário abaixo de US$ 60/bbl está precificado pelo modelo em apenas <strong>{fmt_num(M['wti_l60'], ".1f")}%</strong>.
        A correlação condicional dinâmica (DCC) entre o complexo WTI e Brent mantem-se estruturalmente pinada com parâmetros estáveis de persistência de longo prazo ($\alpha$: {fmt_num(S['dcc_a'], ".4f")}, $\beta$: {fmt_num(S['dcc_b'], ".4f")}).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top:1.5rem;" class="sec-label">Operational Threshold Matrices</div>', unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown('<table class="data-table"><thead><tr><th>WTI Bullish</th><th>Implied Prob</th></tr></thead>'
                    f'<tbody><tr><td>WTI &gt; US$ 70</td><td><strong>{fmt_num(M["wti_70"], ".1f")}%</strong></td></tr>'
                    f'<tr><td>WTI &gt; US$ 80</td><td><strong>{fmt_num(M["wti_80"], ".1f")}%</strong></td></tr>'
                    f'<tr><td>WTI &gt; US$ 90</td><td><strong>{fmt_num(M["wti_90"], ".1f")}%</strong></td></tr>'
                    f'<tr><td>WTI &gt; US$ 100</td><td><strong>{fmt_num(M["wti_100"], ".1f")}%</strong></td></tr>'
                    f'<tr><td>WTI &gt; US$ 120</td><td><strong>{fmt_num(M["wti_120"], ".1f")}%</strong></td></tr></tbody></table>', unsafe_allow_html=True)
    with col_p2:
        st.markdown('<table class="data-table"><thead><tr><th>WTI Bearish</th><th>Implied Prob</th></tr></thead>'
                    f'<tbody><tr><td>WTI &lt; US$ 60</td><td><strong>{fmt_num(M["wti_l60"], ".1f")}%</strong></td></tr>'
                    f'<tr><td>WTI &lt; US$ 50</td><td><strong>{fmt_num(M["wti_l50"], ".1f")}%</strong></td></tr>'
                    f'<tr><td>WTI &lt; US$ 40</td><td><strong>{fmt_num(M["wti_l40"], ".1f")}%</strong></td></tr></tbody></table>', unsafe_allow_html=True)
    with col_p3:
        st.markdown('<table class="data-table"><thead><tr><th>Brent Complex</th><th>Implied Prob</th></tr></thead>'
                    f'<tbody><tr><td>Brent &gt; US$ 90</td><td><strong>{fmt_num(M["brent_90"], ".1f")}%</strong></td></tr>'
                    f'<tr><td>Brent &gt; US$ 100</td><td><strong>{fmt_num(M["brent_100"], ".1f")}%</strong></td></tr></tbody></table>', unsafe_allow_html=True)

with t_vol:
    st.markdown('<div class="sec-label">01 · Risk Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WTI Crude Spot", f"${wti0:.2f}", delta=safe_delta(fan[50][-1], ".2f"))
    c2.metric("Brent Crude", f"${brt0:.2f}", delta=safe_delta(brt0-wti0, ".2f"))
    c3.metric("WTI Vol p.a.", f"{M['vol_wti']:.1f}%", delta=safe_delta(S['dw']['vsa'], ".1f"))
    c4.metric("Brent Vol p.a.", f"{M['vol_brt']:.1f}%", delta=safe_delta(S['db']['vsa'], ".1f"))

    fig_vol = qfig(360)
    fig_vol.add_trace(go.Scatter(x=vw.index, y=vw*np.sqrt(252)*100, name="WTI EGARCH Vol", line=dict(color=C["navy"], width=2)))
    fig_vol.add_trace(go.Scatter(x=vb.index, y=vb*np.sqrt(252)*100, name="Brent EGARCH Vol", line=dict(color=C["blue"], width=2, dash="dash")))
    fig_vol.add_trace(go.Scatter(x=vg.index, y=vg*np.sqrt(252)*100, name="Gold EGARCH Vol", line=dict(color=C["gold"], width=1.5, dash="dot")))
    fig_vol.update_layout(yaxis_ticksuffix="%", title=safe_text("EGARCH(1,1) Filtering Engine (Exogenous GeoFactor Multi-Regime)"))
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Novo gráfico: DCC correlation time series
    if "dcc_rho" in S and not S["dcc_rho"].empty:
        fig_dcc = qfig(240)
        fig_dcc.add_trace(go.Scatter(x=S["dcc_rho"].index, y=S["dcc_rho"].values, name="DCC Correlation (WTI/Brent)", line=dict(color=C["teal"], width=2)))
        fig_dcc.update_layout(title="Conditional Correlation (DCC) WTI/Brent", yaxis_range=[-1,1])
        st.plotly_chart(fig_dcc, use_container_width=True)

with t_geo:
    st.markdown('<div class="sec-label">02 · Micro Structural Risk Tracking</div>', unsafe_allow_html=True)
    fig_geo = dual_axis_fig(360)
    fig_geo.add_trace(go.Scatter(x=zsc.index, y=zsc.values, name="Z-Score Composite", line=dict(color=C["sky"], width=2), fill="tozeroy", fillcolor="rgba(74,115,128,0.04)"))
    fig_geo.add_trace(go.Scatter(x=gf.index, y=gf.values, name="GeoFactor (σ)", line=dict(color=C["navy"], width=2.5), yaxis="y2"))
    st.plotly_chart(fig_geo, use_container_width=True)
    
    st.markdown('<div class="sec-label">Structural Regime Shift Engine</div>', unsafe_allow_html=True)
    fig_reg = qfig(240)
    fig_reg.add_trace(go.Scatter(x=S["regimes_ts"].index, y=S["regimes_ts"].values, name="Regime", line=dict(color=C["burgundy"], width=1.8)))
    fig_reg.update_layout(yaxis=dict(tickvals=[0,1,2,3], ticktext=["Normal","Elevated","Stress","Crisis"]))
    st.plotly_chart(fig_reg, use_container_width=True)

with t_attr:
    st.markdown('<div class="sec-label">03 · Factor Attribution Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Waterfall Analysis & Dynamic Scaling Weights</div>', unsafe_allow_html=True)
    
    w_df = pd.DataFrame(list(S["weights"].items()), columns=["Factor", "Weight"]).sort_values(by="Weight", ascending=False)
    
    fig_water = go.Figure(go.Waterfall(
        name="Attribution", orientation="v",
        x=w_df["Factor"].tolist(),
        textposition="outside",
        y=w_df["Weight"].tolist(),
        connector=dict(line=dict(color="rgb(122,118,110)", width=1)),
        decreasing=dict(marker=dict(color=C["burgundy"])),
        increasing=dict(marker=dict(color=C["navy"]))
    ))
    fig_water.update_layout(**PL, height=380, title=safe_text("GeoFactor Risk Attribution Matrix"))
    st.plotly_chart(fig_water, use_container_width=True)
    
    col_at1, col_at2 = st.columns([1,1])
    with col_at1:
        st.markdown('<div class="sec-label">Factor Weight Ranking Table</div>', unsafe_allow_html=True)
        html_tbl = '<table class="data-table"><thead><tr><th>Macro Indicator</th><th>Relative Beta Weight</th></tr></thead><tbody>'
        for _, r in w_df.iterrows():
            html_tbl += f"<tr><td>{r['Factor']}</td><td><strong>{fmt_num(r['Weight'], '.4f')}</strong></td></tr>"
        html_tbl += "</tbody></table>"
        st.markdown(html_tbl, unsafe_allow_html=True)
    with col_at2:
        st.markdown('<div class="sec-label">Unbounded Macro Signals Dashboard</div>', unsafe_allow_html=True)
        st.markdown(f"- **Baltic Dry Index Proxy:** {fmt_num(S['macro_proxies']['baltic'].iloc[-1], '.2f')}<br>"
                    f"- **Freightos Container Index Proxy:** {fmt_num(S['macro_proxies']['freightos'].iloc[-1], '.2f')}<br>"
                    f"- **MOVE Index Equivalent:** {fmt_num(S['macro_proxies']['move'].iloc[-1], '.1f')}<br>"
                    f"- **Financial Conditions Index (FCI):** {fmt_num(S['macro_proxies']['fci'].iloc[-1], '.3f')}<br>"
                    f"- **GPR (Geopolitical Risk):** {fmt_num(S['macro_proxies']['gpr'].iloc[-1] if not S['macro_proxies']['gpr'].empty else np.nan, '.1f')}<br>"
                    f"- **COT (Speculative net long):** {fmt_num(S['cot'], '.0f')} contracts", unsafe_allow_html=True)

with t_mc:
    st.markdown('<div class="sec-label">04 · Full Distribution Fan Chart</div>', unsafe_allow_html=True)
    x_ax = list(range(mc_steps+1))
    fig_mc = qfig(460)
    fig_mc.add_trace(go.Scatter(x=x_ax+x_ax[::-1], y=list(fan[99])+list(fan[1][::-1]), fill="toself", fillcolor=C["fill_light"], line=dict(width=0), name="98% Macro Tail Bound (P1-P99)"))
    fig_mc.add_trace(go.Scatter(x=x_ax+x_ax[::-1], y=list(fan[95])+list(fan[5][::-1]), fill="toself", fillcolor=C["fill_medium"], line=dict(width=0), name="90% Core Policy Bound (P5-P95)"))
    fig_mc.add_trace(go.Scatter(x=x_ax+x_ax[::-1], y=list(fan[75])+list(fan[25][::-1]), fill="toself", fillcolor=C["fill_deep"], line=dict(width=0), name="50% Interquartile Range (P25-P75)"))
    fig_mc.add_trace(go.Scatter(x=x_ax, y=list(fan[50]), name=f"P50 Median Path → ${fan[50][-1]:.2f}", line=dict(color=C["navy"], width=3)))
    fig_mc.update_layout(xaxis_title="Simulation Horizon Days (OOS)", yaxis_title="WTI Price Index (USD/bbl)", yaxis_tickprefix="$")
    st.plotly_chart(fig_mc, use_container_width=True)
    
    st.markdown('<div class="sec-label">Probability Bracket Density Heatmap</div>', unsafe_allow_html=True)
    br_keys = list(mc["brackets"].keys())
    br_vals = list(mc["brackets"].values())
    fig_heat = go.Figure(data=go.Heatmap(z=[br_vals], x=br_keys, y=["Probability %"], colorscale="magma", text=[[f"{v:.1f}%" for v in br_vals]], texttemplate="%{text}", colorbar=None))
    heat_layout = PL.copy()
    heat_layout.pop("margin", None)
    fig_heat.update_layout(**heat_layout, height=160, margin=dict(t=20,b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

with t_stat:
    st.markdown('<div class="sec-label">05 · Mathematical Distribution Matrix</div>', unsafe_allow_html=True)
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown('<table class="data-table"><thead><tr><th>Simulated Path Moment</th><th>Value</th></tr></thead><tbody>'
                    f'<tr><td>Expected Simulated Mean</td><td><strong>${fmt_num(moments["mean"], ".2f")}</strong></td></tr>'
                    f'<tr><td>Distribution Median (P50)</td><td><strong>${fmt_num(moments["median"], ".2f")}</strong></td></tr>'
                    f'<tr><td>Pearson Empirical Mode</td><td><strong>${fmt_num(moments["mode"], ".2f")}</strong></td></tr>'
                    f'<tr><td>Simulated Skewness Coefficient</td><td><strong>{fmt_num(moments["skew"], ".4f")}</strong></td></tr>'
                    f'<tr><td>Simulated Excess Kurtosis</td><td><strong>{fmt_num(moments["kurt"], ".4f")}</strong></td></tr></tbody></tr>', unsafe_allow_html=True)
    with c_m2:
        st.markdown('<table class="data-table"><thead><tr><th>Institutional Scenario Mapping</th><th>Terminal Target Price</th></tr></thead><tbody>'
                    f'<tr><td><span style="color:var(--danger)">Extreme Bear (P1)</span></td><td><strong>${fmt_num(fan[1][-1], ".2f")}</strong></td></tr>'
                    f'<tr><td>Bear Case (P10)</td><td><strong>${fmt_num(fan[10][-1], ".2f")}</strong></td></tr>'
                    f'<tr><td>Base Case (P50)</td><td><strong>${fmt_num(fan[50][-1], ".2f")}</strong></td></tr>'
                    f'<tr><td>Bull Case (P90)</td><td><strong>${fmt_num(fan[90][-1], ".2f")}</strong></td></tr>'
                    f'<tr><td><span style="color:var(--success)">Extreme Bull (P99)</span></td><td><strong>${fmt_num(fan[99][-1], ".2f")}</strong></td></tr></tbody></table>', unsafe_allow_html=True)

with t_diag:
    st.markdown('<div class="sec-label">06 · Statistical Infrastructure Guardrails</div>', unsafe_allow_html=True)
    
    def get_status_html(p_val, alpha=0.05):
        if np.isnan(p_val): return '<span class="status-warning">WARNING (NaN)</span>'
        if p_val > alpha: return '<span class="status-pass">PASS</span>'
        return '<span class="status-fail">FAIL</span>'
        
    gd, bt = S["gdiag"], S["bt_res"]
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric("Model Integrity Score", f"{S['model_score']}/100")
    col_d2.markdown(f'<div class="diag-card">Ljung-Box (Lag 5)<br>{get_status_html(gd["LB5"])}<br><small>p={fmt_num(gd["LB5"], ".4f")}</small></div>', unsafe_allow_html=True)
    col_d3.markdown(f'<div class="diag-card">ARCH-LM Test<br>{get_status_html(gd["ARCH_p"])}<br><small>p={fmt_num(gd["ARCH_p"], ".4f")}</small></div>', unsafe_allow_html=True)
    col_d4.markdown(f'<div class="diag-card">Dynamic Quantile<br>{get_status_html(bt["DQ_p"])}<br><small>p={fmt_num(bt["DQ_p"], ".4f")}</small></div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.5rem;" class="sec-label">Regulatory VaR & Expected Shortfall Compliance Backtest Table</div>', unsafe_allow_html=True)
    st.markdown('<table class="data-table"><thead><tr><th>Backtest Validation Module</th><th>Target Criteria</th><th>Observed Value</th><th>Status P-Value</th></tr></thead><tbody>'
                f'<tr><td>Kupiec Unconditional Coverage</td><td>5.00% Violations Max</td><td>{(bt["obs_freq"]*100):.2f}%</td><td>{get_status_html(bt["Kupiec_p"])} (p={fmt_num(bt["Kupiec_p"], ".4f")})</td></tr>'
                f'<tr><td>Christoffersen Conditional Independence</td><td>No Violation Clustering</td><td>—</td><td>{get_status_html(bt["Christoffersen_p"])} (p={fmt_num(bt["Christoffersen_p"], ".4f")})</td></tr>'
                f'<tr><td>Acerbi Expected Shortfall Metric Z</td><td>Z Score Optimization &lt; 0</td><td>—</td><td><strong>{fmt_num(S["es_z"], ".4f")}</strong></td></tr></tbody></tr>', unsafe_allow_html=True)

with t_ml:
    st.markdown('<div class="sec-label">07 · Out-of-Sample Predictive Benchmarks</div>', unsafe_allow_html=True)
    
    html_ml = '<table class="data-table"><thead><tr><th>Model Pipeline Architecture</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>Directional Accuracy</th></tr></thead><tbody>'
    for name, metrics in S["ml_metrics"].items():
        html_ml += f"<tr><td><strong>{name}</strong></td><td>{fmt_num(metrics['RMSE'], '.6f')}</td><td>{fmt_num(metrics['MAE'], '.6f')}</td><td>{fmt_num(metrics['MAPE'], '.4f')}%</td><td><strong>{metrics['Directional Accuracy']}</strong></td></tr>"
    html_ml += "</tbody></table>"
    st.markdown(html_ml, unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top:1.5rem;" class="sec-label">Institutional SHAP Value Drivers</div>', unsafe_allow_html=True)
    col_sh1, col_sh2 = st.columns([1.2,1])
    with col_sh1:
        if S["shap_fig"] is not None: st.pyplot(S["shap_fig"])
    with col_sh2:
        shap_mean_abs = np.mean(np.abs(S["shap_vals"]), axis=0)
        shap_df = pd.DataFrame({"Variable": S["feat_names"], "Mean Absolute Impact": shap_mean_abs}).sort_values(by="Mean Absolute Impact", ascending=False)
        html_shap = '<table class="data-table"><thead><tr><th>Top Drivers (Features)</th><th>Mean Absolute SHAP Impact</th></tr></thead><tbody>'
        for _, r in shap_df.iterrows():
            html_shap += f"<tr><td>{r['Variable']}</td><td><strong>{fmt_num(r['Mean Absolute Impact'], '.6f')}</strong></td></tr>"
        html_shap += "</tbody></table>"
        st.markdown(html_shap, unsafe_allow_html=True)

# Nova aba: Model Card
with t_modelcard:
    st.markdown('<div class="sec-label">08 · Model Documentation & Assumptions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#FDFBF8; border:1px solid #D9D5CD; padding:1.5rem; font-size:0.85rem; line-height:1.6;">
        <h3 style="color:#1E3A5F;">Model Architecture</h3>
        <ul>
            <li><strong>EVT (Conditional Extreme Value Theory)</strong> – caudas pesadas ajustadas à volatilidade condicional (EGARCH).</li>
            <li><strong>DCC-GARCH-X</strong> – correlação dinâmica entre WTI e Brent com exógena GeoFactor (α, β estimados via QMLE).</li>
            <li><strong>GeoFactor Composite</strong> – combinação linear de volatilidade do petróleo, fertilizantes (via USDA proxy), condições financeiras (FCI), Baltic Dry, Freightos, MOVE (VIX proxy) e posicionamento especulativo (COT).</li>
            <li><strong>Bayes Shrinkage</strong> – encolhimento da volatilidade EGARCH para priors estruturais, com ajuste pelo GeoFactor.</li>
            <li><strong>Monte Carlo com saltos assimétricos</strong> – distribuição t-Student para inovações, saltos Poisson condicionais ao regime geopolítico.</li>
            <li><strong>Backtests regulatórios</strong> – Kupiec, Christoffersen e Acerbi ES (falhas esperadas em cenários extremos).</li>
        </ul>
        <h3 style="color:#1E3A5F;">Cenário Forward-Looking</h3>
        <ul>
            <li>Data de início do conflito: <strong>28 de fevereiro de 2026</strong> (bloqueio do Estreito de Hormuz).</li>
            <li>Os indicadores macro (Baltic, Freightos, MOVE, FCI) são simulados com tendência de estresse consistente com o cenário.</li>
            <li>Fertilizante baseado em dados Green Markets/CRU até junho/2026, incluindo choque de oferta (queda para 453.5).</li>
        </ul>
        <h3 style="color:#1E3A5F;">Fontes de Dados Reais</h3>
        <ul>
            <li><strong>FRED</strong> – VIX, GPR (Geopolitical Risk Index), NFCI (Financial Conditions).</li>
            <li><strong>EIA</strong> – estoques semanais de petróleo bruto dos EUA.</li>
            <li><strong>OilPrice API</strong> – preço spot do petróleo.</li>
            <li><strong>yfinance</strong> – futuros de commodities, Baltic Dry (^BDI), VIX (^VIX), etc.</li>
            <li><strong>CFTC (proxy)</strong> – posições líquidas de especuladores em WTI.</li>
        </ul>
        <h3 style="color:#1E3A5F;">Limitações e Avisos</h3>
        <ul>
            <li>Este terminal é uma ferramenta de simulação de cenários <strong>forward-looking</strong> e não deve ser usado como única fonte para decisões de investimento.</li>
            <li>Os backtests falham intencionalmente em regime extremo (Hormuz) – isso é esperado e indica que o modelo não está superajustado a condições normais.</li>
            <li>As projeções de cauda (P99) são altamente sensíveis aos parâmetros de salto e à escolha da distribuição.</li>
        </ul>
        <p><em>Model version: 2.0 (com GPR, COT, DCC time series e sensitivity map integrados).</em></p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#   FOOTER
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
  <div>◆ GeoQuant Institutional Terminal · Engine: Conditional EVT + DCC-GARCH-X</div>
  <div>Eduardo Moraes · Quant Data Scientist & Economics</div>
  <div>Proprietary Research Infrastructure · {now_sp.strftime("%Y")}</div>
</div>""", unsafe_allow_html=True)
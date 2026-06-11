"""
GeoQuant – Institutional Macro Research Terminal (v2.0)
EGARCH + Conditional EVT + DCC + GeoFactor + OVX + Regime-Aware
Eduardo Moraes | Quant Data Scientist & Economics
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    "oil": "CL=F", "brent": "BZ=F", "natgas": "NG=F",
    "gold": "GC=F", "silver": "SI=F", "copper": "HG=F",
    "wheat": "ZW=F", "corn": "ZC=F", "soy": "ZS=F",
    "dxy": "DX-Y.NYB", "eur": "EURUSD=X", "tnx": "^TNX",
    "ovx": "^OVX",
}
GEO_W = {"oil_vol":0.20,"gold":0.08,"gold_real":0.08,"dxy":-0.10,"spread":0.08,
          "fert":0.20,"wheat":0.06,"copper":0.04,"natgas_vol":0.06,"ovx":0.10}

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
    navy="#1E3A5F", navy_light="#2A5080", gold="#B49450", gold_light="#D4C094",
    burgundy="#7B3F3F", teal="#2B5F5F", sage="#4A5D4A", gray="#5A554F",
    silver="#9A958A", sky="#4A7380", rust="#8B5A3A",
    fill_light="rgba(30,58,95,0.06)", fill_medium="rgba(30,58,95,0.12)",
)

def qfig(h=420):
    fig = go.Figure(); fig.update_layout(**PL, height=h); return fig

def dual_axis_fig(h=380):
    fig = go.Figure()
    fig.update_layout(**PL, height=h,
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    linecolor="#D9D5CD", zeroline=False,
                    tickfont=dict(size=10, family="JetBrains Mono,monospace", color="#5A554F")))
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
        color:#7A766E;letter-spacing:.12em;margin-top:.2rem;'>Institutional Risk Infrastructure</div>
    </div>""", unsafe_allow_html=True)

    def slabel(t):
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.5rem;letter-spacing:.2em;'
                    f'color:#B49450;text-transform:uppercase;margin:.8rem 0 .3rem;">{t}</div>',
                    unsafe_allow_html=True)
    def ssep():
        st.markdown('<div style="height:1px;background:#D9D5CD;margin:.5rem 0;"></div>',
                    unsafe_allow_html=True)

    slabel("· Simulation")
    mc_sims = st.slider("Monte Carlo paths", 5_000, 50_000, 10_000, 1_000)
    mc_steps = st.slider("Horizon (days)", 5, 60, 10, 1)
    ssep()
    slabel("· Regime Detection")
    vol_threshold = st.slider("Vol threshold (σ)", 1.0, 3.0, 2.0, 0.1)
    ssep()
    slabel("· Data Window")
    data_start = st.date_input("Start date", value=datetime(2018, 1, 1))
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
        EGARCH + Conditional EVT + DCC · Institutional Risk Intelligence</div>
      </div>
    </div>
  </div>
  <div style='text-align:right;'>
    <div style='display:inline-block;background:#1E3A5F;color:#D4C094;padding:.2rem .7rem;
    font-family:"JetBrains Mono",monospace;font-size:.5rem;letter-spacing:.16em;text-transform:uppercase;'>
    ⚑ GEOPOLITICAL AWARE</div>
    <div style='font-family:"JetBrains Mono",monospace;font-size:.52rem;color:#5A554F;
    letter-spacing:.1em;margin-top:.3rem;line-height:1.8;'>
    {now_sp.strftime("%d %B %Y · %H:%M")} (SP)<br>EGARCH Framework · Conditional EVT
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#   QUANT ENGINE (Robust)
# ══════════════════════════════════════════════════════════
def rolling_zscore(s, w=60):
    return (s - s.rolling(w).mean()) / s.rolling(w).std().replace(0, np.nan)

def fill_gaps(s):
    s = s.copy(); valid = s.notna()
    if valid.sum() < 2: return s.ffill()
    try:
        x = s.index[valid].astype(np.int64)
        f = pd.Series(PchipInterpolator(x, s[valid].values)(s.index.astype(np.int64)), index=s.index)
        f[valid] = s[valid]; return f
    except: return s.ffill()

def fit_egarch(ret, exog=None):
    """EGARCH(1,1) com inovação skew-t. Retorna vol diária alinhada com ret."""
    r = ret.dropna()
    if len(r) < 50:
        return pd.Series(r.std(), index=ret.index).ffill().bfill()
    try:
        rc = r * 100
        if exog is not None:
            common = rc.index.intersection(exog.dropna().index)
            if len(common) < 50:
                exog = None
            else:
                rc = rc.loc[common]
                xc = exog.loc[common]
                model = arch_model(rc, x=xc, mean="Constant", vol="EGARCH", p=1, q=1, dist="skewt")
        else:
            model = arch_model(rc, mean="Constant", vol="EGARCH", p=1, q=1, dist="skewt")
        res = model.fit(disp="off")
        vol = res.conditional_volatility / 100
        vol = vol.reindex(ret.index).ffill().bfill()
        vol[vol <= 0] = vol[vol > 0].min()
        return vol
    except:
        return pd.Series(r.rolling(20).std().mean(), index=ret.index).ffill().bfill()

def conditional_evt(returns, vol, q=0.95, min_obs=30):
    """EVT nos resíduos padronizados. Retorna parâmetros ou None se impossível."""
    common = returns.dropna().index.intersection(vol.dropna().index)
    if len(common) < min_obs:
        return None
    r = returns.loc[common]
    v = vol.loc[common].replace(0, np.nan)
    resid = (r / v).dropna()
    resid = resid[np.isfinite(resid)]
    if len(resid) < min_obs:
        return None
    th_up = np.percentile(resid, q*100)
    th_lo = np.percentile(resid, (1-q)*100)
    exc_up = resid[resid > th_up] - th_up
    exc_lo = -resid[resid < th_lo] - th_lo
    shape_up = scale_up = shape_lo = scale_lo = 0.0
    if len(exc_up) >= 10:
        shape_up, _, scale_up = stats.genpareto.fit(exc_up)
    else:
        shape_up, scale_up = 0.2, exc_up.std() if len(exc_up)>0 else 0.1
    if len(exc_lo) >= 10:
        shape_lo, _, scale_lo = stats.genpareto.fit(exc_lo)
    else:
        shape_lo, scale_lo = 0.2, exc_lo.std() if len(exc_lo)>0 else 0.1
    return {"upper": (shape_up, scale_up, th_up), "lower": (shape_lo, scale_lo, th_lo), "resid": resid}

def detect_regime(vol, threshold=2.0):
    """Detecta regime de estresse baseado em vol diária vs média móvel."""
    v = vol.dropna()
    if len(v) < 20:
        return pd.Series(0, index=vol.index)
    mean = v.rolling(60, min_periods=20).mean()
    std = v.rolling(60, min_periods=20).std().replace(0, 1e-8)
    regime = (v > mean + threshold * std).astype(int)
    return regime.reindex(vol.index).fillna(0).astype(int)

def build_geofactor(returns, prices, ovx, weights=None):
    if weights is None: weights = GEO_W
    spread = (prices["brent"] - prices["oil"]) / prices["brent"].replace(0, np.nan)
    geo = pd.Series(0.0, index=returns.index)
    if "oil" in returns.columns:
        geo += weights.get("oil_vol",0) * returns["oil"].rolling(20).std()
    if "gold" in returns.columns:
        geo += weights.get("gold",0) * returns["gold"].rolling(20).mean()
    if "dxy" in returns.columns:
        geo += weights.get("dxy",0) * returns["dxy"].rolling(20).mean()
    if "wheat" in returns.columns:
        geo += weights.get("wheat",0) * returns["wheat"].rolling(20).mean()
    if "copper" in returns.columns:
        geo += weights.get("copper",0) * returns["copper"].rolling(20).mean()
    if "natgas" in returns.columns:
        geo += weights.get("natgas_vol",0) * returns["natgas"].rolling(20).std()
    if "brent" in prices.columns and "oil" in prices.columns:
        geo += weights.get("spread",0) * spread.rolling(20).mean()
    if ovx is not None:
        ovx_ret = ovx.pct_change().rolling(20).mean()
        geo += weights.get("ovx",0) * ovx_ret.reindex(geo.index, method="ffill").fillna(0)
    return geo.dropna()

def gold_signals(prices):
    silver = prices["silver"].replace(0, np.nan)
    if silver.median() > 500: silver /= 100
    gr = prices["gold"] / (1 + prices["tnx"].replace(0, np.nan)/100*5.0)
    sg = silver / prices["gold"].replace(0, np.nan)
    return {"gold_real": gr, "silver_gold": sg,
            "gold_real_ret_roll": np.log(gr/gr.shift(1)).rolling(20).mean(),
            "silver_gold_roll": np.log(sg/sg.shift(1)).rolling(20).mean()}

def fit_dcc(rw, rb, vw, vb):
    common = rw.index.intersection(rb.index).intersection(vw.index).intersection(vb.index)
    ew = (rw[common] / vw[common]).dropna(); eb = (rb[common] / vb[common]).dropna()
    c2 = ew.index.intersection(eb.index)
    e = np.column_stack([np.clip(ew[c2], -3, 3), np.clip(eb[c2], -3, 3)])
    Qb = np.cov(e, rowvar=False)
    def nll(p):
        a,b=p
        if a<=0 or b<=0 or a+b>=1: return 1e10
        Qt=Qb.copy(); ll=0.0
        for t in range(1,len(e)):
            Qt = (1-a-b)*Qb + a*np.outer(e[t-1],e[t-1]) + b*Qt
            d=np.sqrt(np.diag(Qt)); d[d==0]=1e-8
            Rt=Qt/np.outer(d,d); Rt=np.clip(Rt,-0.9999,0.9999)
            try:
                L=np.linalg.cholesky(Rt); z=np.linalg.solve(L,e[t])
                ll+=-0.5*np.sum(z**2)-np.sum(np.log(np.diag(L)))
            except: return 1e10
        return -ll
    res=optimize.minimize(nll,[0.05,0.93],bounds=[(1e-4,0.3),(0.7,0.9999)],method="L-BFGS-B")
    a,b=res.x
    return (0.05,0.93) if a+b>=1 else (float(a),float(b))

def run_mc(wti0, brt0, vol_wti, vol_brt, dcc_a, dcc_b, regime_prob, steps, sims):
    np.random.seed(42)
    rho_base = np.clip(dcc_b/(1-dcc_a), -0.9, 0.9) if dcc_a!=0 else 0.9
    pw = np.zeros((sims, steps+1)); pb = np.zeros((sims, steps+1))
    pw[:,0]=wti0; pb[:,0]=brt0
    prob = max(0, min(1, regime_prob))
    for t in range(steps):
        rho = rho_base + 0.2*(2*prob-1)
        z = np.random.standard_t(5, (sims,2))
        zw = z[:,0]; zb = rho*z[:,0] + np.sqrt(max(0,1-rho**2))*z[:,1]
        jump = np.random.binomial(1, prob*0.1, sims) * np.random.exponential(0.05, sims)
        ret_w = np.clip(zw * vol_wti * np.sqrt(1/252) + jump, -0.15, 0.15)
        ret_b = np.clip(zb * vol_brt * np.sqrt(1/252) + jump*0.9, -0.15, 0.15)
        pw[:,t+1] = pw[:,t] * (1+ret_w)
        pb[:,t+1] = pb[:,t] * (1+ret_b)
    fan = {p: np.percentile(pw, p, axis=0) for p in [5,25,50,75,95]}
    fb = {p: np.percentile(pb, p, axis=0) for p in [5,25,50,75,95]}
    return fan, fb, pw

def backtest_var(returns, var_forecast, alpha=0.05):
    common = returns.index.intersection(var_forecast.index)
    if len(common) < 20: return None
    r = returns.loc[common]; v = var_forecast.loc[common]
    viol = (r < -v).astype(int)
    n=len(viol); nv=viol.sum(); pe=alpha; po=nv/n
    if nv>0 and nv<n:
        LR = -2*np.log(((1-pe)**(n-nv)*pe**nv)/((1-po)**(n-nv)*po**nv))
        kp = 1-chi2.cdf(LR,1)
    else: kp=0.5
    if n>1:
        n00=((viol[:-1]==0)&(viol[1:]==0)).sum(); n01=((viol[:-1]==0)&(viol[1:]==1)).sum()
        n10=((viol[:-1]==1)&(viol[1:]==0)).sum(); n11=((viol[:-1]==1)&(viol[1:]==1)).sum()
        p01=n01/(n00+n01) if (n00+n01)>0 else 0
        p11=n11/(n10+n11) if (n10+n11)>0 else 0
        LRc=-2*np.log(((1-pe)**(n-1-(n01+n11))*pe**(n01+n11))/
                     ((1-p01)**n00*p01**n01*(1-p11)**n10*p11**n11)) if (n01+n11)>0 else 0
        cp=1-chi2.cdf(LRc,1) if LRc>0 else 0.5
    else: cp=0.5
    X=pd.DataFrame({"const":1,"hit":viol.shift(1).fillna(0)})
    try: dq=1-chi2.cdf(Logit(viol,X).fit(disp=0).llr,X.shape[1])
    except: dq=1.0
    return {"Kupiec_p":kp,"Christoffersen_p":cp,"DQ_p":dq,"violations":int(nv),"freq":po}

def walk_forward_validation(returns, train_days=504, step_days=63):
    results=[]
    for start in range(0, len(returns)-train_days-21, step_days):
        train = returns.iloc[start:start+train_days]
        test = returns.iloc[start+train_days:start+train_days+21]
        mu = train.mean()
        rmse = np.sqrt(((mu - test)**2).mean())
        results.append((returns.index[start], returns.index[start+train_days+20], rmse))
    if results:
        return pd.DataFrame(results, columns=["Train Start","Test End","RMSE"])
    return pd.DataFrame()

@st.cache_data(ttl=900)
def fetch_data(start="2018-01-01"):
    df = yf.download(list(TICKERS.values()), start=start, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    df.columns = list(TICKERS.keys())
    return df.ffill().bfill()

# ══════════════════════════════════════════════════════════
#   PIPELINE
# ══════════════════════════════════════════════════════════
if run_btn or "results" not in st.session_state:
    with st.spinner("Calibrando modelos..."):
        prices = fetch_data(data_start.strftime("%Y-%m-%d"))
        prices = prices.ffill().bfill()
        returns = np.log(prices/prices.shift(1)).dropna()
        ovx = prices.get("ovx")

        gf = build_geofactor(returns, prices, ovx)
        gf_std = (gf - gf.mean()) / gf.std() if len(gf)>1 else gf

        vw = fit_egarch(returns["oil"], gf_std)
        vb = fit_egarch(returns["brent"], gf_std)

        evt_wti = conditional_evt(returns["oil"], vw)
        evt_brt = conditional_evt(returns["brent"], vb)

        regime = detect_regime(vw, vol_threshold)
        if regime.sum() == 0:
            regime.iloc[-1] = 0  # garante série não vazia

        dcc_a, dcc_b = fit_dcc(returns["oil"], returns["brent"], vw, vb)

        fan, fb, _ = run_mc(
            wti0=float(prices["oil"].iloc[-1]),
            brt0=float(prices["brent"].iloc[-1]),
            vol_wti=float(vw.iloc[-1]*np.sqrt(252)),
            vol_brt=float(vb.iloc[-1]*np.sqrt(252)),
            dcc_a=dcc_a, dcc_b=dcc_b,
            regime_prob=float(regime.iloc[-1]) if len(regime)>0 else 0.0,
            steps=mc_steps, sims=mc_sims
        )

        var_95 = vw * 1.645
        bt_all = backtest_var(returns["oil"], var_95)
        if regime.sum() > 0 and bt_all is not None:
            bt_calm = backtest_var(returns["oil"][regime==0], var_95[regime==0])
            bt_stress = backtest_var(returns["oil"][regime==1], var_95[regime==1])
        else:
            bt_calm = bt_stress = None

        features = pd.DataFrame({
            "oil_lag1": returns["oil"].shift(1),
            "brent_lag1": returns["brent"].shift(1),
            "vol_egarch": vw,
            "gf": gf_std,
            "spread": (prices["brent"] - prices["oil"]) / prices["brent"],
        }).dropna()
        target = returns["oil"].loc[features.index]
        split = int(len(features)*0.8)
        X_tr, X_te = features.iloc[:split], features.iloc[split:]
        y_tr, y_te = target.iloc[:split], target.iloc[split:]
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
        }
        ml_scores = {}
        for name, mdl in models.items():
            mdl.fit(X_tr, y_tr)
            pred = mdl.predict(X_te)
            ml_scores[name] = {
                "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
                "MAE": mean_absolute_error(y_te, pred)
            }

        wf = walk_forward_validation(returns["oil"])

        st.session_state.update({
            "prices": prices, "returns": returns, "vw": vw, "vb": vb,
            "fan": fan, "fb": fb, "gf": gf_std,
            "regime": regime, "bt_all": bt_all, "bt_calm": bt_calm, "bt_stress": bt_stress,
            "ml_scores": ml_scores, "wf": wf,
            "dcc_a": dcc_a, "dcc_b": dcc_b,
            "evt": evt_wti, "ovx": ovx,
            "results": True
        })

# ══════════════════════════════════════════════════════════
#   RENDER
# ══════════════════════════════════════════════════════════
if "prices" not in st.session_state:
    st.stop()

S = st.session_state
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Market Intelligence", "Geopolitical Risk", "Monte Carlo",
    "Backtesting Suite", "Machine Learning", "Validation"
])

with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("WTI", f"${S['prices']['oil'].iloc[-1]:.2f}")
    c2.metric("Brent", f"${S['prices']['brent'].iloc[-1]:.2f}")
    c3.metric("EGARCH Vol (ann.)", f"{S['vw'].iloc[-1]*np.sqrt(252)*100:.1f}%")
    c4.metric("OVX", f"{S['ovx'].iloc[-1]:.1f}" if S['ovx'] is not None else "N/A")
    fig = qfig()
    fig.add_trace(go.Scatter(x=S['vw'].index, y=S['vw']*np.sqrt(252)*100, name="EGARCH Vol"))
    if S['regime'].sum()>0:
        fig.add_trace(go.Scatter(x=S['regime'].index, y=S['regime']*S['vw'].max()*np.sqrt(252)*100*0.5,
                                 name="Stress Regime", yaxis="y2"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = qfig()
    fig2.add_trace(go.Scatter(x=S['gf'].index, y=S['gf'].values, name="GeoFactor"))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = qfig()
    x = list(range(len(S['fan'][50])))
    fig3.add_trace(go.Scatter(x=x, y=S['fan'][50], name="WTI P50"))
    fig3.add_trace(go.Scatter(x=x, y=S['fb'][50], name="Brent P50", line=dict(dash="dash")))
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    if S['bt_all']:
        st.metric("Calibration Score", f"{1 - np.mean([S['bt_all']['Kupiec_p'], S['bt_all']['Christoffersen_p'], S['bt_all']['DQ_p']]):.2f}")
        st.write("All periods:", S['bt_all'])
        if S['bt_calm']: st.write("Calm regime:", S['bt_calm'])
        if S['bt_stress']: st.write("Stress regime:", S['bt_stress'])

with tab5:
    st.dataframe(pd.DataFrame(S['ml_scores']).T)

with tab6:
    if not S['wf'].empty:
        st.dataframe(S['wf'])
        fig_wf = qfig(300)
        fig_wf.add_trace(go.Bar(x=S['wf']["Test End"], y=S['wf']["RMSE"]))
        st.plotly_chart(fig_wf, use_container_width=True)

st.markdown("<div class='footer'>GeoQuant v2.0 · EGARCH + Conditional EVT · Eduardo Moraes</div>", unsafe_allow_html=True)
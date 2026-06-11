"""
GeoQuant – Macro Research Terminal
Quantitative Geopolitical Intelligence | EVT + DCC + GARCH-X
Eduardo Moraes | Quant Data Scientist & Economics
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, csv, logging, warnings
from datetime import datetime, timedelta
import pytz
from scipy.interpolate import PchipInterpolator
from scipy import stats, optimize
from sklearn.linear_model import LassoCV
from statsmodels.tsa.vector_ar.var_model import VAR
import yfinance as yf
from arch import arch_model

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GeoQuant · Macro Research",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TICKERS = {
    "oil": "CL=F", "brent": "BZ=F", "natgas": "NG=F",
    "gold": "GC=F", "silver": "SI=F", "copper": "HG=F",
    "wheat": "ZW=F", "corn": "ZC=F", "soy": "ZS=F",
    "dxy": "DX-Y.NYB", "eur": "EURUSD=X", "tnx": "^TNX",
}
GEO_WEIGHTS_DEFAULT = {
    "oil_vol": 0.22, "gold": 0.09, "gold_real": 0.09,
    "dxy": -0.10, "spread": 0.09, "fert": 0.22,
    "wheat": 0.07, "copper": 0.04, "natgas_vol": 0.06,
}
ZSCORE_W = {"oil_gold": 0.40, "oil_natgas": 0.35, "gold_real": 0.25}
JUMP_EXTREME = 0.15
JUMP_SKU_NOR = 0.045
JUMP_SKU_EXT = 0.135
JUMP_SKU_DOWN = 0.025
REGIME_NOISE = 0.05
SPREAD_MIN = -0.05
SPREAD_MAX = 0.30
FERT_BS_Z_THR = 1.5
FERT_EVT_Q = 0.90

# Plotly layout – clean, professional
PLOTLY_LAYOUT = {
    "paper_bgcolor": "white",
    "plot_bgcolor": "#F8F9FA",
    "font": {"family": "Helvetica Neue, Arial", "color": "#2C3E50", "size": 11},
    "title_font": {"family": "Helvetica Neue", "size": 14, "color": "#1F4E79"},
    "xaxis": {"gridcolor": "#E5E7EB", "linecolor": "#D1D5DB", "zerolinecolor": "#D1D5DB"},
    "yaxis": {"gridcolor": "#E5E7EB", "linecolor": "#D1D5DB", "zerolinecolor": "#D1D5DB"},
    "legend": {"bgcolor": "rgba(255,255,255,0.9)", "bordercolor": "#D1D5DB", "borderwidth": 1},
    "margin": {"l": 50, "r": 30, "t": 60, "b": 40},
}

COLORS = {
    "wti": "#1F4E79",
    "brent": "#2D6B6B",
    "gold": "#C8A96E",
    "silver": "#9CA3AF",
    "fertilizer": "#5F6B47",
    "natgas": "#2D6B6B",
    "wheat": "#1F4E79",
    "corn": "#2D6B6B",
    "soy": "#6B7280",
    "ci_light": "rgba(31,78,121,0.2)",
    "ci_medium": "rgba(31,78,121,0.4)",
    "stress": "#7A3F30",
}

def quant_fig(height=450):
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig

def quant_subplots(rows=1, cols=1, secondary=False, height=450, **kw):
    specs = [[{"secondary_y": secondary}] * cols for _ in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, specs=specs, **kw)
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## GeoQuant Terminal")
    st.caption("Quantitative Research")
    st.divider()

    mc_sims = st.slider("Monte Carlo paths", 1000, 30000, 5000, 1000)
    mc_steps = st.slider("Horizon (days)", 5, 30, 10, 1)

    st.divider()
    jump_up = st.slider("Jump prob up", 0.01, 0.20, 0.07, 0.01)
    jump_down = st.slider("Jump prob down", 0.01, 0.10, 0.03, 0.01)
    tail_df = st.slider("Tail df", 2.5, 8.0, 3.0, 0.5)

    st.divider()
    prior_wti = st.slider("WTI prior vol (annual)", 0.20, 0.65, 0.35, 0.01)
    prior_brent = st.slider("Brent prior vol (annual)", 0.20, 0.65, 0.35, 0.01)

    st.divider()
    war_start = st.date_input("War start reference", value=datetime(2026, 2, 28))
    run_btn = st.button("Run Full Analysis", type="primary")

    st.caption("FOR PROFESSIONAL USE ONLY\nNOT INVESTMENT ADVICE")

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
now_sp = datetime.now(pytz.timezone("America/Sao_Paulo"))
st.markdown(
    f"""
    <div style="border-bottom: 2px solid #1F4E79; margin-bottom: 1.5rem;">
        <h1 style="color: #1F4E79; font-weight: 400; margin-bottom: 0;">GeoQuant · Macro Research</h1>
        <p style="color: #6B7280; font-size: 0.8rem;">Geopolitical Intelligence · Commodity Markets · EVT+DCC+GARCH-X · {now_sp.strftime('%d %B %Y · %H:%M')} (SP)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Core functions (all)
# ----------------------------------------------------------------------
def rolling_zscore(s, w=60):
    return (s - s.rolling(w).mean()) / s.rolling(w).std().replace(0, np.nan)

def fill_gaps(s):
    s = s.copy()
    valid = s.notna()
    if valid.sum() < 2:
        return s.ffill()
    try:
        x = s.index[valid].astype(np.int64)
        filled = pd.Series(PchipInterpolator(x, s[valid].values)(s.index.astype(np.int64)), index=s.index)
        filled[valid] = s[valid]
        return filled
    except:
        return s.ffill()

# Fertilizer data with real prices (June 2026)
def _fert_csv(path="fertilizer_backup.csv"):
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "urea_price", "dap_price"])
        w.writerows([
            ["2026-01-15", 540, 710], ["2026-02-15", 560, 740],
            ["2026-03-15", 590, 780], ["2026-04-15", 616, 857],
            ["2026-05-01", 720, 900], ["2026-05-06", 810, 920],
            ["2026-05-12", 857, 920], ["2026-06-01", 860, 925],
            ["2026-06-10", 453.5, 920],  # Real market price drop
        ])

def get_usda():
    _fert_csv()
    try:
        df = pd.read_csv("fertilizer_backup.csv", parse_dates=["date"], index_col="date").sort_index()
        last = df.iloc[-1]
        return {
            "urea_price": float(last["urea_price"]),
            "urea_period": str(last.name.date()),
            "dap_price": float(last["dap_price"]),
            "dap_period": str(last.name.date()),
            "source": "Green Markets / CRU",
        }
    except:
        return {"urea_price": 453.5, "urea_period": "2026-06-10", "dap_price": 920, "dap_period": "2026-06-10", "source": "fallback"}

def fert_black_swan(usda):
    _fert_csv()
    try:
        df = pd.read_csv("fertilizer_backup.csv", parse_dates=["date"], index_col="date")
        hist = df["urea_price"].dropna().values
    except:
        hist = []
    cur = usda.get("urea_price")
    if cur is None or len(hist) < 10:
        return 1.0
    rets = np.diff(np.log(hist))
    thr = np.quantile(rets, FERT_EVT_Q)
    exc = rets[rets > thr] - thr
    if len(exc) < 5:
        mu, sig = np.mean(hist), np.std(hist)
        if sig == 0:
            return 1.0
        z = (cur - mu) / sig
        # If price drops > 2 sigma, reduce multiplier (negative black swan)
        if z < -FERT_BS_Z_THR:
            return max(0.5, 1.0 + z * 0.3)
        return min(1.0 + max(0, z - FERT_BS_Z_THR) * 0.8, 3.0)
    try:
        shape, loc, scale = stats.genpareto.fit(exc)
        cr = np.log(cur / hist[-1])
        if cr <= thr:
            # Strong negative return (price collapse)
            if cr < -0.1:
                return 0.6
            return 1.0
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
        "silver_gold_roll": np.log(sg / sg.shift(1)).rolling(20).mean(),
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
        "fert": fi,
    })
    if sd is not None:
        X["silver_demand"] = sd
    y = returns["oil"].shift(-1)
    ci = y.dropna().index.intersection(X.dropna().index)
    X2, y2 = X.loc[ci].dropna(), y.loc[ci]
    if len(X2) < window:
        return GEO_WEIGHTS_DEFAULT.copy()
    Xc, yc = X2.iloc[-window:], y2.iloc[-window:]
    Xm, Xs = Xc.mean(), Xc.std().replace(0, 1)
    try:
        mdl = LassoCV(cv=5, random_state=42, alphas=np.logspace(-4, 0, 20), max_iter=2000).fit((Xc - Xm) / Xs, yc)
        coef = mdl.coef_ / Xs.values
        w = {col: coef[i] for i, col in enumerate(X2.columns)}
        tot = sum(abs(v) for v in w.values())
        return {k: v / tot for k, v in w.items()} if tot > 0 else GEO_WEIGHTS_DEFAULT.copy()
    except:
        return GEO_WEIGHTS_DEFAULT.copy()

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
    return (ZSCORE_W["oil_gold"] * z1 + ZSCORE_W["oil_natgas"] * z2 + ZSCORE_W["gold_real"] * z3).dropna()

def fit_garch(ret, exog):
    rc = ret.loc[ret.index.intersection(exog.index)] * 100
    xc = exog.loc[rc.index]
    try:
        res = arch_model(rc, x=xc, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt").fit(disp="off")
    except:
        res = arch_model(rc, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt").fit(disp="off")
    return res.conditional_volatility / 100

def bayes_shrink(vg, prior_daily, n, geofactor=None, label=""):
    w = np.clip(np.sqrt(n / 252), 0.10, 0.95)
    prior = prior_daily * (1.0 + 0.4 * np.tanh(float(geofactor.iloc[-1]))) if geofactor is not None and not geofactor.empty else prior_daily
    lo, hi = prior * 0.5, prior * 1.5
    v_last = float(vg.iloc[-1])
    vs = vg.copy() if lo <= v_last <= hi else w * vg + (1 - w) * prior
    vga = v_last * np.sqrt(252) * 100
    vsa = float(vs.iloc[-1]) * np.sqrt(252) * 100
    return vs, {"vga": vga, "vsa": vsa, "w": w if not (lo <= v_last <= hi) else 1.0}

def fit_dcc(rw, rb, vw, vb):
    ci = rw.index.intersection(rb.index).intersection(vw.index).intersection(vb.index)
    ew = (rw[ci] / vw[ci]).dropna()
    eb = (rb[ci] / vb[ci]).dropna()
    c2 = ew.index.intersection(eb.index)
    e = np.column_stack([ew[c2], eb[c2]])

    def nll(p):
        a, b = p
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10
        Qb = np.cov(e, rowvar=False)
        Q = Qb.copy()
        ll = 0
        for t in range(1, len(e)):
            Qt = (1 - a - b) * Qb + a * np.outer(e[t - 1], e[t - 1]) + b * Q
            d = np.sqrt(np.diag(Qt))
            d[d == 0] = 1e-8
            R = Qt / np.outer(d, d)
            R = np.clip(R, -0.9999, 0.9999)
            try:
                L = np.linalg.cholesky(R)
                z = np.linalg.inv(L) @ e[t]
                ll += -0.5 * np.sum(z ** 2) - np.sum(np.log(np.diag(L)))
                Q = Qt
            except:
                return 1e10
        return -ll

    res = optimize.minimize(nll, [0.05, 0.93], bounds=[(1e-4, 0.3), (0.7, 0.9999)], method="L-BFGS-B")
    a, b = res.x
    return (0.05, 0.93) if a + b >= 1 else (a, b)

def _tail_jumps(shocks, vol):
    n = len(shocks)
    u = np.random.rand(n)
    ju = np.random.exponential(0.03, n) * vol
    jd = np.random.exponential(0.02, n) * vol
    return shocks + np.where(u < 0.025, ju, 0) - np.where((u >= 0.025) & (u < 0.05), jd, 0)

def _jumps_vec(n, pu, pd_):
    u = np.random.rand(n)
    me = np.random.rand(n) < JUMP_EXTREME
    ju = np.where(me, np.random.exponential(JUMP_SKU_EXT, n), np.random.exponential(JUMP_SKU_NOR, n))
    jd = np.random.exponential(JUMP_SKU_DOWN, n)
    jw = np.where(u < pu, ju, np.where((u >= pu) & (u < pu + pd_), -jd, 0))
    jb = np.where(u < pu, ju * 0.95, np.where((u >= pu) & (u < pu + pd_), -jd * 0.90, 0))
    return jw, jb

def run_mc(wti0, brt0, bvw, bvb, fcast, ocol, bcol,
           rbase, rw, rb, vws, vbs, jpu, tdf, bs=1.0,
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
    ra = 1 + 0.5 * np.clip(rbase + np.random.normal(0, REGIME_NOISE, (sims, steps)), -1, 1)
    max_dv = 0.08
    max_dr = 0.02
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
        vw_ = np.clip(bvw * ra[:, t], 0, max_dv)
        vb_ = np.clip(bvb * ra[:, t], 0, max_dv)
        sw = np.clip(zw * vw_, -4 * vw_, 4 * vw_)
        sb = np.clip(zb * vb_, -4 * vb_, 4 * vb_)
        sw = _tail_jumps(sw, vw_)
        sb = _tail_jumps(sb, vb_)
        jw, jb = _jumps_vec(sims, pu, pd_)
        sw += jw
        sb += jb
        dw = np.clip(fcast[t, ocol] * ra[:, t], -max_dr, max_dr)
        db = np.clip(fcast[t, bcol] * ra[:, t], -max_dr, max_dr)
        nw = pw[:, t] * np.exp(dw + sw)
        nb = pb[:, t] * np.exp(db + sb)
        sp = np.where(nb > 0, (nb - nw) / nb, 0)
        nw = np.where(sp < SPREAD_MIN, nb * (1 + abs(SPREAD_MIN)), nw)
        nw = np.where(sp > SPREAD_MAX, nb * (1 - SPREAD_MAX), nw)
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
    return {"fan": fan, "fan_b": fb, "paths": pw, "metrics": {
        "vol_wti": bvw * np.sqrt(252) * 100, "vol_brt": bvb * np.sqrt(252) * 100,
        "var95": v95, "cvar95": float(np.mean((pw[:, 1] - wti0)[mask])),
        "prob_up": np.mean(term > wti0) * 100,
        "prob_40": np.mean(term < 40) * 100, "prob_150": np.mean(term > 150) * 100,
        "p5": (fan[5][-1] / wti0 - 1) * 100, "p95": (fan[95][-1] / wti0 - 1) * 100,
    }}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_data(start_date=None):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    tickers_list = list(TICKERS.values())
    tickers_keys = list(TICKERS.keys())
    errors = []

    def _extract_close(raw):
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0).unique().tolist()
            field = next((f for f in ["Close", "Adj Close", "Price"] if f in lvl0), None)
            if field:
                out = raw[field].copy()
            else:
                out = raw.iloc[:, :len(tickers_keys)].copy()
        else:
            out = raw.copy()
        return out

    for auto_adj in [True, False]:
        try:
            raw = yf.download(tickers_list, start=start_date, progress=False, auto_adjust=auto_adj)
            out = _extract_close(raw)
            if not out.empty and len(out) > 5:
                out.columns = tickers_keys[:len(out.columns)]
                return out.ffill().dropna()
        except Exception as e:
            errors.append(str(e))

    try:
        raw = yf.download(tickers_list, period="180d", progress=False, auto_adjust=True)
        out = _extract_close(raw)
        if not out.empty and len(out) > 5:
            out.columns = tickers_keys[:len(out.columns)]
            return out.ffill().dropna()
    except:
        pass

    frames = {}
    for key, ticker_sym in TICKERS.items():
        try:
            t = yf.Ticker(ticker_sym)
            df = t.history(start=start_date, auto_adjust=True)
            if df.empty:
                df = t.history(period="180d", auto_adjust=True)
            if not df.empty:
                col = "Close" if "Close" in df.columns else df.columns[0]
                frames[key] = fill_gaps(df[col])
        except:
            pass
    if frames:
        out = pd.DataFrame(frames).ffill().dropna()
        if not out.empty and len(out) > 5:
            return out
    return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live(last_wti=65.0, last_brt=68.0):
    try:
        w = yf.Ticker("CL=F").fast_info
        b = yf.Ticker("BZ=F").fast_info
        wti = float(w.get("last_price", 0))
        brent = float(b.get("last_price", 0))
        if wti > 0 and brent > 0:
            return wti, brent
    except:
        pass
    try:
        d = yf.download(["CL=F", "BZ=F"], period="5d", progress=False, auto_adjust=True, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d = d["Close"]
        wti = float(d["CL=F"].dropna().iloc[-1])
        brent = float(d["BZ=F"].dropna().iloc[-1])
        if wti > 0 and brent > 0:
            return wti, brent
    except:
        pass
    return last_wti, last_brt

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
needs_run = run_btn or "results" not in st.session_state

if needs_run:
    info_placeholder = st.info("Initialising quantitative terminal. Loading market data...")
    prog = st.progress(0)

    prog.progress(10)
    prices = fetch_data()
    if prices.empty or len(prices) < 5:
        info_placeholder.empty()
        st.error("Failed to load market data. Check internet connection and try again.")
        st.stop()

    for key in TICKERS:
        if key not in prices.columns:
            prices[key] = np.nan
    prices = prices.ffill().bfill()

    last_wti = float(prices["oil"].dropna().iloc[-1])
    last_brt = float(prices["brent"].dropna().iloc[-1])
    wti0, brt0 = fetch_live(last_wti, last_brt)
    prices.loc[prices.index[-1], "oil"] = wti0
    prices.loc[prices.index[-1], "brent"] = brt0
    returns = np.log(prices / prices.shift(1)).dropna()

    prog.progress(22)
    usda = get_usda()
    bs_mult = fert_black_swan(usda)
    gs = gold_signals(prices)
    sd = silver_demand_proxy(prices)
    weights = GEO_WEIGHTS_DEFAULT.copy()
    weights["silver_demand"] = 0.02
    tot = sum(abs(v) for v in weights.values())
    weights = {k: v / tot for k, v in weights.items()}
    fi = build_fert_index(returns, usda, bs_mult)

    prog.progress(35)
    dyn_w = calibrate_weights(returns, prices, gs, fi, sd)
    if dyn_w:
        weights = dyn_w
    gf_raw = build_geofactor(returns, prices, gs, fi, weights, sd)
    gf = (gf_raw - gf_raw.mean()) / gf_raw.std() if len(gf_raw) > 1 else gf_raw
    zsc = build_zscore(prices, gs)

    prog.progress(50)
    vw = fit_garch(returns["oil"], gf)
    vb = fit_garch(returns["brent"], gf)
    vg = fit_garch(returns["gold"], gf)
    n = len(returns)
    pw_d = prior_wti / np.sqrt(252)
    pb_d = prior_brent / np.sqrt(252)
    pg_d = 0.18 / np.sqrt(252)
    vw, dw = bayes_shrink(vw, pw_d, n, geofactor=gf, label="WTI")
    vb, db = bayes_shrink(vb, pb_d, n, geofactor=gf, label="BRT")
    vg, _ = bayes_shrink(vg, pg_d, n)
    bvw = float(vw.iloc[-1])
    bvb = float(vb.iloc[-1])

    prog.progress(65)
    dcc_a, dcc_b = fit_dcc(returns["oil"], returns["brent"], vw, vb)
    rv = returns.loc[gf.index.intersection(returns.index)]
    lags = min(5, max(1, len(rv) // 10))
    vm = VAR(rv).fit(lags)
    fcast = vm.forecast(rv.values[-vm.k_ar:], steps=mc_steps)
    cols = list(rv.columns)
    ocol = cols.index("oil")
    bcol = cols.index("brent")
    vr = bvb / (pb_d * 1.5)
    tdf_d = max(2.5, min(6.0, tail_df / np.sqrt(max(vr, 0.5))))
    rbase = float(np.tanh(gf.iloc[-1] / 2)) if not gf.empty else 0.0
    ws = (returns["wheat"].tail(20).mean() + returns["natgas"].tail(20).mean()) / 2
    war_t = bool(ws > 0.005)
    jpu_eff = min(jump_up * 1.5, 0.15) if war_t else jump_up

    prog.progress(75)
    mc_note = st.empty()
    mc_note.markdown("Monte Carlo simulation in progress...")
    mc_bar = st.progress(0)
    mc = run_mc(wti0, brt0, bvw, bvb, fcast, ocol, bcol, rbase,
                returns["oil"], returns["brent"], vw, vb,
                jpu_eff, tdf_d, bs=bs_mult, dcc_a=dcc_a, dcc_b=dcc_b,
                sims=mc_sims, steps=mc_steps, bar=mc_bar)
    mc_note.empty()
    mc_bar.empty()

    try:
        rj = pd.concat([returns["oil"], returns["brent"]], axis=1).dropna()
        ec = rj.ewm(alpha=0.06).cov(pairwise=True)
        lc = ec.loc[ec.index.get_level_values(0)[-1]]
        corr = float(np.clip(lc.loc["oil", "brent"] / np.sqrt(lc.loc["oil", "oil"] * lc.loc["brent", "brent"]), -1, 1))
    except:
        corr = 0.95

    # Additional metrics for new visualizations
    ret_ann = returns[["oil", "brent"]].mean() * 252
    vol_ann = returns[["oil", "brent"]].std() * np.sqrt(252)
    sharpe = ret_ann / vol_ann
    downside = returns[["oil", "brent"]][returns[["oil", "brent"]] < 0].std() * np.sqrt(252)
    sortino = ret_ann / downside
    skew_oil = returns["oil"].skew()
    kurt_oil = returns["oil"].kurtosis()
    skew_brt = returns["brent"].skew()
    kurt_brt = returns["brent"].kurtosis()

    # Dynamic correlation matrix for heatmap
    corr_matrix = returns[["oil", "brent", "gold", "dxy", "tnx"]].dropna().corr()

    # Global stress indicator (composite)
    stress_components = pd.DataFrame({
        "vol_wti": vw.rolling(20).mean() * np.sqrt(252) * 100,
        "vol_brent": vb.rolling(20).mean() * np.sqrt(252) * 100,
        "corr_wti_brent": returns["oil"].rolling(20).corr(returns["brent"]),
        "gold_zscore": rolling_zscore(prices["gold"], 60),
        "geofactor": gf,
    })
    stress_components = stress_components.dropna()
    stress_index = (stress_components["vol_wti"] / 50 + stress_components["vol_brent"] / 50 +
                    np.abs(stress_components["corr_wti_brent"] - 0.8) * 2 +
                    stress_components["gold_zscore"].clip(0, 3) / 3 +
                    stress_components["geofactor"].clip(0, 2) / 2) / 5
    stress_index = stress_index.clip(0, 1)

    prog.progress(100)
    info_placeholder.empty()
    prog.empty()

    st.session_state.update({
        "results": mc, "gf": gf, "zsc": zsc, "vw": vw, "vb": vb, "vg": vg,
        "fi": fi, "gs": gs, "prices": prices, "returns": returns,
        "wti0": wti0, "brt0": brt0, "usda": usda, "bs": bs_mult,
        "dw": dw, "db": db, "tdf": tdf_d, "corr": corr, "rbase": rbase,
        "war_t": war_t, "ws": float(ws), "jpu": jpu_eff, "dcc_a": dcc_a,
        "dcc_b": dcc_b, "weights": weights,
        "sharpe": sharpe, "sortino": sortino,
        "skew_oil": skew_oil, "kurt_oil": kurt_oil,
        "skew_brt": skew_brt, "kurt_brt": kurt_brt,
        "mc_sims": mc_sims, "mc_steps": mc_steps,
        "corr_matrix": corr_matrix, "stress_index": stress_index,
    })

# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------
if "results" in st.session_state:
    # retrieve all variables
    mc = st.session_state["results"]
    gf = st.session_state["gf"]
    zsc = st.session_state["zsc"]
    vw = st.session_state["vw"]
    vb = st.session_state["vb"]
    vg = st.session_state["vg"]
    fi = st.session_state["fi"]
    gs = st.session_state["gs"]
    prices = st.session_state["prices"]
    returns = st.session_state["returns"]
    wti0 = st.session_state["wti0"]
    brt0 = st.session_state["brt0"]
    usda = st.session_state["usda"]
    bs = st.session_state["bs"]
    dw_d = st.session_state["dw"]
    db_d = st.session_state["db"]
    tdf_d = st.session_state["tdf"]
    corr = st.session_state["corr"]
    rbase = st.session_state["rbase"]
    war_t = st.session_state["war_t"]
    ws_val = st.session_state["ws"]
    dcc_a = st.session_state["dcc_a"]
    dcc_b = st.session_state["dcc_b"]
    fan = mc["fan"]
    fan_b = mc["fan_b"]
    M = mc["metrics"]
    sharpe = st.session_state.get("sharpe", pd.Series([0, 0], index=["oil", "brent"]))
    sortino = st.session_state.get("sortino", pd.Series([0, 0], index=["oil", "brent"]))
    skew_oil = st.session_state.get("skew_oil", 0)
    kurt_oil = st.session_state.get("kurt_oil", 0)
    skew_brt = st.session_state.get("skew_brt", 0)
    kurt_brt = st.session_state.get("kurt_brt", 0)
    mc_sims = st.session_state.get("mc_sims", 5000)
    mc_steps = st.session_state.get("mc_steps", 10)
    corr_matrix = st.session_state.get("corr_matrix")
    stress_index = st.session_state.get("stress_index")
    spread = brt0 - wti0

    # Tabs: added "Macro & Correlations" as sixth tab
    tab_market, tab_geo, tab_mc, tab_stats, tab_macro, tab_exec = st.tabs(
        ["Market & Risk", "Geopolitical", "Monte Carlo", "Quantitative Stats", "Macro & Correlations", "Executive Summary"]
    )

    # ----- Tab 1: Market & Risk -----
    with tab_market:
        st.subheader("Live Commodity Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("WTI Crude", f"${wti0:.2f}", f"P50 10d ${fan[50][-1]:.2f}")
        col2.metric("Brent Crude", f"${brt0:.2f}", f"Spread ${spread:.2f} ({spread/wti0*100:.1f}%)")
        col3.metric("WTI Volatility (ann.)", f"{M['vol_wti']:.1f}%", f"shrunk {dw_d['vsa']:.1f}%")
        col4.metric("Brent Volatility (ann.)", f"{M['vol_brt']:.1f}%", f"shrunk {db_d['vsa']:.1f}%")

        st.divider()
        st.subheader("Volatility Surface – GARCH-X with Bayesian Shrinkage")
        if vw is not None and len(vw) > 5:
            fig_vol = quant_fig(480)
            fig_vol.add_trace(go.Scatter(x=vw.index, y=vw*np.sqrt(252)*100, name="WTI", line=dict(color=COLORS["wti"], width=2.5)))
            fig_vol.add_trace(go.Scatter(x=vb.index, y=vb*np.sqrt(252)*100, name="Brent", line=dict(color=COLORS["brent"], width=2.5, dash="dash")))
            fig_vol.add_trace(go.Scatter(x=vg.index, y=vg*np.sqrt(252)*100, name="Gold", line=dict(color=COLORS["gold"], width=2, dash="dot")))
            fig_vol.add_hrect(y0=25, y1=45, fillcolor="rgba(45,107,107,0.05)", line_width=0, annotation_text="Normal band 25–45%")
            fig_vol.update_layout(yaxis_ticksuffix="%", title="Annualised Volatility")
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Volatility data not available.")

        st.divider()
        st.subheader("Stress Indices")
        col_a, col_b = st.columns(2)
        with col_a:
            if fi is not None and len(fi) > 5:
                ng_vol = returns["natgas"].rolling(20).std() * np.sqrt(252) * 100
                fig_f = quant_subplots(secondary=True, height=380)
                fig_f.add_trace(go.Scatter(x=fi.index, y=fi.values, name="Fertilizer Stress", fill="tozeroy",
                                           fillcolor="rgba(95,107,71,0.1)", line=dict(color=COLORS["fertilizer"], width=2)), secondary_y=False)
                fig_f.add_trace(go.Scatter(x=ng_vol.index, y=ng_vol.values, name="NatGas Volatility",
                                           line=dict(color=COLORS["natgas"], width=1.8, dash="dash")), secondary_y=True)
                fig_f.update_yaxes(title_text="Fertilizer Index", secondary_y=False)
                fig_f.update_yaxes(title_text="NatGas Vol %", secondary_y=True)
                st.plotly_chart(fig_f, use_container_width=True)
            bs_str = f" · Black Swan x{bs:.2f}" if bs > 1.2 else (" · Deflationary shock" if bs < 0.8 else "")
            st.caption(f"Urea ${usda['urea_price']:.1f}/t  |  DAP ${usda['dap_price']:.0f}/t  |  {usda['source']}{bs_str}")

        with col_b:
            if gs is not None and not gs["gold_real"].dropna().empty:
                gr_b = float(gs["gold_real"].dropna().iloc[0])
                sg_b = float(gs["silver_gold"].dropna().iloc[0])
                fig_g = quant_subplots(secondary=True, height=380)
                fig_g.add_trace(go.Scatter(x=gs["gold_real"].dropna().index, y=(gs["gold_real"].dropna()/gr_b).values,
                                           name="Gold / Real Yield", line=dict(color=COLORS["gold"], width=2)), secondary_y=False)
                fig_g.add_trace(go.Scatter(x=gs["silver_gold"].dropna().index, y=(gs["silver_gold"].dropna()/sg_b).values,
                                           name="Silver / Gold", line=dict(color=COLORS["silver"], width=1.8, dash="dash")), secondary_y=True)
                fig_g.add_hline(y=1.0, line_dash="dot", line_color="#9CA3AF")
                fig_g.update_yaxes(title_text="Gold/Real Yield (norm)", secondary_y=False)
                fig_g.update_yaxes(title_text="Silver/Gold (norm)", secondary_y=True)
                st.plotly_chart(fig_g, use_container_width=True)
            else:
                st.info("Gold signals not available.")

    # ----- Tab 2: Geopolitical Intelligence -----
    with tab_geo:
        st.subheader("Geopolitical Risk Indicators")
        if zsc is not None and len(zsc) > 5 and gf is not None and len(gf) > 5:
            fig_geo = quant_subplots(secondary=True, height=500)
            fig_geo.add_trace(go.Scatter(x=zsc.index, y=zsc.values, name="Z-Score Composite",
                                         line=dict(color=COLORS["wti"], width=2.5), fill="tozeroy",
                                         fillcolor="rgba(31,78,121,0.1)"), secondary_y=False)
            fig_geo.add_trace(go.Scatter(x=gf.index, y=gf.values, name="GeoFactor (normalized)",
                                         line=dict(color=COLORS["gold"], width=3)), secondary_y=True)
            fig_geo.add_hline(y=1.5, line_dash="dot", line_color=COLORS["gold"], secondary_y=False)
            fig_geo.add_hline(y=-1.5, line_dash="dot", line_color=COLORS["gold"], secondary_y=False)
            fig_geo.update_yaxes(title_text="Z-Score (σ)", secondary_y=False)
            fig_geo.update_yaxes(title_text="GeoFactor (σ)", secondary_y=True)
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.info("Geopolitical data insufficient to generate chart.")

        st.divider()
        st.subheader("Agricultural Commodities – Indexed from War Start")
        fig_ag = quant_fig(450)
        for asset, color, label in [("wheat", COLORS["wheat"], "Wheat"), ("corn", COLORS["corn"], "Corn"), ("soy", COLORS["soy"], "Soy")]:
            bv = float(prices[asset].iloc[0])
            rel = (prices[asset] / bv * 100).dropna()
            if len(rel) > 0:
                fig_ag.add_trace(go.Scatter(x=rel.index, y=rel.values, name=f"{label} (base ${bv:.0f})", line=dict(color=color, width=2)))
        fig_ag.add_hline(y=100, line_dash="dot", line_color="#9CA3AF")
        fig_ag.update_layout(yaxis_title="Price Index (base = 100)")
        st.plotly_chart(fig_ag, use_container_width=True)

    # ----- Tab 3: Monte Carlo Forecast -----
    with tab_mc:
        war_note = " | War boost active" if war_t else ""
        bs_note = f" | Fertilizer Black Swan x{bs:.2f}" if bs > 1.2 else (" | Fertilizer deflation" if bs < 0.8 else "")
        st.subheader(f"Probabilistic Price Forecast – {mc_sims:,} paths, {mc_steps} days{war_note}{bs_note}")
        if fan is not None and len(fan[50]) > 1:
            x_ax = list(range(mc_steps + 1))
            fig_mc = quant_fig(550)
            fig_mc.add_trace(go.Scatter(x=x_ax + x_ax[::-1], y=list(fan[95]) + list(fan[5][::-1]),
                                        fill="toself", fillcolor=COLORS["ci_light"], line=dict(width=0), name="WTI 90% CI"))
            fig_mc.add_trace(go.Scatter(x=x_ax + x_ax[::-1], y=list(fan[75]) + list(fan[25][::-1]),
                                        fill="toself", fillcolor=COLORS["ci_medium"], line=dict(width=0), name="WTI 50% CI"))
            fig_mc.add_trace(go.Scatter(x=x_ax, y=list(fan_b[50]), name=f"Brent P50 → ${fan_b[50][-1]:.2f}",
                                        line=dict(color=COLORS["brent"], width=2.5, dash="dash")))
            fig_mc.add_trace(go.Scatter(x=x_ax, y=list(fan[50]), name=f"WTI P50 → ${fan[50][-1]:.2f}",
                                        line=dict(color=COLORS["wti"], width=4)))
            fig_mc.add_trace(go.Scatter(x=x_ax, y=list(fan[95]), name=f"P95 → ${fan[95][-1]:.2f}",
                                        line=dict(color=COLORS["gold"], width=1.5, dash="dot")))
            fig_mc.add_trace(go.Scatter(x=x_ax, y=list(fan[5]), name=f"P5 → ${fan[5][-1]:.2f}",
                                        line=dict(color=COLORS["gold"], width=1.5, dash="dot")))
            fig_mc.add_hline(y=wti0, line_dash="dash", line_color="#6B7280", annotation_text=f"Current ${wti0:.2f}")
            fig_mc.add_hline(y=40, line_dash="dot", line_color=COLORS["stress"], annotation_text="Stress $40")
            fig_mc.add_hline(y=150, line_dash="dot", line_color=COLORS["stress"], annotation_text="Stress $150")
            fig_mc.update_layout(xaxis_title="Trading Days Ahead", yaxis_title="Price (USD/bbl)", yaxis_tickprefix="$")
            st.plotly_chart(fig_mc, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Probability Up 10d", f"{M['prob_up']:.1f}%")
            c2.metric("VaR 95% 1d", f"${M['var95']:+.2f}", delta=f"CVaR ${M['cvar95']:+.2f}")
            c3.metric("Extreme Probabilities", f"< $40: {M['prob_40']:.2f}%", delta=f"> $150: {M['prob_150']:.2f}%")
        else:
            st.info("Monte Carlo results not available. Run simulation first.")

    # ----- Tab 4: Quantitative Stats -----
    with tab_stats:
        st.subheader("Risk‑Adjusted Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**WTI**")
            st.metric("Sharpe Ratio (ann.)", f"{sharpe['oil']:.2f}")
            st.metric("Sortino Ratio (ann.)", f"{sortino['oil']:.2f}")
            st.metric("Skewness", f"{skew_oil:.3f}", delta="left tail" if skew_oil < 0 else "right tail")
            st.metric("Excess Kurtosis", f"{kurt_oil:.3f}", delta="heavy tails" if kurt_oil > 0 else "thin tails")
        with col2:
            st.markdown("**Brent**")
            st.metric("Sharpe Ratio (ann.)", f"{sharpe['brent']:.2f}")
            st.metric("Sortino Ratio (ann.)", f"{sortino['brent']:.2f}")
            st.metric("Skewness", f"{skew_brt:.3f}")
            st.metric("Excess Kurtosis", f"{kurt_brt:.3f}")

        st.divider()
        st.subheader("DCC Correlation Dynamics")
        st.metric("WTI–Brent Correlation (EWMA)", f"{corr:.4f}")
        st.metric("DCC α (short‑term shock)", f"{dcc_a:.4f}", delta=f"β (persistence): {dcc_b:.4f}")
        st.metric("DCC Persistence (α+β)", f"{dcc_a+dcc_b:.4f}")

    # ----- Tab 5: Macro & Correlations (new) -----
    with tab_macro:
        st.subheader("Global Market Correlations")
        if corr_matrix is not None:
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="Blues",
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
            ))
            fig_heatmap.update_layout(height=450, title="Rolling Correlation Matrix (Oil, Brent, Gold, DXY, 10Y Yield)")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Correlation matrix not available.")

        st.divider()
        st.subheader("Risk-Return Profile")
        # Scatter plot: Sharpe vs Volatility
        risk_return = pd.DataFrame({
            "Asset": ["WTI", "Brent"],
            "Sharpe": [sharpe["oil"], sharpe["brent"]],
            "Volatility": [M["vol_wti"], M["vol_brt"]],
        })
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=risk_return["Volatility"], y=risk_return["Sharpe"],
            mode="markers+text", text=risk_return["Asset"],
            textposition="top center",
            marker=dict(size=20, color=[COLORS["wti"], COLORS["brent"]]),
            showlegend=False,
        ))
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=100, y1=2, line=dict(dash="dot", color="gray"))
        fig_scatter.update_layout(
            xaxis_title="Annualised Volatility (%)",
            yaxis_title="Sharpe Ratio",
            height=450,
            title="Efficient Frontier Implied: WTI vs Brent",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()
        st.subheader("Global Stress Indicator (Composite)")
        if stress_index is not None and len(stress_index) > 0:
            fig_stress = quant_fig(350)
            fig_stress.add_trace(go.Scatter(x=stress_index.index, y=stress_index.values,
                                            fill="tozeroy", line=dict(color=COLORS["stress"], width=2),
                                            name="Stress Index"))
            fig_stress.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Low Stress")
            fig_stress.add_hline(y=0.6, line_dash="dash", line_color="orange", annotation_text="Elevated")
            fig_stress.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Crisis")
            fig_stress.update_layout(yaxis_title="Composite Stress (0-1)", title="Market Stress Indicator")
            st.plotly_chart(fig_stress, use_container_width=True)

            latest_stress = stress_index.iloc[-1]
            stress_color = "green" if latest_stress < 0.3 else ("orange" if latest_stress < 0.6 else "red")
            st.markdown(f"**Current Stress Level:** <span style='color:{stress_color}; font-weight:bold;'>{latest_stress:.2f}</span>", unsafe_allow_html=True)
        else:
            st.info("Stress index not available.")

    # ----- Tab 6: Executive Summary (unchanged) -----
    with tab_exec:
        st.subheader("Model Diagnostics & Key Metrics")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Market & Regime**")
            st.write(f"- WTI Crude: ${wti0:.2f}")
            st.write(f"- Brent Crude: ${brt0:.2f}")
            st.write(f"- Spread: ${spread:.2f} ({spread/wti0*100:.1f}%)")
            st.write(f"- GeoFactor (normalized): {float(gf.iloc[-1]):+.4f}σ")
            st.write(f"- Risk Regime: WAR ({rbase:+.3f})")
            st.write(f"- War Signal: {ws_val:.5f} ({'active' if war_t else 'subdued'})")
            st.write(f"- DCC α/β: {dcc_a:.4f} / {dcc_b:.4f} (persist = {dcc_a+dcc_b:.4f})")
        with col_right:
            st.markdown("**Volatility & Risk**")
            st.write(f"- WTI Vol (ann.): {M['vol_wti']:.1f}% (shrunk {dw_d['vsa']:.1f}%)")
            st.write(f"- Brent Vol (ann.): {M['vol_brt']:.1f}% (shrunk {db_d['vsa']:.1f}%)")
            st.write(f"- WTI–Brent ρ: {corr:.4f} (EWMA)")
            st.write(f"- Dynamic tail df: {tdf_d:.2f}")
            st.write(f"- Probability Up 10d: {M['prob_up']:.1f}%")
            st.write(f"- VaR 95% 1d: ${M['var95']:+.2f}")
            st.write(f"- CVaR 95% 1d: ${M['cvar95']:+.2f}")
            st.write(f"- Z-Composite: {float(zsc.iloc[-1]):+.4f}")
            st.write(f"- Prob WTI < $40: {M['prob_40']:.2f}%")
            st.write(f"- Prob WTI > $150: {M['prob_150']:.2f}%")

        st.info(
            f"**Bayesian Shrinkage** · WTI {dw_d['vga']:.0f}% → {dw_d['vsa']:.0f}% (w={dw_d['w']:.2f}) · "
            f"Brent {db_d['vga']:.0f}% → {db_d['vsa']:.0f}% (w={db_d['w']:.2f}) · "
            f"Fertilizer: Urea ${usda['urea_price']:.1f}/t, DAP ${usda['dap_price']:.0f}/t · {usda['source']}"
        )

    st.divider()
    st.caption(
        f"GeoQuant · EVT + DCC + GARCH-X · {mc_sims:,} Monte Carlo paths · "
        f"Eduardo Moraes · Quant Data Scientist & Economics · {now_sp.strftime('%d %b %Y')}"
    )

else:
    st.info("Configure parameters in the sidebar and press 'Run Full Analysis'.")

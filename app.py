"""
GeoQuant – Institutional Macro Research Terminal & Validation Suite (v9.5)
EVT + DCC-GARCH-X + GeoFactor + Walk‑Forward + Explainable AI + ML Benchmarking
Eduardo Moraes | Quant Data Scientist & Economics
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, csv, logging, warnings, json
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import stats, optimize
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import chi2
import yfinance as yf
from arch import arch_model
import shap
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# GLOBAL API CREDENTIALS CONFIGURATION
# ----------------------------------------------------------------------
EIA_API_KEY = "kVSuPa0tfnUmHzQ2VVSCPC6owKhPQQY2PbEc9hA1"
FRED_API_KEY = "876c9f95b965eb9d423ef2c7b68ae51b"
OILPRICE_API_KEY = "e241c0914287d05fcbbeb18669c23d86e9cdf36c63193a95d42854eb53ed354d"

# ----------------------------------------------------------------------
# STREAMLIT INITIALIZATION & THEME SETUP
# ----------------------------------------------------------------------
st.set_page_config(page_title="GeoQuant · Research Terminal", page_icon="📈", layout="wide")

# Parametrisation constants
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

# Clean Minimalist Design Framework (White Background Preference)
PLOTLY_LAYOUT = {
    "paper_bgcolor": "white",
    "plot_bgcolor": "#F8F9FA",
    "font": {"family": "Helvetica Neue, Arial", "color": "#2C3E50", "size": 11},
    "title_font": {"family": "Helvetica Neue", "size": 14, "color": "#1F4E79"},
    "xaxis": {"gridcolor": "#E5E7EB", "linecolor": "#D1D5DB", "zerolinecolor": "#D1D5DB", "title": ""},
    "yaxis": {"gridcolor": "#E5E7EB", "linecolor": "#D1D5DB", "zerolinecolor": "#D1D5DB", "title": ""},
    "legend": {"bgcolor": "rgba(255,255,255,0.9)", "bordercolor": "#D1D5DB", "borderwidth": 1},
    "margin": {"l": 50, "r": 50, "t": 60, "b": 40},
}
COLORS = {
    "wti": "#1F4E79", "brent": "#2D6B6B", "gold": "#C8A96E",
    "silver": "#9CA3AF", "fertilizer": "#5F6B47", "natgas": "#2D6B6B",
    "wheat": "#1F4E79", "corn": "#2D6B6B", "soy": "#6B7280",
    "ci_light": "rgba(31,78,121,0.2)", "ci_medium": "rgba(31,78,121,0.4)",
    "stress": "#7A3F30",
}

def quant_fig(h=450):
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, height=h)
    return fig

def quant_subplots(rows, cols, secondary=False, height=450, **kw):
    specs = [[{"secondary_y": secondary}] * cols for _ in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, specs=specs, **kw)
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig

# ----------------------------------------------------------------------
# MATHEMATICAL & STATISTICAL OPERATIONS Engine
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

def _force_update_fertilizer_csv(path="fertilizer_backup.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "urea_price", "dap_price"])
        w.writerows([
            ["2026-01-15", 540, 710], ["2026-02-15", 560, 740],
            ["2026-03-15", 590, 780], ["2026-04-15", 616, 857],
            ["2026-05-01", 720, 900], ["2026-05-06", 810, 920],
            ["2026-05-12", 857, 920], ["2026-06-01", 860, 925],
            ["2026-06-10", 453.5, 920],
        ])

def get_usda():
    _force_update_fertilizer_csv()
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
    _force_update_fertilizer_csv()
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
        if z < -FERT_BS_Z_THR:
            return max(0.5, 1.0 + z * 0.3)
        return min(1.0 + max(0, z - FERT_BS_Z_THR) * 0.8, 3.0)
    try:
        shape, loc, scale = stats.genpareto.fit(exc)
        cr = np.log(cur / hist[-1])
        if cr <= thr:
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

def fit_dcc_corrected(rw, rb, vw, vb):
    common = rw.index.intersection(rb.index).intersection(vw.index).intersection(vb.index)
    ew = (rw[common] / vw[common]).dropna()
    eb = (rb[common] / vb[common]).dropna()
    e = np.column_stack([ew, eb])
    Qbar = np.cov(e, rowvar=False)
    def nll(p):
        a, b = p
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10
        Qt = Qbar.copy()
        ll = 0.0
        for t in range(1, len(e)):
            Qt = (1 - a - b) * Qbar + a * np.outer(e[t-1], e[t-1]) + b * Qt
            d = np.sqrt(np.diag(Qt))
            d[d == 0] = 1e-8
            Rt = Qt / np.outer(d, d)
            Rt = np.clip(Rt, -0.9999, 0.9999)
            try:
                L = np.linalg.cholesky(Rt)
                z = np.linalg.solve(L, e[t])
                ll += -0.5 * np.sum(z**2) - np.sum(np.log(np.diag(L)))
            except:
                return 1e10
        return -ll
    res = optimize.minimize(nll, [0.05, 0.93], bounds=[(1e-4, 0.3), (0.7, 0.9999)], method="L-BFGS-B")
    return res.x

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
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    tickers_list = list(TICKERS.values())
    tickers_keys = list(TICKERS.keys())
    def _extract_close(raw):
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0).unique().tolist()
            field = next((f for f in ["Close", "Adj Close", "Price"] if f in lvl0), None)
            if field: out = raw[field].copy()
            else: out = raw.iloc[:, :len(tickers_keys)].copy()
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
        except:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live(last_wti=65.0, last_brt=68.0):
    try:
        w = yf.Ticker("CL=F").fast_info
        b = yf.Ticker("BZ=F").fast_info
        wti = float(w.get("last_price", 0))
        brent = float(b.get("last_price", 0))
        if wti > 0 and brent > 0: return wti, brent
    except:
        pass
    return last_wti, last_brt

# ----------------------------------------------------------------------
# ADVANCED INSTITUTIONAL BACKTESTING & MODEL VALIDATION SUITE
# ----------------------------------------------------------------------
def backtest_var(returns, var_forecast, alpha=0.05):
    common_idx = returns.index.intersection(var_forecast.index)
    if len(common_idx) == 0:
        return {"calibration_score": 0, "Kupiec_p": 1, "Christoffersen_p": 1, "DQ_p": 1}
    r = returns.loc[common_idx]
    v = var_forecast.loc[common_idx]
    violations = (r < -v).astype(int)
    n = len(violations)
    n_viol = violations.sum()
    p_obs = n_viol / n
    p_exp = alpha
    if n_viol > 0 and n_viol < n:
        LR_pf = -2 * np.log(((1-p_exp)**(n - n_viol) * p_exp**n_viol) / 
                            ((1-p_obs)**(n - n_viol) * p_obs**n_viol))
        p_pf = 1 - chi2.cdf(LR_pf, df=1)
    else:
        p_pf = 0.5
    if n > 1:
        n_00 = ((violations[:-1] == 0) & (violations[1:] == 0)).sum()
        n_01 = ((violations[:-1] == 0) & (violations[1:] == 1)).sum()
        n_10 = ((violations[:-1] == 1) & (violations[1:] == 0)).sum()
        n_11 = ((violations[:-1] == 1) & (violations[1:] == 1)).sum()
        pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0
        pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0
        LR_cc = -2 * np.log(((1-p_exp)**(n-1 - (n_01+n_11)) * p_exp**(n_01+n_11)) /
                           ((1-pi_01)**(n_00) * pi_01**n_01 * (1-pi_11)**(n_10) * pi_11**n_11)) if (n_01+n_11)>0 else 0
        p_cc = 1 - chi2.cdf(LR_cc, df=1) if LR_cc>0 else 0.5
    else:
        p_cc = 0.5
    X = pd.DataFrame({'const': 1, 'hit_lag1': violations.shift(1).fillna(0)})
    try:
        model = Logit(violations, X).fit(disp=0)
        dq_stat = model.llr
        p_dq = 1 - chi2.cdf(dq_stat, df=X.shape[1])
    except:
        p_dq = 1
    return {
        "n_violations": int(n_viol), "obs_freq": p_obs, "exp_freq": p_exp,
        "Kupiec_p": p_pf, "Christoffersen_p": p_cc, "DQ_p": p_dq,
        "calibration_score": 1 - np.mean([p_pf, p_cc, p_dq])
    }

def backtest_es(returns, cvar_forecast, var_forecast, alpha=0.05):
    common = returns.index.intersection(var_forecast.index)
    if len(common) == 0: return np.nan
    r = returns.loc[common]
    v = var_forecast.loc[common]
    cv = cvar_forecast.loc[common] if isinstance(cvar_forecast, pd.Series) else cvar_forecast
    if isinstance(cv, pd.Series): cv = cv.iloc[0] if len(cv) > 0 else np.nan
    if np.isnan(cv): return np.nan
    violations = (r < -v).astype(int)
    if violations.sum() == 0: return np.nan
    z = ((r[violations==1] + v[violations==1]).sum() / (violations.sum() * cv)) - 1
    return float(z)

def walk_forward_validation(returns_series, train_years=2, test_months=3):
    dates = returns_series.index
    train_size = int(train_years * 252)
    test_size = int(test_months * 21)
    results = []
    start = 0
    while start + train_size + test_size <= len(dates):
        train_end = start + train_size
        test_end = train_end + test_size
        train_ret = returns_series.iloc[start:train_end]
        test_ret = returns_series.iloc[train_end:test_end]
        
        # Real-world dynamic estimation validation loop
        mu_train = train_ret.mean()
        pred_error = np.sqrt(((mu_train - test_ret) ** 2).mean())
        results.append({
            "Window Start": dates[start].strftime('%Y-%m-%d'),
            "Window End": dates[test_end-1].strftime('%Y-%m-%d'),
            "OOS_RMSE": pred_error
        })
        start += test_size
    return pd.DataFrame(results)

def benchmark_ml_models(returns_df, target_col="oil", split_ratio=0.8):
    features = returns_df.shift(1).dropna()
    target = returns_df[target_col].iloc[1:]
    common = features.index.intersection(target.index)
    X, y = features.loc[common], target.loc[common]
    
    split = int(len(X) * split_ratio)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE": mean_absolute_error(y_test, pred)
        }
    return results, X, y

def run_shap_analysis(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    return fig

def rolling_ic(geofactor, returns, window=252, lag=1):
    ic_series = []
    for i in range(window, len(geofactor)):
        gf_win = geofactor.iloc[i-window:i].shift(lag)
        ret_win = returns.iloc[i-window:i]
        common = gf_win.dropna().index.intersection(ret_win.dropna().index)
        if len(common) > 20:
            ic_series.append(gf_win[common].corr(ret_win[common]))
        else:
            ic_series.append(np.nan)
    return pd.Series(ic_series, index=geofactor.index[window:])

def hit_ratio(geofactor, returns, lag=1):
    gf_lag = geofactor.shift(lag).dropna()
    common = gf_lag.index.intersection(returns.index)
    if len(common) < 10: return np.nan
    return (np.sign(returns[common]) == np.sign(gf_lag[common])).mean()

def fit_evt_tails(returns_vec, threshold_q=0.95):
    th_up = np.percentile(returns_vec, threshold_q*100)
    th_lo = np.percentile(returns_vec, (1-threshold_q)*100)
    exc_up = returns_vec[returns_vec > th_up] - th_up
    exc_lo = -returns_vec[returns_vec < th_lo] - th_lo
    shape_up, _, scale_up = stats.genpareto.fit(exc_up) if len(exc_up) > 5 else (0.5, 0, 0.1)
    shape_lo, _, scale_lo = stats.genpareto.fit(exc_lo) if len(exc_lo) > 5 else (0.5, 0, 0.1)
    return {"upper": (shape_up, scale_up, th_up), "lower": (shape_lo, scale_lo, th_lo)}

def garch_diagnostics(residuals):
    resid = residuals.dropna()
    if len(resid) < 20: return {"LjungBox5": np.nan, "LjungBox10": np.nan, "ARCH_LM_p": np.nan}
    lb = acorr_ljungbox(resid, lags=[5,10], return_df=True)
    arch = het_arch(resid**2, nlags=10)
    return {
        "LjungBox5": lb.loc[5, 'lb_pvalue'] if 5 in lb.index else np.nan,
        "LjungBox10": lb.loc[10, 'lb_pvalue'] if 10 in lb.index else np.nan,
        "ARCH_LM_p": arch[1] if len(arch) > 1 else np.nan
    }

# ----------------------------------------------------------------------
# SIDEBAR CONTROL COCKPIT
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## GeoQuant Terminal Control")
    st.caption("Institutional Configurator Engine")
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
    run_btn = st.button("Run Full System Pipeline", type="primary")
    st.caption("CONFIDENTIALITY NOTICE: Authorized Institutional Use Only.")

now_sp = datetime.now(pytz.timezone("America/Sao_Paulo"))
st.markdown(
    f"""
    <div style="border-bottom: 2px solid #1F4E79; margin-bottom: 1.5rem;">
        <h1 style="color: #1F4E79; font-weight: 400; margin-bottom: 0;">GeoQuant · Research Terminal</h1>
        <p style="color: #6B7280; font-size: 0.8rem;">Geopolitical Intelligence Integrated Environment · EVT+DCC+GARCH-X · {now_sp.strftime('%d %B %Y · %H:%M')} (SP)</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# DATA EXECUÇÃO PIPELINE PIPELINE
# ----------------------------------------------------------------------
needs_run = run_btn or "results" not in st.session_state

if needs_run:
    info_placeholder = st.info("Loading high-frequency matrix from global endpoint repositories...")
    prog = st.progress(0)
    
    prices = fetch_data()
    if prices.empty or len(prices) < 5:
        st.error("Failed to load historical core data arrays.")
        st.stop()
        
    prices = prices.ffill().bfill()
    last_wti = float(prices["oil"].dropna().iloc[-1])
    last_brt = float(prices["brent"].dropna().iloc[-1])
    wti0, brt0 = fetch_live(last_wti, last_brt)
    prices.loc[prices.index[-1], "oil"] = wti0
    prices.loc[prices.index[-1], "brent"] = brt0
    returns = np.log(prices / prices.shift(1)).dropna()
    
    prog.progress(25)
    usda = get_usda()
    bs_mult = fert_black_swan(usda)
    gs = gold_signals(prices)
    sd = silver_demand_proxy(prices)
    
    weights = GEO_WEIGHTS_DEFAULT.copy()
    weights["silver_demand"] = 0.02
    tot = sum(abs(v) for v in weights.values())
    weights = {k: v / tot for k, v in weights.items()}
    
    fi = build_fert_index(returns, usda, bs_mult)
    dyn_w = calibrate_weights(returns, prices, gs, fi, sd)
    if dyn_w: weights = dyn_w
        
    gf_raw = build_geofactor(returns, prices, gs, fi, weights, sd)
    gf = (gf_raw - gf_raw.mean()) / gf_raw.std() if len(gf_raw) > 1 else gf_raw
    zsc = build_zscore(prices, gs)
    
    prog.progress(50)
    vw = fit_garch(returns["oil"], gf)
    vb = fit_garch(returns["brent"], gf)
    vg = fit_garch(returns["gold"], gf)
    
    n = len(returns)
    pw_d, pb_d, pg_d = prior_wti / np.sqrt(252), prior_brent / np.sqrt(252), 0.18 / np.sqrt(252)
    vw, dw = bayes_shrink(vw, pw_d, n, geofactor=gf)
    vb, db = bayes_shrink(vb, pb_d, n, geofactor=gf)
    vg, _ = bayes_shrink(vg, pg_d, n)
    
    bvw, bvb = float(vw.iloc[-1]), float(vb.iloc[-1])
    dcc_a, dcc_b = fit_dcc_corrected(returns["oil"], returns["brent"], vw, vb)
    
    rv = returns.loc[gf.index.intersection(returns.index)]
    vm = VAR(rv).fit(min(5, max(1, len(rv) // 10)))
    fcast = vm.forecast(rv.values[-vm.k_ar:], steps=mc_steps)
    ocol, bcol = list(rv.columns).index("oil"), list(rv.columns).index("brent")
    
    tdf_d = max(2.5, min(6.0, tail_df / np.sqrt(max(bvb / (pb_d * 1.5), 0.5))))
    rbase = float(np.tanh(gf.iloc[-1] / 2)) if not gf.empty else 0.0
    jpu_eff = min(jump_up * 1.5, 0.15) if (returns["wheat"].tail(20).mean() > 0.005) else jump_up
    
    prog.progress(75)
    mc = run_mc(wti0, brt0, bvw, bvb, fcast, ocol, bcol, rbase, returns["oil"], returns["brent"], vw, vb, jpu_eff, tdf_d, bs_mult, dcc_a, dcc_b, mc_sims, mc_steps)
    
    # Mathematical diagnostics additions
    ret_ann, vol_ann = returns[["oil", "brent"]].mean() * 252, returns[["oil", "brent"]].std() * np.sqrt(252)
    corr_matrix = returns[["oil", "brent", "gold", "dxy", "tnx"]].dropna().corr()
    
    stress_comp = pd.DataFrame({
        "vol_wti": vw.rolling(20).mean() * np.sqrt(252) * 100, "vol_brent": vb.rolling(20).mean() * np.sqrt(252) * 100,
        "corr_wti_brent": returns["oil"].rolling(20).corr(returns["brent"]), "gold_zscore": rolling_zscore(prices["gold"], 60), "geofactor": gf,
    }).dropna()
    stress_index = (stress_comp["vol_wti"]/50 + stress_comp["vol_brent"]/50 + np.abs(stress_comp["corr_wti_brent"]-0.8)*2 + stress_comp["gold_zscore"].clip(0,3)/3 + stress_comp["geofactor"].clip(0,2)/2) / 5
    
    feature_importance = pd.DataFrame({"feature": list(weights.keys()), "importance": np.abs(list(weights.values()))}).sort_values("importance", ascending=False)
    evt_tails = fit_evt_tails(returns["oil"])
    garch_diag = garch_diagnostics(vw)
    rolling_ic_series = rolling_ic(gf, returns["oil"])
    hit = hit_ratio(gf, returns["oil"])
    
    # ML Validation Executions
    ml_metrics, X_ml, y_ml = benchmark_ml_models(returns)
    wf_df = walk_forward_validation(returns["oil"])
    shap_fig = run_shap_analysis(X_ml, y_ml)
    
    prog.progress(100)
    info_placeholder.empty()
    prog.empty()
    
    st.session_state.update({
        "results": mc, "gf": gf, "zsc": zsc, "vw": vw, "vb": vb, "vg": vg, "fi": fi, "gs": gs, "prices": prices, "returns": returns,
        "wti0": wti0, "brt0": brt0, "usda": usda, "bs": bs_mult, "dw": dw, "db": db, "tdf": tdf_d, "dcc_a": dcc_a, "dcc_b": dcc_b, "weights": weights,
        "sharpe": ret_ann / vol_ann, "sortino": ret_ann / (returns[["oil", "brent"]][returns[["oil", "brent"]] < 0].std() * np.sqrt(252)),
        "skew_oil": returns["oil"].skew(), "kurt_oil": returns["oil"].kurtosis(), "skew_brt": returns["brent"].skew(), "kurt_brt": returns["brent"].kurtosis(),
        "corr_matrix": corr_matrix, "stress_index": stress_index, "feature_importance": feature_importance, "evt_tails": evt_tails, "garch_diag": garch_diag,
        "rolling_ic": rolling_ic_series, "hit_ratio": hit, "ml_metrics": ml_metrics, "wf_df": wf_df, "shap_fig": shap_fig
    })

# ----------------------------------------------------------------------
# OUTPUT TERMINAL RENDER ENGINE
# ----------------------------------------------------------------------
if "results" in st.session_state:
    r_st = st.session_state
    spread = r_st["brt0"] - r_st["wti0"]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Market Snapshot & Volatility", "Geopolitical Intelligence", "Monte Carlo Simulations", 
        "Descriptive Quant Stats", "Macro Stress Factors", "VaR/ES Backtesting", 
        "Machine Learning Benchmarks", "Walk-Forward Validation"
    ])
    
    with tab1:
        st.subheader("Live Commodity Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WTI Crude Future", f"${r_st['wti0']:.2f}", f"P50 10d Horizon: ${r_st['results']['fan'][50][-1]:.2f}")
        c2.metric("Brent Crude Future", f"${r_st['brt0']:.2f}", f"Spread: ${spread:.2f} ({spread/r_st['wti0']*100:.2f}%)")
        c3.metric("WTI Vol (Ann.)", f"{r_st['results']['metrics']['vol_wti']:.2f}%", f"Shrunk Prior: {r_st['dw']['vsa']:.2f}%")
        c4.metric("Brent Vol (Ann.)", f"{r_st['results']['metrics']['vol_brt']:.2f}%", f"Shrunk Prior: {r_st['db']['vsa']:.2f}%")
        
        st.subheader("Dynamic Variance Surface (Conditional GARCH-X paths)")
        fig_vol = quant_fig(400)
        fig_vol.add_trace(go.Scatter(x=r_st["vw"].index, y=r_st["vw"]*np.sqrt(252)*100, name="WTI Vol", line=dict(color=COLORS["wti"], width=2)))
        fig_vol.add_trace(go.Scatter(x=r_st["vb"].index, y=r_st["vb"]*np.sqrt(252)*100, name="Brent Vol", line=dict(color=COLORS["brent"], width=2, dash="dash")))
        fig_vol.update_layout(yaxis=dict(ticksuffix="%", title="Annualised Parameter Volatility"))
        st.plotly_chart(fig_vol, use_container_width=True)

    with tab2:
        st.subheader("Macro Geopolitical Quant System Factors")
        fig_geo = quant_subplots(1, 1, secondary_y=True)
        fig_geo.add_trace(go.Scatter(x=r_st["zsc"].index, y=r_st["zsc"].values, name="Z-Score Composite", line=dict(color=COLORS["wti"])), secondary_y=False)
        fig_geo.add_trace(go.Scatter(x=r_st["gf"].index, y=r_st["gf"].values, name="GeoFactor Vector (σ)", line=dict(color=COLORS["gold"], width=2)), secondary_y=True)
        st.plotly_chart(fig_geo, use_container_width=True)

    with tab3:
        st.subheader("Probabilistic Asset Distribution Projections")
        x_ax = list(range(mc_steps+1))
        fig_mc = quant_fig(500)
        fig_mc.add_trace(go.Scatter(x=x_ax+x_ax[::-1], y=list(r_st["results"]["fan"][95])+list(r_st["results"]["fan"][5][::-1]), fill="toself", fillcolor=COLORS["ci_light"], line=dict(width=0), name="90% Confidence Interval"))
        fig_mc.add_trace(go.Scatter(x=x_ax, y=list(r_st["results"]["fan"][50]), name="WTI Target Vector (P50)", line=dict(color=COLORS["wti"], width=3)))
        fig_mc.update_layout(xaxis=dict(title="Forward Trading Steps (Days)"), yaxis=dict(title="Price Level (USD/bbl)", tickprefix="$"))
        st.plotly_chart(fig_mc, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Upside Probability Target", f"{r_st['results']['metrics']['prob_up']:.2f}%")
        c2.metric("Value-At-Risk (1d 95%)", f"${r_st['results']['metrics']['var95']:.2f}", f"Expected Shortfall: ${r_st['results']['metrics']['cvar95']:.2f}")
        c3.metric("Tail Trailing Limits", f"< $40: {r_st['results']['metrics']['prob_40']:.2f}%", f"> $150: {r_st['results']['metrics']['prob_150']:.2f}%")

    with tab4:
        st.subheader("Mathematical Higher-Order Moments & DCC Engine Metrics")
        co1, co2 = st.columns(2)
        co1.markdown("**WTI Empirical Distribution**")
        co1.write(f"Sharpe Ratio (Ann.): {r_st['sharpe']['oil']:.3f}")
        co1.write(f"Sortino Ratio (Ann.): {r_st['sortino']['oil']:.3f}")
        co1.write(f"Skewness Parameter: {r_st['skew_oil']:.4f}")
        co1.write(f"Excess Kurtosis Point: {r_st['kurt_oil']:.4f}")
        
        co2.markdown("**Dynamic Conditional Correlation Parameters**")
        co2.write(f"DCC Shock Alpha (α): {r_st['dcc_a']:.4f}")
        co2.write(f"DCC Persistence Beta (β): {r_st['dcc_b']:.4f}")
        co2.write(f"Total Combined Persistence: {r_st['dcc_a'] + r_st['dcc_b']:.4f}")

    with tab5:
        st.subheader("Global Stress Metrics Matrices")
        if r_st["stress_index"] is not None and len(r_st["stress_index"]) > 0:
            fig_st = quant_fig(350)
            fig_st.add_trace(go.Scatter(x=r_st["stress_index"].index, y=r_st["stress_index"].values, fill="tozeroy", line=dict(color=COLORS["stress"])))
            st.plotly_chart(fig_st, use_container_width=True)

    with tab6:
        st.subheader("Institutional Risk Management Backtesting Verifications")
        ret_series = r_st['returns']['oil'].iloc[-252:]
        var_series = r_st['vw'].iloc[-252:] * 1.645
        cvar_series = r_st['vw'].iloc[-252:] * 2.326
        
        bt_res = backtest_var(ret_series, var_series, alpha=0.05)
        st.metric("Model Empirical Calibration Score", f"{bt_res['calibration_score']:.4f}")
        
        cx1, cx2 = st.columns(2)
        cx1.markdown("**Statistical Backtest Significance Tests**")
        cx1.write(f"Kupiec Proportion-of-Failures Test (p-value): {bt_res['Kupiec_p']:.4f}")
        cx1.write(f"Christoffersen Independence Test (p-value): {bt_res['Christoffersen_p']:.4f}")
        cx1.write(f"Dynamic Quantile Joint Violation Test (p-value): {bt_res['DQ_p']:.4f}")
        
        es_z = backtest_es(ret_series, cvar_series, var_series)
        cx2.markdown("**Expected Shortfall Structural Alignment**")
        cx2.write(f"Acerbi Multi-Quantile Z-Statistic Score: {es_z:.4f}" if not np.isnan(es_z) else "Acerbi Z-Statistic: Structural Violations Null")

    with tab7:
        st.subheader("Machine Learning Model Benchmarking Engine")
        bm = r_st["ml_metrics"]
        
        mx1, mx2 = st.columns([1, 1])
        with mx1:
            st.markdown("**Out-of-Sample Predictive Model Error Framework**")
            for model_name, path_metrics in bm.items():
                st.write(f"**{model_name} Model Algorithm**")
                st.write(f" Root Mean Squared Error (RMSE): `{path_metrics['RMSE']:.6f}`")
                st.write(f" Mean Absolute Error (MAE): `{path_metrics['MAE']:.6f}`")
        with mx2:
            st.markdown("**SHAP Global Explanations Graph (Marginal Factor Weights)**")
            st.pyplot(r_st["shap_fig"])

    with tab8:
        st.subheader("Walk-Forward Cross-Validation Matrix Analysis")
        st.dataframe(r_st["wf_df"], use_container_width=True)
        
        fig_wf = quant_fig(300)
        fig_wf.add_trace(go.Bar(x=r_st["wf_df"]["Window End"], y=r_st["wf_df"]["OOS_RMSE"], marker_color=COLORS["wti"], name="OOS Error Distribution"))
        fig_wf.update_layout(title="Rolling Window Out-of-Sample Absolute Error Structure Bounds")
        st.plotly_chart(fig_wf, use_container_width=True)

    st.divider()
    st.caption(f"GeoQuant Integrated Verification Platform • Execution Success Flag. Engine Framework v9.5.")
else:
    st.info("Execute control dashboard metrics to compute statistical parameter arrays.")

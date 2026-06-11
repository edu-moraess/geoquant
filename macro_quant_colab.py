# ╔══════════════════════════════════════════════════════════════════╗
# ║   MACRO GEOPOLITICAL QUANT MODEL  —  Colab Edition               ║
# ║   Eduardo Moraes | Quant Data Scientist & Economics              ║
# ║   Quantitative Research · Professional · Precision               ║
# ╚══════════════════════════════════════════════════════════════════╝
# Execute todas as células em ordem. (Cada célula é delimitada por # %% ──)

# %% ─── CELL 0 · INSTALAÇÃO ──────────────────────────────────────────
!pip install arch pyyaml scipy statsmodels scikit-learn yfinance --quiet 2>/dev/null
print("✔ Dependências instaladas.")

# %% ─── CELL 1 · IMPORTS & CONFIGURAÇÃO INICIAL ──────────────────────
import subprocess, sys, pytz, warnings, os, requests, csv, logging
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from scipy.interpolate import PchipInterpolator
from scipy import stats, optimize
from sklearn.linear_model import LassoCV
from statsmodels.tsa.vector_ar.var_model import VAR
import yfinance as yf
from arch import arch_model
from IPython.display import display, HTML, clear_output

warnings.filterwarnings("ignore")
print("✔ Bibliotecas carregadas.")

# Configurações de estilo (limpo, profissional)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
})

display(HTML("<div style='background:#1F4E79; padding:1rem; color:white;'><b>◆ GeoQuant · Macro Research Terminal</b><br>EVT + DCC + GARCH-X + Adaptive Bayes Shrinkage</div>"))

# %% ─── CELL 2 · CONFIGURAÇÃO CENTRALIZADA ───────────────────────────
CONFIG = {
    "output_dir":           "macro_runs",
    "tickers": {
        "oil": "CL=F", "brent": "BZ=F", "natgas": "NG=F",
        "gold": "GC=F", "silver": "SI=F", "copper": "HG=F",
        "wheat": "ZW=F", "corn": "ZC=F", "soy": "ZS=F",
        "dxy": "DX-Y.NYB", "eur": "EURUSD=X", "tnx": "^TNX",
    },
    "mc_steps":             10,
    "mc_sims":              10_000,
    "mc_seed":              42,
    "max_daily_vol":        0.08,
    "max_drift":            0.02,
    "tail_df_base":         3.0,
    "tail_df_min":          2.5,
    "tail_df_max":          6.0,
    "wti_min":              40,
    "wti_max":              200,
    "spread_min_pct":       -0.05,
    "spread_max_pct":        0.30,
    "guerra_start":         "2026-02-28",
    "jump_prob_up":          0.07,
    "jump_prob_down":        0.03,
    "jump_skew_up_normal":   0.045,
    "jump_skew_up_extreme":  0.135,
    "jump_prob_extreme":     0.15,
    "jump_skew_down":        0.025,
    "regime_noise_std":      0.05,
    "enable_pchip_fill":     True,
    "vol_prior_wti_annual":  0.35,
    "vol_prior_brent_annual":0.35,
    "vol_prior_gold_annual": 0.18,
    "vol_shrink_n_full":     252,
    "geo_weights": {
        "oil_vol":0.22,"gold":0.09,"gold_real":0.09,
        "dxy":-0.10,"spread":0.09,"fert":0.22,
        "wheat":0.07,"copper":0.04,"natgas_vol":0.06,
    },
    "zscore_weights":  {"oil_gold":0.40,"oil_natgas":0.35,"gold_real":0.25},
    "fert_black_swan_z_threshold": 1.5,
    "fert_evt_threshold_q":        0.90,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
print(f"✔ Config OK · Output → {CONFIG['output_dir']}/")

# %% ─── CELL 3 · FUNÇÕES AUXILIARES (TODAS) ──────────────────────────
def rolling_zscore(series, window=60):
    mu = series.rolling(window).mean()
    sig = series.rolling(window).std()
    return (series - mu) / sig.replace(0, np.nan)

def fill_intraday_gaps(series, max_gap_hours=2):
    if not CONFIG["enable_pchip_fill"]:
        return series.ffill()
    series = series.copy()
    if not isinstance(series.index, pd.DatetimeIndex):
        return series.ffill()
    valid = series.notna()
    if valid.sum() < 2:
        return series.ffill()
    x = series.index[valid].astype(np.int64)
    y = series[valid].values
    try:
        filled = pd.Series(PchipInterpolator(x, y)(series.index.astype(np.int64)), index=series.index)
        filled[valid] = series[valid]
        return filled
    except:
        return series.ffill()

def generate_default_fertilizer_csv(csv_path="fertilizer_backup.csv"):
    if os.path.exists(csv_path):
        return
    with open(csv_path, "w", newline="") as f:
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
            ["2026-06-10", 453.5, 920],  # Dado real de junho/2026
        ])

def load_fertilizer_backup(csv_path="fertilizer_backup.csv"):
    generate_default_fertilizer_csv(csv_path)
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date").sort_index()
        last = df.iloc[-1]
        return {"urea_price": float(last["urea_price"]), "urea_period": str(last.name.date()),
                "dap_price": float(last["dap_price"]), "dap_period": str(last.name.date()),
                "source": "local CSV backup"}
    except:
        return None

def get_live_urea_price():
    return load_fertilizer_backup() or {
        "urea_price": 453.5, "urea_period": "2026-06-10",
        "dap_price": 920, "dap_period": "2026-06-10", "source": "hardcoded"
    }

def compute_fertilizer_black_swan(usda_data):
    urea_hist = []
    if os.path.exists("fertilizer_backup.csv"):
        try:
            df = pd.read_csv("fertilizer_backup.csv", parse_dates=["date"], index_col="date")
            urea_hist = df["urea_price"].dropna().values
        except:
            pass
    cur = usda_data.get("urea_price")
    if cur is None or len(urea_hist) < 10:
        return 1.0, 0.0
    rets = np.diff(np.log(urea_hist))
    thr = np.quantile(rets, CONFIG["fert_evt_threshold_q"])
    exc = rets[rets > thr] - thr
    if len(exc) < 5:
        mu, sig = np.mean(urea_hist), np.std(urea_hist)
        if sig == 0:
            return 1.0, 0.0
        z = (cur - mu) / sig
        return min(1.0 + max(0, z - CONFIG["fert_black_swan_z_threshold"]) * 0.8, 3.0), z
    try:
        shape, loc, scale = stats.genpareto.fit(exc)
        cr = np.log(cur / urea_hist[-1])
        if cr <= thr:
            return 1.0, 0.0
        p = 1 - stats.genpareto.cdf(cr - thr, shape, loc=loc, scale=scale)
        return 1.0 + min(p * 5, 2.0), cr
    except:
        return 1.0, 0.0

def build_gold_signals(prices):
    silver = prices["silver"].replace(0, np.nan)
    if silver.median() > 500:
        silver = silver / 100
    gold_real = prices["gold"] / (1 + prices["tnx"].replace(0, np.nan) / 100 * 5.0)
    silver_gold = silver / prices["gold"].replace(0, np.nan)
    return {"gold_real": gold_real, "silver_gold": silver_gold,
            "gold_real_ret_roll": np.log(gold_real / gold_real.shift(1)).rolling(20).mean(),
            "silver_gold_roll": np.log(silver_gold / silver_gold.shift(1)).rolling(20).mean()}

def build_silver_demand_proxy(prices):
    if "copper" not in prices.columns or "brent" not in prices.columns:
        return pd.Series(index=prices.index, data=0.0)
    cr = prices["copper"].pct_change().dropna()
    br = prices["brent"].pct_change().dropna()
    common = cr.index.intersection(br.index)
    return (0.6 * cr[common] + 0.4 * br[common]).rolling(20).mean().reindex(prices.index, method="ffill").fillna(0.0)

def build_geofactor(returns, prices, gold_signals, fert_index, weights, silver_demand=None):
    spread = (prices["brent"] - prices["oil"]) / prices["brent"].replace(0, np.nan)
    geo = (weights.get("oil_vol", 0) * returns["oil"].rolling(20).std() +
           weights.get("gold", 0) * returns["gold"].rolling(20).mean() +
           weights.get("gold_real", 0) * gold_signals["gold_real_ret_roll"] +
           weights.get("dxy", 0) * returns["dxy"].rolling(20).mean() +
           weights.get("spread", 0) * spread.rolling(20).mean() +
           weights.get("wheat", 0) * returns["wheat"].rolling(20).mean() +
           weights.get("copper", 0) * returns["copper"].rolling(20).mean() +
           weights.get("natgas_vol", 0) * returns["natgas"].rolling(20).std())
    if silver_demand is not None:
        common = geo.dropna().index.intersection(silver_demand.dropna().index)
        if len(common) > 0:
            geo.loc[common] += weights.get("silver_demand", 0) * silver_demand.loc[common]
    common = geo.dropna().index.intersection(fert_index.dropna().index)
    geo.loc[common] += weights.get("fert", 0) * fert_index.loc[common]
    return geo.dropna().clip(geo.dropna().quantile(0.05), geo.dropna().quantile(0.95))

def calibrate_geo_weights(returns, prices, gold_signals, fert_index, silver_demand=None, window=60):
    spread = (prices["brent"] - prices["oil"]) / prices["brent"].replace(0, np.nan)
    X = pd.DataFrame({
        "oil_vol": returns["oil"].rolling(20).std(),
        "gold": returns["gold"].rolling(20).mean(),
        "gold_real": gold_signals["gold_real_ret_roll"],
        "dxy": returns["dxy"].rolling(20).mean(),
        "spread": spread.rolling(20).mean(),
        "wheat": returns["wheat"].rolling(20).mean(),
        "copper": returns["copper"].rolling(20).mean(),
        "natgas_vol": returns["natgas"].rolling(20).std(),
        "fert": fert_index,
    })
    if silver_demand is not None:
        X["silver_demand"] = silver_demand
    y = returns["oil"].shift(-1)
    common = y.dropna().index.intersection(X.dropna().index)
    X, y = X.loc[common].dropna(), y.loc[common]
    if len(X) < window:
        return CONFIG["geo_weights"]
    Xc, yc = X.iloc[-window:], y.iloc[-window:]
    Xm, Xs = Xc.mean(), Xc.std().replace(0, 1)
    model = LassoCV(cv=5, random_state=42, alphas=np.logspace(-4, 0, 20), max_iter=2000).fit((Xc - Xm) / Xs, yc)
    coef = model.coef_ / Xs.values
    new_w = {col: coef[i] for i, col in enumerate(X.columns)}
    total = sum(abs(v) for v in new_w.values())
    return {k: v / total for k, v in new_w.items()} if total > 0 else CONFIG["geo_weights"]

def build_composite_zscore(prices, gold_signals, window=60):
    w = min(window, max(20, len(prices) // 2))
    z1 = rolling_zscore(prices["oil"] / prices["gold"].replace(0, np.nan), w)
    z2 = rolling_zscore(prices["oil"] / prices["natgas"].replace(0, np.nan), w)
    z3 = rolling_zscore(gold_signals["gold_real"], w)
    return (CONFIG["zscore_weights"]["oil_gold"] * z1 +
            CONFIG["zscore_weights"]["oil_natgas"] * z2 +
            CONFIG["zscore_weights"]["gold_real"] * z3).dropna()

def build_fertilizer_stress_index(returns, usda_data, black_swan_mult=1.0):
    fert = (0.5 * returns["natgas"].rolling(20).std() +
            0.25 * returns["wheat"].rolling(20).mean() +
            0.25 * returns["corn"].rolling(20).mean())
    if usda_data["urea_price"]:
        fert += np.clip((usda_data["urea_price"] - 380) / 380, -1, 2) * 0.15
    if usda_data["dap_price"]:
        fert += np.clip((usda_data["dap_price"] - 610) / 610, -1, 2) * 0.10
    fert *= black_swan_mult
    return fert.clip(fert.quantile(0.02), fert.quantile(0.98)).dropna()

def fit_garch_x(ret, exog):
    rc = ret.loc[ret.index.intersection(exog.index)] * 100
    xc = exog.loc[rc.index]
    try:
        res = arch_model(rc, x=xc, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt").fit(disp="off")
    except:
        res = arch_model(rc, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt").fit(disp="off")
    return res.conditional_volatility / 100

def bayesian_vol_shrinkage(vol_garch, vol_prior_daily, n_obs, n_full=252, label="", geofactor=None):
    w = np.clip(np.sqrt(n_obs / n_full), 0.10, 0.95)
    v_last = float(vol_garch.iloc[-1])
    prior = vol_prior_daily
    if geofactor is not None and not geofactor.empty:
        prior *= (1.0 + 0.4 * np.tanh(float(geofactor.iloc[-1])))
    lo, hi = prior * 0.50, prior * 1.50
    if lo <= v_last <= hi:
        vs = vol_garch.copy()
        we = 1.0
    else:
        vs = w * vol_garch + (1 - w) * prior
        we = w
    vga = v_last * np.sqrt(252) * 100
    vsa = float(vs.iloc[-1]) * np.sqrt(252) * 100
    return vs, {"label": label, "weight_data": we, "vol_garch_aa": vga, "vol_final_aa": vsa}

def estimate_dcc_params(rw, rb, vw, vb):
    common = rw.index.intersection(rb.index).intersection(vw.index).intersection(vb.index)
    ew = (rw[common] / vw[common]).dropna()
    eb = (rb[common] / vb[common]).dropna()
    c2 = ew.index.intersection(eb.index)
    e = np.column_stack([ew[c2], eb[c2]])

    def nll(p):
        a, b = p
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10
        Qbar = np.cov(e, rowvar=False)
        Q = Qbar.copy()
        ll = 0
        for t in range(1, len(e)):
            Qt = (1 - a - b) * Qbar + a * np.outer(e[t - 1], e[t - 1]) + b * Q
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

def _add_tail_jumps(shocks, vol):
    n = len(shocks)
    u = np.random.rand(n)
    mu = u < 0.025
    md = (u >= 0.025) & (u < 0.05)
    return shocks + np.where(mu, np.random.exponential(0.03, n) * vol, 0) - np.where(md, np.random.exponential(0.02, n) * vol, 0)

def _sample_jumps(n, pu, pd_):
    u = np.random.rand(n)
    mu = u < pu
    md = (u >= pu) & (u < pu + pd_)
    me = np.random.rand(n) < CONFIG["jump_prob_extreme"]
    ju = np.where(me, np.random.exponential(CONFIG["jump_skew_up_extreme"], n), np.random.exponential(CONFIG["jump_skew_up_normal"], n))
    jd = np.random.exponential(CONFIG["jump_skew_down"], n)
    return np.where(mu, ju, np.where(md, -jd, 0)), np.where(mu, ju * 0.95, np.where(md, -jd * 0.90, 0))

def run_monte_carlo(wti_last, brent_last, base_vol, base_vol_brent, forecast,
                    oil_col, brent_col, regime_base, ret_wti, ret_brent,
                    vol_wti_s, vol_brt_s, jpu, tail_df, bs_mult=1.0, dcc_a=0.05, dcc_b=0.93):
    sims = CONFIG["mc_sims"]
    steps = CONFIG["mc_steps"]
    np.random.seed(CONFIG["mc_seed"])
    common = ret_wti.index.intersection(ret_brent.index).intersection(vol_wti_s.index).intersection(vol_brt_s.index)
    ew = (ret_wti[common] / vol_wti_s[common].replace(0, np.nan)).dropna()
    eb = (ret_brent[common] / vol_brt_s[common].replace(0, np.nan)).dropna()
    c2 = ew.index.intersection(eb.index)
    e = np.column_stack([np.clip(ew[c2], -3, 3), np.clip(eb[c2], -3, 3)])
    Qbar = np.cov(e, rowvar=False)
    np.fill_diagonal(Qbar, 1.0)
    eps = e[-1] + np.random.normal(0, 0.05, (sims, 2))
    Qt = np.tile(Qbar, (sims, 1, 1)).copy()
    pu = min(jpu * 1.5, 0.20) if bs_mult > 1.2 else jpu
    pd_ = CONFIG["jump_prob_down"] * (1.3 if bs_mult > 1.2 else 1.0)
    pw = np.zeros((sims, steps + 1))
    pb = np.zeros((sims, steps + 1))
    pw[:, 0] = wti_last
    pb[:, 0] = brent_last
    ra = 1 + 0.5 * np.clip(regime_base + np.random.normal(0, CONFIG["regime_noise_std"], (sims, steps)), -1, 1)
    for t in range(steps):
        outer = np.einsum("si,sj->sij", eps, eps)
        Qt = (1 - dcc_a - dcc_b) * Qbar[np.newaxis] + dcc_a * outer + dcc_b * Qt
        diag = np.clip(np.sqrt(np.diagonal(Qt, axis1=1, axis2=2)), 1e-8, None)
        Rt = Qt / np.einsum("si,sj->sij", diag, diag)
        Rt = np.clip(Rt, -0.9999, 0.9999)
        Rt[:, 0, 0] = Rt[:, 1, 1] = 1.0
        rho = Rt[:, 0, 1]
        sc = np.sqrt(np.clip(1 - rho ** 2, 1e-8, None))
        z = np.random.standard_t(tail_df, (sims, 2))
        zw = z[:, 0]
        zb = rho * z[:, 0] + sc * z[:, 1]
        vw_ = np.clip(base_vol * ra[:, t], 0, CONFIG["max_daily_vol"])
        vb_ = np.clip(base_vol_brent * ra[:, t], 0, CONFIG["max_daily_vol"])
        sw = np.clip(zw * vw_, -4 * vw_, 4 * vw_)
        sb = np.clip(zb * vb_, -4 * vb_, 4 * vb_)
        sw = _add_tail_jumps(sw, vw_)
        sb = _add_tail_jumps(sb, vb_)
        jw, jb = _sample_jumps(sims, pu, pd_)
        sw += jw
        sb += jb
        dw = np.clip(forecast[t, oil_col] * ra[:, t], -CONFIG["max_drift"], CONFIG["max_drift"])
        db = np.clip(forecast[t, brent_col] * ra[:, t], -CONFIG["max_drift"], CONFIG["max_drift"])
        nw = pw[:, t] * np.exp(dw + sw)
        nb = pb[:, t] * np.exp(db + sb)
        sp = np.where(nb > 0, (nb - nw) / nb, 0)
        nw = np.where(sp < CONFIG["spread_min_pct"], nb * (1 + abs(CONFIG["spread_min_pct"])), nw)
        nw = np.where(sp > CONFIG["spread_max_pct"], nb * (1 - CONFIG["spread_max_pct"]), nw)
        nw = np.clip(nw, wti_last * 0.4, wti_last * 2.5)
        nb = np.clip(nb, brent_last * 0.4, brent_last * 2.5)
        pw[:, t + 1] = nw
        pb[:, t + 1] = nb
        eps[:, 0] = np.where(vw_ > 0, sw / vw_, 0)
        eps[:, 1] = np.where(vb_ > 0, sb / vb_, 0)
        eps = np.clip(eps, -5, 5)
    fan = {p: np.percentile(pw, p, axis=0) for p in [5, 25, 50, 75, 95]}
    fan_b = {p: np.percentile(pb, p, axis=0) for p in [5, 25, 50, 75, 95]}
    term = pw[:, -1]
    v95 = np.percentile(pw[:, 1] - wti_last, 5)
    mask = (pw[:, 1] - wti_last) <= v95
    return {"fan": fan, "fan_brent": fan_b, "paths_wti": pw, "paths_brent": pb, "metrics": {
        "vol_wti_aa": base_vol * np.sqrt(252) * 100, "vol_brent_aa": base_vol_brent * np.sqrt(252) * 100,
        "var_95_1d": v95, "cvar_95_1d": float(np.mean((pw[:, 1] - wti_last)[mask])),
        "prob_up_10d": np.mean(term > wti_last) * 100,
        "prob_wti_below_40": np.mean(term < 40) * 100,
        "prob_wti_above_150": np.mean(term > 150) * 100,
        "p95_chg": (fan[95][-1] / wti_last - 1) * 100, "p5_chg": (fan[5][-1] / wti_last - 1) * 100,
    }}

# Funções de plotagem e suporte
def _swiss_ax(ax, title="", ylabel="", xlabel=""):
    ax.set_facecolor("#F8F9FA")
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(True, axis="y", color="#E5E7EB", lw=0.7, ls=":", alpha=0.6)
    ax.grid(False, axis="x")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#D1D5DB")
        ax.spines[spine].set_linewidth(0.8)
    if title:
        ax.set_title(title, pad=14, color="#1F4E79", fontsize=12, fontweight="normal", loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color="#6B7280", fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color="#6B7280", fontsize=9)

def _label(ax, text, x=0.01, y=0.97, fs=7.5):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fs, color="#9E8050", va="top", ha="left",
            fontfamily="monospace", bbox=dict(fc="white", ec="none", alpha=0.9, pad=2))

def save_fig(fig, name):
    ts = datetime.now(pytz.timezone("America/Sao_Paulo")).strftime("%Y-%m-%d_%Hh%M")
    path = os.path.join(CONFIG["output_dir"], f"{name}_{ts}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"  → {path}")
    return path

print("✔ Todas as funções carregadas.")

# %% ─── CELL 4 · DADOS DE MERCADO ────────────────────────────────────
display(HTML("<div style='background:#E5E7EB; border-left:3px solid #1F4E79; padding:0.5rem 1rem;'>⟳ Baixando dados de mercado…</div>"))

prices = yf.download(list(CONFIG["tickers"].values()), start=CONFIG["guerra_start"], progress=False)["Close"]
prices.columns = list(CONFIG["tickers"].keys())
for col in prices.columns:
    prices[col] = fill_intraday_gaps(prices[col])
prices = prices.ffill().dropna()
returns = np.log(prices / prices.shift(1)).dropna()

wti_last = float(yf.Ticker("CL=F").fast_info["last_price"])
brent_last = float(yf.Ticker("BZ=F").fast_info["last_price"])
prices.loc[prices.index[-1], "oil"] = wti_last
prices.loc[prices.index[-1], "brent"] = brent_last

usda_data = get_live_urea_price()
bs_mult, bs_z = compute_fertilizer_black_swan(usda_data)
gold_signals = build_gold_signals(prices)
silver_demand = build_silver_demand_proxy(prices)
if "silver_demand" not in CONFIG["geo_weights"]:
    CONFIG["geo_weights"]["silver_demand"] = 0.02
    t = sum(abs(v) for v in CONFIG["geo_weights"].values())
    for k in CONFIG["geo_weights"]:
        CONFIG["geo_weights"][k] /= t
fert_index = build_fertilizer_stress_index(returns, usda_data, bs_mult)
dyn_w = calibrate_geo_weights(returns, prices, gold_signals, fert_index, silver_demand, window=60)
if dyn_w:
    CONFIG["geo_weights"] = dyn_w
geofactor_raw = build_geofactor(returns, prices, gold_signals, fert_index, CONFIG["geo_weights"], silver_demand)
geofactor = (geofactor_raw - geofactor_raw.mean()) / geofactor_raw.std() if len(geofactor_raw) > 1 else geofactor_raw
z_composite = build_composite_zscore(prices, gold_signals, window=60)

display(HTML(f"""
<div style='background:#1F4E79; padding:1rem; color:white; font-family:monospace;'>
<b>◆ DADOS CARREGADOS</b><br>
Período: {prices.index[0].date()} → {prices.index[-1].date()} ({len(prices)} dias)<br>
WTI: <b>${wti_last:.2f}</b> &nbsp;&nbsp; Brent: <b>${brent_last:.2f}</b><br>
Ureia: ${usda_data['urea_price']:.1f}/t &nbsp; DAP: ${usda_data['dap_price']:.0f}/t &nbsp; [{usda_data['source']}]<br>
BlackSwan multiplier: {bs_mult:.2f} {"⚠ ATIVO" if bs_mult>1.2 else ""}
</div>
"""))

# %% ─── CELL 5 · GARCH-X + BAYESIAN SHRINKAGE + DCC ──────────────────
display(HTML("<div style='background:#E5E7EB; border-left:3px solid #1F4E79; padding:0.5rem 1rem;'>⟳ Ajustando GARCH-X com Shrinkage Bayesiano…</div>"))

vol_oil = fit_garch_x(returns["oil"], geofactor)
vol_brent = fit_garch_x(returns["brent"], geofactor)
vol_gold = fit_garch_x(returns["gold"], geofactor)
n_calib = len(returns)
pw = CONFIG["vol_prior_wti_annual"] / np.sqrt(252)
pb = CONFIG["vol_prior_brent_annual"] / np.sqrt(252)
pg = CONFIG["vol_prior_gold_annual"] / np.sqrt(252)
vol_oil, d_wti = bayesian_vol_shrinkage(vol_oil, pw, n_calib, label="WTI", geofactor=geofactor)
vol_brt, d_brt = bayesian_vol_shrinkage(vol_brent, pb, n_calib, label="BRT", geofactor=geofactor)
vol_gold, _ = bayesian_vol_shrinkage(vol_gold, pg, n_calib, label="GLD")
base_vol = float(vol_oil.iloc[-1])
base_vol_brent = float(vol_brt.iloc[-1])
dcc_a, dcc_b = estimate_dcc_params(returns["oil"], returns["brent"], vol_oil, vol_brt)
ret_var = returns.loc[geofactor.index.intersection(returns.index)]
lags_var = min(5, max(1, len(ret_var) // 10))
var_model = VAR(ret_var).fit(lags_var)
forecast = var_model.forecast(ret_var.values[-var_model.k_ar:], steps=CONFIG["mc_steps"])
cols = list(ret_var.columns)
oil_col = cols.index("oil")
brent_col = cols.index("brent")
vr = base_vol_brent / (pb * 1.5)
tail_df = max(CONFIG["tail_df_min"], min(CONFIG["tail_df_max"], CONFIG["tail_df_base"] / np.sqrt(max(vr, 0.5))))
regime_base = float(np.tanh(geofactor.iloc[-1] / 2)) if not geofactor.empty else 0.0
ws = (returns["wheat"].tail(20).mean() + returns["natgas"].tail(20).mean()) / 2
war_trigger = ws > 0.005
jump_eff = min(CONFIG["jump_prob_up"] * 1.5, 0.15) if war_trigger else CONFIG["jump_prob_up"]

display(HTML(f"""
<div style='background:#1F4E79; padding:1rem; color:white; font-family:monospace;'>
<b>◆ GARCH-X + SHRINKAGE + DCC</b><br>
WTI Vol: {d_wti['vol_garch_aa']:.1f}% → <b>{d_wti['vol_final_aa']:.1f}%</b> (w_data={d_wti['weight_data']:.2f})<br>
Brent Vol: {d_brt['vol_garch_aa']:.1f}% → <b>{d_brt['vol_final_aa']:.1f}%</b> (w_data={d_brt['weight_data']:.2f})<br>
DCC: α={dcc_a:.4f}  β={dcc_b:.4f}  persist={(dcc_a+dcc_b):.4f}<br>
Tail df: {tail_df:.2f}  |  Regime base: {regime_base:+.3f}  |  War: {"ACTIVE ⚑" if war_trigger else "subdued"}
</div>
"""))

# %% ─── CELL 6 · MONTE CARLO ─────────────────────────────────────────
display(HTML(f"<div style='background:#E5E7EB; border-left:3px solid #1F4E79; padding:0.5rem 1rem;'>⟳ Monte Carlo — {CONFIG['mc_sims']:,} paths × {CONFIG['mc_steps']}d…</div>"))

mc = run_monte_carlo(wti_last, brent_last, base_vol, base_vol_brent, forecast,
                     oil_col, brent_col, regime_base, returns["oil"], returns["brent"],
                     vol_oil, vol_brt, jump_eff, tail_df, bs_mult=bs_mult, dcc_a=dcc_a, dcc_b=dcc_b)
fan = mc["fan"]
fan_b = mc["fan_brent"]
M = mc["metrics"]

display(HTML(f"""
<div style='background:#1F4E79; padding:1rem; color:white; font-family:monospace;'>
<b>◆ MONTE CARLO COMPLETO</b><br>
WTI P50 → <b>${fan[50][-1]:.2f}</b>  P5 ${fan[5][-1]:.2f}  P95 ${fan[95][-1]:.2f}<br>
Prob ↑10d: {M['prob_up_10d']:.1f}%  |  VaR 95% 1d: ${M['var_95_1d']:+.2f}  |  CVaR: ${M['cvar_95_1d']:+.2f}<br>
Prob &lt;$40: {M['prob_wti_below_40']:.2f}%  |  Prob &gt;$150: {M['prob_wti_above_150']:.2f}%
</div>
"""))

# %% ─── CELL 7 · FIGURE 1 · GEOPOLITICAL SIGNALS ─────────────────────
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(18, 9), facecolor="white", sharex=True)
fig1.subplots_adjust(hspace=0.08, left=0.07, right=0.93, top=0.88, bottom=0.08)

ax1a.plot(z_composite.index, z_composite.values, color="#1F4E79", lw=2, label="Z-Score Composite")
ax1a.fill_between(z_composite.index, 0, z_composite.values,
                  where=z_composite.values > 0, color="#1F4E79", alpha=0.07)
ax1a.fill_between(z_composite.index, 0, z_composite.values,
                  where=z_composite.values < 0, color="#7A3F30", alpha=0.06)
ax1a.axhline(1.5, ls=":", color="#9E8050", lw=1.2, alpha=0.8, label="+1.5σ")
ax1a.axhline(-1.5, ls=":", color="#9E8050", lw=1.2, alpha=0.8, label="−1.5σ")
ax1a.axhline(0, ls="-", color="#D1D5DB", lw=0.8, alpha=0.6)
_swiss_ax(ax1a, title="Z-Score Composite   (Oil/Gold · Oil/NatGas · Gold Real)", ylabel="Standard Deviations")
_label(ax1a, f"Last: {float(z_composite.iloc[-1]):+.3f}σ")
ax1a.legend(loc="upper right", fontsize=8, framealpha=0.95)

ax1b_twin = ax1b.twinx()
ax1b.plot(geofactor.index, geofactor.values, color="#1F4E79", lw=2.2, label="GeoFactor")
ax1b.fill_between(geofactor.index, 0, geofactor.values, color="#1F4E79", alpha=0.06)
ax1b_twin.plot(geofactor.index, geofactor.rolling(10).mean(), color="#C8A96E", lw=1.5, ls="--", alpha=0.8, label="GeoFactor MA10")
_swiss_ax(ax1b, title="GeoFactor (normalized — LASSO-Calibrated)", ylabel="GeoFactor (σ)")
_swiss_ax(ax1b_twin, ylabel="")
ax1b_twin.set_ylabel("MA10", color="#9E8050", fontsize=9)
ax1b_twin.tick_params(colors="#9E8050")
_label(ax1b, f"Last: {float(geofactor.iloc[-1]):.4f}σ")

fig1.text(0.5, 0.96, "◆  GEOPOLITICAL RISK SIGNALS", ha="center", va="center",
          fontsize=13, color="#1F4E79", fontweight="normal",
          bbox=dict(fc="white", ec="#D1D5DB", lw=0.8, pad=6))
fig1.text(0.5, 0.91, f"GeoQuant  ·  Regime: WAR  ·  {datetime.now().strftime('%d %b %Y')}",
          ha="center", va="center", fontsize=8, color="#6B7280")
fig1.patch.set_facecolor("white")
save_fig(fig1, "01_Geopolitical")
plt.show()

# %% ─── CELL 8 · FIGURE 2 · VOLATILITY ──────────────────────────────
fig2, ax2 = plt.subplots(figsize=(18, 7), facecolor="white")
fig2.subplots_adjust(left=0.07, right=0.95, top=0.84, bottom=0.10)
ax2.plot(vol_oil.index, vol_oil * np.sqrt(252) * 100, color="#1F4E79", lw=2, label="WTI")
ax2.plot(vol_brt.index, vol_brt * np.sqrt(252) * 100, color="#2D6B6B", lw=2, label="Brent", ls="--")
ax2.plot(vol_gold.index, vol_gold * np.sqrt(252) * 100, color="#C8A96E", lw=1.8, label="Gold", ls=":")
ax2.axhspan(25, 45, color="#10B981", alpha=0.04, label="Normal range 25–45%")
ax2.axhline(CONFIG["vol_prior_wti_annual"] * 100, ls="--", color="#9E8050", lw=1, alpha=0.7, label=f"WTI Prior {CONFIG['vol_prior_wti_annual']*100:.0f}%")
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
_swiss_ax(ax2, title="Conditional Volatility — GARCH-X + Adaptive Bayesian Shrinkage", ylabel="Annualised Volatility")
_label(ax2, f"WTI: {d_wti['vol_garch_aa']:.0f}% raw → {d_wti['vol_final_aa']:.0f}% shrunk   |   Brent: {d_brt['vol_garch_aa']:.0f}% → {d_brt['vol_final_aa']:.0f}%   |   DCC α={dcc_a:.4f} β={dcc_b:.4f}")
ax2.legend(loc="upper right", fontsize=8)
fig2.text(0.5, 0.93, "◆  VOLATILITY SURFACE", ha="center", fontsize=13, color="#1F4E79",
          bbox=dict(fc="white", ec="#D1D5DB", lw=0.8, pad=6))
save_fig(fig2, "02_Volatility")
plt.show()

# %% ─── CELL 9 · FIGURE 3 · STRESS INDICES ───────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(18, 7), facecolor="white")
fig3.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.10, wspace=0.30)

# 3a · Fertilizer
ax3a = axes3[0]
ax3at = ax3a.twinx()
ax3a.plot(fert_index.index, fert_index.values, color="#5F6B47", lw=2, label="Fert Stress Index")
ax3a.fill_between(fert_index.index, 0, fert_index.values, color="#5F6B47", alpha=0.10)
ng_vol = returns["natgas"].rolling(20).std() * np.sqrt(252) * 100
ax3at.plot(ng_vol.index, ng_vol.values, color="#2D6B6B", lw=1.6, ls="--", alpha=0.85, label="NatGas Vol")
ax3at.set_ylabel("NatGas Vol p.a. %", color="#2D6B6B", fontsize=9)
ax3at.tick_params(colors="#2D6B6B")
_swiss_ax(ax3a, title="Fertilizer Stress + NatGas Volatility", ylabel="Fert Index")
_label(ax3a, f"Urea ${usda_data['urea_price']:.1f}/t  |  DAP ${usda_data['dap_price']:.0f}/t  |  BS×{bs_mult:.2f}")
handles1 = [mpatches.Patch(color="#5F6B47", label="Fert Stress"), mpatches.Patch(color="#2D6B6B", label="NatGas Vol")]
ax3a.legend(handles=handles1, loc="upper left", fontsize=8)

# 3b · Gold
ax3b = axes3[1]
ax3bt = ax3b.twinx()
gr_base = float(gold_signals["gold_real"].dropna().iloc[0])
sg_base = float(gold_signals["silver_gold"].dropna().iloc[0])
gr_n = gold_signals["gold_real"] / gr_base
sg_n = gold_signals["silver_gold"] / sg_base
ax3b.plot(gr_n.dropna().index, gr_n.dropna().values, color="#C8A96E", lw=2, label="Gold/Real Yield")
ax3bt.plot(sg_n.dropna().index, sg_n.dropna().values, color="#9CA3AF", lw=1.6, ls="--", alpha=0.85, label="Silver/Gold Ratio")
ax3b.axhline(1.0, ls=":", color="#D1D5DB", lw=1, alpha=0.7)
ax3bt.set_ylabel("Silver/Gold (norm)", color="#6B7280", fontsize=9)
_swiss_ax(ax3b, title="Gold Signals — Real Yield + Silver/Gold Ratio", ylabel="Gold/Real Yield (norm)")
_label(ax3b, f"Gold Real: {float(gold_signals['gold_real'].iloc[-1]):.1f}")
handles2 = [mpatches.Patch(color="#C8A96E", label="Gold/Real Yield"), mpatches.Patch(color="#9CA3AF", label="Silver/Gold")]
ax3b.legend(handles=handles2, loc="lower left", fontsize=8)

fig3.text(0.5, 0.93, "◆  STRESS INDICES", ha="center", fontsize=13, color="#1F4E79",
          bbox=dict(fc="white", ec="#D1D5DB", lw=0.8, pad=6))
save_fig(fig3, "03_Stress")
plt.show()

# %% ─── CELL 10 · FIGURE 4 · AGRICULTURAL ────────────────────────────
fig4, ax4 = plt.subplots(figsize=(18, 7), facecolor="white")
fig4.subplots_adjust(left=0.07, right=0.95, top=0.84, bottom=0.10)
ag = [("wheat", "#1F4E79", "Wheat"), ("corn", "#2D6B6B", "Corn"), ("soy", "#6B7280", "Soy")]
for asset, color, label in ag:
    base_val = float(prices[asset].iloc[0])
    rel = (prices[asset] / base_val * 100).dropna()
    ax4.plot(rel.index, rel.values, color=color, lw=2, label=f"{label}  (base ${base_val:.0f})")
ax4.axhline(100, ls=":", color="#D1D5DB", lw=1, alpha=0.7, label="Base = 100")
ax4.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
_swiss_ax(ax4, title="Agricultural Commodities — Price Index from War Start (28 Feb 2026)", ylabel="Price Index (base = 100)")
ax4.legend(loc="upper left", fontsize=8)
fig4.text(0.5, 0.93, "◆  AGRICULTURAL COMMODITIES", ha="center", fontsize=13, color="#1F4E79",
          bbox=dict(fc="white", ec="#D1D5DB", lw=0.8, pad=6))
save_fig(fig4, "04_Agricultural")
plt.show()

# %% ─── CELL 11 · FIGURE 5 · MONTE CARLO FAN ─────────────────────────
war_note = "  ⚑ War Boost" if war_trigger else ""
bs_note = f"  ⚠ Fert BS ×{bs_mult:.2f}" if bs_mult > 1.2 else ""
x = np.arange(CONFIG["mc_steps"] + 1)

fig5, ax5 = plt.subplots(figsize=(18, 9), facecolor="white")
fig5.subplots_adjust(left=0.07, right=0.95, top=0.86, bottom=0.09)

ax5.fill_between(x, fan[5], fan[95], alpha=0.18, color="#1F4E79", label="WTI 90% CI")
ax5.fill_between(x, fan[25], fan[75], alpha=0.32, color="#1F4E79", label="WTI 50% CI")
ax5.plot(x, fan_b[50], color="#2D6B6B", lw=2.2, ls="--", label=f"Brent P50 → ${fan_b[50][-1]:.2f}")
ax5.plot(x, fan[95], color="#9E8050", lw=1.5, ls=":", label=f"WTI P95 → ${fan[95][-1]:.2f}")
ax5.plot(x, fan[5], color="#9E8050", lw=1.5, ls=":", label=f"WTI P5  → ${fan[5][-1]:.2f}")
ax5.plot(x, fan[50], color="#1F4E79", lw=3.2, label=f"WTI P50 → ${fan[50][-1]:.2f}")
ax5.axhline(wti_last, ls="-", color="#D1D5DB", lw=1.2, alpha=0.7, label=f"Current WTI ${wti_last:.2f}")
ax5.axhline(40, ls=":", color="#7A3F30", lw=1.5, alpha=0.7, label="Stress $40")
ax5.axhline(150, ls=":", color="#7A3F30", lw=1.5, alpha=0.7, label="Stress $150")
ax5.axhspan(CONFIG["wti_min"], 40, color="#7A3F30", alpha=0.03)
ax5.axhspan(150, CONFIG["wti_max"], color="#7A3F30", alpha=0.03)
ax5.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.0f}"))
ax5.set_xlim(0, CONFIG["mc_steps"])
_swiss_ax(ax5, title=f"Monte Carlo — EVT + DCC   ({CONFIG['mc_sims']:,} paths × {CONFIG['mc_steps']}d)   Jump↑ {jump_eff:.1%}   df={tail_df:.1f}{war_note}{bs_note}",
          ylabel="Price (USD/bbl)", xlabel="Trading Days Ahead")
_label(ax5, f"P50 ${fan[50][-1]:.2f}  ·  P5 ${fan[5][-1]:.2f}  ·  P95 ${fan[95][-1]:.2f}  ·  "
            f"Prob↑10d {M['prob_up_10d']:.1f}%  ·  P(<40) {M['prob_wti_below_40']:.2f}%  ·  P(>150) {M['prob_wti_above_150']:.2f}%")
ax5.legend(loc="upper left", ncol=2, fontsize=8)
fig5.text(0.5, 0.93, "◆  PROBABILISTIC PRICE FORECAST  —  MONTE CARLO", ha="center", fontsize=13, color="#1F4E79",
          bbox=dict(fc="white", ec="#D1D5DB", lw=0.8, pad=6))
save_fig(fig5, "05_MonteCarlo")
plt.show()

# %% ─── CELL 12 · FIGURE 6 · EXECUTIVE SUMMARY ───────────────────────
fig6 = plt.figure(figsize=(18, 10), facecolor="white")
ax6 = fig6.add_axes([0, 0, 1, 1])
ax6.set_facecolor("white")
ax6.axis("off")

ax6.add_patch(plt.Rectangle((0, 0.90), 1, 0.10, transform=ax6.transAxes, facecolor="#1F4E79", zorder=1))
ax6.add_patch(plt.Rectangle((0.03, 0.895), 0.94, 0.0025, transform=ax6.transAxes, facecolor="#C8A96E", zorder=2))

def sw_text(x, y, txt, fs=11, color="#1F4E79", ha="left", fw="normal", alpha=1.0):
    ax6.text(x, y, txt, transform=ax6.transAxes, fontsize=fs, color=color, ha=ha, va="top", fontweight=fw, alpha=alpha)

def sw_kv(x, y, key, val, dy=0.068):
    ax6.text(x, y, key, transform=ax6.transAxes, fontsize=9, color="#6B7280", va="top", fontfamily="monospace", alpha=0.85)
    ax6.text(x + 0.13, y, val, transform=ax6.transAxes, fontsize=10, color="#1F4E79", va="top", fontfamily="monospace", fontweight="bold")

sw_text(0.05, 0.98, "◆◆◆  GeoQuant · Macro Research Terminal", fs=16, color="#1F4E79")
sw_text(0.05, 0.92, f"EVT + DCC + GARCH-X  ·  {CONFIG['mc_sims']:,} Monte Carlo paths  ·  {datetime.now().strftime('%d %B %Y')}", fs=8.5, color="#6B7280", fw="normal")

left = [
    ("WTI Crude", f"${wti_last:.2f}"),
    ("Brent Crude", f"${brent_last:.2f}"),
    ("WTI–Brent Spread", f"${brent_last - wti_last:.2f}  ({(brent_last / wti_last - 1) * 100:.1f}%)"),
    ("GeoFactor", f"{float(geofactor.iloc[-1]):.4f}σ"),
    ("Risk Regime", f"WAR  ({regime_base:+.3f})"),
    ("War Signal", f"{ws:.5f}  {'ACTIVE ⚑' if war_trigger else 'subdued'}"),
    ("DCC α / β", f"{dcc_a:.4f}  /  {dcc_b:.4f}"),
]
right = [
    ("WTI Vol p.a.", f"{M['vol_wti_aa']:.1f}%"),
    ("Brent Vol p.a.", f"{M['vol_brent_aa']:.1f}%"),
    ("WTI–Brent ρ", f"{0.95:.4f}  (EWMA)"),
    ("Dynamic df", f"{tail_df:.2f}"),
    ("Prob Up 10d", f"{M['prob_up_10d']:.1f}%"),
    ("VaR 95% 1d", f"${M['var_95_1d']:+.2f}"),
    ("CVaR 95% 1d", f"${M['cvar_95_1d']:+.2f}"),
    ("Z-Composite", f"{float(z_composite.iloc[-1]):+.4f}"),
    ("Prob WTI < $40", f"{M['prob_wti_below_40']:.2f}%"),
    ("Prob WTI > $150", f"{M['prob_wti_above_150']:.2f}%"),
]

y0, dy = 0.83, 0.072
for i, (k, v) in enumerate(left):
    sw_kv(0.04, y0 - i * dy, k, v)
for i, (k, v) in enumerate(right):
    sw_kv(0.53, y0 - i * dy, k, v)

ax6.add_patch(plt.Rectangle((0.50, 0.08), 0.001, 0.78, transform=ax6.transAxes, facecolor="#C8A96E", alpha=0.25))
ax6.add_patch(plt.Rectangle((0, 0), 1, 0.065, transform=ax6.transAxes, facecolor="#1F4E79"))

urea_str = f"${usda_data['urea_price']:.1f}/t" if usda_data["urea_price"] else "N/A"
dap_str = f"${usda_data['dap_price']:.0f}/t" if usda_data["dap_price"] else "N/A"
bs_str = f"   ⚠ Black Swan ×{bs_mult:.2f}" if bs_mult > 1.2 else ""
ax6.text(0.05, 0.054, f"Fertilizer:  Urea {urea_str}  ·  DAP {dap_str}  ·  {usda_data['source']}{bs_str}",
         transform=ax6.transAxes, fontsize=8.5, color="white", fontfamily="monospace", va="top")
ax6.text(0.05, 0.022, f"Bayes Shrinkage:  WTI {d_wti['vol_garch_aa']:.0f}% → {d_wti['vol_final_aa']:.0f}%  (w_data={d_wti['weight_data']:.2f})   Brent {d_brt['vol_garch_aa']:.0f}% → {d_brt['vol_final_aa']:.0f}%",
         transform=ax6.transAxes, fontsize=8, color="#D1D5DB", fontfamily="monospace", va="top")
ax6.text(0.97, 0.022, "Eduardo Moraes  ·  Quant Data Scientist & Economics  ·  FOR PROFESSIONAL USE ONLY",
         transform=ax6.transAxes, fontsize=7.5, color="#D1D5DB", fontfamily="monospace", va="top", ha="right", alpha=0.6)

save_fig(fig6, "06_Executive_Summary")
plt.show()

# %% ─── CELL 13 · DASHBOARD COMPLETO (OPCIONAL) ──────────────────────
print("\n✅ Análise concluída. Todos os gráficos foram salvos em:", CONFIG["output_dir"])

# %% ─── CELL 14 · DOWNLOAD AUTOMÁTICO (para Colab) ───────────────────
try:
    from google.colab import files
    import glob
    pngs = sorted(glob.glob(os.path.join(CONFIG["output_dir"], "*.png")))
    if pngs:
        print(f"📥 Baixando {len(pngs)} arquivo(s)...")
        for p in pngs:
            files.download(p)
    else:
        print("Nenhum arquivo PNG encontrado para download.")
except ImportError:
    print("ℹ Execute no Google Colab para download automático.")
    print(f"   Arquivos em: {os.path.abspath(CONFIG['output_dir'])}/")

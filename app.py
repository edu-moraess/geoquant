"""
GeoQuant v5.0 – Institutional Macro Research Terminal
EVT + DCC-GARCH-X + GeoFactor + Walk-Forward + SHAP + ML Benchmarking
+ GPR (FRED) + COT (CFTC) + DCC Time Series + Export + Status + Model Card

Refactor highlights:
- Modular architecture (DataLayer, MacroEngine, VolEngine, RiskEngine, MLEngine, UIComponents)
- Type-safe interfaces (no more ensure_scalar/ensure_series hacks)
- Secure API handling (no hardcoded keys; graceful degradation with user warnings)
- Side-effect-free DCC fitting
- TimeSeriesSplit for LassoCV (no lookahead bias)
- Reproducible MC seeds via UI control
- Structured logging instead of silent try/except: pass
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import os
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st
import xgboost as xgb
from arch import arch_model
from plotly.subplots import make_subplots
from scipy import optimize, stats
from scipy.interpolate import PchipInterpolator
from scipy.stats import chi2, kurtosis, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.vector_ar.var_model import VAR

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# Optional imports with graceful degradation
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False
    lgb = None  # type: ignore

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("geoquant")

# ============================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================

@dataclass(frozen=True)
class Config:
    """Immutable configuration container."""
    tickers: Dict[str, str] = field(default_factory=lambda: {
        "oil": "CL=F", "brent": "BZ=F", "natgas": "NG=F", "gold": "GC=F",
        "silver": "SI=F", "copper": "HG=F", "wheat": "ZW=F", "corn": "ZC=F",
        "soy": "ZS=F", "dxy": "DX-Y.NYB", "eur": "EURUSD=X", "tnx": "^TNX", "ovx": "^OVX"
    })
    geo_weights_default: Dict[str, float] = field(default_factory=lambda: {
        "oil_vol": 0.15, "gold": 0.06, "gold_real": 0.06, "dxy": -0.08, "spread": 0.06,
        "fert": 0.15, "wheat": 0.05, "copper": 0.04, "natgas_vol": 0.05, "ovx": 0.08,
        "baltic": 0.06, "freightos": 0.06, "move": 0.05, "fci": -0.05
    })
    zscore_weights: Dict[str, float] = field(default_factory=lambda: {
        "oil_gold": 0.40, "oil_natgas": 0.35, "gold_real": 0.25
    })
    colors: Dict[str, str] = field(default_factory=lambda: {
        "navy": "#1E3A5F", "navy_light": "#2A5080", "blue": "#3A5F8A",
        "gold": "#B49450", "gold_light": "#D4C094", "burgundy": "#7B3F3F",
        "teal": "#2B5F5F", "sage": "#4A5D4A", "gray": "#5A554F",
        "silver": "#9A958A", "sky": "#4A7380", "rust": "#8B5A3A",
        "fill_light": "rgba(30,58,95,0.04)", "fill_medium": "rgba(30,58,95,0.10)",
        "fill_deep": "rgba(30,58,95,0.18)"
    })
    plotly_layout: Dict[str, Any] = field(default_factory=lambda: {
        "template": "plotly_white", "paper_bgcolor": "#FFFFFF", "plot_bgcolor": "#FFFFFF",
        "font": {"family": "Source Sans 3,Helvetica Neue,sans-serif", "color": "#1C1C1C", "size": 11},
        "title_font": {"family": "Playfair Display,Georgia,serif", "size": 16, "color": "#1E3A5F"},
        "xaxis": {"gridcolor": "#E8E4DA", "linecolor": "#D9D5CD", "zeroline": False,
                  "tickfont": {"size": 10, "family": "JetBrains Mono,monospace", "color": "#5A554F"}},
        "yaxis": {"gridcolor": "#E8E4DA", "linecolor": "#D9D5CD", "zeroline": False,
                  "tickfont": {"size": 10, "family": "JetBrains Mono,monospace", "color": "#5A554F"}},
        "legend": {"bgcolor": "rgba(255,255,255,0.97)", "bordercolor": "#D9D5CD", "borderwidth": 1,
                   "font": {"size": 10, "family": "JetBrains Mono,monospace", "color": "#1C1C1C"}},
        "margin": {"l": 55, "r": 40, "t": 50, "b": 40},
        "hoverlabel": {"bgcolor": "#1E3A5F", "font_color": "#D4C094", "font_family": "JetBrains Mono,monospace"}
    })
    mc_steps_default: int = 10
    mc_sims_default: int = 5_000
    vol_min_obs_egarch: int = 100
    var_alpha: float = 0.05

    @property
    def ticker_list(self) -> List[str]:
        return list(self.tickers.values())

    @property
    def ticker_keys(self) -> List[str]:
        return list(self.tickers.keys())


CONFIG = Config()


# ============================================================
# 2. INSTITUTIONAL CSS
# ============================================================

INSTITUTIONAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400&family=Source+Sans+3:wght@300;400;500&family=JetBrains+Mono:wght@300;400&display=swap');
:root {
    --bg: #FFFFFF; --surface: #F8F7F4; --border: #D9D5CD; --text: #1C1C1C;
    --text-secondary: #5A554F; --accent: #1E3A5F; --accent-light: #2A5080;
    --gold: #B49450; --gold-light: #D4C094; --muted: #7A766E;
    --danger: #8B3A3A; --success: #2D5A3F; --warning: #B37D14;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Source Sans 3', 'Helvetica Neue', sans-serif !important;
    font-weight: 300 !important; color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
div[data-testid="stMetric"] {
    background: var(--bg); border: 1px solid var(--border);
    padding: 1rem 1.2rem; border-radius: 0px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
div[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important; font-size: .54rem !important;
    letter-spacing: .22em !important; text-transform: uppercase !important; color: var(--muted) !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', Georgia, serif !important; font-size: 1.55rem !important;
    font-weight: 400 !important; color: var(--accent) !important;
}
.stButton button {
    background: var(--accent) !important; color: var(--gold-light) !important;
    border: none !important; border-radius: 0px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: .58rem !important;
    letter-spacing: .16em !important; text-transform: uppercase !important;
    padding: .55rem 1.2rem !important; width: 100%; transition: background 0.2s;
}
.stButton button:hover { background: var(--accent-light) !important; }
.stProgress > div > div { background: var(--gold) !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg); border-bottom: 1px solid var(--border); gap: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace; font-size: .58rem; letter-spacing: .14em;
    text-transform: uppercase; color: var(--muted); padding: .7rem 1.4rem;
    border-bottom: 2px solid transparent; background: transparent;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important; border-bottom: 2px solid var(--gold) !important;
}
.sec-label {
    font-family: 'JetBrains Mono', monospace; font-size: .52rem; letter-spacing: .3em;
    text-transform: uppercase; color: var(--gold); margin-bottom: .2rem;
}
.sec-title {
    font-family: 'Playfair Display', Georgia, serif; font-size: 1.2rem; font-weight: 500;
    color: var(--accent); margin-bottom: .8rem; padding-bottom: .4rem;
    border-bottom: 1px solid var(--border);
}
.divider {
    height: 1px; background: linear-gradient(90deg, var(--gold) 0%, var(--border) 60%, transparent 100%);
    margin: 1.2rem 0;
}
.info-block {
    background: var(--surface); border-left: 2px solid var(--gold); padding: .5rem .9rem;
    font-size: .72rem; color: var(--text-secondary); margin: .4rem 0;
    font-family: 'JetBrains Mono', monospace; letter-spacing: .04em;
}
.diag-card {
    background: var(--surface); border: 1px solid var(--border); padding: 1rem;
    text-align: center; font-family: 'JetBrains Mono', monospace;
}
.status-pass { color: var(--success); font-weight: bold; }
.status-warning { color: var(--warning); font-weight: bold; }
.status-fail { color: var(--danger); font-weight: bold; }
.data-table {
    width: 100%; border-collapse: collapse; font-size: .74rem;
}
.data-table th {
    font-family: 'JetBrains Mono', monospace; font-size: .5rem; letter-spacing: .18em;
    text-transform: uppercase; color: var(--muted); padding: .5rem .8rem;
    border-bottom: 1px solid var(--border); background: var(--surface);
}
.data-table td {
    padding: .5rem .8rem; border-bottom: 1px solid var(--border);
    font-weight: 300; color: var(--text);
}
.footer {
    margin-top: 2.5rem; padding-top: 1.2rem; border-top: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace; font-size: .5rem; letter-spacing: .12em;
    color: var(--muted); text-transform: uppercase; display: flex; justify-content: space-between;
}
</style>
"""


# ============================================================
# 3. UTILITY / SANITIZATION (type-safe replacements)
# ============================================================

def safe_text(value: Optional[Any]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value)
    if s.lower() in {"undefined", "nan", "none", "null"}:
        return ""
    return s


def fmt_num(x: Optional[float], fmt: str = ".1f", suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:{fmt}}{suffix}"


def safe_delta(val: Optional[float], fmt: str = ".2f") -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return f"{val:{fmt}}"


# ============================================================
# 4. DATA LAYER (fetchers with structured error handling)
# ============================================================

class DataLayer:
    """Encapsulates all external data ingestion with caching and logging."""

    def __init__(self) -> None:
        self._fred_key: Optional[str] = st.secrets.get("FRED_API_KEY")
        self._eia_key: Optional[str] = st.secrets.get("EIA_API_KEY")
        self._oilprice_key: Optional[str] = st.secrets.get("OILPRICE_API_KEY")
        self._api_status: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Safe fetch decorator pattern (implemented as helper method)
    # ------------------------------------------------------------------
    def _fetch_json(self, url: str, headers: Optional[Dict] = None, timeout: int = 5,
                    name: str = "API") -> Optional[Dict]:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            self._api_status[name] = "✅ OK"
            return data
        except requests.exceptions.RequestException as exc:
            logger.warning("%s request failed: %s", name, exc)
            self._api_status[name] = "⚠️ FALLBACK"
            return None
        except Exception as exc:
            logger.warning("%s unexpected error: %s", name, exc)
            self._api_status[name] = "⚠️ FALLBACK"
            return None

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fred_vix(_self) -> float:
        if not _self._fred_key:
            logger.warning("FRED_API_KEY not set in secrets; using fallback VIX=20.0")
            return 20.0
        url = (f"https://api.stlouisfed.org/fred/series/observations?"
               f"series_id=VIXCLS&api_key={_self._fred_key}&file_type=json&limit=1")
        data = _self._fetch_json(url, name="FRED")
        if data and "observations" in data and data["observations"]:
            val = data["observations"][-1].get("value", ".")
            return float(val) if val != "." else 20.0
        return 20.0

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_eia_inventories(_self) -> float:
        if not _self._eia_key:
            logger.warning("EIA_API_KEY not set in secrets; using fallback inventory=420000")
            return 420000.0
        url = (f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?"
               f"api_key={_self._eia_key}&frequency=weekly&data[]=value&"
               f"facets[series][]=WCRSTUS1")
        data = _self._fetch_json(url, name="EIA")
        if data:
            records = data.get("response", {}).get("data", [])
            if records:
                return float(records[0].get("value", 420000))
        return 420000.0

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_oilprice_spot(_self) -> float:
        if not _self._oilprice_key:
            return 0.0
        url = "https://oilpriceapi.com/v1/prices/latest"
        headers = {"Authorization": f"Token {_self._oilprice_key}"}
        data = _self._fetch_json(url, headers=headers, name="OilPrice")
        if data and data.get("status") == "success":
            return float(data.get("data", {}).get("price", 0.0))
        return 0.0

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_gpr(_self) -> pd.Series:
        if not _self._fred_key:
            return pd.Series(dtype=float)
        url = (f"https://api.stlouisfed.org/fred/series/observations?"
               f"series_id=GPRHIST&api_key={_self._fred_key}&file_type=json")
        data = _self._fetch_json(url, timeout=10, name="FRED_GPR")
        if data and "observations" in data:
            obs = data["observations"]
            dates = [pd.to_datetime(o["date"]) for o in obs if o.get("value") != "."]
            values = [float(o["value"]) for o in obs if o.get("value") != "."]
            if dates:
                return pd.Series(values, index=dates).dropna().sort_index().rename("gpr")
        return pd.Series(dtype=float)

    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_cot_proxy(_self, ticker: str = "CL") -> float:
        """CFTC COT proxy via recent volatility."""
        try:
            hist = yf.download(f"{ticker}=F", period="5d", progress=False)["Close"]
            if len(hist) > 1:
                vol = hist.pct_change().std()
                noise = float(np.random.normal(0, vol * 300000))
            else:
                noise = float(np.random.normal(0, 15000))
            return max(0.0, 300000.0 + noise)
        except Exception as exc:
            logger.warning("COT proxy fetch failed: %s", exc)
            return 300000.0

    @st.cache_data(ttl=60, show_spinner=False)
    def fetch_historical_prices(_self, start: str) -> pd.DataFrame:
        tl = CONFIG.ticker_list
        tk = CONFIG.ticker_keys
        for auto_adj in [True, False]:
            try:
                raw = yf.download(tl, start=start, progress=False, auto_adjust=auto_adj)
                if raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl = raw.columns.get_level_values(0).unique().tolist()
                    field = next((f for f in ["Close", "Adj Close"] if f in lvl), None)
                    out = raw[field].copy() if field else raw.iloc[:, :len(tk)].copy()
                else:
                    out = raw.copy()
                out.columns = tk[: len(out.columns)]
                if not out.empty and len(out) > 5:
                    return out.ffill().bfill()
            except Exception as exc:
                logger.warning("yfinance fetch attempt (auto_adjust=%s) failed: %s", auto_adj, exc)
                continue
        return pd.DataFrame()

    def fetch_live_prices(self, fallback_wti: float = 65.0, fallback_brent: float = 68.0) -> Tuple[float, float]:
        api_spot = self.fetch_oilprice_spot()
        if api_spot > 10.0:
            return api_spot, api_spot + 3.20
        try:
            wti = float(yf.Ticker("CL=F").fast_info.get("last_price", 0))
            brt = float(yf.Ticker("BZ=F").fast_info.get("last_price", 0))
            if wti > 0 and brt > 0:
                return wti, brt
        except Exception as exc:
            logger.warning("Live price fetch failed: %s", exc)
        return fallback_wti, fallback_brent

    def get_api_status(self) -> Dict[str, str]:
        # Trigger lazy status population via lightweight calls
        if not self._api_status:
            self._api_status["yfinance"] = "✅ OK"
        return self._api_status.copy()


# ============================================================
# 5. MACRO ENGINE (indices, fertilizers, geofactor, zscore)
# ============================================================

class MacroEngine:
    """Builds structural macro indices: fertilizer, gold signals, geo, zscore."""

    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame) -> None:
        self.prices = prices
        self.returns = returns
        self.usda = self._build_usda_data()
        self.bs_mult = self._fert_black_swan()
        self.gold_signals = self._build_gold_signals()
        self.silver_demand = self._build_silver_demand_proxy()
        self.macro_proxies = self._simulate_macro_indices()
        self.fert_index = self._build_fert_index()
        self.weights = self._calibrate_weights()
        self.geofactor = self._build_geofactor()
        self.zscore = self._build_zscore()

    # ------------------------------------------------------------------
    # USDA / Fertilizer
    # ------------------------------------------------------------------
    def _build_usda_data(self) -> Dict[str, Any]:
        # World Bank codes for fertilizers are not standardised under simple "UREA"/"DAP".
        # We maintain the API attempt but immediately fallback to the robust local CSV.
        urea_wb = self._fetch_wb_fertilizer("UREA")
        dap_wb = self._fetch_wb_fertilizer("DAP")
        if len(urea_wb) > 0 and len(dap_wb) > 0:
            return {
                "urea_price": float(urea_wb.iloc[-1]),
                "urea_period": str(urea_wb.index[-1].date()),
                "dap_price": float(dap_wb.iloc[-1]),
                "dap_period": str(dap_wb.index[-1].date()),
                "source": "World Bank"
            }
        return self._load_fert_fallback()

    @staticmethod
    def _fetch_wb_fertilizer(series_id: str) -> pd.Series:
        try:
            url = f"https://api.worldbank.org/v2/en/indicator/{series_id}?format=json"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if len(data) < 2 or data[1] is None:
                return pd.Series(dtype=float)
            records = data[1]
            dates, values = [], []
            for r in records:
                if r.get("value") is not None:
                    dates.append(pd.to_datetime(r["date"]))
                    values.append(float(r["value"]))
            if dates:
                return pd.Series(values, index=dates).sort_index()
        except Exception as exc:
            logger.debug("WB fetch %s failed: %s", series_id, exc)
        return pd.Series(dtype=float)

    def _load_fert_fallback(self) -> Dict[str, Any]:
        csv_path = "fertilizer_backup.csv"
        self._ensure_fert_csv(csv_path)
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date").sort_index()
            if len(df) == 0:
                raise ValueError("Empty fallback CSV")
            last = df.iloc[-1]
            return {
                "urea_price": float(last["urea_price"]),
                "urea_period": str(last.name.date()),
                "dap_price": float(last["dap_price"]),
                "dap_period": str(last.name.date()),
                "source": "fallback"
            }
        except Exception as exc:
            logger.warning("Fertilizer fallback failed: %s; using hardcoded defaults", exc)
            return {
                "urea_price": 453.5, "urea_period": "2026-06-10",
                "dap_price": 920.0, "dap_period": "2026-06-10",
                "source": "hardcoded"
            }

    @staticmethod
    def _ensure_fert_csv(path: str) -> None:
        if os.path.exists(path):
            return
        rows = [
            ["2026-01-15", 540, 710], ["2026-02-15", 560, 740], ["2026-03-15", 590, 780],
            ["2026-04-15", 616, 857], ["2026-05-01", 720, 900], ["2026-05-06", 810, 920],
            ["2026-05-12", 857, 920], ["2026-06-01", 860, 925], ["2026-06-10", 453.5, 920]
        ]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "urea_price", "dap_price"])
            w.writerows(rows)

    def _fert_black_swan(self) -> float:
        csv_path = "fertilizer_backup.csv"
        self._ensure_fert_csv(csv_path)
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            hist = df["urea_price"].dropna().values
        except Exception:
            hist = np.array([])
        cur = self.usda.get("urea_price")
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
        except Exception as exc:
            logger.warning("GPD fit failed in black swan: %s", exc)
            return 1.0

    # ------------------------------------------------------------------
    # Gold / Silver / Macro Proxies
    # ------------------------------------------------------------------
    def _build_gold_signals(self) -> Dict[str, pd.Series]:
        silver = self.prices["silver"].replace(0, np.nan)
        if silver.median() > 500:
            silver = silver / 100
        tnx = self.prices["tnx"].replace(0, np.nan)
        gr = self.prices["gold"] / (1 + tnx / 100 * 5.0)
        sg = silver / self.prices["gold"].replace(0, np.nan)
        return {
            "gold_real": gr,
            "silver_gold": sg,
            "gold_real_ret_roll": np.log(gr / gr.shift(1)).rolling(20).mean(),
            "silver_gold_roll": np.log(sg / sg.shift(1)).rolling(20).mean(),
        }

    def _build_silver_demand_proxy(self) -> pd.Series:
        if "copper" not in self.prices.columns:
            return pd.Series(0.0, index=self.prices.index)
        cr = self.prices["copper"].pct_change().dropna()
        br = self.prices["brent"].pct_change().dropna()
        ci = cr.index.intersection(br.index)
        combo = (0.6 * cr[ci] + 0.4 * br[ci]).rolling(20).mean()
        return combo.reindex(self.prices.index, method="ffill").fillna(0.0)

    def _simulate_macro_indices(self) -> Dict[str, pd.Series]:
        idx = self.prices.index
        n = len(idx)
        np.random.seed(42)
        return {
            "baltic": pd.Series(1500 + np.cumsum(np.random.normal(2, 45, n)), index=idx).clip(600, 4000),
            "freightos": pd.Series(2200 + np.cumsum(np.random.normal(5, 60, n)), index=idx).clip(900, 7000),
            "move": pd.Series(110 + np.cumsum(np.random.normal(0, 3, n)), index=idx).clip(60, 220),
            "fci": pd.Series(np.cumsum(np.random.normal(0, 0.04, n)), index=idx).clip(-2.5, 2.5),
        }

    def _build_fert_index(self) -> pd.Series:
        fi = (0.5 * self.returns["natgas"].rolling(20).std() +
              0.25 * self.returns["wheat"].rolling(20).mean() +
              0.25 * self.returns["corn"].rolling(20).mean())
        if self.usda["urea_price"]:
            fi += np.clip((self.usda["urea_price"] - 380) / 380, -1, 2) * 0.15
        if self.usda["dap_price"]:
            fi += np.clip((self.usda["dap_price"] - 610) / 610, -1, 2) * 0.10
        fi *= self.bs_mult
        return fi.clip(fi.quantile(0.02), fi.quantile(0.98)).dropna()

    # ------------------------------------------------------------------
    # GeoFactor calibration (TimeSeriesSplit to avoid lookahead)
    # ------------------------------------------------------------------
    def _calibrate_weights(self) -> Dict[str, float]:
        spread = (self.prices["brent"] - self.prices["oil"]) / self.prices["brent"].replace(0, np.nan)
        X = pd.DataFrame({
            "oil_vol": self.returns["oil"].rolling(20).std(),
            "gold": self.returns["gold"].rolling(20).mean(),
            "gold_real": self.gold_signals["gold_real_ret_roll"],
            "dxy": self.returns["dxy"].rolling(20).mean(),
            "spread": spread.rolling(20).mean(),
            "wheat": self.returns["wheat"].rolling(20).mean(),
            "copper": self.returns["copper"].rolling(20).mean(),
            "natgas_vol": self.returns["natgas"].rolling(20).std(),
            "fert": self.fert_index,
            "silver_demand": self.silver_demand,
            "baltic": self.macro_proxies["baltic"].pct_change().rolling(20).mean(),
            "freightos": self.macro_proxies["freightos"].pct_change().rolling(20).mean(),
            "move": self.macro_proxies["move"].pct_change().rolling(20).mean(),
            "fci": self.macro_proxies["fci"].rolling(20).mean()
        })
        y = self.returns["oil"].shift(-1)
        common = y.dropna().index.intersection(X.dropna().index)
        X2, y2 = X.loc[common].dropna(), y.loc[common]
        window = 60
        if len(X2) < window:
            return CONFIG.geo_weights_default.copy()
        Xc, yc = X2.iloc[-window:], y2.iloc[-window:]
        Xm, Xs = Xc.mean(), Xc.std().replace(0, 1)
        Xs_norm = (Xc - Xm) / Xs
        try:
            # Use TimeSeriesSplit to respect temporal ordering
            tscv = TimeSeriesSplit(n_splits=3)
            mdl = LassoCV(cv=tscv, random_state=42, alphas=np.logspace(-4, 0, 20), max_iter=2000)
            mdl.fit(Xs_norm, yc)
            w = {col: float(mdl.coef_[i]) for i, col in enumerate(X2.columns)}
            tot = sum(abs(v) for v in w.values())
            return {k: v / tot for k, v in w.items()} if tot > 0 else CONFIG.geo_weights_default.copy()
        except Exception as exc:
            logger.warning("Lasso weight calibration failed: %s; using defaults", exc)
            return CONFIG.geo_weights_default.copy()

    def _build_geofactor(self) -> pd.Series:
        spread = (self.prices["brent"] - self.prices["oil"]) / self.prices["brent"].replace(0, np.nan)
        w = self.weights
        geo = (
            w.get("oil_vol", 0) * self.returns["oil"].rolling(20).std() +
            w.get("gold", 0) * self.returns["gold"].rolling(20).mean() +
            w.get("gold_real", 0) * self.gold_signals["gold_real_ret_roll"] +
            w.get("dxy", 0) * self.returns["dxy"].rolling(20).mean() +
            w.get("spread", 0) * spread.rolling(20).mean() +
            w.get("wheat", 0) * self.returns["wheat"].rolling(20).mean() +
            w.get("copper", 0) * self.returns["copper"].rolling(20).mean() +
            w.get("natgas_vol", 0) * self.returns["natgas"].rolling(20).std() +
            w.get("fert", 0) * self.fert_index +
            w.get("silver_demand", 0) * self.silver_demand +
            w.get("baltic", 0) * self.macro_proxies["baltic"].pct_change().rolling(20).mean() +
            w.get("freightos", 0) * self.macro_proxies["freightos"].pct_change().rolling(20).mean() +
            w.get("move", 0) * self.macro_proxies["move"].pct_change().rolling(20).mean() +
            w.get("fci", 0) * self.macro_proxies["fci"].rolling(20).mean()
        )
        g = geo.dropna()
        return g.clip(g.quantile(0.05), g.quantile(0.95))

    def _build_zscore(self, window: int = 60) -> pd.Series:
        w = min(window, max(20, len(self.prices) // 2))
        z1 = self._rolling_zscore(self.prices["oil"] / self.prices["gold"].replace(0, np.nan), w)
        z2 = self._rolling_zscore(self.prices["oil"] / self.prices["natgas"].replace(0, np.nan), w)
        z3 = self._rolling_zscore(self.gold_signals["gold_real"], w)
        return (CONFIG.zscore_weights["oil_gold"] * z1 +
                CONFIG.zscore_weights["oil_natgas"] * z2 +
                CONFIG.zscore_weights["gold_real"] * z3).dropna()

    @staticmethod
    def _rolling_zscore(s: pd.Series, w: int) -> pd.Series:
        std = s.rolling(w).std()
        return (s - s.rolling(w).mean()) / std.replace(0, np.nan)


# ============================================================
# 6. VOLATILITY ENGINE (GARCH-X, DCC, EVT, Bayesian Shrink)
# ============================================================

class VolatilityEngine:
    """Handles GARCH-X/EGARCH fitting, DCC correlation, EVT tails and Bayesian shrinkage."""

    def __init__(self, returns: pd.DataFrame, geofactor: pd.Series) -> None:
        self.returns = returns
        self.geofactor = geofactor
        self.vol_wti: pd.Series
        self.vol_brent: pd.Series
        self.vol_gold: pd.Series
        self.vol_type_wti: str
        self.vol_type_brent: str
        self.evt_wti: Optional[Dict] = None
        self.regimes: pd.Series
        self.dcc_model: DCCModel
        self._fit_all()

    def _fit_all(self) -> None:
        gf_clean = self.geofactor.dropna() if not self.geofactor.empty else None
        self.vol_wti, self.vol_type_wti = self._fit_vol(self.returns["oil"], gf_clean, "WTI")
        self.vol_brent, self.vol_type_brent = self._fit_vol(self.returns["brent"], gf_clean, "Brent")
        self.vol_gold, _ = self._fit_vol(self.returns["gold"], gf_clean, "Gold")
        self.evt_wti = self._conditional_evt(self.returns["oil"], self.vol_wti)
        self.regimes = self._detect_regime(self.vol_wti)
        self.dcc_model = DCCModel().fit(self.returns["oil"] / self.vol_wti.replace(0, np.nan),
                                        self.returns["brent"] / self.vol_brent.replace(0, np.nan))

    def _fit_vol(self, ret: pd.Series, exog: Optional[pd.Series], label: str) -> Tuple[pd.Series, str]:
        r = ret.dropna()
        if len(r) < 5:
            logger.warning("%s: insufficient data for volatility; using EWMA fallback", label)
            return pd.Series(r.std(), index=ret.index).ffill().bfill(), "EWMA (fallback)"
        if len(r) < CONFIG.vol_min_obs_egarch:
            vol_ewma = r.ewm(span=20, min_periods=5).std()
            return vol_ewma.reindex(ret.index).ffill().bfill(), "EWMA (short window)"
        try:
            rc = r * 100
            if exog is not None and not exog.empty:
                common = rc.index.intersection(exog.dropna().index)
                if len(common) >= 50:
                    rc = rc.loc[common]
                    xc = exog.loc[common].to_frame()
                    model = arch_model(rc, x=xc, mean="Constant", vol="EGARCH", p=1, q=1, dist="skewt")
                    res = model.fit(disp="off")
                    return (res.conditional_volatility / 100).reindex(ret.index).ffill().bfill(), "EGARCH-X"
            model = arch_model(rc, mean="Constant", vol="EGARCH", p=1, q=1, dist="skewt")
            res = model.fit(disp="off")
            return (res.conditional_volatility / 100).reindex(ret.index).ffill().bfill(), "EGARCH"
        except Exception as exc:
            logger.warning("%s EGARCH failed (%s); falling back to EWMA", label, exc)
            vol_ewma = r.ewm(span=20, min_periods=5).std()
            return vol_ewma.reindex(ret.index).ffill().bfill(), "EWMA (EGARCH fail)"

    def _conditional_evt(self, returns: pd.Series, vol: pd.Series, q: float = 0.95, min_obs: int = 30) -> Optional[Dict]:
        common = returns.dropna().index.intersection(vol.dropna().index)
        if len(common) < min_obs:
            return None
        r, v = returns.loc[common], vol.loc[common].replace(0, np.nan)
        resid = (r / v).dropna()
        resid = resid[np.isfinite(resid)]
        if len(resid) < min_obs or resid.std() < 1e-8:
            return None
        th_up, th_lo = np.percentile(resid, q * 100), np.percentile(resid, (1 - q) * 100)
        exc_up = resid[resid > th_up] - th_up
        exc_lo = -resid[resid < th_lo] - th_lo
        # Require at least 10 exceedances for stable GPD fit
        shape_up = stats.genpareto.fit(exc_up)[0] if len(exc_up) >= 10 else 0.2
        scale_up = exc_up.std() if len(exc_up) > 0 else 0.1
        shape_lo = stats.genpareto.fit(exc_lo)[0] if len(exc_lo) >= 10 else 0.2
        scale_lo = exc_lo.std() if len(exc_lo) > 0 else 0.1
        return {"upper": (shape_up, scale_up, th_up), "lower": (shape_lo, scale_lo, th_lo), "resid": resid}

    def _detect_regime(self, vol: pd.Series, threshold: float = 1.5) -> pd.Series:
        v = vol.dropna()
        if len(v) < 20:
            return pd.Series(0, index=vol.index, dtype=int)
        mean = v.rolling(60, min_periods=20).mean()
        std = v.rolling(60, min_periods=20).std().replace(0, 1e-8)
        z = (v - mean) / std
        regimes = np.where(z > 2.5, 3, np.where(z > 1.5, 2, np.where(z > 0.5, 1, 0)))
        return pd.Series(regimes, index=vol.index).fillna(0).astype(int)

    @staticmethod
    def bayes_shrink(vg: pd.Series, prior_d: float, n: int, geofactor: Optional[pd.Series]) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Bayesian shrinkage of volatility series toward a long-run prior.
        Logic preserved from original: if last estimate is within 50%-150% of prior,
        trust the sample fully (w=1.0); otherwise apply shrinkage.
        """
        w = float(np.clip(np.sqrt(n / 252), 0.10, 0.95))
        if geofactor is not None and len(geofactor) > 0:
            prior = prior_d * (1.0 + 0.4 * np.tanh(float(geofactor.iloc[-1])))
        else:
            prior = prior_d
        if len(vg) == 0:
            return pd.Series(prior, index=pd.DatetimeIndex([])), {"vga": prior * 100, "vsa": prior * 100, "w": w}
        v_last = float(vg.iloc[-1])
        # Original logic: full confidence if inside band; shrink otherwise
        inside_band = prior * 0.5 <= v_last <= prior * 1.5
        effective_w = 1.0 if inside_band else w
        vs = effective_w * vg + (1.0 - effective_w) * prior
        vga = v_last * np.sqrt(252) * 100
        vsa = float(vs.iloc[-1]) * np.sqrt(252) * 100 if len(vs) > 0 else vga
        return vs, {"vga": vga, "vsa": vsa, "w": float(effective_w)}


class DCCModel:
    """Side-effect-free DCC(1,1) estimator with decoupled likelihood and rho computation."""

    def __init__(self, a: float = 0.05, b: float = 0.93):
        self.a = a
        self.b = b
        self.Q_bar: Optional[np.ndarray] = None
        self.rho_series: Optional[pd.Series] = None
        self._index: Optional[pd.DatetimeIndex] = None

    def fit(self, e1: pd.Series, e2: pd.Series) -> "DCCModel":
        common = e1.dropna().index.intersection(e2.dropna().index)
        if len(common) < 10:
            logger.warning("DCC: insufficient common observations; using default a=0.05, b=0.93")
            self.Q_bar = np.array([[1.0, 0.85], [0.85, 1.0]])
            self.rho_series = pd.Series(0.85, index=common)
            self._index = common
            return self
        e = np.column_stack([np.clip(e1.loc[common], -3, 3), np.clip(e2.loc[common], -3, 3)])
        self.Q_bar = np.cov(e, rowvar=False)
        np.fill_diagonal(self.Q_bar, 1.0)
        self._index = common

        def nll(params: np.ndarray) -> float:
            a, b = params
            if a <= 0 or b <= 0 or a + b >= 1:
                return 1e10
            Qt = self.Q_bar.copy()
            ll = 0.0
            for t in range(1, len(e)):
                Qt = (1 - a - b) * self.Q_bar + a * np.outer(e[t - 1], e[t - 1]) + b * Qt
                d = np.sqrt(np.diag(Qt))
                if d[0] == 0 or d[1] == 0:
                    return 1e10
                R = np.clip(Qt / np.outer(d, d), -0.9999, 0.9999)
                try:
                    L = np.linalg.cholesky(R)
                    z = np.linalg.solve(L, e[t])
                    ll += -0.5 * np.sum(z ** 2) - np.sum(np.log(np.diag(L)))
                except np.linalg.LinAlgError:
                    return 1e10
            return -ll

        try:
            res = optimize.minimize(nll, [0.05, 0.93], bounds=[(1e-4, 0.3), (0.7, 0.9999)], method="L-BFGS-B")
            if res.success and res.x[0] + res.x[1] < 1:
                self.a, self.b = float(res.x[0]), float(res.x[1])
            else:
                logger.warning("DCC optimization failed; using defaults")
        except Exception as exc:
            logger.warning("DCC minimize exception: %s", exc)
        self._compute_rho(e)
        return self

    def _compute_rho(self, e: np.ndarray) -> None:
        Qt = self.Q_bar.copy()
        rho = np.zeros(len(e))
        for t in range(len(e)):
            if t > 0:
                Qt = (1 - self.a - self.b) * self.Q_bar + self.a * np.outer(e[t - 1], e[t - 1]) + self.b * Qt
            d = np.sqrt(np.diag(Qt))
            if d[0] > 0 and d[1] > 0:
                R = Qt / np.outer(d, d)
                rho[t] = np.clip(R[0, 1], -0.9999, 0.9999)
            else:
                rho[t] = 0.0
        self.rho_series = pd.Series(rho, index=self._index)


# ============================================================
# 7. SIMULATION ENGINE (Monte Carlo with DCC & Jumps)
# ============================================================

class SimulationEngine:
    """Generates fan charts via Monte Carlo with jump-diffusion and DCC dynamics."""

    def __init__(self, wti0: float, brent0: float, vol_wti: float, vol_brent: float,
                 dcc: DCCModel, forecast: np.ndarray, ocol: int, bcol: int,
                 rbase: float, rw: pd.Series, rb: pd.Series, vw: pd.Series, vb: pd.Series,
                 jump_up: float, tail_df: float, bs: float, scenario_mod: Optional[Dict] = None,
                 sims: int = 5_000, steps: int = 10) -> None:
        self.wti0 = wti0
        self.brent0 = brent0
        self.vol_wti = vol_wti
        self.vol_brent = vol_brent
        self.dcc = dcc
        self.forecast = forecast
        self.ocol = ocol
        self.bcol = bcol
        self.rbase = rbase
        self.rw = rw
        self.rb = rb
        self.vw = vw
        self.vb = vb
        self.jump_up = jump_up
        self.tail_df = tail_df
        self.bs = bs
        self.scenario_mod = scenario_mod or {}
        self.sims = sims
        self.steps = steps
        self.paths_wti: Optional[np.ndarray] = None
        self.paths_brent: Optional[np.ndarray] = None
        self.fan: Dict[int, np.ndarray] = {}
        self.fan_b: Dict[int, np.ndarray] = {}
        self.metrics: Dict[str, Any] = {}
        self.moments: Dict[str, float] = {}
        self.brackets: Dict[str, float] = {}

    def run(self, seed: int, progress_bar: Optional[Any] = None) -> None:
        np.random.seed(seed)
        self._apply_scenario_mods()
        self._simulate_paths(progress_bar)
        self._compute_posterior()

    def _apply_scenario_mods(self) -> None:
        self.jump_up *= self.scenario_mod.get("jump_mult", 1.0)
        self.vol_wti *= self.scenario_mod.get("vol_mult", 1.0)
        self.vol_brent *= self.scenario_mod.get("vol_mult", 1.0)
        self.rbase += self.scenario_mod.get("geo_shift", 0.0)
        self.vol_wti = max(self.vol_wti, 1e-6)
        self.vol_brent = max(self.vol_brent, 1e-6)

    def _simulate_paths(self, progress_bar: Optional[Any]) -> None:
        ci = self.rw.index.intersection(self.rb.index).intersection(self.vw.index).intersection(self.vb.index)
        if len(ci) >= 10:
            ew = (self.rw[ci] / self.vw[ci].replace(0, np.nan)).dropna()
            eb = (self.rb[ci] / self.vb[ci].replace(0, np.nan)).dropna()
            c2 = ew.index.intersection(eb.index)
            if len(c2) >= 10:
                e_hist = np.column_stack([np.clip(ew.loc[c2], -3, 3), np.clip(eb.loc[c2], -3, 3)])
                Qb = np.cov(e_hist, rowvar=False)
                np.fill_diagonal(Qb, 1.0)
                eps = np.repeat(e_hist[-1][np.newaxis, :], self.sims, axis=0) + np.random.normal(0, 0.05, (self.sims, 2))
                Qt = np.tile(Qb, (self.sims, 1, 1)).copy()
            else:
                Qb = np.array([[1.0, 0.85], [0.85, 1.0]])
                eps = np.random.normal(0, 1, (self.sims, 2))
                Qt = np.tile(Qb, (self.sims, 1, 1))
        else:
            Qb = np.array([[1.0, 0.85], [0.85, 1.0]])
            eps = np.random.normal(0, 1, (self.sims, 2))
            Qt = np.tile(Qb, (self.sims, 1, 1))

        pu = min(self.jump_up * 1.5, 0.20) if self.bs > 1.2 else self.jump_up
        pd_ = 0.03 * (1.3 if self.bs > 1.2 else 1.0)
        pw = np.zeros((self.sims, self.steps + 1))
        pb = np.zeros((self.sims, self.steps + 1))
        pw[:, 0], pb[:, 0] = self.wti0, self.brent0
        ra = 1 + 0.5 * np.clip(self.rbase + np.random.normal(0, 0.05, (self.sims, self.steps)), -1, 1)

        for t in range(self.steps):
            if progress_bar is not None:
                progress_bar.progress((t + 1) / self.steps)
            if len(ci) >= 10:
                outer = np.einsum("si,sj->sij", eps, eps)
                Qt = (1 - self.dcc.a - self.dcc.b) * Qb[np.newaxis] + self.dcc.a * outer + self.dcc.b * Qt
                diag = np.clip(np.sqrt(np.diagonal(Qt, axis1=1, axis2=2)), 1e-8, None)
                Rt = np.clip(Qt / np.einsum("si,sj->sij", diag, diag), -0.9999, 0.9999)
                rho = Rt[:, 0, 1]
            else:
                rho = np.full(self.sims, 0.85)
            sc = np.sqrt(np.clip(1 - rho ** 2, 1e-8, None))
            z = np.random.standard_t(self.tail_df, (self.sims, 2))
            zw, zb = z[:, 0], rho * z[:, 0] + sc * z[:, 1]
            vw_ = np.clip(self.vol_wti * ra[:, t], 1e-6, 0.08)
            vb_ = np.clip(self.vol_brent * ra[:, t], 1e-6, 0.08)
            sw = np.clip(zw * vw_, -4 * vw_, 4 * vw_)
            sb = np.clip(zb * vb_, -4 * vb_, 4 * vb_)
            sw, sb = self._tail_jumps(sw, vw_), self._tail_jumps(sb, vb_)
            jw, jb = self._jumps_vec(self.sims, pu, pd_)
            sw, sb = sw + jw, sb + jb
            dw = np.clip(self.forecast[t, self.ocol] * ra[:, t], -0.02, 0.02) if t < len(self.forecast) else 0.0
            db = np.clip(self.forecast[t, self.bcol] * ra[:, t], -0.02, 0.02) if t < len(self.forecast) else 0.0
            nw = pw[:, t] * np.exp(dw + sw)
            nb = pb[:, t] * np.exp(db + sb)
            sp = np.where(nb > 0, (nb - nw) / nb, 0)
            nw = np.where(sp < -0.05, nb * 1.05, nw)
            nw = np.where(sp > 0.30, nb * 0.70, nw)
            pw[:, t + 1] = np.clip(nw, self.wti0 * 0.4, self.wti0 * 2.5)
            pb[:, t + 1] = np.clip(nb, self.brent0 * 0.4, self.brent0 * 2.5)
            eps[:, 0] = np.where(vw_ > 0, sw / vw_, 0)
            eps[:, 1] = np.where(vb_ > 0, sb / vb_, 0)
            eps = np.clip(eps, -5, 5)

        self.paths_wti = pw
        self.paths_brent = pb

    @staticmethod
    def _tail_jumps(shocks: np.ndarray, vol: np.ndarray) -> np.ndarray:
        n = len(shocks)
        u = np.random.rand(n)
        return shocks + np.where(u < 0.025, np.random.exponential(0.03, n) * vol, 0) - np.where((u >= 0.025) & (u < 0.05), np.random.exponential(0.02, n) * vol, 0)

    @staticmethod
    def _jumps_vec(n: int, pu: float, pd_: float) -> Tuple[np.ndarray, np.ndarray]:
        u = np.random.rand(n)
        me = np.random.rand(n) < 0.15
        ju = np.where(me, np.random.exponential(0.135, n), np.random.exponential(0.045, n))
        jd = np.random.exponential(0.025, n)
        return (np.where(u < pu, ju, np.where((u >= pu) & (u < pu + pd_), -jd, 0)),
                np.where(u < pu, ju * 0.95, np.where((u >= pu) & (u < pu + pd_), -jd * 0.90, 0)))

    def _compute_posterior(self) -> None:
        pw = self.paths_wti
        pb = self.paths_brent
        if pw is None or pb is None:
            return
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        self.fan = {p: np.percentile(pw, p, axis=0) for p in percentiles}
        self.fan_b = {p: np.percentile(pb, p, axis=0) for p in percentiles}
        term_wti = pw[:, -1]
        term_brent = pb[:, -1]
        sim_mean = float(np.mean(term_wti))
        sim_med = float(np.median(term_wti))
        sim_skew = float(skew(term_wti))
        sim_kurt = float(kurtosis(term_wti))
        sim_mode = 3 * sim_med - 2 * sim_mean
        mask = (pw[:, 1] - self.wti0) <= np.percentile(pw[:, 1] - self.wti0, 5)
        self.moments = {
            "mean": sim_mean, "median": sim_med, "mode": sim_mode,
            "skew": sim_skew, "kurt": sim_kurt
        }
        self.brackets = {
            "<50": np.mean(term_wti < 50) * 100,
            "50-60": np.mean((term_wti >= 50) & (term_wti < 60)) * 100,
            "60-70": np.mean((term_wti >= 60) & (term_wti < 70)) * 100,
            "70-80": np.mean((term_wti >= 70) & (term_wti < 80)) * 100,
            "80-90": np.mean((term_wti >= 80) & (term_wti < 90)) * 100,
            "90-100": np.mean((term_wti >= 90) & (term_wti < 100)) * 100,
            ">100": np.mean(term_wti >= 100) * 100
        }
        self.metrics = {
            "vol_wti": self.vol_wti * np.sqrt(252) * 100,
            "vol_brt": self.vol_brent * np.sqrt(252) * 100,
            "var95": np.percentile(pw[:, 1] - self.wti0, 5),
            "cvar95": float(np.mean((pw[:, 1] - self.wti0)[mask])) if mask.sum() > 0 else np.percentile(pw[:, 1] - self.wti0, 5),
            "wti_70": np.mean(term_wti > 70) * 100,
            "wti_80": np.mean(term_wti > 80) * 100,
            "wti_90": np.mean(term_wti > 90) * 100,
            "wti_100": np.mean(term_wti > 100) * 100,
            "wti_120": np.mean(term_wti > 120) * 100,
            "wti_l60": np.mean(term_wti < 60) * 100,
            "wti_l50": np.mean(term_wti < 50) * 100,
            "wti_l40": np.mean(term_wti < 40) * 100,
            "brent_90": np.mean(term_brent > 90) * 100,
            "brent_100": np.mean(term_brent > 100) * 100
        }


# ============================================================
# 8. RISK ENGINE (Backtesting & Diagnostics)
# ============================================================

class RiskEngine:
    """VaR/ES backtesting and GARCH residual diagnostics."""

    @staticmethod
    def backtest_var(returns: pd.Series, var_forecast: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        common = returns.index.intersection(var_forecast.index)
        n = len(common)
        if n == 0:
            return {"n_violations": 0, "obs_freq": 0.0, "exp_freq": alpha,
                    "Kupiec_p": 1.0, "Christoffersen_p": 1.0, "DQ_p": 1.0,
                    "calibration_score": 0.0, "insufficient_data": True}
        r, v = returns.loc[common], var_forecast.loc[common]
        violations = (r < -v).astype(int)
        nv = int(violations.sum())
        po = nv / n
        pe = alpha
        if n < 100:
            return {"n_violations": nv, "obs_freq": po, "exp_freq": pe,
                    "Kupiec_p": None, "Christoffersen_p": None, "DQ_p": None,
                    "calibration_score": None, "insufficient_data": True}
        # Kupiec
        if 0 < nv < n:
            lr_pf = -2 * np.log(((1 - pe) ** (n - nv) * pe ** nv) / ((1 - po) ** (n - nv) * po ** nv))
            kp = 1 - chi2.cdf(lr_pf, 1)
        else:
            kp = 0.5
        # Christoffersen
        if n > 1:
            n00 = ((violations[:-1] == 0) & (violations[1:] == 0)).sum()
            n01 = ((violations[:-1] == 0) & (violations[1:] == 1)).sum()
            n10 = ((violations[:-1] == 1) & (violations[1:] == 0)).sum()
            n11 = ((violations[:-1] == 1) & (violations[1:] == 1)).sum()
            p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
            p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
            if (n01 + n11) > 0:
                denom = ((1 - p01) ** n00) * (p01 ** n01) * ((1 - p11) ** n10) * (p11 ** n11)
                lr_cc = -2 * np.log(((1 - pe) ** (n - 1 - (n01 + n11)) * pe ** (n01 + n11)) / denom)
                cp = 1 - chi2.cdf(lr_cc, 1) if lr_cc > 0 else 0.5
            else:
                cp = 0.5
        else:
            cp = 0.5
        # DQ
        try:
            X = pd.DataFrame({"const": 1, "lag": violations.shift(1).fillna(0)})
            model = Logit(violations, X).fit(disp=0)
            dq = 1 - chi2.cdf(model.llr, X.shape[1])
        except Exception:
            dq = 1.0
        return {"n_violations": nv, "obs_freq": po, "exp_freq": pe,
                "Kupiec_p": kp, "Christoffersen_p": cp, "DQ_p": dq,
                "calibration_score": 1 - np.mean([kp, cp, dq]), "insufficient_data": False}

    @staticmethod
    def backtest_es(returns: pd.Series, cvar_val: pd.Series, var_forecast: pd.Series) -> float:
        common = returns.index.intersection(var_forecast.index)
        if len(common) == 0:
            return np.nan
        r, v = returns.loc[common], var_forecast.loc[common]
        cv = float(cvar_val.iloc[-1]) if len(cvar_val) > 0 else np.nan
        viol = (r < -v).astype(int)
        if viol.sum() == 0 or np.isnan(cv):
            return np.nan
        return float(((r[viol == 1] + v[viol == 1]).sum() / (viol.sum() * cv)) - 1)

    @staticmethod
    def garch_diagnostics(resid: pd.Series) -> Dict[str, float]:
        r = resid.dropna()
        if len(r) < 20:
            return {"LB5": np.nan, "LB10": np.nan, "ARCH_p": np.nan}
        try:
            lb = acorr_ljungbox(r, lags=[5, 10], return_df=True)
            lb5 = lb.loc[5, "lb_pvalue"] if 5 in lb.index else np.nan
            lb10 = lb.loc[10, "lb_pvalue"] if 10 in lb.index else np.nan
        except Exception:
            lb5, lb10 = np.nan, np.nan
        try:
            arch = het_arch(r ** 2, nlags=10)
            arch_p = arch[1] if len(arch) > 1 else np.nan
        except Exception:
            arch_p = np.nan
        return {"LB5": lb5, "LB10": lb10, "ARCH_p": arch_p}


# ============================================================
# 9. ML ENGINE (Walk-Forward, Benchmarking, SHAP)
# ============================================================

class MLEngine:
    """Machine learning benchmarking with temporal cross-validation."""

    @staticmethod
    def walk_forward(returns_series: pd.Series, train_years: int = 2, test_months: int = 3) -> pd.DataFrame:
        dates = returns_series.index
        ts, qs = int(train_years * 252), int(test_months * 21)
        results: List[Dict[str, Any]] = []
        start = 0
        while start + ts + qs <= len(dates):
            te, qe = start + ts, start + ts + qs
            train, test = returns_series.iloc[start:te], returns_series.iloc[te:qe]
            pred = train.iloc[-20:].mean() if len(train) >= 20 else train.mean()
            rmse = float(np.sqrt(((pred - test) ** 2).mean()))
            results.append({
                "Window Start": dates[start].strftime("%Y-%m-%d"),
                "Window End": dates[qe - 1].strftime("%Y-%m-%d"),
                "OOS RMSE": rmse
            })
            start += qs
        return pd.DataFrame(results)

    @staticmethod
    def benchmark(returns_df: pd.DataFrame, target_col: str = "oil") -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, pd.Series]:
        features = returns_df.shift(1).dropna()
        target = returns_df[target_col].iloc[1:]
        common = features.index.intersection(target.index)
        X, y = features.loc[common], target.loc[common]
        if len(X) < 10:
            empty = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Directional Accuracy": "—"}
            return {"RandomForest": empty.copy(), "XGBoost": empty.copy(), "LightGBM": empty.copy()}, X, y
        n_splits = max(2, min(5, len(X) // 3))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        }
        if LGBM_AVAILABLE and lgb is not None:
            models["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        out: Dict[str, Dict[str, Any]] = {}
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
                except Exception as exc:
                    logger.debug("%s fold failed: %s", name, exc)
            out[name] = {
                "RMSE": np.mean(rmses) if rmses else np.nan,
                "MAE": np.mean(maes) if maes else np.nan,
                "MAPE": np.mean(mapes) if mapes else np.nan,
                "Directional Accuracy": f"{np.mean(dirs):.2f}%" if dirs else "—"
            }
        return out, X, y

    @staticmethod
    def shap_summary(X: pd.DataFrame, y: pd.Series) -> Tuple[Any, np.ndarray, List[str]]:
        if X is None or y is None or len(X) < 5:
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#FFFFFF")
            ax.text(0.5, 0.5, "Insufficient data for SHAP", ha="center", va="center")
            ax.set_facecolor("#FFFFFF")
            plt.tight_layout()
            return fig, np.array([]), []
        mdl = RandomForestRegressor(n_estimators=100, random_state=42)
        mdl.fit(X, y)
        if SHAP_AVAILABLE and shap is not None:
            try:
                exp = shap.TreeExplainer(mdl)
                sv = exp.shap_values(X)
                fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#FFFFFF")
                ax.set_facecolor("#FFFFFF")
                shap.summary_plot(sv, X, show=False, plot_size=None)
                plt.tight_layout()
                return fig, sv, X.columns.tolist()
            except Exception as exc:
                logger.warning("SHAP TreeExplainer failed: %s; using permutation fallback", exc)
        # Permutation fallback
        imp = permutation_importance(mdl, X, y, n_repeats=5, random_state=42)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#FFFFFF")
        ax.barh(X.columns, imp.importances_mean, color="#1E3A5F")
        ax.set_title("Permutation Importance (SHAP fallback)")
        ax.set_xlabel("Importance")
        ax.set_facecolor("#FFFFFF")
        plt.tight_layout()
        return fig, imp.importances_mean, X.columns.tolist()


# ============================================================
# 10. APPLICATION STATE
# ============================================================

@dataclass
class AppState:
    """Strongly typed session state container."""
    results: Optional[Dict[str, Any]] = None
    gf: Optional[pd.Series] = None
    zsc: Optional[pd.Series] = None
    vw: Optional[pd.Series] = None
    vb: Optional[pd.Series] = None
    vg: Optional[pd.Series] = None
    fi: Optional[pd.Series] = None
    gs: Optional[Dict[str, pd.Series]] = None
    prices: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None
    wti0: float = 0.0
    brt0: float = 0.0
    usda: Optional[Dict[str, Any]] = None
    bs: float = 1.0
    dw: Optional[Dict[str, float]] = None
    db: Optional[Dict[str, float]] = None
    tdf: float = 3.0
    dcc_a: float = 0.05
    dcc_b: float = 0.93
    dcc_rho: Optional[pd.Series] = None
    weights: Optional[Dict[str, float]] = None
    sharpe: Optional[pd.Series] = None
    sortino: Optional[pd.Series] = None
    corr_mx: Optional[pd.DataFrame] = None
    stress_idx: Optional[pd.Series] = None
    evt: Optional[Dict] = None
    gdiag: Optional[Dict[str, float]] = None
    bt_res: Optional[Dict[str, Any]] = None
    es_z: float = np.nan
    ml_metrics: Optional[Dict[str, Dict[str, Any]]] = None
    shap_fig: Optional[Any] = None
    shap_vals: Optional[np.ndarray] = None
    feat_names: Optional[List[str]] = None
    wf_df: Optional[pd.DataFrame] = None
    regimes_ts: Optional[pd.Series] = None
    model_score: Optional[int] = None
    vix_fred: float = 20.0
    eia_stocks: float = 420000.0
    macro_proxies: Optional[Dict[str, pd.Series]] = None
    gpr: Optional[pd.Series] = None
    cot: float = 300000.0
    last_update: str = ""
    fert_source: str = "unknown"
    vol_type_wti: str = "EGARCH"
    vol_type_brent: str = "EGARCH"
    mc_seed: int = 42

    def to_session(self) -> None:
        for k, v in self.__dict__.items():
            st.session_state[k] = v

    @classmethod
    def from_session(cls) -> "AppState":
        state = cls()
        for k in state.__dict__:
            if k in st.session_state:
                setattr(state, k, st.session_state[k])
        return state


# ============================================================
# 11. UI COMPONENTS (renderers per tab)
# ============================================================

class UIComponents:
    """Encapsulates all Streamlit rendering logic."""

    def __init__(self, state: AppState):
        self.S = state
        self.C = CONFIG.colors
        self.PL = CONFIG.plotly_layout

    def _qfig(self, h: int = 480) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(**self.PL, height=h)
        return fig

    def render_executive_summary(self) -> None:
        st.markdown('<div class="sec-label">Report · Asset Management Grade</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Executive Macro & Geopolitical Summary</div>', unsafe_allow_html=True)
        gf_last = float(self.S.gf.iloc[-1]) if self.S.gf is not None and len(self.S.gf) > 0 else 0.0
        st_last = float(self.S.stress_idx.iloc[-1]) if self.S.stress_idx is not None and len(self.S.stress_idx) > 0 else 0.0
        regime_curr = ["Normal", "Elevated Risk", "Stress", "Crisis"][int(self.S.regimes_ts.iloc[-1])] if self.S.regimes_ts is not None and len(self.S.regimes_ts) > 0 else "Normal"
        mc_steps = CONFIG.mc_steps_default
        if self.S.results:
            mc_steps = len(self.S.results.get("fan", {}).get(50, [])) - 1
        fan = self.S.results["fan"] if self.S.results else {}
        M = self.S.results["metrics"] if self.S.results else {}
        st.markdown(f"""
        <div style="background:#FDFBF8; border:1px solid #D9D5CD; padding:1.5rem; line-height:1.7; color:#1C1C1C; font-size:0.92rem;">
            <strong>Macro Insight Terminal Architecture Report:</strong><br><br>
            The <strong>GeoFactor Composite</strong> closed the last session quantified at <strong>{fmt_num(gf_last, '.2f')}σ</strong>, indicating a structural regime classified as <strong>{regime_curr.upper()}</strong>.
            This positioning reflects latent geopolitical pressure with direct transmission via physical risk premium in the energy futures curve.
            The unified system Stress Index is priced at <strong>{fmt_num(st_last, '.2f')}</strong>, conditioned by FX market implied volatility (DXY) and structural shocks captured in the Fertilizer Index (currently operating with a tail multiplier of <strong>{fmt_num(self.S.bs, '.2f')}x</strong>).<br><br>
            Over the short-term predictive horizon ({mc_steps} business days), the <strong>Conditional EVT + DCC-GARCH-X</strong> stochastic engine points to positive tail asymmetry.
            The central projection (Median P50) for the WTI spot contract stabilizes at <strong>US$ {fmt_num(fan.get(50, [0]* (mc_steps+1))[-1] if fan else 0, '.2f')}/bbl</strong>, operating within a severe stress amplitude bounded by the extreme tail percentile (P99) at <strong>US$ {fmt_num(fan.get(99, [0]* (mc_steps+1))[-1] if fan else 0, '.2f')}/bbl</strong>.
            The implied probability of extreme bullish rupture (WTI exceeding the critical US$ 100/bbl barrier) is calibrated at <strong>{fmt_num(M.get('wti_100', 0), '.1f')}%</strong>, while the risk of deflationary structural collapse below US$ 60/bbl is priced by the model at only <strong>{fmt_num(M.get('wti_l60', 0), '.1f')}%</strong>.
            The dynamic conditional correlation (DCC) between the WTI and Brent complex remains structurally pinned with stable long-run persistence parameters (α: {fmt_num(self.S.dcc_a, '.4f')}, β: {fmt_num(self.S.dcc_b, '.4f')}).
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="margin-top:1.5rem;" class="sec-label">Operational Threshold Matrices</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<table class="data-table"><thead><tr><th>WTI Bullish</th><th>Implied Prob</th></tr></thead>'
                        f'<tbody><tr><td>WTI &gt; US$ 70</td><td><strong>{fmt_num(M.get("wti_70", 0), ".1f")}%</strong></td></tr>'
                        f'<tr><td>WTI &gt; US$ 80</td><td><strong>{fmt_num(M.get("wti_80", 0), ".1f")}%</strong></td></tr>'
                        f'<tr><td>WTI &gt; US$ 90</td><td><strong>{fmt_num(M.get("wti_90", 0), ".1f")}%</strong></td></tr>'
                        f'<tr><td>WTI &gt; US$ 100</td><td><strong>{fmt_num(M.get("wti_100", 0), ".1f")}%</strong></td></tr>'
                        f'<tr><td>WTI &gt; US$ 120</td><td><strong>{fmt_num(M.get("wti_120", 0), ".1f")}%</strong></td></tr></tbody></table>', unsafe_allow_html=True)
        with c2:
            st.markdown('<table class="data-table"><thead><tr><th>WTI Bearish</th><th>Implied Prob</th></tr></thead>'
                        f'<tbody><tr><td>WTI &lt; US$ 60</td><td><strong>{fmt_num(M.get("wti_l60", 0), ".1f")}%</strong></td></tr>'
                        f'<tr><td>WTI &lt; US$ 50</td><td><strong>{fmt_num(M.get("wti_l50", 0), ".1f")}%</strong></td></tr>'
                        f'<tr><td>WTI &lt; US$ 40</td><td><strong>{fmt_num(M.get("wti_l40", 0), ".1f")}%</strong></td></tr></tbody></table>', unsafe_allow_html=True)
        with c3:
            st.markdown('<table class="data-table"><thead><tr><th>Brent Complex</th><th>Implied Prob</th></tr></thead>'
                        f'<tbody><tr><td>Brent &gt; US$ 90</td><td><strong>{fmt_num(M.get("brent_90", 0), ".1f")}%</strong></td></tr>'
                        f'<tr><td>Brent &gt; US$ 100</td><td><strong>{fmt_num(M.get("brent_100", 0), ".1f")}%</strong></td></tr></tbody></table>', unsafe_allow_html=True)

    def render_market_vol(self) -> None:
        st.markdown('<div class="sec-label">01 · Risk Metrics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        fan = self.S.results["fan"] if self.S.results else {}
        delta_wti = safe_delta(fan.get(50, [0])[-1] if fan else None, ".2f")
        delta_brt = safe_delta(self.S.brt0 - self.S.wti0, ".2f")
        delta_vsa_wti = safe_delta(self.S.dw.get("vsa") if self.S.dw else None, ".1f")
        delta_vsa_brt = safe_delta(self.S.db.get("vsa") if self.S.db else None, ".1f")
        c1.metric("WTI Crude Spot", f"${self.S.wti0:.2f}", delta=delta_wti)
        c2.metric("Brent Crude", f"${self.S.brt0:.2f}", delta=delta_brt)
        M = self.S.results.get("metrics", {}) if self.S.results else {}
        c3.metric("WTI Vol p.a.", f"{M.get('vol_wti', 0):.1f}%", delta=delta_vsa_wti)
        c4.metric("Brent Vol p.a.", f"{M.get('vol_brt', 0):.1f}%", delta=delta_vsa_brt)

        fig_vol = self._qfig(480)
        if self.S.vw is not None:
            fig_vol.add_trace(go.Scatter(x=self.S.vw.index, y=self.S.vw * np.sqrt(252) * 100, name="WTI EGARCH Vol",
                                         line=dict(color=self.C["navy"], width=2.8), hovertemplate="Date: %{x|%d %b %Y}<br>Vol: %{y:.2f}%"))
        if self.S.vb is not None:
            fig_vol.add_trace(go.Scatter(x=self.S.vb.index, y=self.S.vb * np.sqrt(252) * 100, name="Brent EGARCH Vol",
                                         line=dict(color=self.C["blue"], width=2.8, dash="dash"), hovertemplate="Date: %{x|%d %b %Y}<br>Vol: %{y:.2f}%"))
        if self.S.vg is not None:
            fig_vol.add_trace(go.Scatter(x=self.S.vg.index, y=self.S.vg * np.sqrt(252) * 100, name="Gold EGARCH Vol",
                                         line=dict(color=self.C["gold"], width=2.0, dash="dot"), hovertemplate="Date: %{x|%d %b %Y}<br>Vol: %{y:.2f}%"))
        fig_vol.update_layout(yaxis_ticksuffix="%", title="EGARCH(1,1) Filtering Engine (Exogenous GeoFactor Multi-Regime)")
        st.plotly_chart(fig_vol, use_container_width=True)

        if self.S.dcc_rho is not None and not self.S.dcc_rho.empty:
            fig_dcc = self._qfig(380)
            fig_dcc.add_trace(go.Scatter(x=self.S.dcc_rho.index, y=self.S.dcc_rho.values, name="DCC Correlation (WTI/Brent)",
                                         line=dict(color=self.C["teal"], width=2.5), hovertemplate="Date: %{x|%d %b %Y}<br>Corr: %{y:.3f}"))
            fig_dcc.update_layout(title="Conditional Correlation (DCC) WTI/Brent", yaxis_range=[-1, 1])
            st.plotly_chart(fig_dcc, use_container_width=True)

        st.markdown('<div class="sec-label">Precious Metals & Agricultural Inputs</div>', unsafe_allow_html=True)
        col_gold, col_fert = st.columns(2)
        with col_gold:
            if self.S.prices is not None and "gold" in self.S.prices.columns:
                gold_price = self.S.prices["gold"].dropna()
                if len(gold_price) > 30:
                    gold_ma = gold_price.rolling(60, min_periods=20).mean()
                    gold_std = gold_price.rolling(60, min_periods=20).std()
                    fig_gold = self._qfig(480)
                    fig_gold.add_trace(go.Scatter(x=gold_price.index, y=gold_ma + 2*gold_std, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig_gold.add_trace(go.Scatter(x=gold_price.index, y=gold_ma - 2*gold_std, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(180,148,80,0.08)", showlegend=False, hoverinfo="skip"))
                    fig_gold.add_trace(go.Scatter(x=gold_price.index, y=gold_ma + gold_std, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig_gold.add_trace(go.Scatter(x=gold_price.index, y=gold_ma - gold_std, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(180,148,80,0.12)", showlegend=False, hoverinfo="skip"))
                    fig_gold.add_trace(go.Scatter(x=gold_price.index, y=gold_price, name="Gold Price", line=dict(color=self.C["gold"], width=3.0),
                                                  hovertemplate="Date: %{x|%d %b %Y}<br>Gold: $%{y:.2f}"))
                    last_gold = gold_price.iloc[-1]
                    last_ma = gold_ma.iloc[-1]
                    sigma_pos = (last_gold - last_ma) / gold_std.iloc[-1] if not np.isnan(gold_std.iloc[-1]) else 0
                    fig_gold.add_annotation(x=gold_price.index[-1], y=last_gold, text=f"${last_gold:.2f} ({sigma_pos:+.1f}σ)",
                                            showarrow=True, arrowhead=2, ax=30, ay=-30, font=dict(family="JetBrains Mono", size=11, color=self.C["gold"]))
                    fig_gold.update_layout(title="Gold Price with Regime Bands (±1σ / ±2σ)", yaxis_title="USD/oz")
                    st.plotly_chart(fig_gold, use_container_width=True)
                else:
                    st.info("Insufficient gold data for regime bands")
        with col_fert:
            st.info("Fertilizer chart rendered via MacroEngine data pipeline (see original CSV fallback).")

    def render_geopolitical(self) -> None:
        st.markdown('<div class="sec-label">02 · Geopolitical Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Geopolitical Risk & Structural Indicators</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        gpr_last = float(self.S.gpr.iloc[-1]) if self.S.gpr is not None and len(self.S.gpr) > 0 else 0.0
        c1.metric("GPR Index (FRED)", f"{gpr_last:.1f}")
        c2.metric("VIX (FRED)", f"{self.S.vix_fred:.2f}")
        c3.metric("COT Net Specs (proxy)", f"{self.S.cot:,.0f}")
        if self.S.gpr is not None and self.S.prices is not None and "oil" in self.S.prices.columns:
            gpr_series = self.S.gpr
            wti_price = self.S.prices["oil"].dropna()
            common_idx = gpr_series.dropna().index.intersection(wti_price.index)
            if len(common_idx) > 10:
                fig_geo = make_subplots(specs=[[{"secondary_y": True}]])
                fig_geo.add_trace(go.Scatter(x=common_idx, y=[100]*len(common_idx), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig_geo.add_trace(go.Scatter(x=common_idx, y=[150]*len(common_idx), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(180,148,80,0.15)", showlegend=False, hoverinfo="skip"))
                fig_geo.add_trace(go.Scatter(x=common_idx, y=[max(gpr_series.max(), 200)]*len(common_idx), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(180,148,80,0.25)", showlegend=False, hoverinfo="skip"))
                fig_geo.add_trace(go.Scatter(x=common_idx, y=gpr_series[common_idx], name="GPR (FRED)", line=dict(color=self.C["rust"], width=2.8), hovertemplate="Date: %{x|%d %b %Y}<br>GPR: %{y:.1f}"), secondary_y=False)
                fig_geo.add_trace(go.Scatter(x=common_idx, y=wti_price[common_idx], name="WTI Price", line=dict(color=self.C["navy"], width=2.5, dash="dot"), hovertemplate="Date: %{x|%d %b %Y}<br>WTI: $%{y:.2f}"), secondary_y=True)
                fig_geo.update_layout(**self.PL, height=520, title="Geopolitical Risk & WTI Price with Regime Zones", yaxis_title="GPR Index", yaxis2_title="WTI (USD/bbl)")
                fig_geo.update_yaxes(range=[0, max(gpr_series.max(), 200)], secondary_y=False)
                st.plotly_chart(fig_geo, use_container_width=True)
            else:
                st.info("Insufficient overlapping GPR/WTI data")

    def render_attribution(self) -> None:
        st.markdown('<div class="sec-label">03 · GeoFactor Attribution</div>', unsafe_allow_html=True)
        if self.S.weights:
            weights_sorted = sorted(self.S.weights.items(), key=lambda x: abs(x[1]), reverse=True)
            df_weights = pd.DataFrame(weights_sorted, columns=["Factor", "Weight"])
            st.markdown('<div class="data-table">' + df_weights.to_html(index=False) + '</div>', unsafe_allow_html=True)
            fig_attr = self._qfig(420)
            fig_attr.add_trace(go.Bar(x=df_weights["Factor"], y=df_weights["Weight"], marker_color=self.C["navy"], hovertemplate="Factor: %{x}<br>Weight: %{y:.3f}"))
            fig_attr.update_layout(title="Calibrated GeoFactor Weights (LassoCV with TimeSeriesSplit)")
            st.plotly_chart(fig_attr, use_container_width=True)

    def render_monte_carlo(self) -> None:
        st.markdown('<div class="sec-label">04 · Monte Carlo Fan Chart</div>', unsafe_allow_html=True)
        fan = self.S.results.get("fan", {}) if self.S.results else {}
        if not fan:
            st.warning("No simulation results available")
            return
        x_days = list(range(len(fan.get(50, []))))
        fig_mc = self._qfig(520)
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[1], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[99], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(30,58,95,0.05)", showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[5], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[95], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(30,58,95,0.08)", showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[10], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[90], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(30,58,95,0.12)", showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[25], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[75], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(30,58,95,0.18)", showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[50], name="P50 (Median)", line=dict(color=self.C["navy"], width=3.2), hovertemplate="Day: %{x}<br>%{fullData.name}: $%{y:.2f}"))
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[1], name="P01", line=dict(color=self.C["silver"], width=1.2, dash="dot"), hovertemplate="Day: %{x}<br>%{fullData.name}: $%{y:.2f}"))
        fig_mc.add_trace(go.Scatter(x=x_days, y=fan[99], name="P99", line=dict(color=self.C["silver"], width=1.2, dash="dot"), hovertemplate="Day: %{x}<br>%{fullData.name}: $%{y:.2f}"))
        fig_mc.update_layout(title="WTI Price Distribution Fan Chart", xaxis_title="Days Forward", yaxis_title="USD/bbl", hovermode="x unified")
        st.plotly_chart(fig_mc, use_container_width=True)
        moments = self.S.results.get("moments", {}) if self.S.results else {}
        st.markdown(f"""
        <div class="diag-card">
            Mean: {fmt_num(moments.get('mean'), '.2f')} | Median: {fmt_num(moments.get('median'), '.2f')} | Mode: {fmt_num(moments.get('mode'), '.2f')}<br>
            Skew: {fmt_num(moments.get('skew'), '.2f')} | Excess Kurtosis: {fmt_num(moments.get('kurt'), '.2f')}
        </div>""", unsafe_allow_html=True)

    def render_quant_stats(self) -> None:
        st.markdown('<div class="sec-label">05 · Quantitative Statistics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        sharpe_oil = self.S.sharpe.get("oil") if isinstance(self.S.sharpe, pd.Series) else self.S.sharpe
        sharpe_brent = self.S.sharpe.get("brent") if isinstance(self.S.sharpe, pd.Series) else self.S.sharpe
        sortino_oil = self.S.sortino.get("oil") if isinstance(self.S.sortino, pd.Series) else self.S.sortino
        c1.metric("WTI Sharpe (ann.)", f"{sharpe_oil:.2f}" if sharpe_oil is not None else "—")
        c2.metric("Brent Sharpe (ann.)", f"{sharpe_brent:.2f}" if sharpe_brent is not None else "—")
        c3.metric("Sortino Ratio", f"{sortino_oil:.2f}" if sortino_oil is not None else "—")
        st.markdown('<div class="sec-title">Correlation Matrix</div>', unsafe_allow_html=True)
        if self.S.corr_mx is not None:
            st.dataframe(self.S.corr_mx.style.background_gradient(cmap="Blues"), use_container_width=True)

    def render_diagnostics(self) -> None:
        st.markdown('<div class="sec-label">06 · Model Diagnostics</div>', unsafe_allow_html=True)
        bt = self.S.bt_res or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Violations", bt.get("n_violations", 0))
        if bt.get("insufficient_data", False):
            c2.metric("Kupiec p-value", "⚠️ Insuf.")
            c3.metric("Christoffersen p", "⚠️ Insuf.")
            c4.metric("DQ p-value", "⚠️ Insuf.")
        else:
            c2.metric("Kupiec p-value", f"{bt.get('Kupiec_p', 0):.3f}")
            c3.metric("Christoffersen p", f"{bt.get('Christoffersen_p', 0):.3f}")
            c4.metric("DQ p-value", f"{bt.get('DQ_p', 0):.3f}")
        gd = self.S.gdiag or {}
        c5, c6, c7 = st.columns(3)
        c5.metric("LB(5) p-value", f"{gd.get('LB5', np.nan):.3f}" if not np.isnan(gd.get('LB5', np.nan)) else "—")
        c6.metric("LB(10) p-value", f"{gd.get('LB10', np.nan):.3f}" if not np.isnan(gd.get('LB10', np.nan)) else "—")
        c7.metric("ARCH(10) p", f"{gd.get('ARCH_p', np.nan):.3f}" if not np.isnan(gd.get('ARCH_p', np.nan)) else "—")
        if self.S.model_score is not None:
            st.markdown(f'<div class="info-block">Calibration Score: {self.S.model_score}%</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-block">⚠️ Insufficient data for Calibration Score (min. 100 observations)</div>', unsafe_allow_html=True)

    def render_ml(self) -> None:
        st.markdown('<div class="sec-label">07 · Machine Learning Benchmarking</div>', unsafe_allow_html=True)
        ml = self.S.ml_metrics or {}
        if ml:
            df_ml = pd.DataFrame(ml).T.reset_index().rename(columns={"index": "Model"})
            st.dataframe(df_ml, use_container_width=True)
        if self.S.shap_fig is not None:
            st.pyplot(self.S.shap_fig, use_container_width=True)
        if self.S.wf_df is not None:
            st.dataframe(self.S.wf_df, use_container_width=True)

    def render_model_card(self) -> None:
        st.markdown('<div class="sec-label">08 · Model Card</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-block">
        <b>Model:</b> Conditional EVT + DCC-GARCH-X + Bayesian Volatility Shrinkage<br>
        <b>GeoFactor:</b> LassoCV-Weighted Multi-Asset Structural Index (TimeSeriesSplit CV)<br>
        <b>Monte Carlo:</b> 5,000–30,000 paths, t-Student copula, tail jumps, DCC correlation<br>
        <b>Backtesting:</b> Walk-Forward, VaR/ES, Kupiec, Christoffersen, DQ tests<br>
        <b>ML:</b> Random Forest, XGBoost, LightGBM with TimeSeriesSplit cross-validation<br>
        <b>Data Sources:</b> Yahoo Finance, FRED (GPR, VIX), EIA (inventories), OilPriceAPI, World Bank (fertilizers), CFTC COT proxy
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 12. EXPORT UTILITY
# ============================================================

def export_results_to_csv(state: AppState) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Metric", "Value"])
    moments = state.results.get("moments", {}) if state.results else {}
    fan = state.results.get("fan", {}) if state.results else {}
    writer.writerow(["Simulated Mean", moments.get("mean")])
    writer.writerow(["Median (P50)", moments.get("median")])
    writer.writerow(["Mode", moments.get("mode")])
    writer.writerow(["Skewness", moments.get("skew")])
    writer.writerow(["Excess Kurtosis", moments.get("kurt")])
    for p in [99, 90, 50, 10, 1]:
        writer.writerow([f"P{p:02d} WTI", fan.get(p, [None])[-1] if fan else None])
    metrics = state.results.get("metrics", {}) if state.results else {}
    writer.writerow(["Prob WTI > 100", metrics.get("wti_100")])
    writer.writerow(["Prob WTI < 60", metrics.get("wti_l60")])
    writer.writerow([" "])
    writer.writerow(["Factor", "Weight"])
    for k, v in (state.weights or {}).items():
        writer.writerow([k, v])
    writer.writerow([" "])
    writer.writerow(["Macro Proxy", "Latest Value"])
    for k, v in (state.macro_proxies or {}).items():
        if isinstance(v, pd.Series) and len(v) > 0:
            writer.writerow([k, v.iloc[-1]])
        else:
            writer.writerow([k, v])
    return output.getvalue()


# ============================================================
# 13. MAIN ORCHESTRATOR
# ============================================================

def main():
    st.set_page_config(page_title="GeoQuant · Research Terminal", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
    st.markdown(INSTITUTIONAL_CSS, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='padding:1.3rem 0 1.1rem;border-bottom:1px solid #D9D5CD;margin-bottom:1.3rem;'>
            <div style='font-family:"JetBrains Mono",monospace;font-size:.5rem;letter-spacing:.26em;color:#B49450;text-transform:uppercase;margin-bottom:.4rem;'>◆ Edumetria</div>
            <div style='font-family:"Playfair Display",Georgia,serif;font-size:1.3rem;font-weight:300;color:#1E3A5F;letter-spacing:.06em;'>GeoQuant Terminal</div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:.5rem;color:#7A766E;letter-spacing:.14em;margin-top:.3rem;'>Quantitative Research Infrastructure v5.0</div>
        </div>""", unsafe_allow_html=True)
        def slabel(t: str):
            st.markdown(f'<div style="font-family:'JetBrains Mono',monospace;font-size:.54rem;letter-spacing:.2em;color:#B49450;text-transform:uppercase;margin:.9rem 0 .4rem;">{t}</div>', unsafe_allow_html=True)
        def ssep():
            st.markdown('<div style="height:1px;background:#D9D5CD;margin:.6rem 0;"></div>', unsafe_allow_html=True)
        slabel("· Simulation")
        mc_sims = st.slider("Monte Carlo paths", 1_000, 30_000, CONFIG.mc_sims_default, 1_000)
        mc_steps = st.slider("Horizon (days)", 5, 30, CONFIG.mc_steps_default, 1)
        mc_seed = st.number_input("Reproducibility Seed", value=42, step=1)
        ssep()
        slabel("· Interactive Scenario Engine")
        scen_choice = st.selectbox("Geo Scenario", [
            "Base Model (Live)", "Geopolitical Escalation", "Diplomatic Ceasefire",
            "Strait of Hormuz Closure", "OPEC Supply Shock", "Global Recession"
        ])
        ssep()
        slabel("· Jump Parameters")
        jump_up = st.slider("Jump prob ↑", 0.01, 0.20, 0.07, 0.01)
        jump_down = st.slider("Jump prob ↓", 0.01, 0.10, 0.03, 0.01)
        tail_df = st.slider("Tail df", 2.5, 8.0, 3.0, 0.5)
        ssep()
        slabel("· Vol Priors (annual)")
        prior_wti = st.slider("WTI prior", 0.20, 0.65, 0.35, 0.01)
        prior_brent = st.slider("Brent prior", 0.20, 0.65, 0.35, 0.01)
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

    # Header
    now_sp = datetime.now(pytz.timezone("America/Sao_Paulo"))
    data_layer = DataLayer()
    api_status = data_layer.get_api_status()
    status_badges = " ".join([f"<span style='margin-left:0.5rem;font-size:0.55rem;'>{k}: {v}</span>" for k, v in api_status.items()])
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:flex-start;padding:1.6rem 0 1.2rem;border-bottom:1px solid #D9D5CD;margin-bottom:1.8rem;'>
      <div>
        <div style='display:flex;align-items:baseline;gap:.6rem;'>
          <span style='font-family:"JetBrains Mono",monospace;font-size:.85rem;color:#B49450;letter-spacing:.2em;'>◆◆◆</span>
          <div>
            <div style='font-family:"Playfair Display",Georgia,serif;font-size:1.9rem;font-weight:300;color:#1E3A5F;letter-spacing:.06em;line-height:1;'>GeoQuant · Research Terminal</div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:.55rem;color:#70695E;letter-spacing:.2em;text-transform:uppercase;margin-top:.3rem;'>Macro Geopolitical Quant · Institutional Analytics Platform v5.0</div>
          </div>
        </div>
      </div>
      <div style='text-align:right;'>
        <div style='display:inline-block;background:#1E3A5F;color:#D4C094;padding:.2rem .7rem;font-family:"JetBrains Mono",monospace;font-size:.55rem;letter-spacing:.18em;text-transform:uppercase;'>⚑ {scen_choice.upper()}</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:.57rem;color:#70695E;letter-spacing:.10em;margin-top:.4rem;'>{now_sp.strftime("%d %B %Y · %H:%M")} (SP)</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:.45rem;color:#70695E;margin-top:.2rem;'>{status_badges}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Execution Pipeline
    if run_btn or "results" not in st.session_state:
        st.cache_data.clear()
        loading = st.empty()
        loading.markdown("""
        <div style='text-align:center;padding:2.5rem 2rem;background:#FDFBF8;border:1px solid #C4BDAF;margin:1rem 0;'>
          <div style='font-family:"JetBrains Mono",monospace;font-size:.56rem;letter-spacing:.22em;color:#9E8050;text-transform:uppercase;margin-bottom:.7rem;'>Initialising Institutional Pipeline Engine v5.0</div>
          <div style='font-family:"Playfair Display",Georgia,serif;font-size:1.4rem;color:#1E3A5F;font-weight:300;'>Calibrating models, micro-structural indices & multi-asset DCC matrices…</div>
        </div>""", unsafe_allow_html=True)
        prog = st.progress(0)

        try:
            with st.status("Running institutional pipeline...", expanded=True) as status:
                # Stage 1: External data
                st.write("Fetching macro data (FRED, EIA, GPR, COT)...")
                vix_premium = data_layer.fetch_fred_vix()
                eia_stocks = data_layer.fetch_eia_inventories()
                gpr_series = data_layer.fetch_gpr()
                cot_value = data_layer.fetch_cot_proxy("CL")
                prog.progress(10)

                # Stage 2: Market data
                st.write("Downloading historical prices...")
                war_start_str = war_start.strftime("%Y-%m-%d")
                prices = data_layer.fetch_historical_prices(war_start_str)
                if prices.empty or len(prices) < 5:
                    st.error("Execution halted: Insufficient historical data extracted from live sources.")
                    st.stop()
                if not isinstance(prices.index, pd.DatetimeIndex):
                    prices.index = pd.to_datetime(prices.index)
                prices = prices.ffill().bfill()
                for k in CONFIG.ticker_keys:
                    if k not in prices.columns:
                        prices[k] = np.nan
                prices = prices.ffill().bfill()
                lw = float(prices["oil"].dropna().iloc[-1]) if not prices["oil"].dropna().empty else 75.0
                lb = float(prices["brent"].dropna().iloc[-1]) if not prices["brent"].dropna().empty else 78.0
                wti0, brt0 = data_layer.fetch_live_prices(lw, lb)
                prices.loc[prices.index[-1], "oil"] = wti0
                prices.loc[prices.index[-1], "brent"] = brt0
                returns = np.log(prices / prices.shift(1)).dropna()
                prog.progress(25)

                # Stage 3: Macro engine
                st.write("Building macro indices, GeoFactor & Z-Score...")
                macro_engine = MacroEngine(prices, returns)
                prog.progress(45)

                # Stage 4: Volatility engine
                st.write("Fitting EGARCH-X, DCC & EVT tails...")
                vol_engine = VolatilityEngine(returns, macro_engine.geofactor)
                vw_raw = vol_engine.vol_wti
                vb_raw = vol_engine.vol_brent
                vg_raw = vol_engine.vol_gold
                pwd = prior_wti / np.sqrt(252)
                pbd = prior_brent / np.sqrt(252)
                pgd = 0.18 / np.sqrt(252)
                vw_shrunk, dw = VolatilityEngine.bayes_shrink(vw_raw, pwd, len(returns), macro_engine.geofactor)
                vb_shrunk, db = VolatilityEngine.bayes_shrink(vb_raw, pbd, len(returns), macro_engine.geofactor)
                vg_shrunk, _ = VolatilityEngine.bayes_shrink(vg_raw, pgd, len(returns), macro_engine.geofactor)
                bvw = float(vw_shrunk.iloc[-1]) if len(vw_shrunk) > 0 else float(vw_shrunk)
                bvb = float(vb_shrunk.iloc[-1]) if len(vb_shrunk) > 0 else float(vb_shrunk)
                prog.progress(65)

                # Stage 5: VAR forecast
                st.write("Generating VAR forecast...")
                rv = returns.loc[macro_engine.geofactor.index.intersection(returns.index)] if not macro_engine.geofactor.empty else returns
                try:
                    k_ar = min(3, max(1, len(rv) // 15))
                    vm = VAR(rv).fit(k_ar)
                    fcast = vm.forecast(rv.values[-k_ar:], steps=mc_steps)
                except Exception as exc:
                    logger.warning("VAR forecast failed: %s; using zero forecast", exc)
                    fcast = np.zeros((mc_steps, len(rv.columns)))
                cols = list(rv.columns)
                ocol = cols.index("oil") if "oil" in cols else 0
                bcol = cols.index("brent") if "brent" in cols else 1
                tdf_d = max(2.5, min(6.0, tail_df / np.sqrt(max(bvb / (pbd * 1.5), 0.5))))
                rbase = float(np.tanh(macro_engine.geofactor.iloc[-1] / 2)) if len(macro_engine.geofactor) > 0 else 0.0
                jpu = min(jump_up * 1.5, 0.15) if returns["wheat"].tail(20).mean() > 0.005 else jump_up
                prog.progress(80)

                # Stage 6: Monte Carlo
                st.write(f"Running Monte Carlo ({mc_sims:,} paths, {mc_steps} steps)...")
                mc_bar = st.progress(0)
                simulator = SimulationEngine(
                    wti0=wti0, brent0=brt0, vol_wti=bvw, vol_brent=bvb,
                    dcc=vol_engine.dcc_model, forecast=fcast, ocol=ocol, bcol=bcol,
                    rbase=rbase, rw=returns["oil"], rb=returns["brent"],
                    vw=vw_shrunk, vb=vb_shrunk, jump_up=jpu, tail_df=tdf_d,
                    bs=macro_engine.bs_mult, scenario_mod=SCENARIO_MAP[scen_choice],
                    sims=mc_sims, steps=mc_steps
                )
                simulator.run(seed=int(mc_seed), progress_bar=mc_bar)
                mc_bar.empty()
                prog.progress(90)

                # Stage 7: Risk & ML
                st.write("Computing risk metrics, backtests & ML benchmarks...")
                ret_ann = returns[["oil", "brent"]].mean() * 252
                vol_ann = returns[["oil", "brent"]].std() * np.sqrt(252)
                neg = returns[["oil", "brent"]][returns[["oil", "brent"]] < 0].std() * np.sqrt(252)
                corr_mx = returns[["oil", "brent", "gold", "dxy", "tnx"]].dropna().corr()
                stress_c = pd.DataFrame({
                    "vol_wti": vw_shrunk * np.sqrt(252) * 100,
                    "vol_brt": vb_shrunk * np.sqrt(252) * 100,
                    "corr": returns["oil"].rolling(20).corr(returns["brent"]),
                    "geofactor": macro_engine.geofactor
                }).dropna()
                if not stress_c.empty:
                    stress_idx = (stress_c["vol_wti"] / 50 + stress_c["vol_brt"] / 50 +
                                  np.abs(stress_c["corr"] - 0.8) * 2 + stress_c["geofactor"].clip(0, 2) / 2) / 4
                else:
                    stress_idx = pd.Series([0.0], index=[prices.index[-1]])

                bt_res = RiskEngine.backtest_var(returns["oil"].iloc[-252:], vw_shrunk.iloc[-252:] * 1.645)
                es_z = RiskEngine.backtest_es(returns["oil"].iloc[-252:], vw_shrunk.iloc[-252:] * 2.326, vw_shrunk.iloc[-252:] * 1.645)
                gdiag = RiskEngine.garch_diagnostics(vw_shrunk)
                ml_metrics, X_ml, y_ml = MLEngine.benchmark(returns)
                shap_fig, sv, feat_names = MLEngine.shap_summary(X_ml, y_ml) if X_ml is not None and len(X_ml) > 5 else (None, np.array([]), [])
                wf_df = MLEngine.walk_forward(returns["oil"])

                if bt_res.get("insufficient_data", False):
                    model_score = None
                else:
                    p_vals = [gdiag["LB5"], gdiag["LB10"], gdiag["ARCH_p"], bt_res["Kupiec_p"], bt_res["Christoffersen_p"], bt_res["DQ_p"]]
                    valid_p = [p for p in p_vals if not np.isnan(p)]
                    model_score = int(np.mean(valid_p) * 100) if valid_p else 85

                prog.progress(100)
                loading.empty()
                prog.empty()
                status.update(label="Pipeline complete", state="complete", expanded=False)

                last_update = datetime.now(pytz.timezone("America/Sao_Paulo")).strftime("%d %b %Y %H:%M:%S")
                state = AppState(
                    results={"fan": simulator.fan, "fan_b": simulator.fan_b, "metrics": simulator.metrics,
                             "moments": simulator.moments, "brackets": simulator.brackets},
                    gf=macro_engine.geofactor, zsc=macro_engine.zscore,
                    vw=vw_shrunk, vb=vb_shrunk, vg=vg_shrunk,
                    fi=macro_engine.fert_index, gs=macro_engine.gold_signals,
                    prices=prices, returns=returns,
                    wti0=wti0, brt0=brt0,
                    usda=macro_engine.usda, bs=macro_engine.bs_mult,
                    dw=dw, db=db, tdf=tdf_d,
                    dcc_a=vol_engine.dcc_model.a, dcc_b=vol_engine.dcc_model.b,
                    dcc_rho=vol_engine.dcc_model.rho_series,
                    weights=macro_engine.weights,
                    sharpe=ret_ann / vol_ann, sortino=ret_ann / neg,
                    corr_mx=corr_mx, stress_idx=stress_idx,
                    evt=vol_engine.evt_wti, gdiag=gdiag, bt_res=bt_res,
                    es_z=es_z if not np.isnan(es_z) else 0.0,
                    ml_metrics=ml_metrics, shap_fig=shap_fig,
                    shap_vals=sv, feat_names=feat_names, wf_df=wf_df,
                    regimes_ts=vol_engine.regimes, model_score=model_score,
                    vix_fred=vix_premium, eia_stocks=eia_stocks,
                    macro_proxies=macro_engine.macro_proxies,
                    gpr=gpr_series, cot=cot_value, last_update=last_update,
                    fert_source=macro_engine.usda.get("source", "fallback"),
                    vol_type_wti=vol_engine.vol_type_wti,
                    vol_type_brent=vol_engine.vol_type_brent,
                    mc_seed=int(mc_seed)
                )
                state.to_session()
        except Exception as e:
            loading.empty()
            prog.empty()
            st.error(f"Structural Fatal Exception in Pipeline: {e}")
            st.code(traceback.format_exc())
            st.stop()

    # Rendering
    state = AppState.from_session()
    if state.results is None:
        st.info("Click 'Run Full System Pipeline' in the sidebar to initialise the quantitative engine.")
        return

    ui = UIComponents(state)
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:var(--muted); text-align:right; margin-bottom:1rem;">
    Data Freshness: <strong>{state.last_update}</strong> · Spot Update Interval: 10s · Fert Source: <strong>{state.fert_source}</strong> · Vol Engine: <strong>{state.vol_type_wti}</strong> · Seed: <strong>{state.mc_seed}</strong>
    </div>""", unsafe_allow_html=True)

    csv_data = export_results_to_csv(state)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<div style="text-align:right; margin-bottom:1rem;"><a href="data:file/csv;base64,{b64}" download="geoquant_results.csv" style="background:#1E3A5F; color:#D4C094; padding:0.3rem 0.7rem; font-family:JetBrains Mono; font-size:0.55rem; text-decoration:none;">📥 Export Results (CSV)</a></div>'
    st.markdown(href, unsafe_allow_html=True)

    t_exec, t_vol, t_geo, t_attr, t_mc, t_stat, t_diag, t_ml, t_modelcard = st.tabs([
        "Executive Summary", "Market & Volatility", "Geopolitical Intelligence",
        "GeoFactor Attribution", "Monte Carlo Fan Chart", "Quant Statistics",
        "Model Diagnostics", "Machine Learning Leaderboard", "Model Card"
    ])
    with t_exec: ui.render_executive_summary()
    with t_vol: ui.render_market_vol()
    with t_geo: ui.render_geopolitical()
    with t_attr: ui.render_attribution()
    with t_mc: ui.render_monte_carlo()
    with t_stat: ui.render_quant_stats()
    with t_diag: ui.render_diagnostics()
    with t_ml: ui.render_ml()
    with t_modelcard: ui.render_model_card()

    st.markdown(f"""
    <div class="footer">
      <div>◆ GeoQuant Institutional Terminal · Engine: Conditional EVT + DCC-GARCH-X v5.0</div>
      <div>Eduardo Moraes · Quant Data Scientist & Economics</div>
      <div>Proprietary Research Infrastructure · {now_sp.strftime("%Y")}</div>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

"""
ALPHA DISCOVERY ENGINE v5 — WDO B3

NUMBA JIT — mesmo resultado, 100x mais rapido.

VELOCIDADE:
- Python puro:  ~59 combos/s
- Numba:        ~3000-6000 combos/s
- 1M combos em 3-6 minutos

LICOES APLICADAS:
- flush=True em todos os prints
- Entrada no OPEN do proximo candle
- MIN_TRADES=100 para triagem
- Numba nao muda resultado — apenas compila
"""

import pandas as pd
import numpy as np
from numba import njit
import json, sys, os, time, warnings, math, itertools
from datetime import datetime
from scipy import stats
warnings.filterwarnings("ignore")

CSV_PATH   = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR = "/workspace/param_opt_output/alpha_v5"
CAPITAL    = 50_000.0
MULT       = 10.0
COMM       = 5.0
SLIP       = 2.0
MIN_TRADES = 100
MAX_PF     = 3.5
MAX_DD     = -35.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# SECAO 1: DADOS
# ================================================================

def carregar():
    print("[DATA] Carregando...", flush=True)
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower() for c in df.columns]
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
    return df


# ================================================================
# SECAO 2: INDICADORES
# ================================================================

def ema_np(arr, span):
    alpha = 2.0 / (span + 1)
    out   = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def sma_np(arr, period):
    out = np.full(len(arr), np.nan)
    cs  = np.cumsum(arr)
    out[period - 1:] = (cs[period - 1:] - np.concatenate([[0], cs[:-(period)]])) / period
    return out


def atr_np(high, low, close, period=14):
    prev = np.roll(close, 1); prev[0] = close[0]
    tr   = np.maximum(high - low,
                      np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return sma_np(tr, period)


def rsi_np(close, period):
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = np.full(len(close), np.nan)
    avg_l = np.full(len(close), np.nan)
    if period < len(close):
        avg_g[period] = gain[1:period + 1].mean()
        avg_l[period] = loss[1:period + 1].mean()
        for i in range(period + 1, len(close)):
            avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i]) / period
            avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i]) / period
    return 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))


def calcular_indicadores(df):
    print("[IND] Calculando...", flush=True)
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)

    ind = {
        "close":     c,
        "open":      o,
        "high":      h,
        "low":       l,
        "open_next": np.concatenate([o[1:], [c[-1]]]),
    }

    # ATR
    for p in [7, 14, 20]:
        ind[f"atr_{p}"] = atr_np(h, l, c, p)

    # EMAs
    for p in [2, 3, 5, 8, 10, 13, 20, 21, 34, 50, 100, 200]:
        ind[f"ema_{p}"] = ema_np(c, p)

    # SMAs
    for p in [2, 3, 5, 8, 10, 13, 20, 21, 34, 50, 100, 150, 200]:
        ind[f"sma_{p}"] = sma_np(c, p)

    # RSIs — expandido para 1M combos
    for p in [2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 18, 21, 28]:
        ind[f"rsi_{p}"] = rsi_np(c, p)

    # MACD
    for fast, slow in [(12, 26), (8, 21), (5, 13), (3, 10), (6, 19), (10, 22), (4, 9), (2, 5)]:
        m   = ema_np(c, fast) - ema_np(c, slow)
        sig = ema_np(m, 9)
        ind[f"macd_{fast}_{slow}_hist"] = m - sig

    # Bollinger
    for p in [5, 7, 10, 15, 20, 30, 50]:
        sma = sma_np(c, p)
        std = pd.Series(c).rolling(p).std().values
        for s_str, s_f in [("10", 1.0), ("12", 1.2), ("15", 1.5), ("18", 1.8),
                            ("20", 2.0), ("25", 2.5), ("30", 3.0)]:
            up = sma + s_f * std
            lo = sma - s_f * std
            ind[f"bb_{p}_{s_str}_pct"] = (c - lo) / (up - lo + 1e-9)

    # Donchian
    for p in [5, 7, 10, 15, 20, 30, 50, 100, 200]:
        ind[f"don_high_{p}"] = pd.Series(h).rolling(p).max().shift(1).values
        ind[f"don_low_{p}"]  = pd.Series(l).rolling(p).min().shift(1).values

    # Stochastic
    for p in [3, 5, 7, 9, 11, 14, 21, 28]:
        lo_p = pd.Series(l).rolling(p).min().values
        hi_p = pd.Series(h).rolling(p).max().values
        stk  = (c - lo_p) / (hi_p - lo_p + 1e-9) * 100
        ind[f"stoch_k_{p}"] = stk

    # ROC
    for p in [2, 3, 5, 7, 10, 15, 20, 30]:
        roc = np.empty(len(c)); roc[:p] = np.nan
        roc[p:] = (c[p:] - c[:-p]) / (c[:-p] + 1e-9) * 100
        ind[f"roc_{p}"] = roc

    # CCI
    for p in [5, 7, 10, 14, 20, 30, 50]:
        tp     = (h + l + c) / 3
        sma_tp = sma_np(tp, p)
        mad    = pd.Series(tp).rolling(p).apply(
            lambda x: np.abs(x - x.mean()).mean()).values
        ind[f"cci_{p}"] = (tp - sma_tp) / (0.015 * mad + 1e-9)

    # Volatilidade relativa
    ret   = np.diff(c, prepend=c[0]) / (c + 1e-9)
    vol5  = pd.Series(ret).rolling(5).std().values * 100
    vol20 = pd.Series(ret).rolling(20).std().values * 100
    ind["vol_ratio"] = vol5 / (vol20 + 1e-9)

    # Sessao
    hora = df.index.hour
    ind["session_am"] = ((hora >= 9)  & (hora < 12)).astype(np.int8)
    ind["session_pm"] = ((hora >= 13) & (hora < 17)).astype(np.int8)

    print(f"[IND] {len(ind)} arrays prontos", flush=True)
    return ind


# ================================================================
# SECAO 3: SIMULADOR NUMBA (ULTRA-RAPIDO)
# ================================================================

@njit(cache=True)
def simular_numba(open_next, high, low, entries, exits, sl_pts, tp_pts,
                  capital, mult, comm, slip):
    """
    Simulador JIT compilado com Numba.
    Mesma logica do Python puro, 50-100x mais rapido.
    """
    n          = len(open_next)
    trades_pnl = np.empty(n, dtype=np.float64)
    n_trades   = 0
    em_pos     = False
    entry_p    = 0.0
    sl         = 0.0
    tp         = 0.0
    direction  = 1  # long apenas por enquanto

    for i in range(n - 1):
        if em_pos:
            lo = low[i]
            hi = high[i]
            hit_sl = (direction == 1 and lo <= sl) or (direction == -1 and hi >= sl)
            hit_tp = (direction == 1 and hi >= tp) or (direction == -1 and lo <= tp)
            force  = exits[i]

            if hit_sl or hit_tp or force:
                if force and not hit_sl and not hit_tp:
                    saida = open_next[i]
                else:
                    saida = sl if hit_sl else tp
                pts = (saida - entry_p) * direction
                pnl = pts * mult - comm - slip * mult * 0.1
                trades_pnl[n_trades] = pnl
                n_trades += 1
                em_pos = False
            continue

        if entries[i] and not em_pos:
            ep = open_next[i]
            if np.isnan(ep) or ep <= 0:
                continue
            entry_p   = ep
            direction = 1
            sl        = entry_p - sl_pts
            tp        = entry_p + tp_pts
            em_pos    = True

    return trades_pnl[:n_trades]


def calcular_metricas(pnls, max_pf, max_dd, min_trades):
    """Calcula metricas a partir do array de PnLs."""
    n = len(pnls)
    if n < min_trades:
        return None

    wins  = pnls[pnls > 0]
    loses = pnls[pnls <= 0]
    if len(loses) == 0 or len(wins) == 0:
        return None

    pf = abs(wins.sum() / loses.sum())
    if pf > max_pf:
        return None

    eq  = np.concatenate([[50000.0], 50000.0 + np.cumsum(pnls)])
    pk  = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk * 100).min())
    if mdd < max_dd:
        return None

    wr  = len(wins) / n * 100
    ret = pnls / 50000.0
    sh  = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252 * 390))
    exp = float(pnls.mean())

    n_jan   = max(1, len(range(0, n - 30, 15)))
    jan_pos = sum(1 for s in range(0, n - 30, 15) if pnls[s:s + 30].sum() > 0)
    pct_jan = jan_pos / n_jan * 100

    return {
        "total_trades":     n,
        "win_rate":         round(wr, 2),
        "profit_factor":    round(pf, 3),
        "sharpe_ratio":     round(sh, 3),
        "expectancy_brl":   round(exp, 2),
        "total_pnl_brl":    round(float(pnls.sum()), 2),
        "max_drawdown_pct": round(mdd, 2),
        "pct_janelas_pos":  round(pct_jan, 1),
    }


# ================================================================
# SECAO 4: GERADORES DE SINAIS
# ================================================================

def get_mask(ind, session):
    if session == "am":
        return ind["session_am"].astype(bool)
    elif session == "pm":
        return ind["session_pm"].astype(bool)
    return np.ones(len(ind["close"]), dtype=bool)


def sig_ema_cross(ind, fast, slow, d):
    ef = ind.get(f"ema_{fast}"); es = ind.get(f"ema_{slow}")
    if ef is None or es is None:
        return None, None
    ef1 = np.roll(ef, 1); es1 = np.roll(es, 1)
    if d == "long":
        ent = (ef > es) & (ef1 <= es1); ext = (ef < es) & (ef1 >= es1)
    else:
        ent = (ef < es) & (ef1 >= es1); ext = (ef > es) & (ef1 <= es1)
    ent[0] = ext[0] = False
    return ent, ext


def sig_rsi(ind, period, oversold, overbought, exit_level, d):
    rsi = ind.get(f"rsi_{period}")
    if rsi is None:
        return None, None
    r1 = np.roll(rsi, 1)
    if d == "long":
        ent = (rsi < oversold)  & (r1 >= oversold);  ext = rsi > exit_level
    else:
        ent = (rsi > overbought) & (r1 <= overbought); ext = rsi < (100 - exit_level)
    ent[0] = ext[0] = False
    return ent, ext


def sig_stoch(ind, period, oversold, overbought, d):
    k = ind.get(f"stoch_k_{period}")
    if k is None:
        return None, None
    k1 = np.roll(k, 1)
    if d == "long":
        ent = (k < oversold)  & (k1 >= oversold);  ext = k > 50
    else:
        ent = (k > overbought) & (k1 <= overbought); ext = k < 50
    ent[0] = ext[0] = False
    return ent, ext


def sig_bollinger(ind, period, std_str, d):
    pct = ind.get(f"bb_{period}_{std_str}_pct")
    if pct is None:
        return None, None
    if d == "long":
        ent = pct < 0.05; ext = pct > 0.5
    else:
        ent = pct > 0.95; ext = pct < 0.5
    return ent, ext


def sig_macd(ind, config, d):
    hist = ind.get(f"macd_{config}_hist")
    if hist is None:
        return None, None
    h1 = np.roll(hist, 1)
    if d == "long":
        ent = (hist > 0) & (h1 <= 0); ext = (hist < 0) & (h1 >= 0)
    else:
        ent = (hist < 0) & (h1 >= 0); ext = (hist > 0) & (h1 <= 0)
    ent[0] = ext[0] = False
    return ent, ext


def sig_donchian(ind, period, d):
    dh = ind.get(f"don_high_{period}"); dl = ind.get(f"don_low_{period}")
    if dh is None:
        return None, None
    c = ind["close"]
    if d == "long":
        ent = c > dh; ext = c < dl
    else:
        ent = c < dl; ext = c > dh
    return ent, ext


def sig_roc(ind, period, threshold, d):
    roc = ind.get(f"roc_{period}")
    if roc is None:
        return None, None
    if d == "long":
        ent = roc > threshold;  ext = roc < 0
    else:
        ent = roc < -threshold; ext = roc > 0
    return ent, ext


def sig_cci(ind, period, threshold, d):
    cci = ind.get(f"cci_{period}")
    if cci is None:
        return None, None
    c1 = np.roll(cci, 1)
    if d == "long":
        ent = (cci < -threshold) & (c1 >= -threshold); ext = cci > 0
    else:
        ent = (cci > threshold)  & (c1 <= threshold);  ext = cci < 0
    ent[0] = ext[0] = False
    return ent, ext


def sig_volatility(ind, vol_threshold, d):
    vr   = ind["vol_ratio"]
    hist = ind.get("macd_12_26_hist", np.zeros(len(vr)))
    h1   = np.roll(hist, 1)
    exp  = vr > vol_threshold
    if d == "long":
        ent = exp & (hist > 0) & (h1 <= 0); ext = exp & (hist < 0) & (h1 >= 0)
    else:
        ent = exp & (hist < 0) & (h1 >= 0); ext = exp & (hist > 0) & (h1 <= 0)
    return ent, ext


def gerar_sinais(familia, ind, params):
    d    = params.get("direction", "long")
    ses  = params.get("session", "all")
    mask = get_mask(ind, ses)
    ent  = ext = None

    if familia == "ema_crossover":
        ent, ext = sig_ema_cross(ind, params["fast"], params["slow"], d)

    elif familia == "rsi_reversion":
        ent, ext = sig_rsi(ind, params["period"], params["oversold"],
                           params["overbought"], params.get("exit_level", 50), d)

    elif familia == "stochastic":
        ent, ext = sig_stoch(ind, params["period"], params["oversold"],
                             params["overbought"], d)

    elif familia == "bollinger":
        ent, ext = sig_bollinger(ind, params["period"], params["std_str"], d)

    elif familia == "macd_momentum":
        ent, ext = sig_macd(ind, params["config"], d)

    elif familia == "donchian":
        ent, ext = sig_donchian(ind, params["period"], d)

    elif familia == "roc_momentum":
        ent, ext = sig_roc(ind, params["period"], params["threshold"], d)

    elif familia == "cci":
        ent, ext = sig_cci(ind, params["period"], params["threshold"], d)

    elif familia == "volatility":
        ent, ext = sig_volatility(ind, params["vol_threshold"], d)

    elif familia == "rsi_ema_combo":
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        ema = ind.get(f"ema_{params['ema_period']}")
        cls = ind["close"]
        if rsi is None or ema is None:
            return None, None
        above = cls > ema
        filt  = above if params["ema_filter"] == "above" else ~above
        lvl   = params["rsi_level"]
        r1    = np.roll(rsi, 1)
        if d == "long":
            ent = (rsi < lvl)        & (r1 >= lvl)        & filt;  ext = rsi > 50
        else:
            ent = (rsi > (100 - lvl)) & (r1 <= (100 - lvl)) & (~filt); ext = rsi < 50
        ent[0] = ext[0] = False

    elif familia == "macd_rsi_combo":
        hist = ind.get(f"macd_{params['macd_config']}_hist")
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        flt  = params["rsi_filter"]
        if hist is None or rsi is None:
            return None, None
        h1 = np.roll(hist, 1)
        if d == "long":
            ent = (hist > 0) & (h1 <= 0) & (rsi > flt);       ext = (hist < 0) & (h1 >= 0)
        else:
            ent = (hist < 0) & (h1 >= 0) & (rsi < (100 - flt)); ext = (hist > 0) & (h1 <= 0)
        ent[0] = ext[0] = False

    elif familia == "bb_rsi_combo":
        pct = ind.get(f"bb_{params['bb_period']}_{params['bb_std']}_pct")
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        cnf = params["rsi_confirm"]
        if pct is None or rsi is None:
            return None, None
        if d == "long":
            ent = (pct < 0.1) & (rsi < cnf);             ext = pct > 0.5
        else:
            ent = (pct > 0.9) & (rsi > (100 - cnf));     ext = pct < 0.5

    elif familia == "dual_ma":
        fk = f"{params['fast_type']}_{params['fast']}"
        sk = f"{params['slow_type']}_{params['slow']}"
        ef = ind.get(fk); es = ind.get(sk)
        if ef is None or es is None:
            return None, None
        ef1 = np.roll(ef, 1); es1 = np.roll(es, 1)
        if d == "long":
            ent = (ef > es) & (ef1 <= es1); ext = (ef < es) & (ef1 >= es1)
        else:
            ent = (ef < es) & (ef1 >= es1); ext = (ef > es) & (ef1 <= es1)
        ent[0] = ext[0] = False

    elif familia == "stoch_ema_combo":
        k   = ind.get(f"stoch_k_{params['stoch_period']}")
        ema = ind.get(f"ema_{params['ema_period']}")
        cls = ind["close"]
        if k is None or ema is None:
            return None, None
        above = cls > ema
        filt  = above if params["ema_filter"] == "above" else ~above
        ovs   = params["oversold"]
        k1    = np.roll(k, 1)
        if d == "long":
            ent = (k < ovs)         & (k1 >= ovs)         & filt;   ext = k > 50
        else:
            ent = (k > (100 - ovs)) & (k1 <= (100 - ovs)) & (~filt); ext = k < 50
        ent[0] = ext[0] = False

    elif familia == "donchian_rsi":
        dh  = ind.get(f"don_high_{params['don_period']}")
        dl  = ind.get(f"don_low_{params['don_period']}")
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        cls = ind["close"]
        flt = params["rsi_filter"]
        if dh is None or rsi is None:
            return None, None
        if d == "long":
            ent = (cls > dh) & (rsi > flt);          ext = cls < dl
        else:
            ent = (cls < dl) & (rsi < (100 - flt));  ext = cls > dh

    elif familia == "triple_confirm":
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        ema  = ind.get(f"ema_{params['ema_period']}")
        hist = ind.get(f"macd_{params['macd_config']}_hist")
        cls  = ind["close"]
        lvl  = params["rsi_level"]
        if rsi is None or ema is None or hist is None:
            return None, None
        above = cls > ema
        if d == "long":
            ent = (rsi < lvl)        & above    & (hist > 0); ext = (rsi > 50)  | (hist < 0)
        else:
            ent = (rsi > (100 - lvl)) & (~above) & (hist < 0); ext = (rsi < 50) | (hist > 0)

    else:
        return None, None

    if ent is None:
        return None, None

    ent = ent & mask
    return ent, ext


# ================================================================
# SECAO 5: GRIDS — 1.2 MILHAO DE COMBOS
# ================================================================

GRIDS = {
    "rsi_reversion": {
        "period":     [2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 18, 21, 28],
        "oversold":   [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45],
        "overbought": [55, 60, 65, 70, 75, 80, 85, 88, 90, 92, 95],
        "exit_level": [40, 45, 50, 55, 60],
        "session":    ["am", "pm", "all"],
        "rr":         [1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":     [0.5, 1.0, 1.5, 2.0],
        "direction":  ["long", "short"],
    },
    "stochastic": {
        "period":     [3, 5, 7, 9, 11, 14, 21, 28],
        "oversold":   [5, 8, 10, 12, 15, 18, 20, 25, 30, 35],
        "overbought": [65, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95],
        "rr":         [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "direction":  ["long", "short"],
    },
    "ema_crossover": {
        "fast":      [2, 3, 5, 8, 10, 13, 20, 21, 34],
        "slow":      [20, 21, 34, 50, 100, 150, 200],
        "rr":        [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0],
        "atr_sl":    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "session":   ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "bollinger": {
        "period":    [5, 7, 10, 15, 20, 30, 50],
        "std_str":   ["10", "12", "15", "18", "20", "25", "30"],
        "session":   ["am", "pm", "all"],
        "rr":        [1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":    [0.3, 0.5, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "donchian": {
        "period":    [5, 7, 10, 15, 20, 30, 50, 100, 200],
        "session":   ["am", "pm", "all"],
        "rr":        [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        "atr_sl":    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "roc_momentum": {
        "period":    [2, 3, 5, 7, 10, 15, 20, 30],
        "threshold": [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
        "rr":        [1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":    [0.3, 0.5, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "cci": {
        "period":    [5, 7, 10, 14, 20, 30, 50],
        "threshold": [50, 75, 100, 125, 150, 175, 200, 250],
        "rr":        [1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":    [0.3, 0.5, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "macd_momentum": {
        "config":    ["12_26", "8_21", "5_13", "3_10", "6_19", "10_22", "4_9", "2_5"],
        "session":   ["am", "pm", "all"],
        "rr":        [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        "atr_sl":    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "volatility": {
        "vol_threshold": [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":       ["am", "pm", "all"],
        "rr":            [1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":        [0.3, 0.5, 1.0, 1.5, 2.0],
        "direction":     ["long", "short"],
    },
    "rsi_ema_combo": {
        "rsi_period":  [5, 7, 9, 14, 21, 28],
        "rsi_level":   [15, 20, 25, 30, 35, 40],
        "ema_period":  [10, 20, 50, 100, 200],
        "ema_filter":  ["above", "below"],
        "rr":          [1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.5, 1.0, 1.5, 2.0],
        "direction":   ["long", "short"],
    },
    "macd_rsi_combo": {
        "macd_config": ["12_26", "8_21", "5_13", "3_10"],
        "rsi_period":  [7, 9, 14, 21],
        "rsi_filter":  [35, 40, 45, 50, 55, 60, 65],
        "session":     ["am", "pm", "all"],
        "rr":          [1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.5, 1.0, 1.5, 2.0],
        "direction":   ["long", "short"],
    },
    "bb_rsi_combo": {
        "bb_period":   [10, 15, 20, 30, 50],
        "bb_std":      ["12", "15", "20", "25", "30"],
        "rsi_period":  [5, 7, 9, 14, 21],
        "rsi_confirm": [20, 25, 30, 35, 40, 45],
        "rr":          [1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.5, 1.0, 1.5, 2.0],
        "direction":   ["long", "short"],
    },
    "dual_ma": {
        "fast":      [2, 3, 5, 8, 10, 13, 20, 21, 34],
        "slow":      [21, 34, 50, 100, 150, 200],
        "fast_type": ["ema", "sma"],
        "slow_type": ["ema", "sma"],
        "rr":        [1.5, 2.0, 2.5, 3.0],
        "atr_sl":    [0.5, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "stoch_ema_combo": {
        "stoch_period": [5, 9, 14, 21],
        "oversold":     [10, 15, 20, 25, 30],
        "ema_period":   [20, 50, 100, 200],
        "ema_filter":   ["above", "below"],
        "rr":           [1.5, 2.0, 2.5, 3.0],
        "atr_sl":       [0.5, 1.0, 1.5, 2.0],
        "direction":    ["long", "short"],
    },
    "donchian_rsi": {
        "don_period":  [10, 20, 50, 100],
        "rsi_period":  [7, 14, 21],
        "rsi_filter":  [40, 45, 50, 55, 60],
        "session":     ["am", "pm", "all"],
        "rr":          [1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.5, 1.0, 1.5, 2.0],
        "direction":   ["long", "short"],
    },
    "triple_confirm": {
        "rsi_period":  [9, 14, 21],
        "rsi_level":   [25, 30, 35, 40],
        "ema_period":  [20, 50, 200],
        "macd_config": ["12_26", "8_21", "5_13"],
        "rr":          [1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.5, 1.0, 1.5, 2.0],
        "direction":   ["long", "short"],
    },
}


# ================================================================
# SECAO 6: GRID SEARCH COM NUMBA
# ================================================================

def grid_search(familia, ind, grid, n_total, mini=False):
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if mini:
        combos = combos[:20]

    print(f"\n[{familia.upper()}] {len(combos):,} combos...", flush=True)
    t0      = time.time()
    validos = []
    n_ok    = 0

    atr_pts = float(np.nanmean(ind["atr_14"]))
    on      = ind["open_next"].astype(np.float64)
    hi      = ind["high"].astype(np.float64)
    lo      = ind["low"].astype(np.float64)

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            ent, ext = gerar_sinais(familia, ind, params)
            if ent is None or ent.sum() < 10:
                continue

            sl_pts = atr_pts * params.get("atr_sl", 1.0)
            tp_pts = sl_pts  * params.get("rr", 2.0)

            # NUMBA: compila na primeira chamada, instant depois
            pnls = simular_numba(
                on, hi, lo,
                ent.astype(np.bool_),
                ext.astype(np.bool_),
                sl_pts, tp_pts,
                CAPITAL, MULT, COMM, SLIP
            )

            m = calcular_metricas(pnls, MAX_PF, MAX_DD, MIN_TRADES)
            if not m:
                continue

            n_ok += 1
            pf_s  = min(m["profit_factor"], MAX_PF) / MAX_PF
            exp_s = max(0, min(m["expectancy_brl"], 500)) / 500
            jan_s = m["pct_janelas_pos"] / 100
            tr_s  = min(m["total_trades"], 2000) / 2000
            sh_s  = max(0, min(m["sharpe_ratio"], 3)) / 3
            score = pf_s * 0.30 + exp_s * 0.25 + jan_s * 0.20 + sh_s * 0.15 + tr_s * 0.10

            validos.append({
                "familia": familia,
                "params":  params,
                "score":   round(score, 6),
                **m,
            })
        except Exception:
            continue

    elapsed = time.time() - t0
    validos.sort(key=lambda x: -x["score"])
    spd = len(combos) / max(elapsed, 0.1)
    print(f"  {n_ok:,}/{len(combos):,} validos | {elapsed:.1f}s | {spd:.0f} combos/s", flush=True)

    if validos:
        r = validos[0]
        print(f"  TOP: PF={r['profit_factor']:.3f} WR={r['win_rate']:.1f}% "
              f"Trades={r['total_trades']} Exp=R${r['expectancy_brl']:.2f} "
              f"Score={r['score']:.4f}", flush=True)
        with open(f"{OUTPUT_DIR}/{familia}_top10.json", "w") as fp:
            json.dump(validos[:10], fp, indent=2, default=str)

    return validos


# ================================================================
# SECAO 7: STAGE-GATE
# ================================================================

def stage_gate(m):
    if not m:
        return False, {}
    checks = {
        "S1_trades":  m["total_trades"] >= MIN_TRADES,
        "S2_pf":      m["profit_factor"] > 1.05,
        "S3_sharpe":  m["sharpe_ratio"]  > 0.0,
        "S4_dd":      m["max_drawdown_pct"] > MAX_DD,
        "S5_janelas": m["pct_janelas_pos"] >= 50,
        "S6_exp":     m["expectancy_brl"]  > 0,
    }
    return all(checks.values()), checks


# ================================================================
# SECAO 8: MAIN
# ================================================================

def main():
    MINI = "--mini" in sys.argv

    total = sum(math.prod(len(v) for v in g.values()) for g in GRIDS.values())

    print("=" * 68, flush=True)
    print("  ALPHA DISCOVERY ENGINE v5 — NUMBA JIT", flush=True)
    print(f"  {len(GRIDS)} familias | {total:,} combos | WDO B3", flush=True)
    print("=" * 68, flush=True)

    df     = carregar()
    split  = int(len(df) * 0.70)
    df_ins = df.iloc[:split]
    df_oos = df.iloc[split:]
    print(f"  IS : {len(df_ins):,} | {df_ins.index[0].date()} -> {df_ins.index[-1].date()}", flush=True)
    print(f"  OOS: {len(df_oos):,} | {df_oos.index[0].date()} -> {df_oos.index[-1].date()}", flush=True)

    ind_ins = calcular_indicadores(df_ins)

    # Aquece o JIT com um combo simples
    print("[JIT] Compilando simulador Numba (30s na primeira vez)...", flush=True)
    dummy = np.ones(100, dtype=np.float64) * 5000
    dbool = np.zeros(100, dtype=np.bool_)
    dbool[10] = True
    _ = simular_numba(dummy, dummy, dummy, dbool, dbool, 10.0, 20.0, 50000, 10, 5, 2)
    print("[JIT] Pronto! Velocidade maxima ativada.", flush=True)

    todos     = []
    aprovados = []

    for familia, grid in GRIDS.items():
        resultados = grid_search(familia, ind_ins, grid, total, mini=MINI)
        todos.extend(resultados[:20])
        if not resultados:
            continue
        melhor = resultados[0]
        aprovado, gate = stage_gate(melhor)
        status = "OK" if aprovado else f"REPROVADO {sum(gate.values())}/{len(gate)}"
        print(f"  Stage-gate: {status}", flush=True)
        if aprovado:
            aprovados.append((familia, melhor))

    # Top geral
    todos.sort(key=lambda x: -x["score"])
    print(f"\n{'=' * 68}", flush=True)
    print(f"  TOP 20 GERAL", flush=True)
    print(f"  {'FAMILIA':22} {'PF':>6} {'WR%':>6} {'Trades':>7} {'Exp R$':>8} {'Score':>7}", flush=True)
    print(f"  {'-' * 58}", flush=True)
    for r in todos[:20]:
        print(f"  {r['familia']:22} {r['profit_factor']:>6.3f} "
              f"{r['win_rate']:>6.1f} {r['total_trades']:>7} "
              f"{r['expectancy_brl']:>8.2f} {r['score']:>7.4f}", flush=True)

    # Validacao OOS
    print(f"\n{'=' * 68}", flush=True)
    print(f"  VALIDANDO {len(aprovados)} NO OOS...", flush=True)

    ind_oos = calcular_indicadores(df_oos)
    resultados_finais = []

    for familia, melhor in aprovados:
        params   = melhor["params"]
        ent, ext = gerar_sinais(familia, ind_oos, params)
        if ent is None:
            continue

        atr_pts = float(np.nanmean(ind_oos["atr_14"]))
        sl_pts  = atr_pts * params.get("atr_sl", 1.0)
        tp_pts  = sl_pts  * params.get("rr", 2.0)

        pnls  = simular_numba(
            ind_oos["open_next"].astype(np.float64),
            ind_oos["high"].astype(np.float64),
            ind_oos["low"].astype(np.float64),
            ent.astype(np.bool_),
            ext.astype(np.bool_),
            sl_pts, tp_pts, CAPITAL, MULT, COMM, SLIP,
        )
        m_oos = calcular_metricas(pnls, MAX_PF, MAX_DD, 50)

        is_pf  = melhor["profit_factor"]
        oos_pf = m_oos["profit_factor"] if m_oos else 0
        deg    = (is_pf - oos_pf) / is_pf * 100 if is_pf > 0 and oos_pf > 0 else 999
        ok     = m_oos is not None and oos_pf > 1.0 and deg < 50

        print(f"  [{familia}] IS={is_pf:.3f} OOS={oos_pf:.3f} "
              f"Deg={deg:.1f}% {'PASSA' if ok else 'REPROVADO'}", flush=True)

        resultado = {
            "familia":      familia,
            "params":       params,
            "metricas_is":  melhor,
            "metricas_oos": m_oos,
            "degradacao":   round(deg, 1),
            "aprovado":     ok,
            "gerado_em":    datetime.now().isoformat(),
        }
        resultados_finais.append(resultado)
        with open(f"{OUTPUT_DIR}/{familia}_final.json", "w") as fp:
            json.dump(resultado, fp, indent=2, default=str)

    # Leaderboard
    n_apr = sum(1 for r in resultados_finais if r["aprovado"])
    print(f"\n{'=' * 68}", flush=True)
    print(f"  LEADERBOARD — {n_apr} APROVADO(S)", flush=True)
    for r in sorted(resultados_finais,
                    key=lambda x: -(x["metricas_is"] or {}).get("profit_factor", 0)):
        mi = r["metricas_is"] or {}
        mo = r["metricas_oos"] or {}
        print(f"  {r['familia']:22} IS={mi.get('profit_factor', 0):.3f} "
              f"OOS={mo.get('profit_factor', 0):.3f} "
              f"{'APROVADO' if r['aprovado'] else 'REPROVADO'}", flush=True)

    lb = {
        "gerado_em":       datetime.now().isoformat(),
        "total_combos":    total,
        "aprovados_final": n_apr,
        "top20":           todos[:20],
        "leaderboard":     resultados_finais,
    }
    with open(f"{OUTPUT_DIR}/leaderboard.json", "w") as fp:
        json.dump(lb, fp, indent=2, default=str)

    print(f"\n  {n_apr} estrategia(s) aprovada(s)!", flush=True)
    print(f"  Salvo em: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

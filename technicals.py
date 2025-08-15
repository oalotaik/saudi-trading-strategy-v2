from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

import config

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(period).mean()
    loss = down.rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(period).mean()

def roc(series: pd.Series, lookback: int = 63) -> pd.Series:
    return 100.0 * (series / series.shift(lookback) - 1.0)

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    tr = true_range(high, low, close)
    atr_ = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx_ = dx.rolling(period).mean()
    return adx_

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = ema(out["Close"], config.EMA_SHORT)
    out["EMA50"] = ema(out["Close"], config.EMA_MED)
    out["SMA50"] = sma(out["Close"], 50)
    out["SMA200"] = sma(out["Close"], config.SMA_LONG)
    out["RSI14"] = rsi(out["Close"], config.RSI_PERIOD)
    out["ROC63"] = roc(out["Close"], config.ROC_LOOKBACK)
    out["ATR14"] = atr(out["High"], out["Low"], out["Close"], config.ATR_PERIOD)
    out["ADX14"] = adx(out["High"], out["Low"], out["Close"], config.ADX_PERIOD)
    out["VolAvg20"] = out["Volume"].rolling(config.VOLUME_AVG).mean()
    out["VValue"] = out["Close"] * out["Volume"]
    out["VValueAvg20"] = out["VValue"].rolling(config.VOLUME_AVG).mean()
    out["High20"] = out["High"].rolling(20).max()
    out["Low20"] = out["Low"].rolling(20).min()
    return out

def technical_posture(out: pd.DataFrame) -> Dict[str, bool]:
    row = out.iloc[-1]
    posture = {
        "Uptrend": bool(row["EMA20"] > row["EMA50"] > row["SMA200"] and row["Close"] > row["EMA50"]),
        "AboveSMA200": bool(row["Close"] > row["SMA200"]),
    }
    return posture

def trigger_D1(out: pd.DataFrame) -> bool:
    last = out.iloc[-1]
    conds = [
        last["EMA20"] > last["EMA50"] > last["SMA200"],
        last["Close"] >= last["High20"],
        last["Volume"] >= config.D1_VOLUME_MULT * (last["VolAvg20"] if not np.isnan(last["VolAvg20"]) else 0),
        last["RSI14"] >= config.D1_MIN_RSI,
        last["ADX14"] >= config.D1_MIN_ADX,
        last["ROC63"] >= config.D1_MIN_ROC,
        ((last["Close"] - last["EMA50"]) / last["EMA50"] * 100.0) <= config.D1_MAX_EXT_ABOVE_EMA50_ATR
    ]
    return all(conds)

def trigger_D2(out: pd.DataFrame) -> bool:
    if len(out) < config.D2_PULLBACK_DAYS_MAX + 5:
        return False
    recent = out.iloc[-(config.D2_PULLBACK_DAYS_MAX + 3):]
    if not (recent["EMA20"].iloc[-1] > recent["EMA50"].iloc[-1] and recent["Close"].iloc[-1] > recent["EMA50"].iloc[-1]):
        return False
    near_ema20 = (recent["Low"] <= (recent["EMA20"] + 0.5 * recent["ATR14"]))
    pb_count = int(near_ema20.tail(config.D2_PULLBACK_DAYS_MAX).sum())
    if pb_count < config.D2_PULLBACK_DAYS_MIN:
        return False
    if (recent["Close"] < recent["EMA50"]).any():
        return False
    last = out.iloc[-1]
    prev = out.iloc[-2]
    if last["Close"] > prev["High"] and last["RSI14"] >= 50.0:
        return True
    return False

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import config
from technicals import compute_indicators, technical_posture, trigger_D1, trigger_D2


def liquidity_screen(ind_df: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
    """Return dict of pass/fail liquidity per ticker based on the latest 20-day averages."""
    res = {}
    for ticker, df in ind_df.items():
        if df.empty or len(df) < config.VOLUME_AVG:
            res[ticker] = False
            continue

        last = df.iloc[-1]

        # Use precomputed columns from technicals.compute_indicators()
        vvalue_avg20 = last.get("VValueAvg20")
        vol_avg20 = last.get("VolAvg20")
        close_px = last.get("Close")

        # Guard against NaNs (e.g., short history)
        vvalue_ok = (
            pd.notna(vvalue_avg20) and vvalue_avg20 >= config.MIN_AVG_DAILY_VALUE_SAR
        )
        volume_ok = pd.notna(vol_avg20) and vol_avg20 >= config.MIN_AVG_DAILY_VOLUME
        price_ok = pd.notna(close_px) and close_px >= config.MIN_PRICE_SAR

        res[ticker] = bool(vvalue_ok and volume_ok and price_ok)
    return res


def compute_breadth(universe_inds: Dict[str, pd.DataFrame]) -> float:
    flags = []
    for t, df in universe_inds.items():
        if df.empty:
            continue
        last = df.iloc[-1]
        flags.append(float(last["Close"] > last["EMA50"]))
    if not flags:
        return 0.0
    return 100.0 * (sum(flags) / len(flags))


def market_regime(index_df: pd.DataFrame, breadth_pct: float) -> Dict[str, bool]:
    last = index_df.iloc[-1]
    slope = (
        (index_df["SMA200"].diff(config.REGIME_SLOPE_WINDOW).iloc[-1])
        if "SMA200" in index_df.columns
        else 0.0
    )
    cond1 = (last["Close"] > last["SMA200"]) and (slope > 0)
    cond2 = last["Close"] > index_df["EMA50"].iloc[-1]
    cond3 = breadth_pct >= config.BREADTH_MIN_PCT_ABOVE_SMA50
    return {
        "cond1": cond1,
        "cond2": cond2,
        "cond3": cond3,
        "true_count": sum([cond1, cond2, cond3]),
    }


def sector_strength(returns_20d: pd.Series, index_ret_20d: float) -> pd.Series:
    """Compute sector RS vs index: sector 20d return minus index 20d return, then rank to percentiles."""
    rel = returns_20d - index_ret_20d
    ranks = rel.rank(pct=True) * 100.0
    return ranks


def technical_screen(ind_df: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, bool]]:
    out = {}
    for t, df in ind_df.items():
        if df.empty or len(df) < 220:
            out[t] = {"PostureUptrend": False, "D1": False, "D2": False}
            continue
        post = technical_posture(df)
        out[t] = {
            "PostureUptrend": post["Uptrend"],
            "D1": trigger_D1(df),
            "D2": trigger_D2(df),
        }
    return out

from typing import Dict, Tuple
import numpy as np
import pandas as pd

import config

def tech_score(ind: pd.DataFrame, sector_rel_series_value: float) -> float:
    last = ind.iloc[-1]
    rs_pct = float(sector_rel_series_value) if sector_rel_series_value is not None else 50.0
    adx = last["ADX14"]
    adx_clip = min(max(adx, 15.0), 35.0)
    adx_pct = (adx_clip - 15.0) / (35.0 - 15.0) * 100.0
    prox = abs((last["Close"] - last["EMA20"]) / (last["ATR14"] if last["ATR14"]>0 else 1.0))
    prox = max(0.0, 1.0 - min(prox, 1.0)) * 100.0
    return float(np.mean([rs_pct, rs_pct, adx_pct, prox]))

def composite_rank(tech: float, fund: float, sect: float) -> float:
    return round(config.TECH_WEIGHT * tech + config.FUND_WEIGHT * fund + config.SECTOR_WEIGHT * sect, 2)

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import config

def position_size(equity: float, entry: float, stop: float) -> int:
    risk_per_trade = config.RISK_PER_TRADE * equity
    per_share_risk = max(entry - stop, 0.0)
    if per_share_risk <= 0:
        return 0
    shares = int(risk_per_trade // per_share_risk)
    return max(shares, 0)

def cap_weight(shares: int, entry: float, equity: float) -> int:
    value = shares * entry
    cap = config.MAX_POSITION_WEIGHT * equity
    if value <= cap:
        return shares
    return int(cap // entry)

def correlation_filter(price_hist: Dict[str, pd.Series]) -> Dict[Tuple[str,str], float]:
    df = pd.DataFrame({t: s for t, s in price_hist.items()}).pct_change()
    corr = df.corr()
    pairs = {}
    for i, t1 in enumerate(corr.index):
        for j, t2 in enumerate(corr.columns):
            if j <= i:
                continue
            c = corr.loc[t1, t2]
            if pd.notna(c) and c > config.MAX_CORRELATION:
                pairs[(t1, t2)] = float(c)
    return pairs

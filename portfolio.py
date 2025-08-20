
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import os, json, time
from datetime import datetime
import pandas as pd
import numpy as np

from reporting import warn

import config
from risk import position_size, cap_weight

CACHE_DIR = os.path.join(os.getcwd(), "cache")
STATE_PATH = os.path.join(CACHE_DIR, "portfolio_state.json")
HIST_PATH = os.path.join(CACHE_DIR, "equity_history.csv")
TRADES_PATH = os.path.join(CACHE_DIR, "trades_log.csv")

@dataclass
class PositionState:
    ticker: str
    entry_price: float
    shares: int
    stop: float
    trail: float
    r: float
    sector: str
    entry_date: str
    bars_held: int = 0
    scaled: bool = False

@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, PositionState] = field(default_factory=dict)
    peak_equity: float = 0.0
    cooldown: int = 0
    last_date: Optional[str] = None

def load_state(default_cash: float = 100000.0) -> PortfolioState:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            pos = {t: PositionState(**p) for t, p in raw.get("positions", {}).items()}
            return PortfolioState(
                cash=float(raw.get("cash", default_cash)),
                positions=pos,
                peak_equity=float(raw.get("peak_equity", default_cash)),
                cooldown=int(raw.get("cooldown", 0)),
                last_date=raw.get("last_date"),
            )
        except Exception:
            pass
    st = PortfolioState(cash=default_cash, positions={}, peak_equity=default_cash, cooldown=0, last_date=None)
    save_state(st)
    return st

def save_state(state: PortfolioState) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = {
        "cash": state.cash,
        "positions": {t: asdict(p) for t, p in state.positions.items()},
        "peak_equity": state.peak_equity,
        "cooldown": state.cooldown,
        "last_date": state.last_date,
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

def _chandelier(high_close_20: float, atr: float, mult: float = 3.0) -> float:
    return high_close_20 - mult * atr

def mark_to_market(state: PortfolioState, ind: Dict[str, pd.DataFrame], dt: Optional[pd.Timestamp] = None):
    if dt is None:
        dates = [df.index[-1] for df in ind.values() if not df.empty]
        if not dates:
            return state.cash, 0.0
        dt = min(dates)
    mkt_value = 0.0
    for t, p in state.positions.items():
        df = ind.get(t)
        if df is None or df.empty:
            continue
        sub = df.loc[:dt]
        if sub.empty:
            continue
        close = float(sub["Close"].iloc[-1])
        mkt_value += close * p.shares
    equity = state.cash + mkt_value
    state.peak_equity = max(state.peak_equity, equity)
    dd = (equity / state.peak_equity) - 1.0 if state.peak_equity > 0 else 0.0
    return equity, dd

def _append_history(dt: pd.Timestamp, equity: float) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    row = {"Date": dt.strftime("%Y-%m-%d"), "Equity": float(equity)}
    if os.path.exists(HIST_PATH):
        import csv
        with open(HIST_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Date","Equity"])
            w.writerow(row)
    else:
        pd.DataFrame([row]).to_csv(HIST_PATH, index=False)

def _append_trades(trades: List[dict]) -> None:
    if not trades:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = pd.DataFrame(trades)
    if os.path.exists(TRADES_PATH):
        df.to_csv(TRADES_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(TRADES_PATH, index=False)

def manage_positions(state: PortfolioState, ind: Dict[str, pd.DataFrame], dt: pd.Timestamp):
    executed = []
    to_close = []
    for t, p in list(state.positions.items()):
        df = ind.get(t)
        if df is None or df.empty:
            continue
        sub = df.loc[:dt]
        if sub.empty:
            continue
        p.bars_held += 1
        last = sub.iloc[-1]
        if (not p.scaled) and (last["Close"] >= p.entry_price + 2 * p.r):
            sell = max(p.shares // 2, 0)
            if sell > 0:
                state.cash += sell * float(last["Close"])
                executed.append({"Date": dt.strftime("%Y-%m-%d"), "Ticker": t, "Action": "SELL", "Price": float(last["Close"]), "Shares": int(sell), "Reason": "SCALE_2R"})
                p.shares -= sell
                p.scaled = True
                p.stop = max(p.stop, p.entry_price)
        idx_here = sub.index.get_indexer([sub.index[-1]], method="pad")[0]
        high_close_20 = sub["Close"].iloc[max(0, idx_here - 19): idx_here + 1].max()
        p.trail = max(p.trail, _chandelier(float(high_close_20), float(last["ATR14"]), 3.0))
        exit_reason = None
        if last["Close"] <= p.stop:
            exit_reason = "STOP"
        else:
            prev = sub.iloc[-2] if len(sub) >= 2 else None
            if prev is not None and (last["Close"] < last["EMA20"] and prev["Close"] < prev["EMA20"]):
                exit_reason = "EMA20xDOWN"
            elif (last["Close"] < last["EMA50"]) and (last["ADX14"] < 15.0):
                exit_reason = "WEAK"
            elif last["Close"] < p.trail:
                exit_reason = "TRAIL"
            elif p.bars_held >= 60:
                exit_reason = "TIME"
        if exit_reason:
            state.cash += p.shares * float(last["Close"])
            executed.append({"Date": dt.strftime("%Y-%m-%d"), "Ticker": t, "Action": "SELL", "Price": float(last["Close"]), "Shares": int(p.shares), "Reason": exit_reason})
            to_close.append(t)
    for t in to_close:
        state.positions.pop(t, None)
    return state, executed

def select_candidates(ind: Dict[str, pd.DataFrame],
                      sectors: Dict[str, str],
                      triggers: Dict[str, Dict[str, bool]],
                      sector_rank_pct: Dict[str, float],
                      rs_pct_within_sector: Dict[str, float],
                      fs_map: Dict[str, float]) -> List[Tuple[str, str, str]]:
    """Mirror backtest selection ordering via CompositeRank."""
    out = []
    from ranking import tech_score, composite_rank
    for t, df in ind.items():
        if df.empty:
            continue
        tt = triggers.get(t, {})
        trig = "D1" if tt.get("D1") else ("D2" if tt.get("D2") else ("DB55" if tt.get("DB55") else None))
        if trig is None:
            continue
        sec = sectors.get(t, "Unknown")
        if sector_rank_pct.get(sec, 50.0) < config.SECTOR_TOP_PERCENTILE:
            continue
        if rs_pct_within_sector.get(t, 50.0) < 70.0:
            continue
        rs = rs_pct_within_sector.get(t, 50.0)
        tech = tech_score(df, rs)
        fund = float(fs_map.get(t, 70.0))
        sect = sector_rank_pct.get(sec, 50.0)
        comp = composite_rank(tech, fund, sect)
        out.append((t, trig, sec, comp))
    out.sort(key=lambda x: x[3], reverse=True)
    return [(t, trig, sec) for (t, trig, sec, _c) in out]

def build_action_list(state: PortfolioState,
                      ind: Dict[str, pd.DataFrame],
                      selection: List[Tuple[str, str, str]],
                      sectors: Dict[str, str],
                      equity_for_sizing: float,
                      regime_score: int = 3) -> Tuple[List[dict], List[dict]]:
    from collections import defaultdict
    entry_actions = []
    mgmt_actions = []
    by_sec = defaultdict(int)
    held = set(state.positions.keys())

    # Regime-gated capacity (same as backtest)
    if regime_score >= 3:
        max_today = config.MAX_CONCURRENT_POSITIONS
    elif regime_score == 2:
        max_today = max(1, config.MAX_CONCURRENT_POSITIONS - 1)
    elif regime_score == 1:
        max_today = 1
    else:
        max_today = 0
    capacity = max(0, max_today - len(held))

    def _max_corr_with_held(ticker: str) -> float:
        if not held:
            return 0.0
        df = pd.DataFrame()
        for ht in held:
            if ht in ind and not ind[ht].empty:
                df[ht] = ind[ht]["Close"].pct_change().tail(config.CORRELATION_LOOKBACK).reset_index(drop=True)
        df[ticker] = ind[ticker]["Close"].pct_change().tail(config.CORRELATION_LOOKBACK).reset_index(drop=True)
        corr = df.corr().loc[ticker, list(held)].max()
        return float(corr) if pd.notna(corr) else 0.0

    for t, trig, sec in selection:
        if capacity <= 0:
            break
        if by_sec[sec] >= config.MAX_PER_SECTOR:
            continue
        if _max_corr_with_held(t) > config.MAX_CORRELATION:
            continue
        last = ind[t].iloc[-1]
        entry = float(last["Close"])
        atr = float(last["ATR14"])
        swing_low = float(ind[t]["Low"].tail(10).min())
        stop = max(entry - 2 * atr, swing_low - 0.5 * atr)
        if stop >= entry:
            continue
        shares = position_size(equity_for_sizing, entry, stop)
        shares = cap_weight(shares, entry, equity_for_sizing)
        if shares <= 0:
            continue
        entry_actions.append({
            "Ticker": t, "Action": "MOC", "Price": round(entry, 2), "Shares": int(shares),
            "Stop": round(stop, 2), "Sector": sec, "Reason": trig
        })
        by_sec[sec] += 1
        capacity -= 1

    # Management suggestions (trailing/2R heads-up)
    for t, p in state.positions.items():
        last = ind[t].iloc[-1]
        new_trail = max(p.trail, _chandelier(float(ind[t]["Close"].tail(20).max()), float(last["ATR14"]), 3.0))
        if new_trail > p.trail:
            mgmt_actions.append({"Ticker": t, "Action": "RAISE_TRAIL", "To": round(new_trail, 2)})
        if (not p.scaled) and (last["Close"] >= p.entry_price + 2 * p.r):
            mgmt_actions.append({"Ticker": t, "Action": "REDUCE_HALF_AT_2R"})
        if p.bars_held >= 59:
            mgmt_actions.append({"Ticker": t, "Action": "TIME_STOP_TOMORROW"})
    return entry_actions, mgmt_actions

def apply_entries(state: PortfolioState, ind: Dict[str, pd.DataFrame], entries: List[dict], dt: pd.Timestamp):
    executed = []
    for e in entries:
        t = e["Ticker"]
        if t in state.positions:
            continue
        px = float(e["Price"]); sh = int(e["Shares"])
        if sh <= 0: continue
        cost = sh * px
        if cost > state.cash:
            warn(f"Skipped BUY {t} — insufficient cash (need {cost:.2f}, have {state.cash:.2f})")
            continue
        state.cash -= cost
        last = ind[t].iloc[-1]
        atr = float(last["ATR14"])
        r = px - float(e["Stop"])
        trail = max(float(ind[t]["Close"].tail(20).max()), px) - 3.0 * atr
        state.positions[t] = PositionState(
            ticker=t, entry_price=px, shares=sh, stop=float(e["Stop"]), trail=float(trail),
            r=float(r), sector=e.get("Sector",""), entry_date=dt.strftime(config.DATE_FORMAT), bars_held=0, scaled=False
        )
        executed.append({"Date": dt.strftime("%Y-%m-%d"), "Ticker": t, "Action": "BUY", "Price": px, "Shares": sh, "Reason": e.get("Reason","")})
    return state, executed

def summarize_state(state: PortfolioState, ind: Dict[str, pd.DataFrame]) -> List[List[str]]:
    rows = []
    from data_fetcher import get_company_name
    for t, p in state.positions.items():
        last = ind.get(t)
        close = float(last["Close"].iloc[-1]) if last is not None and not last.empty else float("nan")
        name = ""
        try:
            nm = get_company_name(t)
            name = nm if isinstance(nm, str) else ""
        except Exception:
            name = ""
        disp = f"{t} — {name}" if name else t
        rows.append([disp, f"{close:.2f}", f"{p.entry_price:.2f}", f"{p.stop:.2f}", f"{p.trail:.2f}", p.shares, p.sector])
    return rows

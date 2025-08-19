
# Portfolio-level backtest (robust to missing bars; regime-aware; DB55 entries; CSV/plot hooks live in main.py)
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import config
from ranking import composite_rank, tech_score
from risk import position_size, cap_weight

@dataclass
class Position:
    ticker: str
    entry: float
    shares: int
    stop: float
    trail: float
    r: float
    scaled: bool = False
    bars_held: int = 0
    sector: str = ""

@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    action: str
    price: float
    shares: int
    reason: str = ""

@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    trades: List[Trade] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)
    bench_equity: Optional[pd.Series] = None
    bench_stats: Optional[Dict[str, float]] = None
    open_positions: List[Position] = field(default_factory=list)

def _chandelier_trail(close_roll_max: float, atr: float, multiple: float = 3.0) -> float:
    return close_roll_max - multiple * atr

def _perf_metrics(equity: pd.Series) -> Dict[str, float]:
    rets = equity.pct_change().dropna()
    if rets.empty:
        return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252.0 / len(rets)) - 1
    vol = rets.std() * np.sqrt(252.0)
    sharpe = (rets.mean() / rets.std() * np.sqrt(252.0)) if rets.std() > 0 else 0.0
    dd = (equity / equity.cummax() - 1.0).min()
    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": float(dd)}

# --- Helpers to be robust to missing bars on a given dt ---
def _row_on_or_before(df: pd.DataFrame, dt: pd.Timestamp) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    # fast path
    try:
        if dt in df.index:
            return df.loc[dt]
    except Exception:
        pass
    sub = df.loc[:dt]
    if sub.empty:
        return None
    return sub.iloc[-1]

def _close_on_or_before(df: pd.DataFrame, dt: pd.Timestamp) -> float:
    row = _row_on_or_before(df, dt)
    return float(row["Close"]) if row is not None and not pd.isna(row["Close"]) else np.nan

def portfolio_backtest(
    price: Dict[str, pd.DataFrame],
    ind: Dict[str, pd.DataFrame],
    sectors: Dict[str, str],
    index_ind: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    init_equity: float = 300000.0,
    fund_filter_mode: str = "none",
):

    # Build date range from index, clipped to [start,end]
    dates = index_ind.index
    if start:
        dates = dates[dates >= pd.to_datetime(start)]
    if end:
        dates = dates[dates <= pd.to_datetime(end)]

    # Pre-compute 20d returns
    ret20 = {t: ind[t]["Close"].pct_change(config.SECTOR_RS_LOOKBACK) for t in ind}
    # Sector mean 20d returns
    unique_sectors = set(sectors.values())
    sector_ret20 = {}
    for sec in unique_sectors:
        members = [t for t, s in sectors.items() if s == sec and t in ret20]
        if not members:
            continue
        df = pd.DataFrame({t: ret20[t] for t in members})
        sector_ret20[sec] = df.mean(axis=1)

    idx_ret20 = index_ind["Close"].pct_change(config.SECTOR_RS_LOOKBACK)

    # Fundamentals (static gating if requested)
    fs_static_pass = {t: True for t in price.keys()}
    if fund_filter_mode == "static":
        from fundamentals import compute_fundamental_metrics, sector_relative_scores
        from data_fetcher import get_fundamentals_cached_or_fetch

        fund_raw = {}
        for t in price.keys():
            try:
                fund_raw[t] = get_fundamentals_cached_or_fetch(
                    t, compute_func=compute_fundamental_metrics, manual_df=None
                )
            except Exception:
                fund_raw[t] = {"metrics": {}, "improvements": {}, "FS": None}
        fs_map = sector_relative_scores(fund_raw, sectors)
        # Sector FS percentile for top-40% rule
        sec_groups = {}
        for t, sec in sectors.items():
            sec_groups.setdefault(sec, []).append((t, fs_map.get(t, 0.0)))
        fs_pct = {}
        for sec, items in sec_groups.items():
            ser = pd.Series({t: fs for t, fs in items})
            ranks = ser.rank(pct=True) * 100.0
            for t, _ in items:
                fs_pct[t] = float(ranks.loc[t])
        for t in price.keys():
            fs_static_pass[t] = (fs_map.get(t, 0.0) >= config.FUNDAMENTAL_MIN_FS) and (
                fs_pct.get(t, 0.0) >= config.FUNDAMENTAL_MIN_FS_SECTOR_TOP
            )

    # Portfolio state
    cash = init_equity
    equity_curve = []
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    cooldown = 0
    peak_equity = init_equity

    for dt in dates:
        # Breadth at dt
        above50_flags = []
        for t, df in ind.items():
            row = _row_on_or_before(df, dt)
            if row is None:
                continue
            above50_flags.append(float(row["Close"] > row.get("SMA50", np.nan)))
        breadth = 100.0 * (np.mean(above50_flags) if above50_flags else 0.0)

        # Regime at dt
        row_idx = _row_on_or_before(index_ind, dt)
        if row_idx is None:
            continue
        i = index_ind.index.get_indexer([dt], method="pad")[0]  # nearest <= dt
        if i >= config.REGIME_SLOPE_WINDOW:
            slope = (
                index_ind["SMA200"].iloc[i]
                - index_ind["SMA200"].iloc[i - config.REGIME_SLOPE_WINDOW]
            )
        else:
            slope = 0.0
        cond1 = (row_idx["Close"] > row_idx["SMA200"]) and (slope > 0)
        cond2 = row_idx["Close"] > row_idx.get("SMA50", row_idx.get("EMA50", row_idx["Close"]))
        cond3 = breadth >= config.BREADTH_MIN_PCT_ABOVE_SMA50
        regime_true = sum([cond1, cond2, cond3])

        # Sector RS percentile today
        sector_rs_pct = {}
        for sec, sret in sector_ret20.items():
            if dt in sret.index and dt in idx_ret20.index:
                rel = sret.loc[dt] - idx_ret20.loc[dt]
                sector_rs_pct[sec] = rel
        if sector_rs_pct:
            ser_rel = pd.Series(sector_rs_pct)
            ranks = ser_rel.rank(pct=True) * 100.0
            sector_rs_pct = ranks.to_dict()
        else:
            sector_rs_pct = {sec: 50.0 for sec in unique_sectors}

        # Update positions (scale/trail/time/exit)
        to_close = []
        for t, pos in list(positions.items()):
            df = ind[t]
            row = _row_on_or_before(df, dt)
            if row is None:
                continue
            pos.bars_held += 1
            # scale at +2R once
            if (not pos.scaled) and (row["Close"] >= pos.entry + 2 * pos.r):
                sell_sh = pos.shares // 2
                if sell_sh > 0:
                    proceeds = sell_sh * row["Close"]
                    cash += proceeds
                    trades.append(Trade(dt, t, "SELL", float(row["Close"]), int(sell_sh), "SCALE_2R"))
                    pos.shares -= sell_sh
                    pos.scaled = True
                    pos.stop = max(pos.stop, pos.entry)
            # update trail
            idx_here = df.index.get_indexer([dt], method="pad")[0]
            high_close_20 = df["Close"].iloc[max(0, idx_here - 19) : idx_here + 1].max()
            pos.trail = max(pos.trail, _chandelier_trail(high_close_20, row["ATR14"], 3.0))
            # exits
            exit_reason = None
            # protective stop
            if row["Close"] <= pos.stop:
                exit_reason = "STOP"
            else:
                # trend exit: two closes below EMA20
                idx_prev = max(0, idx_here - 1)
                if (
                    df["Close"].iloc[idx_here] < df["EMA20"].iloc[idx_here]
                    and df["Close"].iloc[idx_prev] < df["EMA20"].iloc[idx_prev]
                ):
                    exit_reason = "EMA20xDOWN"
                elif (row["Close"] < row["EMA50"]) and (row["ADX14"] < 15.0):
                    exit_reason = "WEAK"
                elif row["Close"] < pos.trail:
                    exit_reason = "TRAIL"
                elif pos.bars_held >= 60:
                    exit_reason = "TIME"

            if exit_reason:
                proceeds = pos.shares * row["Close"]
                cash += proceeds
                trades.append(Trade(dt, t, "SELL", float(row["Close"]), int(pos.shares), exit_reason))
                to_close.append(t)
        for t in to_close:
            positions.pop(t, None)

        # Equity and drawdown using as-of close (pad last value)
        mkt_value = 0.0
        for t, pos in positions.items():
            close = _close_on_or_before(ind[t], dt)
            if not np.isnan(close):
                mkt_value += close * pos.shares
        equity = cash + mkt_value
        peak_equity = max(peak_equity, equity)
        dd = (equity / peak_equity) - 1.0

        # Drawdown controls
        if dd <= -config.PORTFOLIO_MAX_DRAWDOWN and len(positions) > 0:
            for t, pos in list(positions.items()):
                price_close = _close_on_or_before(ind[t], dt)
                if np.isnan(price_close):
                    continue
                cash += pos.shares * float(price_close)
                trades.append(Trade(dt, t, "SELL", float(price_close), int(pos.shares), "MAX_DD"))
                positions.pop(t, None)
            cooldown = config.DRAWDOWN_COOLDOWN_DAYS

        # Cooldown decrement
        if cooldown > 0:
            cooldown -= 1

        # New entries capacity today + dynamic risk
        risk_frac = 0.0
        max_names_today = 0
        if regime_true == 3 and cooldown == 0:
            max_names_today = config.MAX_CONCURRENT_POSITIONS
            risk_frac = config.RISK_PER_TRADE
        elif regime_true == 2 and cooldown == 0:
            max_names_today = max(1, config.MAX_CONCURRENT_POSITIONS - 1)
            risk_frac = max(0.5 * config.RISK_PER_TRADE, 0.005)
        elif regime_true == 1 and cooldown == 0:
            max_names_today = 1
            risk_frac = max(0.5 * config.RISK_PER_TRADE, 0.005)
        else:
            max_names_today = 0
            risk_frac = 0.0

        # Select and enter new positions if capacity
        if max_names_today > len(positions):
            # RS percentile within sector at dt
            rs_pct = {}
            for sec in unique_sectors:
                members = [t for t, s in sectors.items() if s == sec and t in ind]
                vals = {}
                for t in members:
                    series = ind[t]["Close"].pct_change(config.SECTOR_RS_LOOKBACK)
                    if dt in series.index:
                        vals[t] = series.loc[dt]
                    else:
                        vals[t] = series.loc[:dt].iloc[-1] if not series.loc[:dt].empty else np.nan
                ser = pd.Series(vals).dropna()
                if not ser.empty:
                    ranks = ser.rank(pct=True) * 100.0
                    for t in members:
                        rs_pct[t] = float(ranks.get(t, np.nan))

            # Build candidates
            candidates = []
            from technicals import technical_posture, trigger_D1, trigger_D2, trigger_DB55

            for t, df in ind.items():
                if t in positions:
                    continue
                # require at least some history before dt
                if _row_on_or_before(df, dt) is None:
                    continue
                if fund_filter_mode == "static" and not fs_static_pass.get(t, False):
                    continue
                sec = sectors.get(t, "Unknown")
                # sector RS gate
                sec_strength = sector_rs_pct.get(sec, 50.0)
                if sec_strength < config.SECTOR_TOP_PERCENTILE:
                    continue
                last = _row_on_or_before(df, dt)
                volavg20 = last.get("VolAvg20", np.nan)
                vvalue_avg20 = (last["Close"] * volavg20) if not pd.isna(volavg20) else np.nan
                # liquidity screens
                if not (
                    (not pd.isna(vvalue_avg20) and vvalue_avg20 >= config.MIN_AVG_DAILY_VALUE_SAR)
                    and (not pd.isna(volavg20) and volavg20 >= config.MIN_AVG_DAILY_VOLUME)
                    and last["Close"] >= config.MIN_PRICE_SAR
                ):
                    continue
                sub = df.loc[:dt]
                post = technical_posture(sub)
                if not post["Uptrend"]:
                    continue
                d1 = trigger_D1(sub)
                d2 = trigger_D2(sub)
                db = trigger_DB55(sub)
                if not (d1 or d2 or db):
                    continue
                # RS>=70 within sector gate
                if rs_pct.get(t, 50.0) < 70.0:
                    continue
                tech = tech_score(sub, rs_pct.get(t, 50.0))
                fund = 70.0
                sect_score = sec_strength
                comp = composite_rank(tech, fund, sect_score)
                trig = "D1" if d1 else ("D2" if d2 else "DB55")
                candidates.append((t, comp, trig, sec))

            # Correlation & sector caps; greedily add
            from collections import defaultdict
            by_sec = defaultdict(int)
            add_needed = max_names_today - len(positions)
            selected = []
            held = list(positions.keys())

            for t, comp, trig, sec in sorted(candidates, key=lambda x: x[1], reverse=True):
                if by_sec[sec] >= config.MAX_PER_SECTOR:
                    continue
                # Correlation filter vs current holdings (as-of dt)
                ok_corr = True
                if held:
                    corr_df = pd.DataFrame()
                    for ht in held:
                        s = ind[ht]["Close"].pct_change().loc[:dt].tail(config.CORRELATION_LOOKBACK)
                        if not s.empty:
                            corr_df[ht] = s.reset_index(drop=True)
                    s_new = ind[t]["Close"].pct_change().loc[:dt].tail(config.CORRELATION_LOOKBACK)
                    if not s_new.empty:
                        corr_df[t] = s_new.reset_index(drop=True)
                    if not corr_df.empty and t in corr_df.columns:
                        corr = corr_df.corr()
                        for ht in held:
                            if ht in corr.index and t in corr.columns:
                                if pd.notna(corr.loc[ht, t]) and corr.loc[ht, t] > config.MAX_CORRELATION:
                                    ok_corr = False
                                    break
                if not ok_corr:
                    continue
                selected.append((t, comp, trig, sec))
                by_sec[sec] += 1
                if len(selected) >= add_needed:
                    break

            # Execute entries at as-of close
            for t, comp, trig, sec in selected:
                row = _row_on_or_before(ind[t], dt)
                if row is None:
                    continue
                entry_price = float(row["Close"])
                swing_low = ind[t]["Low"].loc[:dt].rolling(10).min().iloc[-1]
                init_stop = max(entry_price - 2 * row["ATR14"], swing_low - 0.5 * row["ATR14"])
                per_share_risk = max(entry_price - init_stop, 0.0)
                if per_share_risk <= 0:
                    continue
                shares = int((risk_frac * equity) // per_share_risk)
                shares = cap_weight(shares, entry_price, equity)
                if shares <= 0 or (shares * entry_price) > cash:
                    continue
                cash -= shares * entry_price
                r = entry_price - init_stop
                high_close_20 = ind[t]["Close"].loc[:dt].tail(20).max()
                trail = _chandelier_trail(high_close_20, row["ATR14"], 3.0)
                positions[t] = Position(t, entry_price, int(shares), float(init_stop), float(trail), float(r),
                                        scaled=False, bars_held=0, sector=sec)
                trades.append(Trade(dt, t, "BUY", float(entry_price), int(shares), f"ENTRY_{trig}"))

        # Record equity at end of dt
        mkt_value = 0.0
        for t, pos in positions.items():
            close = _close_on_or_before(ind[t], dt)
            if not np.isnan(close):
                mkt_value += close * pos.shares
        equity = cash + mkt_value
        equity_curve.append((dt, equity))
    equity_series = pd.Series({d: v for d, v in equity_curve}).sort_index()
    stats = _perf_metrics(equity_series)

    # Benchmark (^TASI.SR) buy&hold aligned to equity dates
    idx_close = index_ind["Close"].reindex(equity_series.index).ffill()
    bench_rets = idx_close.pct_change().fillna(0.0)
    bench_equity = (1.0 + bench_rets).cumprod() * float(init_equity)
    bench_stats = _perf_metrics(bench_equity)

    return PortfolioResult(
        equity_curve=equity_series,
        trades=trades,
        stats=stats,
        bench_equity=bench_equity,
        bench_stats=bench_stats,
        open_positions=list(positions.values()),
    )

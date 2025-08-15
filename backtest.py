
# Portfolio-level backtest (updated with new triggers, trend exits, CSV export hooks)
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import os

import config
from technicals import compute_indicators, trigger_DB55
from screening import technical_screen
from ranking import composite_rank, tech_score
from risk import position_size, cap_weight, correlation_filter


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


def _chandelier_trail(close_roll_max: float, atr: float, multiple: float = 3.0) -> float:
    return close_roll_max - multiple * atr


def _perf_metrics(equity: pd.Series) -> Dict[str, float]:
    rets = equity.pct_change().dropna()
    if rets.empty:
        return {}
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252.0 / len(rets)) - 1
    vol = rets.std() * np.sqrt(252.0)
    sharpe = (rets.mean() / rets.std() * np.sqrt(252.0)) if rets.std() > 0 else 0.0
    dd = (equity / equity.cummax() - 1.0).min()
    return {
        "CAGR": float(cagr),
        "Vol": float(vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(dd),
    }


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
    """
    Portfolio-level simulator with revised rules (RADICAL CHANGES):
    - Regime gating (>=2 of 3) controls max names and risk per trade (half risk at 2/3).
    - Entries: accept any of {D1 breakout, D2 pullback-and-go, DB55 Donchian breakout} + RS>=70 within sector.
    - 2% risk per trade at regime=3; 1% risk at regime=2; 0 new entries at regime<2.
    - Scale half at +2R; trail = Chandelier(20, 3*ATR). Hard trend exit: Close<EMA20 for 2 bars OR Close<EMA50 & ADX<15.
    - Time stop at 60 bars.
    - Drawdown controls (-20% flatten + cooldown).
    """
    dates = index_ind.index
    if start:
        dates = dates[dates >= pd.to_datetime(start)]
    if end:
        dates = dates[dates <= pd.to_datetime(end)]

    # Pre-compute 20d returns
    ret20 = {t: ind[t]["Close"].pct_change(20) for t in ind}
    # Sector mean 20d returns
    unique_sectors = set(sectors.values())
    sector_ret20 = {}
    for sec in unique_sectors:
        members = [t for t, s in sectors.items() if s == sec and t in ret20]
        if not members:
            continue
        df = pd.DataFrame({t: ret20[t] for t in members})
        sector_ret20[sec] = df.mean(axis=1)

    idx_ret20 = index_ind["Close"].pct_change(20)

    # Fundamentals static gating (optional; lookahead-prone)
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
            if dt in df.index:
                row = df.loc[dt]
                above50_flags.append(float(row["Close"] > row["SMA50"]))
        breadth = 100.0 * (np.mean(above50_flags) if above50_flags else 0.0)

        # Regime
        row_idx = index_ind.loc[dt]
        i = index_ind.index.get_loc(dt)
        if i >= config.REGIME_SLOPE_WINDOW:
            slope = (
                index_ind["SMA200"].iloc[i]
                - index_ind["SMA200"].iloc[i - config.REGIME_SLOPE_WINDOW]
            )
        else:
            slope = 0.0
        cond1 = (row_idx["Close"] > row_idx["SMA200"]) and (slope > 0)
        cond2 = row_idx["Close"] > row_idx["SMA50"]
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
            if dt not in df.index:
                continue
            row = df.loc[dt]
            pos.bars_held += 1
            # scale at +2R once
            if (not pos.scaled) and (row["Close"] >= pos.entry + 2 * pos.r):
                sell_sh = pos.shares // 2
                if sell_sh > 0:
                    proceeds = sell_sh * row["Close"]
                    cash += proceeds
                    trades.append(
                        Trade(
                            dt, t, "SELL", float(row["Close"]), int(sell_sh), "SCALE_2R"
                        )
                    )
                    pos.shares -= sell_sh
                    pos.scaled = True
                    # move protective stop to breakeven for remainder
                    pos.stop = max(pos.stop, pos.entry)
            # update trail
            idx_here = df.index.get_loc(dt)
            high_close_20 = df["Close"].iloc[max(0, idx_here - 19) : idx_here + 1].max()
            pos.trail = max(
                pos.trail, _chandelier_trail(high_close_20, row["ATR14"], 3.0)
            )
            # exits
            exit_reason = None
            # hard protective stop
            if row["Close"] <= pos.stop:
                exit_reason = "STOP"
            # trend exit: two closes below EMA20
            elif idx_here >= 1 and (df["Close"].iloc[idx_here] < df["EMA20"].iloc[idx_here]) and (df["Close"].iloc[idx_here-1] < df["EMA20"].iloc[idx_here-1]):
                exit_reason = "EMA20xDOWN"
            # weakening / loss of momentum
            elif (row["Close"] < row["EMA50"]) and (row["ADX14"] < 15.0):
                exit_reason = "WEAK"
            # trailing stop
            elif row["Close"] < pos.trail:
                exit_reason = "TRAIL"
            # time stop
            elif pos.bars_held >= 60:
                exit_reason = "TIME"

            if exit_reason:
                proceeds = pos.shares * row["Close"]
                cash += proceeds
                trades.append(
                    Trade(
                        dt, t, "SELL", float(row["Close"]), int(pos.shares), exit_reason
                    )
                )
                to_close.append(t)
        for t in to_close:
            positions.pop(t, None)

        # Equity and drawdown
        mkt_value = sum(
            ind[t].loc[dt]["Close"] * pos.shares
            for t, pos in positions.items()
            if dt in ind[t].index
        )
        equity = cash + mkt_value
        peak_equity = max(peak_equity, equity)
        dd = (equity / peak_equity) - 1.0

        # Drawdown controls
        if dd <= -config.PORTFOLIO_MAX_DRAWDOWN and len(positions) > 0:
            for t, pos in list(positions.items()):
                price_close = ind[t].loc[dt]["Close"]
                cash += pos.shares * price_close
                trades.append(
                    Trade(dt, t, "SELL", float(price_close), int(pos.shares), "MAX_DD")
                )
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
            risk_frac = max(0.5 * config.RISK_PER_TRADE, 0.005)  # half risk
        else:
            max_names_today = 0
            risk_frac = 0.0

        # Select and enter new positions if capacity
        if max_names_today > len(positions):
            # RS percentile within sector at dt
            rs_pct = {}
            for sec in unique_sectors:
                members = [
                    t
                    for t, s in sectors.items()
                    if s == sec and dt in ind.get(t, pd.DataFrame()).index
                ]
                vals = {}
                for t in members:
                    r = ind[t]["Close"].pct_change(20).loc[dt]
                    vals[t] = r if not np.isnan(r) else np.nan
                ser = pd.Series(vals).dropna()
                if not ser.empty:
                    ranks = ser.rank(pct=True) * 100.0
                    for t in members:
                        rs_pct[t] = float(ranks.get(t, np.nan))

            # Build candidates
            candidates = []
            from technicals import technical_posture, trigger_D1, trigger_D2, trigger_DB55

            for t, df in ind.items():
                if t in positions or dt not in df.index:
                    continue
                if fund_filter_mode == "static" and not fs_static_pass.get(t, False):
                    continue
                sec = sectors.get(t, "Unknown")
                if sector_rs_pct.get(sec, 0.0) < config.SECTOR_TOP_PERCENTILE:
                    continue
                last = df.loc[dt]
                vvalue_avg20 = df["VValue"].rolling(config.VOLUME_AVG).mean().loc[dt]
                vol_avg20 = df["Volume"].rolling(config.VOLUME_AVG).mean().loc[dt]
                if not (
                    vvalue_avg20 >= config.MIN_AVG_DAILY_VALUE_SAR
                    and vol_avg20 >= config.MIN_AVG_DAILY_VOLUME
                    and last["Close"] >= config.MIN_PRICE_SAR
                ):
                    continue
                sub = df.loc[:dt]
                post = technical_posture(sub)
                if not post["Uptrend"]:
                    continue
                d1 = trigger_D1(sub)
                d2 = trigger_D2(sub)
                db = trigger_DB55(sub)  # NEW radical breakout
                if not (d1 or d2 or db):
                    continue
                # RS>=70 within sector gate
                if rs_pct.get(t, 50.0) < 70.0:
                    continue
                tech = tech_score(sub, rs_pct.get(t, 50.0))
                fund = 70.0
                sect_score = sector_rs_pct.get(sec, 50.0)
                comp = composite_rank(tech, fund, sect_score)
                trig = "D1" if d1 else ("D2" if d2 else "DB55")
                candidates.append((t, comp, trig, sec))

            # Correlation & sector caps; greedily add
            from collections import defaultdict

            by_sec = defaultdict(int)
            add_needed = max_names_today - len(positions)
            selected = []
            held = list(positions.keys())
            for t, comp, trig, sec in sorted(
                candidates, key=lambda x: x[1], reverse=True
            ):
                if by_sec[sec] >= config.MAX_PER_SECTOR:
                    continue
                # Correlation check vs current holdings
                ok_corr = True
                if held:
                    corr_df = pd.DataFrame(
                        {
                            ht: ind[ht]["Close"]
                            .pct_change()
                            .loc[:dt]
                            .iloc[-config.CORRELATION_LOOKBACK :]
                            for ht in held
                            if dt in ind[ht].index
                        }
                    )
                    corr_df[t] = (
                        ind[t]["Close"]
                        .pct_change()
                        .loc[:dt]
                        .iloc[-config.CORRELATION_LOOKBACK :]
                    )
                    corr = corr_df.corr()
                    for ht in held:
                        if ht in corr.index and t in corr.columns:
                            if (
                                pd.notna(corr.loc[ht, t])
                                and corr.loc[ht, t] > config.MAX_CORRELATION
                            ):
                                ok_corr = False
                                break
                if not ok_corr:
                    continue
                selected.append((t, comp, trig, sec))
                by_sec[sec] += 1
                if len(selected) >= add_needed:
                    break

            # Execute entries at close
            for t, comp, trig, sec in selected:
                row = ind[t].loc[dt]
                entry_price = float(row["Close"])
                swing_low = ind[t]["Low"].loc[:dt].rolling(10).min().iloc[-1]
                init_stop = max(
                    entry_price - 2 * row["ATR14"], swing_low - 0.5 * row["ATR14"]
                )
                per_share_risk = max(entry_price - init_stop, 0.0)
                if per_share_risk <= 0:
                    continue
                # dynamic risk
                shares = int((risk_frac * equity) // per_share_risk)
                shares = cap_weight(shares, entry_price, equity)
                if shares <= 0:
                    continue
                cost = shares * entry_price
                if cost > cash:
                    continue
                cash -= cost
                r = entry_price - init_stop
                high_close_20 = ind[t]["Close"].loc[:dt].iloc[-20:].max()
                trail = _chandelier_trail(high_close_20, row["ATR14"], 3.0)
                positions[t] = Position(
                    t,
                    entry_price,
                    int(shares),
                    float(init_stop),
                    float(trail),
                    float(r),
                    scaled=False,
                    bars_held=0,
                    sector=sec,
                )
                trades.append(
                    Trade(dt, t, "BUY", float(entry_price), int(shares), f"ENTRY_{trig}")
                )

        # Record equity
        mkt_value = sum(
            ind[t].loc[dt]["Close"] * pos.shares
            for t, pos in positions.items()
            if dt in ind[t].index
        )
        equity = cash + mkt_value
        equity_curve.append((dt, equity))

    equity_series = pd.Series({d: v for d, v in equity_curve}).sort_index()
    stats = _perf_metrics(equity_series)
    return PortfolioResult(equity_series, trades, stats)

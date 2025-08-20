import argparse

import pandas as pd
import numpy as np

import config
import data_fetcher as dfetch
from technicals import compute_indicators, trigger_D1, trigger_D2, trigger_DB55
from screening import liquidity_screen, technical_screen
from fundamentals import compute_fundamental_metrics, sector_relative_scores
from ranking import tech_score, composite_rank
from risk import position_size, cap_weight
from reporting import (
    print_panel,
    print_table,
    info,
    warn,
    error,
    colorize_status,
    fmt_ticker_and_name,
)
from portfolio import (
    load_state,
    save_state,
    mark_to_market,
    manage_positions,
    select_candidates,
    build_action_list,
    apply_entries,
    summarize_state,
)

# --- Progress formatting to match other modules ---
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def _make_progress(disabled: bool = False):
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
        disable=disabled,
    )


def weekend_refresh(universe: list, no_progress: bool = False):
    info("Weekend refresh: fundamentals + sector RS + regime + watchlist...")
    price, ind, sectors, names = {}, {}, {}, {}
    with _make_progress(disabled=no_progress) as progress:
        t_prices = progress.add_task("Prices & indicators", total=len(universe))
        for t in universe:
            p = dfetch.get_price_history(t, lookback_days=1500, refresh=False)
            if not p.empty:
                price[t] = p
                ind[t] = compute_indicators(p)
                sectors[t] = dfetch.get_sector(t) or "Unknown"
                names[t] = dfetch.get_company_name(t) or ""
            progress.advance(t_prices)
    if not ind:
        error("No data for weekend refresh.")
        return

    idx = dfetch.get_index_history(lookback_days=1500, refresh=False)
    idx_ind = compute_indicators(idx)
    breadth = 100.0 * np.mean(
        [float(df.iloc[-1]["Close"] > df.iloc[-1]["EMA50"]) for df in ind.values()]
    )
    cond1 = (idx_ind.iloc[-1]["Close"] > idx_ind.iloc[-1]["SMA200"]) and (
        idx_ind["SMA200"].iloc[-1] > idx_ind["SMA200"].iloc[-20]
    )
    cond2 = idx_ind.iloc[-1]["Close"] > idx_ind.iloc[-1]["EMA50"]
    cond3 = breadth >= config.BREADTH_MIN_PCT_ABOVE_SMA50
    regime_score = int(cond1) + int(cond2) + int(cond3)

    # Print Market Regime (same as main.py)
    r_txt = (
        f"Cond1 (Idx>SMA200 & SMA200↑): {'✔' if cond1 else '✖'}\n"
        f"Cond2 (Idx>EMA50): {'✔' if cond2 else '✖'}\n"
        f"Cond3 (Breadth≥{config.BREADTH_MIN_PCT_ABOVE_SMA50:.0f}%): {'✔' if cond3 else '✖'}\n"
        f"Score: {regime_score}/3   |   Breadth: {breadth:.1f}%"
    )
    print_panel("Market Regime", r_txt)
    # Fundamentals snapshot (static) for watchlist
    try:
        manual_df = pd.read_csv(config.MANUAL_FUNDAMENTALS_CSV)
    except Exception:
        manual_df = pd.DataFrame(columns=["ticker", "metric", "value", "period"])
    fund_raw = {
        t: dfetch.get_fundamentals_cached_or_fetch(
            t, compute_fundamental_metrics, manual_df
        )
        for t in universe
        if t in ind
    }
    fs_map = sector_relative_scores(fund_raw, sectors)

    dfetch.persist_fs_to_cache(fs_map, sectors)
    # Sector strength 20d RS vs index
    idx_ret20 = idx_ind["Close"].pct_change(config.SECTOR_RS_LOOKBACK).iloc[-1]
    sec_returns = {}
    for sec in set(sectors.values()):
        members = [
            t for t, s in sectors.items() if s == sec and t in ind and len(ind[t]) >= 25
        ]
        if not members:
            continue
        rlist = [
            ind[t]["Close"].pct_change(config.SECTOR_RS_LOOKBACK).iloc[-1]
            for t in members
        ]
        rlist = [r for r in rlist if pd.notna(r)]
        if rlist:
            sec_returns[sec] = float(
                np.nanmean(rlist) - (idx_ret20 if pd.notna(idx_ret20) else 0.0)
            )
    sec_pct = (
        (pd.Series(sec_returns).rank(pct=True) * 100.0).to_dict() if sec_returns else {}
    )

    rows = []
    for t in universe:
        if t not in ind:
            continue
        ts = tech_score(ind[t], 50.0)
        fs = fs_map.get(t, 0.0)
        sc = sec_pct.get(sectors.get(t, "Unknown"), 50.0)
        comp = composite_rank(ts, fs, sc)
        rows.append(
            [
                fmt_ticker_and_name(t, names.get(t)),
                sectors.get(t, "Unknown"),
                round(ts, 2),
                round(fs, 2),
                round(sc, 2),
                round(comp, 2),
            ]
        )
    rows.sort(key=lambda r: r[-1], reverse=True)

    print_panel(
        "Market Regime (Weekend)",
        f"Score: {regime_score}/3 | Breadth SMA50: {breadth:.1f}%",
    )
    print_table(
        "Watchlist (pre-ranked by CompositeRank)",
        ["Ticker — Name", "Sector", "Tech", "Fund", "Sector", "Composite"],
        rows[:25],
    )


def daily_after_close(
    universe: list, initial_equity: float = 100000.0, no_progress: bool = False
):
    info(
        "Daily workflow: update technicals, triggers, select, size, and manage live positions."
    )
    state = load_state(default_cash=initial_equity)

    ind, sectors, names = {}, {}, {}
    with _make_progress(disabled=no_progress) as progress:
        t_prices = progress.add_task("Prices & indicators", total=len(universe))
        for t in universe:
            p = dfetch.get_price_history(t, lookback_days=1500, refresh=False)
            if not p.empty:
                ind[t] = compute_indicators(p)
                sectors[t] = dfetch.get_sector(t) or "Unknown"
                names[t] = dfetch.get_company_name(t) or ""
            progress.advance(t_prices)
    if not ind:
        error("No price/indicator data today.")
        return

    liq = liquidity_screen(ind)
    tech = technical_screen(ind)

    idx = dfetch.get_index_history(lookback_days=1500, refresh=False)
    idx_ind = compute_indicators(idx)
    idx_ret20 = (
        idx_ind["Close"].pct_change(config.SECTOR_RS_LOOKBACK).iloc[-1]
        if len(idx_ind) >= 21
        else np.nan
    )

    unique_secs = list(set(sectors.values()))
    with _make_progress(disabled=no_progress) as progress:
        t_sec = progress.add_task("Sector strength", total=len(unique_secs))
        sec_returns = {}
        for sec in unique_secs:
            members = [
                t
                for t, s in sectors.items()
                if s == sec and t in ind and len(ind[t]) >= 25
            ]
            if members:
                import pandas as pd

                rlist = [
                    ind[t]["Close"].pct_change(config.SECTOR_RS_LOOKBACK).iloc[-1]
                    for t in members
                ]
                rlist = [r for r in rlist if pd.notna(r)]
                if rlist:
                    import pandas as pd

                    sec_returns[sec] = float(
                        np.nanmean(rlist) - (idx_ret20 if pd.notna(idx_ret20) else 0.0)
                    )
            progress.advance(t_sec)
    sec_pct = (
        (pd.Series(sec_returns).rank(pct=True) * 100.0).to_dict()
        if sec_returns
        else {s: 50.0 for s in unique_secs}
    )

    rs_pct_within = {}
    for sec in unique_secs:
        members = [
            t for t, s in sectors.items() if s == sec and t in ind and len(ind[t]) >= 25
        ]
        if not members:
            continue
        ser = pd.Series(
            {
                t: ind[t]["Close"].pct_change(config.SECTOR_RS_LOOKBACK).iloc[-1]
                for t in members
            }
        ).dropna()
        if ser.empty:
            continue
        ranks = ser.rank(pct=True) * 100.0
        for t in members:
            rs_pct_within[t] = float(ranks.get(t, np.nan))

    # Candidate pool: liquidity + uptrend posture
    selection_pool = {
        t: ind[t]
        for t in ind
        if liq.get(t, False) and tech.get(t, {}).get("PostureUptrend", False)
    }

    # Fundamentals: cached/static (ignore manual overrides to mirror backtest default)
    fund_raw = {
        t: dfetch.get_fundamentals_cached_or_fetch(t, compute_fundamental_metrics, None)
        for t in selection_pool.keys()
    }
    fs_map = sector_relative_scores(fund_raw, sectors)

    dfetch.persist_fs_to_cache(fs_map, sectors)
    fs_pct = {}
    sec_groups = {}
    for t in selection_pool.keys():
        sec = sectors.get(t, "Unknown")
        sec_groups.setdefault(sec, []).append((t, fs_map.get(t, 0.0)))
    for sec, items in sec_groups.items():
        ser = pd.Series({t: fs for t, fs in items})
        ranks = ser.rank(pct=True) * 100.0
        for t, _ in items:
            fs_pct[t] = float(ranks.get(t, 0.0))
    # Gate on FS absolute + sector-top-percentile (same thresholds as backtest's static mode)
    selection_pool = {
        t: selection_pool[t]
        for t in selection_pool.keys()
        if (fs_map.get(t, 0.0) >= config.FUNDAMENTAL_MIN_FS)
        and (fs_pct.get(t, 0.0) >= config.FUNDAMENTAL_MIN_FS_SECTOR_TOP)
    }

    # Triggers: compute D1/D2/DB55 just like backtest
    triggers = {}
    for t in selection_pool:
        sub = ind[t]
        triggers[t] = {
            "D1": bool(trigger_D1(sub)),
            "D2": bool(trigger_D2(sub)),
            "DB55": bool(trigger_DB55(sub)),
        }

    candidates = select_candidates(
        selection_pool, sectors, triggers, sec_pct, rs_pct_within, fs_map
    )

    # Mark to market and risk controls
    dt = min([df.index[-1] for df in ind.values()])
    equity, dd = mark_to_market(state, ind, dt=dt)
    trades_today = []

    # Regime gating identical to backtest
    breadth = 100.0 * np.mean(
        [float(df.iloc[-1]["Close"] > df.iloc[-1]["EMA50"]) for df in ind.values()]
    )
    i = len(idx_ind) - 1
    slope = (
        (
            idx_ind["SMA200"].iloc[i]
            - idx_ind["SMA200"].iloc[i - config.REGIME_SLOPE_WINDOW]
        )
        if i >= config.REGIME_SLOPE_WINDOW
        else 0.0
    )
    cond1 = (idx_ind.iloc[-1]["Close"] > idx_ind.iloc[-1]["SMA200"]) and (slope > 0)
    cond2 = idx_ind.iloc[-1]["Close"] > idx_ind.iloc[-1]["EMA50"]
    cond3 = breadth >= config.BREADTH_MIN_PCT_ABOVE_SMA50
    regime_score = int(cond1) + int(cond2) + int(cond3)

    # Print Market Regime (same as main.py)
    r_txt = (
        f"Cond1 (Idx>SMA200 & SMA200↑): {'✔' if cond1 else '✖'}\n"
        f"Cond2 (Idx>EMA50): {'✔' if cond2 else '✖'}\n"
        f"Cond3 (Breadth≥{config.BREADTH_MIN_PCT_ABOVE_SMA50:.0f}%): {'✔' if cond3 else '✖'}\n"
        f"Score: {regime_score}/3   |   Breadth: {breadth:.1f}%"
    )
    print_panel("Market Regime", r_txt)

    # === Preview: IF a trigger fires today, what enters the action list first? ===
    # Treat every stock in selection_pool as if it had a trigger (D1=True).
    fake_triggers = {
        t: {"D1": True, "D2": False, "DB55": False} for t in selection_pool
    }

    # Reuse your normal candidate ranking (CompositeRank ordering, sector RS gates, FS gates, etc.)
    hypo_candidates = select_candidates(
        selection_pool, sectors, fake_triggers, sec_pct, rs_pct_within, fs_map
    )  # list of (t, trig_dict, sector) already sorted in your usual order

    # Apply the same *action-list* gates (capacity by regime, sector cap, correlation to held),
    # but DO NOT execute or cash-check; we only preview the order and size.
    from collections import Counter

    held = set(state.positions.keys())
    by_sec = Counter(sectors.get(t, "Unknown") for t in held)

    capacity = max(1, config.MAX_CONCURRENT_POSITIONS)

    def _max_corr_with_held_local(ticker):
        # Same idea as portfolio correlation check, but local/small for preview
        import pandas as pd

        if not held:
            return 0.0
        df = pd.DataFrame()
        for ht in held:
            if ht in ind and not ind[ht].empty:
                df[ht] = (
                    ind[ht]["Close"]
                    .pct_change()
                    .tail(config.CORRELATION_LOOKBACK)
                    .reset_index(drop=True)
                )
        if ticker not in ind or ind[ticker].empty or df.empty:
            return float("nan")
        df[ticker] = (
            ind[ticker]["Close"]
            .pct_change()
            .tail(config.CORRELATION_LOOKBACK)
            .reset_index(drop=True)
        )
        try:
            return float(df.corr().loc[ticker, list(held)].max())
        except Exception:
            return float("nan")

    preview_rows = []
    for t, trig, sec in hypo_candidates:
        if capacity <= 0:
            break

        # Sector cap gate
        if by_sec[sec] >= config.MAX_PER_SECTOR:
            continue

        # Correlation-to-held gate
        mc = _max_corr_with_held_local(t)
        if mc == mc and mc > config.MAX_CORRELATION:
            continue

        # Vol-adjusted size (equity-based), no cash check (preview only)
        df_t = ind.get(t)
        if df_t is None or df_t.empty:
            continue
        last = df_t.iloc[-1]
        entry = float(last["Close"])
        atr = float(last["ATR14"])
        swing_low = float(df_t["Low"].tail(10).min())
        init_stop = max(entry - 2 * atr, swing_low - 0.5 * atr)
        sh = position_size(equity, entry, init_stop)
        sh = cap_weight(sh, entry, equity)

        preview_rows.append(
            [
                fmt_ticker_and_name(t, names.get(t)),
                sec,
                f"{entry:.2f}",
                f"{init_stop:.2f}",
                int(sh),
            ]
        )

        by_sec[sec] += 1
        capacity -= 1

    print_table(
        "Preview (IF Trigger Fires): Action Order & Sizing",
        ["Ticker — Name", "Sector", "Entry", "InitStop", "Shares"],
        preview_rows or [["(none)", "", "", "", ""]],
    )
    # === End preview ===

    # Debug: Excluded by Sector RS gate (passed liquidity+posture+FS+triggers but sector RS below threshold)
    excluded_sector = []
    for t in selection_pool.keys():
        tr = triggers.get(t, {})
        if not (tr.get("D1") or tr.get("D2") or tr.get("DB55")):
            continue
        if rs_pct_within.get(t, 50.0) < 70.0:
            continue
        sec2 = sectors.get(t, "Unknown")
        sec_pct_val = sec_pct.get(sec2, 50.0)
        if sec_pct_val < config.SECTOR_TOP_PERCENTILE:
            excluded_sector.append(
                [fmt_ticker_and_name(t, names.get(t)), sec2, f"{sec_pct_val:.1f}%"]
            )

    print_table(
        "Excluded (Sector RS Gate)",
        ["Ticker — Name", "Sector", "SectorRS%"],
        excluded_sector,
    )

    if dd <= -config.PORTFOLIO_MAX_DRAWDOWN and state.positions:
        for t, p in list(state.positions.items()):
            px = float(ind[t].loc[:dt].iloc[-1]["Close"])
            state.cash += p.shares * px
            trades_today.append(
                {
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Ticker": t,
                    "Action": "SELL",
                    "Price": px,
                    "Shares": p.shares,
                    "Reason": "MAX_DD",
                }
            )
            state.positions.pop(t, None)
        state.cooldown = config.DRAWDOWN_COOLDOWN_DAYS

    if state.positions:
        state, exec_mgmt = manage_positions(state, ind, dt=dt)
        trades_today.extend(exec_mgmt)

    if state.cooldown > 0:
        state.cooldown -= 1
        entry_actions, mgmt_actions = [], []
    else:
        entry_actions, mgmt_actions = build_action_list(
            state,
            selection_pool,
            candidates,
            sectors,
            equity_for_sizing=equity,
            regime_score=regime_score,
        )

    # Execute entries at today's close (MOC), mirroring backtest
    # Debug: Excluded (Correlation Gate) — candidates dropped because max corr with held > threshold
    excluded_corr = []
    held = set(state.positions.keys())
    if held:
        import pandas as pd

        for t, trig, sec in candidates:
            if any(e.get("Ticker") == t for e in entry_actions):
                continue
            df_corr = pd.DataFrame()
            for ht in held:
                if ht in ind and not ind[ht].empty:
                    df_corr[ht] = (
                        ind[ht]["Close"]
                        .pct_change()
                        .tail(config.CORRELATION_LOOKBACK)
                        .reset_index(drop=True)
                    )
            if t not in ind or ind[t].empty or df_corr.empty:
                continue
            df_corr[t] = (
                ind[t]["Close"]
                .pct_change()
                .tail(config.CORRELATION_LOOKBACK)
                .reset_index(drop=True)
            )
            try:
                mc = float(df_corr.corr().loc[t, list(held)].max())
            except Exception:
                mc = float("nan")
            if mc == mc and mc > config.MAX_CORRELATION:
                excluded_corr.append(
                    [
                        fmt_ticker_and_name(t, names.get(t)),
                        sectors.get(t, "Unknown"),
                        f"{mc:.2f}",
                    ]
                )

    print_table(
        "Excluded (Correlation Gate)",
        ["Ticker — Name", "Sector", "MaxCorrWithHeld"],
        excluded_corr,
    )

    state, exec_entries = apply_entries(state, ind, entry_actions, dt=dt)
    trades_today.extend(exec_entries)

    equity, dd = mark_to_market(state, ind, dt=dt)
    state.last_date = dt.strftime(config.DATE_FORMAT)
    save_state(state)

    from portfolio import _append_history, _append_trades

    _append_history(dt, equity)
    _append_trades(trades_today)

    rows_state = summarize_state(state, ind)
    print_table(
        "Portfolio Status",
        ["Ticker — Name", "Close", "Entry", "Stop", "Trail", "Shares", "Sector"],
        rows_state,
    )

    next_actions = []
    for a in entry_actions:
        nm = names.get(a["Ticker"], "")
        disp = f"{a['Ticker']} — {nm}" if nm else a["Ticker"]
        next_actions.append(
            [
                disp,
                a["Action"],
                f"{a['Price']:.2f}",
                a["Shares"],
                f"Stop={a['Stop']:.2f}",
            ]
        )
    for a in mgmt_actions:
        nm = names.get(a["Ticker"], "")
        disp = f"{a['Ticker']} — {nm}" if nm else a["Ticker"]
        if a["Action"] == "RAISE_TRAIL":
            next_actions.append([disp, "RAISE_TRAIL", f"{a['To']:.2f}", "-", "-"])
        else:
            next_actions.append([disp, a["Action"], "-", "-", "-"])
    print_table(
        "Next-Day Action List",
        ["Ticker — Name", "What", "Price/To", "Shares", "Notes"],
        next_actions,
    )


def monthly_dashboard():
    from reporting import print_table, print_panel

    try:
        eq = pd.read_csv("cache/equity_history.csv")
        eq["Date"] = pd.to_datetime(eq["Date"])
        eq = eq.sort_values("Date")
    except Exception:
        print_panel("Dashboard", "No equity history yet.")
        return
    eq["Ret"] = eq["Equity"].pct_change()
    cagr = (eq["Equity"].iloc[-1] / eq["Equity"].iloc[0]) ** (
        252.0 / max(len(eq) - 1, 1)
    ) - 1
    vol = eq["Ret"].std() * (252**0.5)
    sharpe = (eq["Ret"].mean() / (eq["Ret"].std() + 1e-9)) * (252**0.5)
    dd = (eq["Equity"] / eq["Equity"].cummax() - 1.0).min()
    try:
        trades = pd.read_csv("cache/trades_log.csv")
        buys = trades[trades["Action"] == "BUY"].shape[0]
        sells = trades[trades["Action"] == "SELL"].shape[0]
    except Exception:
        buys, sells = 0, 0
    rows = [
        ["Rolling Sharpe (approx)", f"{sharpe:.2f}"],
        ["Max Drawdown", f"{dd:.2%}"],
        ["CAGR (approx)", f"{cagr:.2%}"],
        ["Vol (ann.)", f"{vol:.2%}"],
        ["# Buys", buys],
        ["# Sells", sells],
    ]
    print_table("Strategy Health (Monthly Review)", ["Metric", "Value"], rows)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", type=str, default="universe.txt")
    p.add_argument(
        "--equity",
        type=float,
        default=100000.0,
        help="Initial equity if portfolio is new",
    )
    p.add_argument("--daily", action="store_true")
    p.add_argument("--weekly", action="store_true")
    p.add_argument("--monthly", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p


def main(args):
    universe = dfetch.load_universe(args.universe)
    if args.weekly:
        weekend_refresh(universe, no_progress=args.no_progress)
    if args.daily:
        daily_after_close(
            universe, initial_equity=args.equity, no_progress=args.no_progress
        )
    if args.monthly:
        monthly_dashboard()


if __name__ == "__main__":
    main(build_parser().parse_args())

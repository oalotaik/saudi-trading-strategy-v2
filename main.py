# main.py — updated with price caching CLI, company names, color-coding, plotting & CSV export
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
import data_fetcher as dfetch

from technicals import compute_indicators, technical_posture, trigger_D1, trigger_D2
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

# Progress UI (disabled with --no-progress)
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


# -----------------------------
# Progress factory
# -----------------------------
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


# -----------------------------
# Simple helpers for regime/breadth
# -----------------------------
def compute_breadth(ind: dict) -> float:
    """% of universe with Close > EMA50 (latest bar)."""
    flags = []
    for t, df in ind.items():
        if df.empty:
            continue
        last = df.iloc[-1]
        if "EMA50" in last and "Close" in last:
            flags.append(float(last["Close"] > last["EMA50"]))
    return 100.0 * (np.mean(flags) if flags else 0.0)


def market_regime(idx_ind: pd.DataFrame, breadth_pct: float) -> dict:
    """
    3-condition regime:
      1) Index Close > SMA200 and SMA200 slope up (lookback REGIME_SLOPE_WINDOW)
      2) Index Close > EMA50
      3) Universe breadth >= BREADTH_MIN_PCT_ABOVE_SMA50
    """
    if idx_ind is None or idx_ind.empty:
        return {"Cond1": False, "Cond2": False, "Cond3": False, "Score": 0}
    i = len(idx_ind) - 1
    if i < 0:
        return {"Cond1": False, "Cond2": False, "Cond3": False, "Score": 0}

    last = idx_ind.iloc[i]
    if i >= config.REGIME_SLOPE_WINDOW:
        slope = (
            idx_ind["SMA200"].iloc[i]
            - idx_ind["SMA200"].iloc[i - config.REGIME_SLOPE_WINDOW]
        )
    else:
        slope = 0.0

    cond1 = (last["Close"] > last["SMA200"]) and (slope > 0)
    cond2 = last["Close"] > last["EMA50"]
    cond3 = breadth_pct >= config.BREADTH_MIN_PCT_ABOVE_SMA50
    score = int(cond1) + int(cond2) + int(cond3)
    return {"Cond1": cond1, "Cond2": cond2, "Cond3": cond3, "Score": score}


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", type=str, default="universe.txt")
    p.add_argument("--verbose", action="store_true")

    # Price cache flag (requirement #1)
    p.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Force fresh price download, ignoring the 1-day cache (default uses cached if fresh).",
    )

    p.add_argument(
        "--backtest",
        action="store_true",
        help="Run simplified per-ticker backtest (original).",
    )
    p.add_argument(
        "--portfolio-backtest",
        action="store_true",
        help="Run portfolio-level backtest (updated).",
    )
    p.add_argument(
        "--bt-start",
        type=str,
        default="2022-01-01",
        help="Portfolio backtest start date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--bt-end",
        type=str,
        default="2025-02-06",
        help="Portfolio backtest end date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--bt-fundamentals",
        type=str,
        default="static",
        choices=["none", "static"],
        help="Fundamentals gating in portfolio backtest: none (no fundamentals) or static (use current FS; look-ahead).",
    )
    p.add_argument(
        "--equity",
        type=float,
        default=100000.0,
        help="Starting equity for sizing/backtests.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Plot equity curve and save under ./output when running a backtest.",
    )
    p.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars and spinners."
    )
    return p


# -----------------------------
# Universe
# -----------------------------
def load_universe(path: str) -> list:
    return dfetch.load_universe(path)


# -----------------------------
# Main pipeline
# -----------------------------
def main(args):
    if args.verbose:
        info(f"Universe file: {args.universe}")

    universe = load_universe(args.universe)
    if not universe:
        error("Universe is empty or file not found.")
        return

    # 1) Prices & indicators
    info(
        "Fetching price data and computing indicators (cached for 1 day unless --refresh-prices is set)..."
    )
    price = {}
    ind = {}
    sectors = {}
    names = {}
    with _make_progress(disabled=args.no_progress) as progress:
        t_prices = progress.add_task("Prices & indicators", total=len(universe))
        for t in universe:
            try:
                p = dfetch.get_price_history(
                    t, lookback_days=1500, refresh=args.refresh_prices
                )
                if p.empty:
                    warn(f"No price data for {t}. Skipping.")
                    progress.advance(t_prices)
                    continue
                p_ind = compute_indicators(p)
                price[t] = p
                ind[t] = p_ind
                sectors[t] = dfetch.get_sector(t) or "Unknown"
                names[t] = dfetch.get_company_name(t) or ""
            except Exception as e:
                warn(f"Failed {t}: {e}")
            finally:
                progress.advance(t_prices)

    if not ind:
        error("No price/indicator data; cannot proceed.")
        return

    # 2) Index & regime
    info(f"Fetching index ({config.INDEX_TICKER}) and computing market regime...")
    with _make_progress(disabled=args.no_progress) as progress:
        t_reg = progress.add_task("Market regime", total=1)
        idx = dfetch.get_index_history(lookback_days=1500, refresh=args.refresh_prices)
        if idx.empty:
            error("Index data not available; cannot compute regime.")
            return
        idx_ind = compute_indicators(idx)
        breadth = compute_breadth(ind)
        regime = market_regime(idx_ind, breadth)
        progress.advance(t_reg)

    # Regime panel
    r_txt = (
        f"Cond1 (Idx>SMA200 & SMA200↑): {'✔' if regime['Cond1'] else '✖'}\n"
        f"Cond2 (Idx>EMA50): {'✔' if regime['Cond2'] else '✖'}\n"
        f"Cond3 (Breadth≥{config.BREADTH_MIN_PCT_ABOVE_SMA50:.0f}%): {'✔' if regime['Cond3'] else '✖'}\n"
        f"Score: {regime['Score']}/3   |   Breadth: {breadth:.1f}%"
    )
    print_panel("Market Regime", r_txt)

    # 3) Fundamentals (cached; FS computed later by sector_relative_scores)
    info("Computing/caching fundamentals & FS...")
    try:
        manual_df = pd.read_csv(config.MANUAL_FUNDAMENTALS_CSV)
    except Exception:
        manual_df = pd.DataFrame(columns=["ticker", "metric", "value", "period"])

    fund_data = {}
    with _make_progress(disabled=args.no_progress) as progress:
        t_fund = progress.add_task("Fundamentals & FS", total=len(universe))
        for t in universe:
            try:
                fund_data[t] = dfetch.get_fundamentals_cached_or_fetch(
                    t, compute_func=compute_fundamental_metrics, manual_df=manual_df
                )
            except Exception as e:
                warn(f"Fundamentals failed for {t}: {e}")
            finally:
                progress.advance(t_fund)

    # Sector-relative FS (0..100)
    fs_map = sector_relative_scores(fund_data, sectors)

    # 4) Liquidity & technical screens
    info("Running liquidity screen...")
    liq_pass = liquidity_screen(ind)

    info("Running technical screen...")
    with _make_progress(disabled=args.no_progress) as progress:
        t_tech = progress.add_task("Technical screening", total=1)
        tech_res = technical_screen(ind)  # dict per ticker with PostureUptrend, D1, D2
        progress.advance(t_tech)

    # 5) Sector strength (20d return vs index) for sector score
    info("Computing sector strength (20d RS vs index) ...")
    sec_returns = {}
    unique_secs = list(set(sectors.values()))
    with _make_progress(disabled=args.no_progress) as progress:
        t_sec = progress.add_task("Sector strength", total=len(unique_secs))
        # Precompute 20d index return
        idx_ret20 = (
            idx_ind["Close"].pct_change(20).iloc[-1] if len(idx_ind) >= 21 else np.nan
        )
        for sec in unique_secs:
            members = [
                t
                for t, s in sectors.items()
                if s == sec and t in ind and len(ind[t]) >= 25
            ]
            if members:
                rlist = []
                for t in members:
                    r = ind[t]["Close"].pct_change(20).iloc[-1]
                    if pd.notna(r):
                        rlist.append(r)
                if rlist:
                    sec_returns[sec] = float(
                        np.nanmean(rlist) - (idx_ret20 if pd.notna(idx_ret20) else 0.0)
                    )
            progress.advance(t_sec)

    # Convert sector RS to percentiles (higher is better)
    if sec_returns:
        ser_sec = pd.Series(sec_returns)
        sec_pct = (ser_sec.rank(pct=True) * 100.0).to_dict()
    else:
        sec_pct = {s: 50.0 for s in unique_secs}

    # 6) Build candidate ranking table (ticker + company name)
    rank_rows = []
    rs_pct_within_sector = {}  # 20d return percentile within each sector
    for sec in unique_secs:
        members = [
            t for t, s in sectors.items() if s == sec and t in ind and len(ind[t]) >= 25
        ]
        if not members:
            continue
        ser = pd.Series({t: ind[t]["Close"].pct_change(20).iloc[-1] for t in members})
        ser = ser.dropna()
        if ser.empty:
            continue
        ranks = ser.rank(pct=True) * 100.0
        for t in members:
            rs_pct_within_sector[t] = float(ranks.get(t, np.nan))

    for t in universe:
        if t not in ind or ind[t].empty:
            continue
        df = ind[t]
        # Tech score uses recent window + RS within sector
        ts = tech_score(df, rs_pct_within_sector.get(t, 50.0))
        fs = fs_map.get(t, 0.0)
        sscore = sec_pct.get(sectors.get(t, "Unknown"), 50.0)
        comp = composite_rank(ts, fs, sscore)
        trigs = tech_res.get(t, {})
        trig_str = f"D1={trigs.get('D1', False)}, D2={trigs.get('D2', False)}"
        rank_rows.append(
            [
                fmt_ticker_and_name(t, names.get(t)),
                sectors.get(t, "Unknown"),
                round(ts, 2),
                round(fs, 2),
                round(sscore, 2),
                round(comp, 2),
                trig_str,
            ]
        )

    # Sort by composite desc
    rank_rows.sort(key=lambda r: r[5], reverse=True)
    print_table(
        "Candidate Ranking",
        [
            "Ticker — Name",
            "Sector",
            "TechScore",
            "FundScore",
            "SectorScore",
            "Composite",
            "Triggers",
        ],
        rank_rows[:10],
    )

    # 7) Show screening overview (pass/fail per stage) with color coding
    overview = []
    for t in universe:
        lp = "PASS" if liq_pass.get(t, False) else "FAIL"
        tr = tech_res.get(t, {})
        up = "PASS" if bool(tr.get("PostureUptrend", False)) else "FAIL"
        fs_val = fs_map.get(t, 0.0)
        fs_ok = (
            "PASS" if fs_val >= getattr(config, "FUNDAMENTAL_MIN_FS", 60.0) else "FAIL"
        )
        overview.append(
            [
                fmt_ticker_and_name(t, names.get(t)),
                sectors.get(t, "Unknown"),
                colorize_status(lp),
                f"{fs_val:.1f} ({colorize_status(fs_ok)})",
                colorize_status(up),
            ]
        )
    print_table(
        "Screening Overview",
        ["Ticker — Name", "Sector", "Liquidity", "FS", "TechUptrend"],
        overview,
    )

    # 8) Final selection (pass all + top composite, sector-capped)
    # NOTE: selection still uses raw values (not strings with markup)
    selected = []
    by_sector = {}
    max_names = config.MAX_CONCURRENT_POSITIONS
    # We map from display string back to ticker for sizing
    comp_map = {row[0].split(" — ")[0]: row for row in rank_rows}
    for row in rank_rows:
        t_display = row[0]
        t = t_display.split(" — ")[0]
        sec = row[1]
        if len(selected) >= max_names:
            break
        if not liq_pass.get(t, False):
            continue
        if fs_map.get(t, 0.0) < getattr(config, "FUNDAMENTAL_MIN_FS", 60.0):
            continue
        if not tech_res.get(t, {}).get("PostureUptrend", False):
            continue
        if sec not in by_sector:
            by_sector[sec] = 0
        if by_sector[sec] >= config.MAX_PER_SECTOR:
            continue
        selected.append(t)
        by_sector[sec] += 1

    if not selected:
        warn("No stocks passed all stages today under current thresholds.")
    else:
        pretty = ", ".join([fmt_ticker_and_name(t, names.get(t)) for t in selected])
        print_panel("Selected Stocks", pretty)

    # 9) Risk-based sizing preview
    sizing_rows = []
    for t in selected:
        df = ind[t]
        last = df.iloc[-1]
        entry = float(last["Close"])
        atr = float(last["ATR14"])
        # swing low stop heuristic: min low over last 10 bars
        swing_low = float(df["Low"].iloc[-10:].min())
        init_stop = max(entry - 2 * atr, swing_low - 0.5 * atr)
        shares = position_size(args.equity, entry, init_stop)
        shares = cap_weight(shares, entry, args.equity)
        sizing_rows.append(
            [
                fmt_ticker_and_name(t, names.get(t)),
                sectors.get(t, "Unknown"),
                f"{entry:.2f}",
                f"{init_stop:.2f}",
                int(shares),
            ]
        )
    if sizing_rows:
        print_table(
            "Risk-Based Position Sizing",
            ["Ticker — Name", "Sector", "Entry", "InitStop", "Shares"],
            sizing_rows,
        )

    # 10) Optional: simplified per-ticker backtest (original) — unchanged

    # 11) Optional: portfolio-level backtest (updated) + plotting & CSV export
    if args.portfolio_backtest:
        from backtest import portfolio_backtest

        print_panel("Portfolio Backtest", "Running portfolio-level simulation...")
        with _make_progress(disabled=args.no_progress) as progress:
            t_pbt = progress.add_task("Portfolio simulation", total=1)
            res = portfolio_backtest(
                price,
                ind,
                sectors,
                idx_ind,
                args.bt_start,
                args.bt_end,
                init_equity=args.equity,
                fund_filter_mode=args.bt_fundamentals,
            )
            progress.advance(t_pbt)

        rows = [
            [k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in res.stats.items()
        ]
        print_table("Portfolio Stats", ["Metric", "Value"], rows)

        # Last 10 equity points
        ec_rows = [
            [d.strftime(config.DATE_FORMAT), f"{res.equity_curve.loc[d]:.0f}"]
            for d in res.equity_curve.index[-10:]
        ]
        print_table("Equity Curve (last 10)", ["Date", "Equity"], ec_rows)

        # Recent trades
        trows = []
        for tr in res.trades[-20:]:
            trows.append(
                [
                    tr.date.strftime(config.DATE_FORMAT),
                    fmt_ticker_and_name(tr.ticker, names.get(tr.ticker)),
                    tr.action,
                    f"{tr.price:.2f}",
                    tr.shares,
                    tr.reason,
                ]
            )
        print_table(
            "Recent Trades (last 20)",
            ["Date", "Ticker — Name", "Action", "Price", "Shares", "Reason"],
            trows,
        )

        # === New: persist full trades CSV & plot equity curve ===
        os.makedirs("output", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trades_csv = os.path.join("output", f"trades_{ts}.csv")
        if res.trades:
            df_trades = pd.DataFrame(
                [
                    {
                        "date": tr.date.strftime(config.DATE_FORMAT),
                        "ticker": tr.ticker,
                        "company": names.get(tr.ticker, ""),
                        "action": tr.action,
                        "price": tr.price,
                        "shares": tr.shares,
                        "reason": tr.reason,
                    }
                    for tr in res.trades
                ]
            )
            df_trades.to_csv(trades_csv, index=False)
            info(f"Saved trades to {trades_csv}")

        if args.plot and not res.equity_curve.empty:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(10, 4))
            res.equity_curve.plot()
            plt.title("Equity Curve")
            plt.xlabel("Date")
            plt.ylabel("Equity")
            png_path = os.path.join("output", f"equity_curve_{ts}.png")
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close(fig)
            info(f"Saved equity curve plot to {png_path}")


# -----------------------------
if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)

import argparse
import pandas as pd
import numpy as np

import config
import data_fetcher as dfetch
from technicals import compute_indicators
from screening import liquidity_screen, technical_screen
from fundamentals import compute_fundamental_metrics, sector_relative_scores
from reporting import print_panel, print_table, info, warn, error

# NEW: rich progress imports
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
    """Factory for a consistent progress UI; set disabled=True to turn it off."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,  # clears bar/spinner when the task finishes
        disable=disabled,  # honor --no-progress
    )


def diagnose(args):
    tickers = dfetch.load_universe(args.universe)
    if not tickers:
        error("Universe is empty.")
        return

    # 1) Prices & indicators
    info("Fetching prices & indicators...")
    ind = {}
    sectors = {}
    with _make_progress(disabled=args.no_progress) as progress:
        t_prices = progress.add_task("Prices & indicators", total=len(tickers))
        for t in tickers:
            try:
                p = dfetch.get_price_history(t, lookback_days=500)
                if p.empty:
                    warn(f"No price for {t}")
                    progress.advance(t_prices)
                    continue
                ind[t] = compute_indicators(p)
                sectors[t] = dfetch.get_sector(t) or "Unknown"
            except Exception as e:
                warn(f"{t}: {e}")
            finally:
                progress.advance(t_prices)

    # 2) Fundamentals (cached)
    info("Computing fundamentals & FS...")
    try:
        manual_df = pd.read_csv(config.MANUAL_FUNDAMENTALS_CSV)
    except Exception:
        manual_df = pd.DataFrame(columns=["ticker", "metric", "value", "period"])

    fund_raw = {}
    with _make_progress(disabled=args.no_progress) as progress:
        t_fund = progress.add_task("Fundamentals & FS", total=len(tickers))
        for t in tickers:
            try:
                fund_raw[t] = dfetch.get_fundamentals_cached_or_fetch(
                    t, compute_func=compute_fundamental_metrics, manual_df=manual_df
                )
            except Exception as e:
                warn(f"Fund fail {t}: {e}")
            finally:
                progress.advance(t_fund)

    # 3) Sector-relative FS percentile
    info("Computing sector-relative FS percentiles...")
    fs_map = sector_relative_scores(fund_raw, sectors)
    sector_fs = {}
    for t, s in sectors.items():
        sector_fs.setdefault(s, []).append((t, fs_map.get(t, 0.0)))
    fs_pct = {}

    unique_secs = list(sector_fs.keys())
    with _make_progress(disabled=args.no_progress) as progress:
        t_fs = progress.add_task(
            "Sector FS percentiles", total=len(unique_secs) if unique_secs else 1
        )
        for s in unique_secs:
            items = sector_fs.get(s, [])
            if items:
                ser = pd.Series({t: fs for t, fs in items})
                ranks = ser.rank(pct=True) * 100.0
                for t, _ in items:
                    fs_pct[t] = float(ranks.loc[t])
            progress.advance(t_fs)

    # 4) Technical & Liquidity screen (no heavy loops inside; just run, then assemble table rows with a bar)
    info("Running technical & liquidity screens...")
    with _make_progress(disabled=args.no_progress) as progress:
        t_screens = progress.add_task("Screens", total=3)
        liq = liquidity_screen(ind)
        progress.advance(t_screens)
        tech = technical_screen(ind)
        progress.advance(t_screens)
        # assemble rows
        rows = []
        near_miss_pct = args.near_miss_pct
        # Build table rows (progress over tickers)
        t_rows = progress.add_task("Build report", total=len(tickers))
        for t in tickers:
            fs = fs_map.get(t, 0.0)
            fs_pass = fs >= config.FUNDAMENTAL_MIN_FS
            fs_near = (
                fs >= config.FUNDAMENTAL_MIN_FS * (1 - near_miss_pct / 100.0)
            ) and not fs_pass

            tech_post = tech.get(t, {}).get("PostureUptrend", False)
            tech_near = False
            if t in ind and not ind[t].empty:
                last = ind[t].iloc[-1]
                if (
                    last["EMA20"] > last["EMA50"] > last["SMA200"]
                    and abs(last["Close"] / last["EMA50"] - 1.0) <= 0.01
                ):
                    tech_near = True

            rows.append(
                [
                    t,
                    sectors.get(t, "Unknown"),
                    "PASS" if liq.get(t, False) else "FAIL",
                    f"{fs:.1f} ({'PASS' if fs_pass else 'NEAR' if fs_near else 'FAIL'})",
                    "PASS" if tech_post else ("NEAR" if tech_near else "FAIL"),
                    f"D1={tech.get(t, {}).get('D1', False)}, D2={tech.get(t, {}).get('D2', False)}",
                    f"FS pct in sector: {fs_pct.get(t, 0.0):.1f}%",
                ]
            )
            progress.advance(t_rows)

    print_table(
        "Diagnosis: Cross-Analysis of Screening Phases",
        [
            "Ticker",
            "Sector",
            "Liquidity",
            "Fundamentals",
            "Tech Posture",
            "Triggers",
            "FS Sector %",
        ],
        rows,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=str, default="universe.txt")
    ap.add_argument(
        "--phase",
        type=str,
        default="all",
        help="Reserved for future: choose which phase(s) to diagnose.",
    )
    ap.add_argument(
        "--near-miss-pct",
        type=float,
        default=10.0,
        help="Threshold for near-miss classification in % below the pass threshold.",
    )
    # NEW: toggle progress bars
    ap.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars and spinners."
    )
    args = ap.parse_args()
    diagnose(args)

# Saudi Trading Strategy — README

A rules-based position-trading system for TASI stocks. It combines **fundamentals (FS)**, **sector strength (RS)**, and **technicals** (D1/D2/DB55 setups), then enforces **correlation** and **sector caps**, and sizes positions by risk.

All console reports keep **“Ticker — Name”** next to every ticker.

---

## What’s new in this version
- **Incremental price updates** now depend on the **last `Date` in the CSV**, not file modified time. The loader backfills a **small buffer (5 days)**, appends only new rows, dedupes by date, and keeps the **full local history**.
- **FS is persisted** to cached fundamentals (`FS`, `fs_date`, `fs_sector`) after weekly/daily recomputation.
- **CompositeRank uses real FS** during daily ordering (FS still gates eligibility).
- **Sector RS lookback** is read from `config.SECTOR_RS_LOOKBACK` (default: 20d) rather than a hardcoded value.
- Added a lightweight **diagnostic** utility: `diagnose_quickcheck.py`.

---

## Strategy (high level)
1. **Universe** from `universe.txt` (one ticker per line, e.g., `1321.SR`).  
2. **Regime** filter using ^TASI.SR + breadth (% above SMA50).  
3. **Sectors** ranked by RS over `SECTOR_RS_LOOKBACK`; within-sector RS percentiles for members.  
4. **Fundamentals (FS)**: weighted, sector-relative score; must pass absolute and sector-top thresholds.  
5. **Technicals**: uptrend posture + setups (D1 / D2 / DB55).  
6. **CompositeRank** = weighted sum of TechScore, FundScore (FS), and SectorScore.  
7. **Selection**: apply correlation cap & sector caps, regime-aware capacity.  
8. **Risk & Orders**: volatility-adjusted sizing, MOC or next-day buy-stops, stops/trailing, reductions, time stops.

---

## Installation
```bash
pip install -U pandas numpy yfinance rich matplotlib
```

---

## Configuration (`config.py`)
Key items (not exhaustive):
- `INDEX_TICKER` (default: `^TASI.SR`)
- `SECTOR_RS_LOOKBACK` — sector and within-sector RS horizon (default 20)
- FS gates: `FUNDAMENTAL_MIN_FS`, `FUNDAMENTAL_MIN_FS_SECTOR_TOP`
- Liquidity filters: `MIN_AVG_DAILY_VALUE_SAR`, `MIN_AVG_DAILY_VOLUME`, `MIN_PRICE_SAR`
- Composite weights: `TECH_WEIGHT`, `FUND_WEIGHT`, `SECTOR_WEIGHT`
- Risk & caps: `RISK_PER_TRADE`, `MAX_POSITION_WEIGHT`, `MAX_CONCURRENT_POSITIONS`, `MAX_PER_SECTOR`
- Correlation filter: `CORRELATION_LOOKBACK`, `MAX_CORRELATION`
- **Note on `PRICE_CACHE_DAYS`**: now **optional failsafe**. Incremental updates key off the **last `Date`** in CSV, not mtime. Use `--refresh-prices` for a forced full re-download.

---

## Project structure
> Folders created during runs (e.g., `cache/`, `output/`) are **untracked by git**.

```
.
├─ backtest.py
├─ config.py
├─ data_fetcher.py
├─ diagnose.py
├─ diagnose_quickcheck.py           # new: quick cache & FS checker
├─ fundamentals.py
├─ main.py
├─ portfolio.py
├─ ranking.py
├─ reporting.py
├─ risk.py
├─ screening.py
├─ technicals.py
├─ workflow.py
├─ universe.txt                     # one ticker per line
├─ cache/                           # (runtime) local persistent cache
│  ├─ prices/                       # CSV per ticker (full history; appended incrementally)
│  ├─ fundamentals/                 # JSON per ticker (includes FS, fs_date, fs_sector)
│  └─ company_meta.json             # names/sectors map
└─ output/                          # (runtime) equity plots, trades CSVs
```

---

## Typical run order

### One-time hydration / overview
```bash
python main.py --universe universe.txt --refresh-prices
```

### Weekly (after the week ends)
```bash
python workflow.py --weekly
```
- Recomputes fundamentals → FS (sector-relative) and **persists FS** to cache.
- Updates sector RS and watchlist.

### Daily (after market close)
```bash
python workflow.py --daily
```
- **Incrementally** updates prices/indicators from the last `Date` in each CSV (with a small backfill buffer).
- Refreshes TS/technicals and **CompositeRank (uses actual FS)**.
- Applies correlation/sector caps, sizes positions, and prints the action list.

---

## Diagnostics

### Quick cache & FS check
```bash
python diagnose_quickcheck.py --universe universe.txt
```
Shows, per ticker: **Ticker — Name**, Cache Span, Last Date, **Appended Today?**, **FS**, fs_date, fs_sector.

### Full pipeline health check
```bash
python diagnose.py --universe universe.txt
```
Progressive checks for prices, indicators, screens, fundamentals, and near-miss thresholds.

---

## Backtest (optional)
```bash
python backtest.py --universe universe.txt
```
- Plots equity to `output/`.  
- (If you add the snippet from the docstring comment) saves **trades CSV** to `output/trades_*.csv`.

> Backtests are **not** part of `workflow.py` daily/weekly runs.

---

## Notes & troubleshooting
- **Manual CSV edits**: safe. The loader uses **last `Date`** to decide what to append and backfills a few days to capture vendor corrections.
- **No new rows today?** Probably a non-trading day; cache stays as-is.
- **Force full refresh**: add `--refresh-prices` to the command.
- **Missing names next to tickers?** Ensure `cache/company_meta.json` exists. Running `main.py` or `workflow.py --weekly` typically populates it.

---

## Disclaimer
This project is for research/education. Markets involve risk. Use at your own discretion.
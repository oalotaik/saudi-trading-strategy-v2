# Saudi Swing Strategy (Personal Use)

> **Note:** This repository is for **personal use**. I’m not seeking or accepting external contributions or issues.

This project contains my end-to-end pipeline for **regime-aware swing/position trading** on the Saudi market (Tadawul). The system is pragmatic: focus on **healthy regimes**, buy **leaders among leaders** on actionable triggers, and **cut risk fast**.

---

## What’s New (Aug 2025)

- **Workflow automation**: `workflow.py` now provides **weekly**, **daily**, and **monthly** routines.
  - **Weekly (weekend)**: refresh Fundamentals Score (FS), sector RS, market regime, and rebuild the watchlist (pre-ranked by CompositeRank).
  - **Daily (after close)**: update technicals, recompute triggers (**D1/D2/DB55**), apply **correlation** and **sector caps**, **regime-gated capacity**, risk-based sizing, and **MOC** entries. Manage live positions (raise trails, 2R reductions, time stop).
  - **Monthly**: a light health dashboard (Sharpe, MaxDD, etc.).
- **Backtest ↔ Workflow alignment**: live workflow mirrors the backtest selection logic:
  - Triggers include **D1**, **D2**, and **DB55**.
  - Entries are **MOC at today’s close** (no buy-stops).
  - **Regime gating** of capacity: **3/3 → full**, **2/3 → -1 slot**, **1/3 → 1 slot**, **0/3 → 0**.
  - Static fundamentals gate in workflow to match backtest default (no manual override during daily run).
- **Benchmark everywhere**: portfolio backtest prints a **side-by-side stat table** vs **^TASI.SR** and overlays the benchmark on the plot.
- **Improved equity plot**: thousands separator (SAR) on left axis, **% change** on right axis, grid, and **Max Drawdown annotation**.
- **Permanent, incremental price cache**: historical prices are saved permanently; new bars are **appended** (no daily full refetch). Weekend (Fri/Sat) spans are tolerated without warnings.
- **“Ticker — Name” in tables**: company name is printed alongside the ticker in reports.
- **Backtest end snapshot**: always prints an **Active Positions at End of Backtest** section (shows a table or a “None” panel).
- **Preview clarity**: in `main.py`, the sizing table is labeled **“What-If Position Sizing (Preview — no orders placed)”** to avoid confusion.

---

## Strategy Overview (unchanged core)

### 1) Market Regime & Exposure
Three daily checks on the index and breadth:
1. **Idx > SMA200 & SMA200 rising (20 bars slope)**  
2. **Idx > EMA50**  
3. **Breadth** ≥ threshold (**% of universe above SMA50**)

Capacity by regime score:
- **3/3** → full capacity (`MAX_CONCURRENT_POSITIONS`)
- **2/3** → capacity reduced by 1
- **1/3** → **1 slot**
- **0/3** → **no new entries**

A **portfolio max drawdown** triggers **flatten** and a **cooldown** period before re-entry.

### 2) Universe & Liquidity
From `universe.txt`. Minimum **20d value traded**, **20d volume**, and **min price** filters.

### 3) Sector & RS
- Only consider **top-percentile sectors** by 20d sector RS vs index.
- Within those sectors, require **within-sector RS ≥ 70th percentile**.

### 4) Entries (Any trigger can fire)
- **D1** (breakout confirmation), **D2** (pullback-and-go), **DB55** (Donchian breakout).  
- **Workflow and backtest both use MOC** fills for entries.

### 5) Risk & Allocation
- **Risk-per-trade** sizing off an **initial stop** (ATR/swing-low based).
- **Max position weight**, **correlation limit**, and **sector cap**.

### 6) Exits & Management
- **Initial stop** from entry; **scale ½ at +2R** and raise to **breakeven**.
- **Chandelier trail** (~20, 3×ATR).
- **Trend deterioration**: two closes below **EMA20**; **weakness**: Close < EMA50 and low ADX.
- **Time stop** after max holding days.

> Fundamentals: A static fundamentals gate is available (used by default in backtests and mirrored by the daily workflow for alignment). For unbiased historical tests consider turning it off.

---

## Updated Results

**Backtest window:** 2022‑01‑01 → 2025‑08‑14  
**Benchmark:** **^TASI.SR** (buy & hold)

**Portfolio Stats (vs Benchmark)**

| Metric | Portfolio | Benchmark (^TASI.SR) |
|:--:|:--:|:--:|
| **CAGR** | **0.1446** | **-0.0126** |
| **Vol** | **0.1192** | **0.1395** |
| **Sharpe** | **1.1926** | **-0.0210** |
| **MaxDD** | **-0.1046** | **-0.2781** |

**Notes**: The improvement (vs older settings) comes from allowing **1 slot at a 1/3 regime**, which increases participation while keeping risk controls intact. Flat equity near the end of the window is expected when the regime score drops to **0/3** (no new entries).
![Backtest results](https://github.com/oalotaik/saudi-swing-strategy/blob/main/assets/equity_curve_with_bench_20250817_142346.png)

---

## From‑Zero Sequence (Fresh Start Without Re‑Downloading History)

> Use this if you’ve **deleted state files** and want to restart live tracking **without refetching history**.

1) **Prep**
   - Ensure `universe.txt` (one ticker per line, e.g., `1321.SR`).
   - (Optional) `manual_fundamentals.csv` for overrides (ticker, metric, value, period).

2) **(Optional) One‑shot warm‑up** — only when you truly have no local prices:
   ```bash
   python main.py --refresh-prices --universe universe.txt
   ```
   - Downloads and saves **permanent** price history, then subsequent runs append **incrementally**.
   - Rebuilds local company **names/sectors** metadata and computes fundamentals (cached per `FUNDAMENTAL_CACHE_DAYS`).

3) **Weekly refresh (watchlist & sector RS)** — uses cached data:
   ```bash
   python workflow.py --weekly --universe universe.txt
   ```

4) **First daily run (initializes live state)**:
   ```bash
   python workflow.py --daily --universe universe.txt --equity 100000
   ```
   This creates and maintains:
   - `cache/portfolio_state.json` (cash/positions)
   - `cache/equity_history.csv` (daily equity)
   - `cache/trades_log.csv` (executed actions)

**Resetting only the portfolio state** (do **not** delete price/fundamental caches):
```bash
# Windows PowerShell
Remove-Item .\cache\portfolio_state.json -ErrorAction SilentlyContinue
Remove-Item .\cache\equity_history.csv -ErrorAction SilentlyContinue
Remove-Item .\cache	rades_log.csv -ErrorAction SilentlyContinue

# macOS/Linux
rm -f cache/portfolio_state.json cache/equity_history.csv cache/trades_log.csv
```
Then run the daily workflow again with `--equity` to seed cash.

---

## CLI Cheatsheet

**Backtest (with benchmark overlay & plot):**
```bash
python main.py --portfolio-backtest --bt-start 2022-01-01 --bt-end 2025-08-14 --equity 100000 --bt-fundamentals static --plot
```

**Daily workflow (after close):**
```bash
python workflow.py --daily --universe universe.txt --equity 100000
```

**Weekly refresh (weekend):**
```bash
python workflow.py --weekly --universe universe.txt
```

**Monthly dashboard:**
```bash
python workflow.py --monthly
```

---

## Data & Caching Behavior

- **Prices**: permanently cached; runs perform **incremental** appends only. Non‑trading‑day spans (Fri/Sat) are tolerated without errors.
- **Fundamentals**: cached for `FUNDAMENTAL_CACHE_DAYS` (default **7**); weekly run recomputes FS and sector‑relative percentiles.
- **Index ticker**: `^TASI.SR` as defined in `config.py`.

---

## Project Structure (key files)

```
.
├── workflow.py              # Daily/weekly/monthly automation (mirrors backtest logic)
├── main.py                  # End-to-end pipeline, screening/ranking, backtest runner & reporting
├── backtest.py              # Portfolio simulator (MOC entries; D1/D2/DB55; regime-gated capacity)
├── data_fetcher.py          # Permanent incremental price cache, fundamentals & company metadata
├── technicals.py            # Indicators & D1/D2/DB55 triggers
├── fundamentals.py          # FS computation + sector-relative scoring (cached)
├── ranking.py               # CompositeRank: TECH/FUND/SECTOR weights from config
├── screening.py             # Liquidity & posture
├── risk.py                  # Risk-based sizing, caps, correlations
├── reporting.py             # Rich TUI helpers; colored tables/panels
├── config.py                # Thresholds & knobs
├── universe.txt             # One ticker per line
└── cache/                   # Local persistent caches (prices, fundamentals, state)
```

---

## Caveats

- **No slippage/fees** modeled by default. Live results will differ.
- Yahoo Finance data gaps/outliers may occur; code uses as‑of logic for robustness.
- This is **not financial advice**; for research/education only.

---

## License & Contact

- **License**: Private / personal use.
- **Contributions**: Not accepted.
- **Contact**: Reach out privately if you have non‑code questions about the strategy.

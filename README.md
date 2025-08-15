# Saudi Swing Strategy (Personal Use)

> **Note:** This repository is for **personal use**. I’m not seeking or accepting external contributions or issues.

This project contains my end-to-end pipeline for **regime-aware swing/position trading** on the Saudi market. The focus is on **finding the strongest names during healthy market regimes**, entering on **breakouts or momentum-friendly pullbacks**, and **exiting quickly** when trend conditions deteriorate. The code is intentionally pragmatic rather than academic; the strategy description below is the main event, while the code is only here to make it run.

---

## Strategy Overview

### 1) Market Regime & Exposure
The portfolio opens new trades only when the **market tape** is supportive. Three signals are evaluated daily:
1. **Index above SMA200 with SMA200 slope rising** (trend confirmation)  
2. **Index above EMA50** (near-term support)  
3. **Breadth**: ≥ a threshold of the universe above SMA50 (internal strength)

- **3/3 signals true** → full capacity and full per-trade risk  
- **2/3 signals true** → reduced capacity and **half** per-trade risk  
- **<2/3** → **no new entries**  
- If portfolio drawdown hits a max threshold, **all positions are flattened** and a cooldown period begins before new entries are allowed.

### 2) Universe & Liquidity
From a user-specified universe, names must satisfy basic **liquidity & price** constraints (e.g., 20-day value traded, 20-day volume, minimum price). This keeps the portfolio implementable and avoids micro illiquidity.

### 3) Sector Strength & Relative Strength (RS)
- Compute **sector strength** (e.g., 20-day sector return vs the index) and only consider sectors in the top percentile bucket.  
- Within those sectors, rank tickers by **within-sector RS** and require **RS ≥ 70th percentile** for entry consideration. This pushes exposure into **leaders among leaders**.

### 4) Entries (Any Trigger Can Fire)
The strategy blends **breakout and pullback** styles to adapt to what the market offers:
- **D1 – Breakout**: Price confirming momentum with volume and trend filters in place.  
- **D2 – Pullback-and-Go**: A shallow pullback that holds above key MAs, then re-accelerates.  
- **DB55 – Donchian Breakout** (configurable lookback, default 55): Close reaches the highest high over the lookback window, with minimum ADX and volume confirmation and price above SMA200.

This trio covers **fresh breakouts**, **continuations after shallow dips**, and **persistent trend resumption**.

### 5) Position Sizing & Risk
- **Risk-per-trade** based sizing using **initial stop** (e.g., ATR-based or recent swing low).  
- **Max position weight** caps.  
- **Correlation filter** to avoid clustering similar names the same day.  
- **Sector cap** to prevent concentration.

### 6) Exits & Scaling
- **Initial stop** from entry (risk-based).  
- **Scale-out**: take partial profits at **+2R**, then raise stop to breakeven on the remainder.  
- **Chandelier trailing stop** (20, 3×ATR) to lock in trends.  
- **Trend exit**: **two closes below EMA20**.  
- **Weakness exit**: Close below EMA50 **and** ADX < 15.  
- **Time stop** after a maximum holding period.

These rules are designed to **let winners run** but **cut losers/laggards** without hesitation.

> **Important:** An optional “static” fundamentals gate is available for experimentation, but using today’s fundamentals to judge past trades can **introduce look-ahead bias**. For unbiased backtests use fundamentals gating **off**.

---

## Results (Backtest)

**Backtest window:** 2022-01-01 → 2025-02-06  
**Benchmark:** TASI’s CAGR during the same period ≈ **3.14%**

**Portfolio Stats**

| Metric | Value  |
|:------:|:------:|
| CAGR   | 0.0767 |
| Vol    | 0.1044 |
| Sharpe | 0.7597 |
| MaxDD  | -0.1047 |

**Brief interpretation**  
- **CAGR 7.67%**: Annualized growth rate of the equity curve over the backtest. This **outpaced TASI (~3.14%)** in the same window.  
- **Vol 10.44%**: Annualized volatility; reflects the swing in returns. Lower is generally smoother.  
- **Sharpe 0.76**: Return per unit of volatility (≈ excess return normalized by risk). Values around **0.7–1.0** are considered acceptable for active strategies.  
- **MaxDD -10.5%**: Worst peak-to-trough drawdown; indicates risk of loss from equity highs.

### Equity Curve
![Equity Curve](https://github.com/oalotaik/saudi-swing-strategy/blob/main/assets/equity_curve_20250815_131846.png)

> Backtests are not guarantees of future results. Slippage, fees, data errors, halts, and structural market shifts will impact live performance.

---

## Project Structure

```
.
├── main.py                  # CLI: end-to-end pipeline; screening, ranking, backtesting, reporting
├── backtest.py              # Portfolio simulator (regime-aware, entries/exits, risk, DD controls)
├── data_fetcher.py          # Data I/O with 1-day price cache + company names & sectors cache
├── screening.py             # Liquidity & technical screening helpers (diagnostics/overview)
├── technicals.py            # Indicators & entry triggers (EMA/SMA/RSI/ATR/ADX/Donchian, etc.)
├── fundamentals.py          # Fundamental score (FS) & sector-relative scoring (optional)
├── ranking.py               # Tech/Fund/Sector composite scoring
├── reporting.py             # Rich TUI helpers for colored tables/panels
├── risk.py                  # Position sizing, weight caps, simple correlation utilities
├── config.py                # All thresholds & knobs (lookbacks, risks, limits, gates)
├── universe.txt             # One ticker per line (e.g., 1321.SR); comments start with '#'
├── assets/                  # Git-tracked images for README (published plot lives here)
│   └── equity_curve.png
├── output/                  # Untracked artifacts (plots & CSVs generated locally)
├── cache/                   # Cached data (prices, fundamentals, metadata)
│   ├── prices/
│   └── fundamentals/
└── manual_fundamentals.csv  # Optional overrides (ticker, metric, value, period)
```


---

## Quick Start (Minimal)

> This section is intentionally short. The code is for personal use; only the essentials are listed for anyone who still wants to try it.

### 1) Python Env
- Python **3.10+** recommended
- Create a venv and install the basics:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  python -m pip install --upgrade pip wheel
  python -m pip install pandas numpy yfinance rich matplotlib
  ```
  > `matplotlib` is optional; it’s only needed if you want to save equity curve plots.

### 2) Universe & Config
- Put your tickers in `universe.txt`, one per line (e.g., `1321.SR`).  
- Adjust thresholds in `config.py` (liquidity, regime, risk, etc.).  
  - If your TASI index symbol differs on Yahoo, set `INDEX_TICKER` appropriately (e.g., `"^TASI"`).

### 3) Run a Diagnosis
```bash
python main.py --universe universe.txt
```

### 4) Portfolio Backtest
```bash
python main.py --universe universe.txt   --portfolio-backtest   --bt-start 2022-01-01 --bt-end 2025-02-06   --equity 100000   --plot
```
- Use `--refresh-prices` to ignore the **1-day cache** on prices.  
- Use `--bt-fundamentals none` for unbiased backtests (recommended) or `static` for experimentation.  
- If you want more history, add a CLI flag for `--lookback-days` (the default may be 500 in some versions).

**Outputs**
- CSV of trades: `output/trades_YYYYMMDD_HHMMSS.csv`  
- Equity plot (if matplotlib present): `output/equity_curve_YYYYMMDD_HHMMSS.png`

---

## Assumptions, Data, and Caveats

- Prices & company metadata use **Yahoo Finance**. Data quality varies across tickers/holidays and may have missing bars. The backtester uses **as-of** logic (last available close ≤ date) to handle gaps.  
- No transaction costs, taxes, or slippage are modeled by default unless you add them.  
- The optional fundamentals gate can introduce **look-ahead bias** in backtests.  
- This is **not financial advice**. It’s a personal research tool; use at your own risk.

---

## Contributions & License

- **Contributions:** Not accepted. This is a personal project.  
- **Issues/PRs:** Closed by default.  
- **License:** Private use. If you want to reuse any part, please reach out first.

---

## Contact

If you’ve cloned this for personal learning and have a question about the **strategy** (not the code internals), feel free to ask privately.

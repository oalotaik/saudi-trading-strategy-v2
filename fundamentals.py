# fundamentals.py  —  updated to handle Yahoo label variants for Tadawul tickers
from typing import Dict, Optional
import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _latest_series_value(frame: pd.DataFrame, row_label: str) -> Optional[float]:
    """Return newest non-null value from a single named row (Series) in a DataFrame."""
    if frame is None or frame.empty:
        return None
    try:
        row = frame.loc[row_label]
        if isinstance(row, pd.Series) and not row.empty:
            return float(row.dropna().iloc[0])
        return None
    except Exception:
        return None


def _latest_two_series_values(frame: pd.DataFrame, row_label: str):
    """Return newest and previous values from a single named row, if available."""
    if frame is None or frame.empty:
        return None, None
    try:
        row = frame.loc[row_label]
        if isinstance(row, pd.Series):
            vals = row.dropna().tolist()
            if len(vals) >= 2:
                return float(vals[0]), float(vals[1])
            elif len(vals) == 1:
                return float(vals[0]), None
    except Exception:
        pass
    return None, None


def _pct_growth(curr, prev):
    if curr is None or prev is None or prev == 0:
        return None
    return 100.0 * (curr - prev) / abs(prev)


# ---------------------------------------------------------------------------
# NEW: flexible label helpers for Yahoo balance sheet variants
# ---------------------------------------------------------------------------
def _pick_row(df: pd.DataFrame, label_options: list[str]) -> pd.Series:
    """Return first matching row Series (newest-first) among label_options, else empty Series."""
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    for lbl in label_options:
        if lbl in df.index:
            s = df.loc[lbl].dropna()
            if not s.empty:
                return s
    return pd.Series(dtype="float64")


def _latest_value_flex(df: pd.DataFrame, label_options: list[str]) -> Optional[float]:
    s = _pick_row(df, label_options)
    return float(s.iloc[0]) if not s.empty else None


def _latest_two_values_flex(df: pd.DataFrame, label_options: list[str]):
    s = _pick_row(df, label_options)
    if s.empty:
        return None, None
    vals = s.iloc[:2].tolist()
    if len(vals) == 1:
        return float(vals[0]), None
    return float(vals[0]), float(vals[1])


# Yahoo label variants commonly seen on Tadawul
EQ_LABELS = ["Common Stock Equity", "Stockholders Equity", "Total Stockholder Equity"]
CURR_ASSETS_LABELS = ["Current Assets", "Total Current Assets"]
CURR_LIAB_LABELS = ["Current Liabilities", "Total Current Liabilities"]
TOTAL_DEBT_LABELS = ["Total Debt"]
LONG_DEBT_LABELS = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
SHORT_DEBT_LABELS = ["Current Debt", "Current Debt And Capital Lease Obligation"]


# ---------------------------------------------------------------------------
def compute_fundamental_metrics(
    ticker: str,
    fin: pd.DataFrame,
    bs: pd.DataFrame,
    info: Dict,
    manual_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Compute the 12 metrics + improvement flags for a ticker.
    Notes:
      - This function intentionally leaves FS=None; sector_relative_scores computes FS later.
      - Uses flexible label mapping for equity/current items/debt to accommodate Yahoo variants.
    """
    out = {"ticker": ticker, "metrics": {}, "improvements": {}, "FS": None}

    def fin_latest(label):
        return _latest_series_value(fin, label)

    def fin_latest2(label):
        return _latest_two_series_values(fin, label)

    def bs_latest(label):
        return _latest_series_value(bs, label)

    def bs_latest2(label):
        return _latest_two_series_values(bs, label)

    # Manual overrides (optional)
    manual_map = {}
    if manual_df is not None and not manual_df.empty:
        msub = manual_df[manual_df["ticker"] == ticker]
        for _, r in msub.iterrows():
            manual_map[(r["metric"])] = float(r["value"])

    # -------------------------------
    # Income-statement driven metrics
    # -------------------------------
    rev = fin_latest("Total Revenue")
    cogs = fin_latest("Cost Of Revenue")

    gm = None
    if rev and rev != 0 and cogs is not None:
        gm = 100.0 * (rev - cogs) / rev
    out["metrics"]["Gross Margin"] = manual_map.get("Gross Margin", gm)

    ni = fin_latest("Net Income")
    nm = None
    if rev and rev != 0 and ni is not None:
        nm = 100.0 * (ni / rev)
    out["metrics"]["Net Margin"] = manual_map.get("Net Margin", nm)

    assets_curr, assets_prev = bs_latest2("Total Assets")
    roa = None
    if ni is not None and assets_curr is not None:
        denom = (
            assets_curr if assets_prev is None else 0.5 * (assets_curr + assets_prev)
        )
        if denom != 0:
            roa = 100.0 * (ni / denom)
    out["metrics"]["Return on Assets (ROA)"] = manual_map.get(
        "Return on Assets (ROA)", roa
    )

    at = None
    if rev is not None and assets_curr is not None:
        denom = (
            assets_curr if assets_prev is None else 0.5 * (assets_curr + assets_prev)
        )
        if denom != 0:
            at = rev / denom
    out["metrics"]["Asset Turnover"] = manual_map.get("Asset Turnover", at)

    # -------------------------------
    # Flexible balance-sheet metrics (label mismatch fix)
    # -------------------------------
    # Equity (book value proxy) & ROE denominator
    eq_curr, eq_prev = _latest_two_values_flex(bs, EQ_LABELS)

    roe = None
    if ni is not None and eq_curr is not None:
        denom = eq_curr if eq_prev is None else 0.5 * (eq_curr + eq_prev)
        if denom != 0:
            roe = 100.0 * (ni / denom)
    out["metrics"]["Return on Equity (ROE)"] = manual_map.get(
        "Return on Equity (ROE)", roe
    )

    # Current Ratio (Current Assets / Current Liabilities)
    ca = _latest_value_flex(bs, CURR_ASSETS_LABELS)
    cl = _latest_value_flex(bs, CURR_LIAB_LABELS)
    cr = None
    if ca is not None and cl and cl != 0:
        cr = ca / cl
    out["metrics"]["Current Ratio"] = manual_map.get("Current Ratio", cr)

    # Book Value (≈ equity)
    out["metrics"]["Book Value"] = manual_map.get("Book Value", eq_curr)

    # Debt-to-Equity: prefer Total Debt, else sum of Short+Long if available
    total_debt = _latest_value_flex(bs, TOTAL_DEBT_LABELS)
    if total_debt is None:
        sld = _latest_value_flex(bs, SHORT_DEBT_LABELS)
        ltd = _latest_value_flex(bs, LONG_DEBT_LABELS)
        if sld is not None or ltd is not None:
            total_debt = (sld or 0.0) + (ltd or 0.0)

    de = None
    if total_debt is not None and eq_curr and eq_curr != 0:
        de = total_debt / eq_curr
    out["metrics"]["Debt-to-Equity"] = manual_map.get("Debt-to-Equity", de)

    # -------------------------------
    # Valuation / growth
    # -------------------------------
    teps = info.get("trailingEps", None)
    so = info.get("sharesOutstanding", None)
    eps = teps
    if eps is None and ni is not None and so:
        if so != 0:
            eps = ni / so
    out["metrics"]["Earnings Per Share (EPS)"] = manual_map.get(
        "Earnings Per Share (EPS)", eps
    )

    pe = info.get("trailingPE", None)
    out["metrics"]["Price-to-Earnings (P/E)"] = manual_map.get(
        "Price-to-Earnings (P/E)", pe
    )

    rev_growth = info.get("revenueGrowth", None)
    if rev_growth is not None:
        rev_growth = (
            float(rev_growth) * 100.0 if abs(rev_growth) < 1.0 else float(rev_growth)
        )
    else:
        rev_curr, rev_prev = fin_latest2("Total Revenue")
        rev_growth = _pct_growth(rev_curr, rev_prev)
    out["metrics"]["Revenue Growth"] = manual_map.get("Revenue Growth", rev_growth)

    eq_growth = _pct_growth(eq_curr, eq_prev)
    out["metrics"]["Equity Growth"] = manual_map.get("Equity Growth", eq_growth)

    # -------------------------------
    # Improvements (simple)
    # -------------------------------
    out["improvements"]["Revenue Growth YoY > 0"] = (
        rev_growth is not None and rev_growth > 0
    )

    roe_prev = None
    if ni is not None and eq_prev is not None and eq_prev != 0:
        roe_prev = 100.0 * (ni / eq_prev)
    out["improvements"]["ROE improved YoY"] = (
        roe is not None and roe_prev is not None and roe > roe_prev
    )

    # Keep as-is to minimize changes; can be refined later to use prior-period debt/equity
    out["improvements"]["Debt/Equity decreased YoY"] = False

    # Gross margin trend (uses IS data)
    cogs_curr, cogs_prev = _latest_two_series_values(fin, "Cost Of Revenue")
    rev_curr, rev_prev = fin_latest2("Total Revenue")
    gm_prev = None
    if rev_prev and rev_prev != 0 and cogs_prev is not None:
        gm_prev = 100.0 * (rev_prev - cogs_prev) / rev_prev
    gm_curr = out["metrics"]["Gross Margin"]
    out["improvements"]["Gross Margin improved YoY"] = (
        gm_curr is not None and gm_prev is not None and gm_curr > gm_prev
    )

    # FS is computed later from sector-relative percentiles
    out["FS"] = None
    return out


def sector_relative_scores(
    fundamental_data: Dict[str, Dict], sectors: Dict[str, str]
) -> Dict[str, float]:
    """
    Convert the 12 metrics to sector-relative percentiles and aggregate using the
    provided weights to produce a 0–100 Fundamental Score (FS) for each ticker.
    """
    sector_groups = {}
    for ticker, data in fundamental_data.items():
        sec = sectors.get(ticker, "Unknown")
        sector_groups.setdefault(sec, []).append((ticker, data["metrics"]))

    fs_out = {}
    for sec, items in sector_groups.items():
        if not items:
            continue

        metric_names = list(items[0][1].keys())
        metric_arrays = {m: [] for m in metric_names}
        for ticker, metrics in items:
            for m in metric_names:
                metric_arrays[m].append(metrics.get(m))

        metric_percentiles = {m: {} for m in metric_names}
        for m, vals in metric_arrays.items():
            series = pd.Series(vals, dtype="float64")
            valid = series.dropna()
            if valid.empty:
                for i, (ticker, _) in enumerate(items):
                    metric_percentiles[m][ticker] = 0.0
                continue
            ranks = valid.rank(pct=True, method="average")
            val_map = {k: v for k, v in zip(valid.index, ranks.values)}
            for i, (ticker, _) in enumerate(items):
                v = series.iloc[i]
                if pd.isna(v):
                    metric_percentiles[m][ticker] = 0.0
                else:
                    metric_percentiles[m][ticker] = float(val_map[i]) * 100.0

        # weights and direction (from your JSON spec)
        weight_map = {
            "Gross Margin": 0.10,
            "Net Margin": 0.08,
            "Return on Assets (ROA)": 0.15,
            "Asset Turnover": 0.06,
            "Return on Equity (ROE)": 0.15,
            "Current Ratio": 0.07,
            "Book Value": 0.05,
            "Debt-to-Equity": 0.08,
            "Earnings Per Share (EPS)": 0.04,
            "Price-to-Earnings (P/E)": 0.04,
            "Revenue Growth": 0.10,
            "Equity Growth": 0.08,
        }
        better_map = {
            "Gross Margin": "high",
            "Net Margin": "high",
            "Return on Assets (ROA)": "high",
            "Asset Turnover": "high",
            "Return on Equity (ROE)": "high",
            "Current Ratio": "high",
            "Book Value": "high",
            "Debt-to-Equity": "low",
            "Earnings Per Share (EPS)": "high",
            "Price-to-Earnings (P/E)": "low",
            "Revenue Growth": "high",
            "Equity Growth": "high",
        }

        for ticker, _ in items:
            fs = 0.0
            for m in metric_names:
                pct = metric_percentiles[m].get(ticker, 0.0)
                if better_map.get(m, "high") == "low":
                    pct = 100.0 - pct
                fs += weight_map.get(m, 0.0) * (pct / 100.0)
            fs_out[ticker] = round(fs * 100.0, 2)

    return fs_out

import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

import config

# -----------------------------
# Caching directories
# -----------------------------
CACHE_DIR = os.path.join(os.getcwd(), "cache")
FUND_CACHE_DIR = os.path.join(CACHE_DIR, "fundamentals")
PRICE_CACHE_DIR = os.path.join(CACHE_DIR, "prices")
META_CACHE_PATH = os.path.join(CACHE_DIR, "company_meta.json")
os.makedirs(FUND_CACHE_DIR, exist_ok=True)
os.makedirs(PRICE_CACHE_DIR, exist_ok=True)

# Cache freshness
PRICE_CACHE_DAYS = 1  # requirement: cache price data for 1 day by default


def _safe_yf_download(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_retries: int = 3,
    sleep_sec: float = 1.0,
) -> pd.DataFrame:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                interval="1d",
                group_by="ticker",
                threads=False,
            )
            # If empty (e.g., non-trading days Fri/Sat for KSA), treat as "no new bars"
            if not isinstance(df, pd.DataFrame) or df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(0, axis=1)
            df = df.rename(columns=str.title)
            cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            keep = [c for c in cols if c in df.columns]
            df = df[keep].copy()
            df.index = pd.to_datetime(df.index, utc=False).tz_localize(None)
            return df
        except Exception as e:
            # Tolerate yfinance weekend spans: "no price data found"
            if "no price data found" in str(e).lower():
                return pd.DataFrame()
            last_exc = e
        time.sleep(sleep_sec * attempt)
    if last_exc:
        raise last_exc
    return pd.DataFrame()


# -----------------------------
# Meta helpers (company name/sector) with light cache
# -----------------------------
def _load_meta_cache() -> Dict[str, Dict]:
    if os.path.exists(META_CACHE_PATH):
        try:
            with open(META_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_meta_cache(d: Dict[str, Dict]) -> None:
    os.makedirs(os.path.dirname(META_CACHE_PATH), exist_ok=True)
    with open(META_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def _yf_ticker_obj(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


def get_info(ticker: str) -> Dict:
    t = _yf_ticker_obj(ticker)
    info = {}
    try:
        if getattr(t, "fast_info", None):
            info.update(t.fast_info)
    except Exception:
        pass
    try:
        legacy = t.info
        if isinstance(legacy, dict):
            info.update(legacy)
    except Exception:
        pass
    return info or {}


def get_company_name(ticker: str) -> Optional[str]:
    meta = _load_meta_cache()
    if ticker in meta and "name" in meta[ticker]:
        return meta[ticker]["name"]
    info = get_info(ticker)
    name = info.get("shortName") or info.get("longName") or info.get("displayName")
    if isinstance(name, str) and name.strip():
        meta.setdefault(ticker, {})["name"] = name.strip()
        _save_meta_cache(meta)
        return name.strip()
    return None


def get_sector(ticker: str) -> Optional[str]:
    meta = _load_meta_cache()
    if ticker in meta and "sector" in meta[ticker]:
        return meta[ticker]["sector"]
    info = get_info(ticker)
    sector = info.get("sector") or info.get("industry") or info.get("industryDisp")
    if isinstance(sector, str) and sector.strip():
        meta.setdefault(ticker, {})["sector"] = sector.strip()
        _save_meta_cache(meta)
        return sector.strip()
    return None


# -----------------------------
# Price cache helpers
# -----------------------------
def _price_cache_path(ticker: str) -> str:
    safe = ticker.replace("^", "").replace("/", "_")
    return os.path.join(PRICE_CACHE_DIR, f"{safe}.csv")


def _is_cache_fresh(path: str, days: int) -> bool:
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime) <= timedelta(days=days)


def _load_cached_prices(ticker: str) -> Optional[pd.DataFrame]:
    path = _price_cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        # Normalize/validate Date column or index
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
        else:
            # Fallback: try to parse the existing index as dates
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[~df.index.isna()]
            except Exception:
                return None
        # Ensure datetime index (naive), dedupe and sort
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception:
        return None


def _save_cached_prices(ticker: str, df: pd.DataFrame) -> None:
    path = _price_cache_path(ticker)
    to_save = df.copy()
    to_save = to_save.reset_index().rename(columns={"index": "Date"})
    to_save.to_csv(path, index=False)


# -----------------------------
# Public price fetchers (with 1-day cache and CLI override)
# -----------------------------
def get_price_history(
    ticker: str, lookback_days: int = 400, refresh: bool = False
) -> pd.DataFrame:
    """
    Return OHLCV history for `ticker`.

    Behavior:
    - If `refresh=True`: force a full download covering ~1.5x lookback window and overwrite cache.
    - Else: load any existing cached history (ignoring file mtime), compute the last available
      date in the cache, backfill a small buffer window from that date, **append only new rows**,
      dedupe by Date, and persist.
    """
    today = datetime.utcnow().date()

    # Helper to trim the returned *view* while keeping full cache on disk
    def _trim_view(df: pd.DataFrame) -> pd.DataFrame:
        lb_start = pd.Timestamp.today().normalize() - pd.Timedelta(
            days=int(lookback_days * 1.5)
        )
        return df[df.index >= lb_start]

    # Full refresh path
    if refresh:
        start = today - timedelta(days=int(lookback_days * 1.5))
        df = _safe_yf_download(
            ticker, start=start.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d")
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        _save_cached_prices(ticker, df)
        return _trim_view(df)

    # Incremental path: read any cache unconditionally (ignore mtime)
    cached = _load_cached_prices(ticker)

    # No cache → do a regular download window
    if cached is None or cached.empty:
        start = today - timedelta(days=int(lookback_days * 1.5))
        df = _safe_yf_download(
            ticker, start=start.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d")
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        _save_cached_prices(ticker, df)
        return _trim_view(df)

    # We have cache: compute last date and backfill buffer to catch vendor backfills
    last_date = cached.index.max().date()
    fetch_start = last_date - timedelta(days=5)  # buffer days to catch splits/backfills
    start_str = fetch_start.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    new_df = _safe_yf_download(ticker, start=start_str, end=end_str)
    if new_df is None or new_df.empty:
        # Nothing new (weekend/holiday), just return cached view
        return _trim_view(cached)

    if "Date" in new_df.columns:
        new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
        new_df = new_df.dropna(subset=["Date"]).set_index("Date")
    new_df.index = pd.DatetimeIndex(new_df.index).tz_localize(None)
    new_df = new_df[~new_df.index.duplicated(keep="last")].sort_index()

    # Append/dedupe/sort and persist
    combined = pd.concat([cached, new_df], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    _save_cached_prices(ticker, combined)

    return _trim_view(combined)


def get_index_history(lookback_days: int = 400, refresh: bool = False) -> pd.DataFrame:
    return get_price_history(
        config.INDEX_TICKER, lookback_days=lookback_days, refresh=refresh
    )


# -----------------------------
# Fundamentals cache (unchanged)
# -----------------------------
def _fund_cache_path(ticker: str) -> str:
    return os.path.join(FUND_CACHE_DIR, f"{ticker}.json")


def _is_fund_cache_fresh(path: str, days: int) -> bool:
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime) <= timedelta(days=days)


def load_cached_fundamentals(ticker: str) -> Optional[Dict]:
    path = _fund_cache_path(ticker)
    if _is_fund_cache_fresh(path, config.FUNDAMENTAL_CACHE_DAYS):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_cached_fundamentals(ticker: str, data: Dict) -> None:
    path = _fund_cache_path(ticker)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def persist_fs_to_cache(
    fs_map: Dict[str, float], sector_map: Dict[str, str], as_of: Optional[str] = None
) -> None:
    """
    Write FS, fs_date, and fs_sector into each company's cached fundamentals JSON.
    Best-effort: silently skips tickers that fail to write.
    """
    os.makedirs(FUND_CACHE_DIR, exist_ok=True)
    stamp = as_of or datetime.now().strftime("%Y-%m-%d")
    for ticker, fs in fs_map.items():
        path = _fund_cache_path(ticker)
        try:
            data = {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            if not isinstance(data, dict):
                data = {}
            data["FS"] = float(fs) if fs is not None else None
            data["fs_date"] = stamp
            data["fs_sector"] = sector_map.get(ticker, None)
            save_cached_fundamentals(ticker, data)
        except Exception:
            continue  # best-effort


def get_fundamental_frames(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = _yf_ticker_obj(ticker)
    fin = pd.DataFrame()
    bs = pd.DataFrame()
    try:
        fin = t.financials if t.financials is not None else pd.DataFrame()
    except Exception:
        fin = pd.DataFrame()
    try:
        bs = t.balance_sheet if t.balance_sheet is not None else pd.DataFrame()
    except Exception:
        bs = pd.DataFrame()
    return fin, bs


def get_fundamentals_cached_or_fetch(
    ticker: str, compute_func, manual_df: Optional[pd.DataFrame] = None
) -> Dict:
    cached = load_cached_fundamentals(ticker)
    if cached is not None:
        return cached
    fin, bs = get_fundamental_frames(ticker)
    info = get_info(ticker)
    data = compute_func(ticker, fin, bs, info, manual_df=manual_df)
    save_cached_fundamentals(ticker, data)
    return data


def load_universe(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        tickers = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    return tickers

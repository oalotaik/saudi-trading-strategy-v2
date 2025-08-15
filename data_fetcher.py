
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


def _safe_yf_download(ticker: str, start: Optional[str] = None, end: Optional[str] = None,
                      max_retries: int = 3, sleep_sec: float = 1.0) -> pd.DataFrame:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker, start=start, end=end, auto_adjust=False,
                progress=False, interval="1d", group_by="ticker", threads=False
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(0, axis=1)
                return df.rename(columns=str.title)
        except Exception as e:
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
    if _is_cache_fresh(path, PRICE_CACHE_DAYS):
        try:
            df = pd.read_csv(path)
            if not df.empty and "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
            return df
        except Exception:
            return None
    return None

def _save_cached_prices(ticker: str, df: pd.DataFrame) -> None:
    path = _price_cache_path(ticker)
    to_save = df.copy()
    to_save = to_save.reset_index().rename(columns={"index": "Date"})
    to_save.to_csv(path, index=False)


# -----------------------------
# Public price fetchers (with 1-day cache and CLI override)
# -----------------------------
def get_price_history(ticker: str, lookback_days: int = 400, refresh: bool = False) -> pd.DataFrame:
    if not refresh:
        cached = _load_cached_prices(ticker)
        if cached is not None and not cached.empty:
            # Trim to lookback
            lb_start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(lookback_days * 1.5))
            cached = cached[cached.index >= lb_start]
            return cached

    end = datetime.utcnow().date()
    start = end - timedelta(days=int(lookback_days * 1.5))
    df = _safe_yf_download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        # fall back to stale cache if present
        cached = _load_cached_prices(ticker)
        if cached is not None:
            return cached
        return pd.DataFrame()

    df = df[['Open','High','Low','Close','Adj Close','Volume']].copy()
    df.dropna(subset=['Close','Volume'], inplace=True)
    df.index = pd.to_datetime(df.index)
    _save_cached_prices(ticker, df)
    return df

def get_index_history(lookback_days: int = 400, refresh: bool = False) -> pd.DataFrame:
    return get_price_history(config.INDEX_TICKER, lookback_days=lookback_days, refresh=refresh)


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

def get_fundamentals_cached_or_fetch(ticker: str, compute_func, manual_df: Optional[pd.DataFrame] = None) -> Dict:
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
        tickers = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return tickers

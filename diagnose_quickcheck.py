import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd

try:
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Console
    from rich import box
except Exception:
    Table = Panel = Console = None

DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), "cache")
PRICE_DIR = os.path.join(DEFAULT_CACHE_DIR, "prices")
FUND_DIR = os.path.join(DEFAULT_CACHE_DIR, "fundamentals")
META_PATH = os.path.join(DEFAULT_CACHE_DIR, "company_meta.json")


def load_universe(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def load_names(meta_path: str) -> Dict[str, str]:
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # expect mapping: {ticker: {"name": "...", ...}} OR {ticker: "Company Name"}
        out = {}
        for t, v in data.items():
            if isinstance(v, dict):
                out[t] = v.get("name") or v.get("Name") or v.get("company_name") or ""
            elif isinstance(v, str):
                out[t] = v
        return out
    except Exception:
        return {}


def fmt_ticker_and_name(t: str, n: Optional[str]) -> str:
    return f"{t} — {n}" if n else t


def quick_check(universe_path: str, cache_dir: Optional[str] = None):
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    price_dir = os.path.join(cache_dir, "prices")
    fund_dir = os.path.join(cache_dir, "fundamentals")
    meta_path = os.path.join(cache_dir, "company_meta.json")
    names = load_names(meta_path)

    rows = []
    today = datetime.now().date()

    for t in load_universe(universe_path):
        # Prices
        p_csv = os.path.join(price_dir, f"{t.replace('^', '').replace('/', '_')}.csv")
        span, last, appended_today = ("—", "—", "—")
        if os.path.exists(p_csv):
            try:
                df = pd.read_csv(p_csv)
                if "Date" in df.columns:
                    # normalize date col
                    df["Date"] = pd.to_datetime(df["Date"]).dt.date
                    if not df.empty:
                        dmin = df["Date"].min()
                        dmax = df["Date"].max()
                        span = f"{dmin} → {dmax}"
                        last = str(dmax)
                        appended_today = "Yes" if dmax == today else "No"
            except Exception:
                pass

        # Fundamentals FS
        f_json = os.path.join(fund_dir, f"{t}.json")
        fs_str, fs_date, fs_sector = ("—", "—", "—")
        if os.path.exists(f_json):
            try:
                with open(f_json, "r", encoding="utf-8") as f:
                    dat = json.load(f)
                fs = dat.get("FS", None)
                fs_str = (
                    f"{fs:.1f}"
                    if isinstance(fs, (int, float))
                    else ("None" if fs is None else str(fs))
                )
                fs_date = dat.get("fs_date", "—")
                fs_sector = dat.get("fs_sector", "—")
            except Exception:
                pass

        rows.append(
            [
                fmt_ticker_and_name(t, names.get(t, "")),
                span,
                last,
                appended_today,
                fs_str,
                fs_date,
                fs_sector,
            ]
        )

    title = "Quick Cache & FS Check"
    cols = [
        "Ticker — Name",
        "Cache Span",
        "Last Date",
        "Appended Today?",
        "FS",
        "FS Date",
        "FS Sector",
    ]

    if Console:
        console = Console()
        table = Table(title=title, box=box.SIMPLE_HEAVY)
        for c in cols:
            table.add_column(c, overflow="fold")
        for r in rows:
            table.add_row(*[str(x) for x in r])
        console.print(table)
    else:
        # basic fallback
        print(title)
        print("\t".join(cols))
        for r in rows:
            print("\t".join([str(x) for x in r]))


def build_parser():
    p = argparse.ArgumentParser(
        description="Quick check for price cache span & FS presence"
    )
    p.add_argument("--universe", type=str, default="universe.txt")
    p.add_argument(
        "--cache-dir", type=str, default=None, help="Override cache dir if not ./cache"
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    quick_check(args.universe, args.cache_dir)

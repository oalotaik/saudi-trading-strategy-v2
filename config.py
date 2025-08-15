# Strategy configuration and thresholds

from datetime import timedelta

# -------- General --------
INDEX_TICKER = "^TASI.SR"

# Universe & basic filters
MIN_AVG_DAILY_VALUE_SAR = 5000000  # 20-day average traded value
MIN_AVG_DAILY_VOLUME = 100000  # 20-day average volume
MIN_PRICE_SAR = 5

# Breadth & regime
BREADTH_LOOKBACK = 50  # SMA50 breadth
BREADTH_MIN_PCT_ABOVE_SMA50 = 45.0  # percent
REGIME_SMA_LONG = 200
REGIME_SMA_MED = 50
REGIME_SLOPE_WINDOW = 20
REGIME_MIN_TRUE = 2  # need >=2 of 3 for active

# Sector strength
SECTOR_RS_LOOKBACK = 20  # days
SECTOR_TOP_PERCENTILE = 60  # buy only top 50% sectors

# Fundamentals
FUNDAMENTAL_CACHE_DAYS = 7  # days to cache fundies
FUNDAMENTAL_MIN_FS = 50  # min Fundamental Score (0-100)
FUNDAMENTAL_MIN_FS_SECTOR_TOP = 60.0  # sector percentile threshold (top 40% == pct>=60)
FUNDAMENTAL_NEAR_MISS_PCT = 10.0  # within 10% of threshold considered near miss

# Technicals
EMA_SHORT = 20
EMA_MED = 50
SMA_LONG = 200
RSI_PERIOD = 14
ROC_LOOKBACK = 63
ADX_PERIOD = 14
ATR_PERIOD = 14
VOLUME_AVG = 20

# D1 Breakout
D1_MIN_RSI = 55.0
D1_MIN_ADX = 20.0
D1_MIN_ROC = 5.0
D1_VOLUME_MULT = 1.5
D1_MAX_EXT_ABOVE_EMA50_ATR = 10.0  # percent above EMA50 max

# D2 Pullback-and-Go
D2_PULLBACK_DAYS_MIN = 2
D2_PULLBACK_DAYS_MAX = 5

# Risk
RISK_PER_TRADE = 0.02  # 2% of portfolio
MAX_POSITION_WEIGHT = 0.40  # 40% cap per position
PORTFOLIO_MAX_DRAWDOWN = 0.20  # 20% max drawdown
CORRELATION_LOOKBACK = 60
MAX_CORRELATION = 0.60

# Selection
MAX_CONCURRENT_POSITIONS = 3
MAX_PER_SECTOR = 1  # max 1 position per sector

# Ranking weights
TECH_WEIGHT = 0.50
FUND_WEIGHT = 0.30
SECTOR_WEIGHT = 0.20

# Reporting
DATE_FORMAT = "%Y-%m-%d"

# Manual fundamentals file (optional override or fill-in)
MANUAL_FUNDAMENTALS_CSV = (
    "manual_fundamentals.csv"  # optional, columns: ticker,metric,value,period
)

# Backtest (optional defaults)
BACKTEST_START = None  # e.g., "2022-01-01"
BACKTEST_END = None


# Cooldown days after a max-DD flatten event (no new entries)
DRAWDOWN_COOLDOWN_DAYS = 7


# --- Donchian breakout (DB55) ---
DONCHIAN_LOOKBACK = 55
DB55_MIN_ADX = 18.0
DB55_VOL_MULT = 1.2

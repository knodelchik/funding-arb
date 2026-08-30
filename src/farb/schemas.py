"""Dataset schemas and storage layout — the single source of truth for both
the collector (writer) and the backtest engine (reader).

Layout: data/binance/<dataset>/<SYMBOL>.parquet, one file per symbol,
sorted by the time column, no duplicate timestamps (incremental appends
must dedupe on the time column).

All times are UTC milliseconds from Binance, stored as Datetime("ms", "UTC").
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

# Symbols reach this module from CLI args and config files and become file
# names; a strict allowlist keeps "../../x" or an absolute path from ever
# escaping data_root (real Binance symbols are short [A-Z0-9] strings).
# SECURITY: must stay explicit ASCII ranges (\w/\d are unicode-aware and admit
# lookalikes) and must be applied with .fullmatch() with NO trailing $ — both
# .match() and a $-terminated pattern silently re-admit "BTCUSDT\n".
_SYMBOL_RE = re.compile(r"[A-Z0-9]{2,20}")

TS = pl.Datetime("ms", "UTC")

# fapi/v1/fundingRate — one row per settled funding event
FUNDING_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "funding_time": TS,
    "funding_rate": pl.Float64,
    "mark_price": pl.Float64,  # may be NaN for very old rows
}

# klines (spot / perp / mark / premium) share one shape; volume is 0.0 where
# the endpoint has no real volume (mark, premium)
KLINE_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "open_time": TS,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

# fapi/v1/fundingInfo — only symbols with a non-default interval appear on the
# endpoint; the collector must still write a row for every requested symbol,
# filling 8 for the absent ones.
FUNDING_INFO_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "funding_interval_hours": pl.Int64,
}

# Datasets keyed by directory name under data/binance/
DATASETS = {
    "funding": FUNDING_SCHEMA,
    "klines_spot_1h": KLINE_SCHEMA,
    "klines_perp_1h": KLINE_SCHEMA,
    "klines_mark_1h": KLINE_SCHEMA,
    "premium_1h": KLINE_SCHEMA,
}

TIME_COLUMN = {
    "funding": "funding_time",
    "klines_spot_1h": "open_time",
    "klines_perp_1h": "open_time",
    "klines_mark_1h": "open_time",
    "premium_1h": "open_time",
}

DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "BCHUSDT", "TRXUSDT", "NEARUSDT", "ATOMUSDT",
    "FILUSDT", "ETCUSDT", "APTUSDT", "SUIUSDT", "ARBUSDT",
]


def dataset_dir(root: Path, dataset: str) -> Path:
    if dataset not in DATASETS and dataset != "funding_info":
        raise ValueError(f"unknown dataset: {dataset}")
    return root / "binance" / dataset


def dataset_path(root: Path, dataset: str, symbol: str) -> Path:
    if not _SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"invalid symbol {symbol!r}: must match [A-Z0-9]{{2,20}}")
    return dataset_dir(root, dataset) / f"{symbol}.parquet"


def funding_info_path(root: Path) -> Path:
    return root / "binance" / "funding_info.parquet"

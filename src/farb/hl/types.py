"""Typed records for the Hyperliquid public API client.

Mirrors farb.paper.models's style (pydantic BaseModel, extra="forbid") since
these values flow straight into the paper engine's signal/PnL math — the
same "no silent extra/missing field" discipline applies here.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AssetCtx(BaseModel):
    """One coin's current market context, from metaAndAssetCtxs."""

    model_config = ConfigDict(extra="forbid")

    coin: str
    funding_hourly: float  # fraction, e.g. 0.0001 = 1bp per hour (HL settles hourly)
    mark_px: float
    open_interest: float
    oracle_px: float


class FundingEvent(BaseModel):
    """One settled hourly funding event for a coin, from fundingHistory."""

    model_config = ConfigDict(extra="forbid")

    coin: str
    time: datetime  # UTC, ms precision
    rate: float  # fraction, per-1h — unlike Binance's 8h-settled rate


class BookTop(BaseModel):
    """Best bid/ask for a coin, from l2Book."""

    model_config = ConfigDict(extra="forbid")

    coin: str
    best_bid: float
    best_ask: float
    ts: datetime  # UTC; the book snapshot's server time when available


class CoinSpec(BaseModel):
    """One coin's static trading spec, from meta — used for order sizing."""

    model_config = ConfigDict(extra="forbid")

    coin: str
    sz_decimals: int  # qty must round to this many decimal places
    max_leverage: int


class Candle(BaseModel):
    """One OHLC candle for a coin, from candleSnapshot."""

    model_config = ConfigDict(extra="forbid")

    coin: str
    open_time: datetime  # UTC
    open: float
    high: float
    low: float
    close: float
    volume: float  # 0.0 when the endpoint doesn't report volume for this series

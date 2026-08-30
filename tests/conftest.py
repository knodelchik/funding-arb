"""Shared synthetic-data builders. No network anywhere in tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from farb.schemas import FUNDING_SCHEMA, KLINE_SCHEMA

T0 = datetime(2025, 1, 1, tzinfo=UTC)


def make_funding(
    symbol: str,
    rates: list[float],
    start: datetime = T0,
    interval_hours: int = 8,
    mark_price: float = 100.0,
) -> pl.DataFrame:
    """Settled funding events at fixed interval with the given rates."""
    times = [start + timedelta(hours=interval_hours * i) for i in range(len(rates))]
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(rates),
            "funding_time": times,
            "funding_rate": rates,
            "mark_price": [mark_price] * len(rates),
        },
        schema=FUNDING_SCHEMA,
    )


def make_klines(
    symbol: str,
    closes: list[float],
    start: datetime = T0,
    interval_hours: int = 1,
) -> pl.DataFrame:
    """1h-style klines with flat OHLC == close and unit volume."""
    times = [start + timedelta(hours=interval_hours * i) for i in range(len(closes))]
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "open_time": times,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        },
        schema=KLINE_SCHEMA,
    )


@pytest.fixture
def t0() -> datetime:
    return T0

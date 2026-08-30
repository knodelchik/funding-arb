"""Binance public-data collector.

Pulls funding-rate history, funding intervals, and perp/mark/premium/spot 1h
klines into per-symbol parquet files under `data/binance/` (layout defined in
farb.schemas). Incremental by construction: each dataset resumes from an
on-disk checkpoint rather than from --since, so re-running the collector
appends new rows without duplicating history already collected. One bad
symbol never aborts the whole run — each (symbol, dataset) fetch is isolated.
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from farb.data.client import FAPI_BASE, SPOT_BASE, BinanceClient
from farb.schemas import (
    DATASETS,
    FUNDING_INFO_SCHEMA,
    TIME_COLUMN,
    dataset_path,
    funding_info_path,
)

FUNDING_LIMIT = 1000
PERP_KLINE_LIMIT = 1500
SPOT_KLINE_LIMIT = 1000
DEFAULT_FUNDING_INTERVAL_HOURS = 8
HOUR_MS = 3_600_000

# dataset name -> (base URL, endpoint path, page limit); all share the same
# kline array shape and forward-pagination rule.
_KLINE_ENDPOINTS: dict[str, tuple[str, str, int]] = {
    "klines_perp_1h": (FAPI_BASE, "/fapi/v1/klines", PERP_KLINE_LIMIT),
    "klines_mark_1h": (FAPI_BASE, "/fapi/v1/markPriceKlines", PERP_KLINE_LIMIT),
    "premium_1h": (FAPI_BASE, "/fapi/v1/premiumIndexKlines", PERP_KLINE_LIMIT),
    "klines_spot_1h": (SPOT_BASE, "/api/v3/klines", SPOT_KLINE_LIMIT),
}

class ParseError(ValueError):
    """A numeric value from Binance was malformed or non-finite.

    A ValueError subclass so callers that only care "was this bad data" can
    still catch ValueError, but run_collect's per-(symbol, dataset)
    isolation catches ParseError specifically — not bare ValueError — so a
    genuine programming bug that happens to raise plain ValueError
    elsewhere (e.g. a bug in _write_incremental) still propagates instead
    of being silently downgraded to a warning and an empty dataset.
    """


# Errors that must not abort a single (symbol, dataset) collection: a bad
# HTTP response, or a malformed/non-finite value from Binance. A bad symbol
# is a separate, narrower case (see run_collect's pre-check); anything else
# — including a bare ValueError from deeper in the pipeline — is a bug and
# must propagate.
_SKIPPABLE_ERRORS = (httpx.HTTPError, ParseError)


def _utcnow() -> datetime:
    """Wall-clock now, as its own function so tests can monkeypatch it."""
    return datetime.now(UTC)


def _to_ms(dt: datetime) -> int:
    """Milliseconds since epoch, matching the units Binance sends/expects."""
    return round(dt.timestamp() * 1000)


def _parse_finite_float(raw: Any, *, field: str, symbol: str) -> float:
    """Parse a Binance numeric string, rejecting garbage and NaN/Infinity survivors.

    Binance shouldn't send unparseable or non-finite numbers, but either one
    silently propagating as inf/nan would corrupt every downstream
    computation (APR, EMA, PnL) without ever raising anywhere obvious. This
    raises ParseError, which the per-(symbol, dataset) isolation in
    run_collect catches, warns about, and skips past.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"{symbol}: unparseable {field} value from Binance: {raw!r}") from exc
    if not math.isfinite(value):
        raise ParseError(f"{symbol}: non-finite {field} value from Binance: {raw!r}")
    return value


def _funding_rows(symbol: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map raw /fundingRate rows to the FUNDING_SCHEMA record shape.

    markPrice is a string that may be missing or empty on very old rows;
    both cases become NaN rather than a parse error (NaN here is a
    deliberate "absent" sentinel, so it is not run through the finite check).
    """
    rows = []
    for row in batch:
        mark_price_raw = row.get("markPrice")
        mark_price = (
            _parse_finite_float(mark_price_raw, field="mark_price", symbol=symbol)
            if mark_price_raw
            else float("nan")
        )
        rows.append(
            {
                "symbol": symbol,
                "funding_time": int(row["fundingTime"]),
                "funding_rate": _parse_finite_float(
                    row["fundingRate"], field="funding_rate", symbol=symbol
                ),
                "mark_price": mark_price,
            }
        )
    return rows


def _kline_rows(symbol: str, batch: list[list[Any]]) -> list[dict[str, Any]]:
    """Map raw kline arrays (indices 0 open_time..5 volume) to KLINE_SCHEMA."""
    return [
        {
            "symbol": symbol,
            "open_time": int(row[0]),
            "open": _parse_finite_float(row[1], field="open", symbol=symbol),
            "high": _parse_finite_float(row[2], field="high", symbol=symbol),
            "low": _parse_finite_float(row[3], field="low", symbol=symbol),
            "close": _parse_finite_float(row[4], field="close", symbol=symbol),
            "volume": _parse_finite_float(row[5], field="volume", symbol=symbol),
        }
        for row in batch
    ]


def _drop_unclosed_final_bar(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the last bar if its close time (open_time + 1h) hasn't happened yet.

    Binance's most recent kline is often the in-progress candle for the
    current hour (observed on real data: the last bar can carry ~20% of a
    typical bar's volume). Writing it would freeze a partial bar, since the
    on-disk resume point only re-fetches the *last written* bar, not
    whatever was in progress at collection time — this filter is what keeps
    that bar out until it has actually closed.
    """
    if not rows:
        return rows
    close_time_ms = rows[-1]["open_time"] + HOUR_MS
    if close_time_ms > _to_ms(_utcnow()):
        return rows[:-1]
    return rows


def _paginate_funding(client: BinanceClient, symbol: str, start_ms: int) -> list[dict[str, Any]]:
    """Forward-paginate /fundingRate: next startTime = last fundingTime + 1."""
    rows: list[dict[str, Any]] = []
    next_start = start_ms
    while True:
        batch = client.get_list(
            FAPI_BASE,
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": next_start, "limit": FUNDING_LIMIT},
        )
        if not batch:
            break
        rows.extend(_funding_rows(symbol, batch))
        next_start = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < FUNDING_LIMIT:
            break
    return rows


def _paginate_klines(
    client: BinanceClient,
    base: str,
    path: str,
    symbol: str,
    start_ms: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Forward-paginate a klines-shaped endpoint: next startTime = last open_time + 1.

    This in-pagination rule is unrelated to the on-disk resume point (see
    _next_start_ms) — it always advances past the last row of a page. Only
    the very last bar of the whole result can still be in progress, so the
    unclosed-bar filter runs once, after pagination is done.
    """
    rows: list[dict[str, Any]] = []
    next_start = start_ms
    while True:
        batch = client.get_list(
            base,
            path,
            {"symbol": symbol, "interval": "1h", "startTime": next_start, "limit": limit},
        )
        if not batch:
            break
        rows.extend(_kline_rows(symbol, batch))
        next_start = int(batch[-1][0]) + 1
        if len(batch) < limit:
            break
    return _drop_unclosed_final_bar(rows)


def _existing_max_time_ms(path: Path, time_column: str) -> int | None:
    """Max timestamp (ms) already on disk for this dataset/symbol, if any."""
    if not path.exists():
        return None
    existing = pl.read_parquet(path)
    if existing.is_empty():
        return None
    return _to_ms(existing[time_column].max())


def _next_start_ms(path: Path, time_column: str, since: datetime, *, refetch_last: bool) -> int:
    """Resume point for the next fetch.

    Settled funding events never change once written, so funding resumes at
    max + 1ms (refetch_last=False). Klines can still be in progress when
    written — the final bar of a run is often a partial candle — so klines
    resume AT the on-disk max (refetch_last=True), re-requesting that bar so
    a now-completed version can overwrite the partial one (see the dedup
    rule in _write_incremental, which keeps the most recently fetched row).
    """
    existing_max = _existing_max_time_ms(path, time_column)
    if existing_max is None:
        return _to_ms(since)
    return existing_max if refetch_last else existing_max + 1


def _write_incremental(
    path: Path,
    new_rows: list[dict[str, Any]],
    schema: dict[str, pl.DataType],
    time_column: str,
) -> None:
    """Merge new_rows onto the existing parquet (if any); dedupe, sort, atomic write.

    Dedup keeps the last occurrence on the time column, so a freshly
    re-collected row wins over what was already on disk for the same
    timestamp. This is not just theoretical: klines deliberately re-fetch
    their last on-disk bar (see _next_start_ms), so this is the path that
    turns a stored partial candle into the completed one once new data
    arrives with the same open_time. Writing to a .tmp file and renaming
    avoids ever leaving a half-written parquet behind.
    """
    new_df = pl.DataFrame(new_rows, schema=schema) if new_rows else pl.DataFrame(schema=schema)
    if path.exists():
        combined = pl.concat([pl.read_parquet(path), new_df], how="vertical")
    else:
        combined = new_df
    combined = combined.unique(subset=[time_column], keep="last").sort(time_column)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    combined.write_parquet(tmp_path)
    tmp_path.replace(path)


def _collect_funding(
    client: BinanceClient, symbol: str, since: datetime, data_root: Path
) -> None:
    dataset = "funding"
    path = dataset_path(data_root, dataset, symbol)
    time_column = TIME_COLUMN[dataset]
    start_ms = _next_start_ms(path, time_column, since, refetch_last=False)
    new_rows = _paginate_funding(client, symbol, start_ms)
    _write_incremental(path, new_rows, DATASETS[dataset], time_column)
    print(f"{symbol} {dataset}: +{len(new_rows)} rows")


def _collect_klines(
    client: BinanceClient,
    dataset: str,
    base: str,
    endpoint_path: str,
    limit: int,
    symbol: str,
    since: datetime,
    data_root: Path,
) -> None:
    path = dataset_path(data_root, dataset, symbol)
    time_column = TIME_COLUMN[dataset]
    start_ms = _next_start_ms(path, time_column, since, refetch_last=True)
    new_rows = _paginate_klines(client, base, endpoint_path, symbol, start_ms, limit)
    _write_incremental(path, new_rows, DATASETS[dataset], time_column)
    print(f"{symbol} {dataset}: +{len(new_rows)} rows")


def _collect_funding_info(client: BinanceClient, symbols: list[str], data_root: Path) -> None:
    """Upsert one row per requested symbol: interval from /fundingInfo if
    listed, else 8h default.

    Not a time series but a small current-state lookup table, keyed by
    symbol. `--symbols` is a documented subset flag, so a subset run must
    upsert only the requested symbols and leave every other symbol's
    previously recorded interval untouched rather than wiping the table.
    """
    response = client.get_list(FAPI_BASE, "/fapi/v1/fundingInfo", None)
    intervals = {row["symbol"]: int(row["fundingIntervalHours"]) for row in response}
    new_rows = pl.DataFrame(
        [
            {
                "symbol": symbol,
                "funding_interval_hours": intervals.get(symbol, DEFAULT_FUNDING_INTERVAL_HOURS),
            }
            for symbol in symbols
        ],
        schema=FUNDING_INFO_SCHEMA,
    )
    path = funding_info_path(data_root)
    if path.exists():
        combined = pl.concat([pl.read_parquet(path), new_rows], how="vertical")
    else:
        combined = new_rows
    combined = combined.unique(subset=["symbol"], keep="last").sort("symbol")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    combined.write_parquet(tmp_path)
    tmp_path.replace(path)
    print(f"funding_info: {len(symbols)} symbols")


# funding + one entry per kline dataset; used to charge a whole symbol's
# worth of attempts as failed when its symbol format is invalid.
_DATASETS_PER_SYMBOL = 1 + len(_KLINE_ENDPOINTS)


def run_collect(symbols: list[str], since: datetime, data_root: Path) -> int:
    """Collect (or incrementally refresh) all datasets for every symbol.

    Each (symbol, dataset) fetch is isolated: an HTTP failure or a
    malformed/non-finite value from Binance (ParseError) is warned about on
    stderr and skipped, rather than aborting the whole run. An invalid
    configured symbol is checked once per symbol, before any of its
    datasets are attempted, via the same narrow ValueError that
    schemas.dataset_path raises — so that check, and only that check, is
    skip-with-warning; any OTHER exception (a genuine programming bug, e.g.
    in _write_incremental) is not caught here and propagates. funding_info
    is always attempted at the end regardless of which per-symbol fetches
    failed above.

    Returns the number of (symbol, dataset) collections that failed. Raises
    RuntimeError if every attempted collection failed — a systemic outage
    or misconfiguration, not a handful of bad symbols — so a cron caller
    can detect total failure via a non-zero exit instead of a silently
    "successful" empty run.
    """
    attempted = 0
    failed = 0
    with BinanceClient() as client:
        for symbol in symbols:
            try:
                dataset_path(data_root, "funding", symbol)  # symbol-format pre-check
            except ValueError as exc:
                attempted += _DATASETS_PER_SYMBOL
                failed += _DATASETS_PER_SYMBOL
                print(f"WARNING: {symbol} skipped (invalid symbol): {exc}", file=sys.stderr)
                continue

            attempted += 1
            try:
                _collect_funding(client, symbol, since, data_root)
            except _SKIPPABLE_ERRORS as exc:
                failed += 1
                print(f"WARNING: {symbol} funding failed: {exc}", file=sys.stderr)

            for dataset, (base, endpoint_path, limit) in _KLINE_ENDPOINTS.items():
                attempted += 1
                try:
                    _collect_klines(
                        client, dataset, base, endpoint_path, limit, symbol, since, data_root
                    )
                except _SKIPPABLE_ERRORS as exc:
                    failed += 1
                    print(f"WARNING: {symbol} {dataset} failed: {exc}", file=sys.stderr)

        try:
            _collect_funding_info(client, symbols, data_root)
        except httpx.HTTPError as exc:
            print(f"WARNING: funding_info failed: {exc}", file=sys.stderr)

    succeeded = attempted - failed
    print(f"collect summary: {succeeded}/{attempted} (symbol, dataset) collections succeeded")
    if attempted > 0 and failed == attempted:
        raise RuntimeError(f"all {attempted} collections failed for {symbols}; see warnings above")
    return failed

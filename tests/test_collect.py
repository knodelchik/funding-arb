"""Collector tests. respx mocks httpx at the transport level — no real network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx

from farb.data import collect
from farb.data.client import FAPI_BASE, SPOT_BASE, BinanceClient
from farb.data.collect import (
    HOUR_MS,
    PERP_KLINE_LIMIT,
    _collect_funding_info,
    _paginate_funding,
    _paginate_klines,
    _to_ms,
    run_collect,
)
from farb.schemas import (
    FUNDING_INFO_SCHEMA,
    FUNDING_SCHEMA,
    KLINE_SCHEMA,
    dataset_path,
    funding_info_path,
)

FUNDING_URL = f"{FAPI_BASE}/fapi/v1/fundingRate"
FUNDING_INFO_URL = f"{FAPI_BASE}/fapi/v1/fundingInfo"
PERP_KLINES_URL = f"{FAPI_BASE}/fapi/v1/klines"
MARK_KLINES_URL = f"{FAPI_BASE}/fapi/v1/markPriceKlines"
PREMIUM_KLINES_URL = f"{FAPI_BASE}/fapi/v1/premiumIndexKlines"
SPOT_KLINES_URL = f"{SPOT_BASE}/api/v3/klines"

SINCE = datetime(2025, 1, 1, tzinfo=UTC)
START_MS = _to_ms(SINCE)
FUNDING_INTERVAL_MS = 8 * HOUR_MS


def _funding_row(t: int, rate: float = 0.0001, mark: str | None = "100.0") -> dict:
    row: dict = {"symbol": "BTCUSDT", "fundingTime": t, "fundingRate": str(rate)}
    if mark is not None:
        row["markPrice"] = mark
    return row


def _kline_row(
    open_time: int,
    open_price: float = 100.0,
    high_price: float = 103.0,
    low_price: float = 99.0,
    close_price: float = 101.0,
    volume: str = "1.0",
) -> list:
    """Raw Binance kline array with DISTINCT o/h/l/c by default, so tests
    relying on this helper double as a guard against an OHLC index swap."""
    return [open_time, str(open_price), str(high_price), str(low_price), str(close_price), volume]


def _mock_untested_endpoints(
    start_ms: int = START_MS, exclude: frozenset[str] = frozenset()
) -> None:
    """Register minimal, fixed mocks for the datasets not under direct test.

    Two rows each (below every page limit) so pagination stops after one
    request; funding_info returns no overrides so every symbol defaults.
    """
    two_rows = [_kline_row(start_ms), _kline_row(start_ms + HOUR_MS)]
    for url in (PERP_KLINES_URL, MARK_KLINES_URL, PREMIUM_KLINES_URL, SPOT_KLINES_URL):
        if url not in exclude:
            respx.get(url).mock(return_value=httpx.Response(200, json=two_rows))
    if FUNDING_INFO_URL not in exclude:
        respx.get(FUNDING_INFO_URL).mock(return_value=httpx.Response(200, json=[]))


@respx.mock
def test_paginate_funding_full_then_short_batch_collects_all_no_dupes() -> None:
    """A full 1000-row batch followed by a short batch: keep paginating until short."""
    route = respx.get(FUNDING_URL)
    full_batch = [_funding_row(START_MS + i) for i in range(1000)]
    short_batch = [_funding_row(START_MS + 1000 + i) for i in range(7)]
    route.side_effect = [
        httpx.Response(200, json=full_batch),
        httpx.Response(200, json=short_batch),
    ]

    with BinanceClient() as client:
        rows = _paginate_funding(client, "BTCUSDT", START_MS)

    assert route.call_count == 2
    assert len(rows) == 1007
    assert len({r["funding_time"] for r in rows}) == 1007
    assert int(route.calls[0].request.url.params["startTime"]) == START_MS
    # next startTime must be last fundingTime of the first batch + 1
    assert int(route.calls[1].request.url.params["startTime"]) == START_MS + 1000


@respx.mock
def test_paginate_klines_full_then_short_batch_uses_last_plus_one_within_pagination() -> None:
    """Multi-page klines: the in-pagination next page always starts at last open_time + 1,
    independent of the (different) on-disk resume-point rule tested elsewhere."""
    route = respx.get(PERP_KLINES_URL)
    full_batch = [_kline_row(START_MS + i * HOUR_MS) for i in range(PERP_KLINE_LIMIT)]
    # next startTime = last open_time + 1, not the next hour boundary
    short_start = START_MS + (PERP_KLINE_LIMIT - 1) * HOUR_MS + 1
    short_batch = [_kline_row(short_start + i * HOUR_MS) for i in range(3)]
    route.side_effect = [
        httpx.Response(200, json=full_batch),
        httpx.Response(200, json=short_batch),
    ]

    with BinanceClient() as client:
        rows = _paginate_klines(
            client, FAPI_BASE, "/fapi/v1/klines", "BTCUSDT", START_MS, PERP_KLINE_LIMIT
        )

    assert route.call_count == 2
    assert int(route.calls[0].request.url.params["startTime"]) == START_MS
    assert int(route.calls[1].request.url.params["startTime"]) == short_start
    assert len(rows) == PERP_KLINE_LIMIT + 3
    assert len({r["open_time"] for r in rows}) == PERP_KLINE_LIMIT + 3


@respx.mock
def test_resume_point_klines_use_ondisk_max_funding_uses_max_plus_one(tmp_path: Path) -> None:
    """On-disk resume differs by dataset kind: klines re-fetch the last (possibly
    partial) bar; funding, whose settled events never change, does not."""
    _mock_untested_endpoints(exclude=frozenset({PERP_KLINES_URL}))
    funding_route = respx.get(FUNDING_URL)
    funding_route.mock(return_value=httpx.Response(200, json=[_funding_row(START_MS)]))
    perp_route = respx.get(PERP_KLINES_URL)
    perp_route.mock(
        return_value=httpx.Response(
            200, json=[_kline_row(START_MS), _kline_row(START_MS + HOUR_MS)]
        )
    )

    run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    funding_max = _to_ms(
        pl.read_parquet(dataset_path(tmp_path, "funding", "BTCUSDT"))["funding_time"].max()
    )
    perp_max = _to_ms(
        pl.read_parquet(dataset_path(tmp_path, "klines_perp_1h", "BTCUSDT"))["open_time"].max()
    )

    run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    second_funding_start = int(funding_route.calls[1].request.url.params["startTime"])
    second_perp_start = int(perp_route.calls[1].request.url.params["startTime"])
    assert second_funding_start == funding_max + 1
    assert second_perp_start == perp_max


@respx.mock
def test_run_collect_twice_dedupes_sorts_and_keeps_latest_value(tmp_path: Path) -> None:
    """Re-running collect with an overlapping mocked response must not duplicate
    rows, and the freshly re-collected value must win (keep='last' dedup)."""
    _mock_untested_endpoints(exclude=frozenset({PERP_KLINES_URL}))
    funding_route = respx.get(FUNDING_URL)
    first_funding = [_funding_row(START_MS + i * FUNDING_INTERVAL_MS) for i in range(3)]
    funding_route.mock(return_value=httpx.Response(200, json=first_funding))
    perp_route = respx.get(PERP_KLINES_URL)
    perp_route.mock(
        return_value=httpx.Response(
            200,
            json=[
                _kline_row(START_MS, close_price=100.0),
                _kline_row(START_MS + HOUR_MS, close_price=101.0),
            ],
        )
    )

    run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    # second mocked response overlaps the last two funding rows and adds one new one
    second_funding = [_funding_row(START_MS + i * FUNDING_INTERVAL_MS) for i in range(1, 4)]
    funding_route.mock(return_value=httpx.Response(200, json=second_funding))
    # klines resume WITHOUT +1ms, so the last bar is re-fetched; give it a
    # different close to prove the freshly re-collected value wins.
    perp_route.mock(
        return_value=httpx.Response(
            200, json=[_kline_row(START_MS + HOUR_MS, close_price=999.0)]
        )
    )

    run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    funding_df = pl.read_parquet(dataset_path(tmp_path, "funding", "BTCUSDT"))
    times = funding_df["funding_time"].to_list()
    assert len(times) == len(set(times)) == 4
    assert times == sorted(times)

    perp_df = pl.read_parquet(dataset_path(tmp_path, "klines_perp_1h", "BTCUSDT"))
    assert len(perp_df) == 2
    updated_close = perp_df.filter(pl.col("open_time") == perp_df["open_time"].max())["close"][0]
    assert updated_close == 999.0


@respx.mock
def test_kline_ohlc_columns_map_to_correct_indices(tmp_path: Path) -> None:
    """Regression guard for an open/high/low/close index-mapping bug."""
    _mock_untested_endpoints(exclude=frozenset({PERP_KLINES_URL}))
    respx.get(FUNDING_URL).mock(return_value=httpx.Response(200, json=[_funding_row(START_MS)]))
    respx.get(PERP_KLINES_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                _kline_row(
                    START_MS, open_price=10.0, high_price=40.0, low_price=5.0, close_price=20.0
                )
            ],
        )
    )

    run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    row = pl.read_parquet(dataset_path(tmp_path, "klines_perp_1h", "BTCUSDT")).row(
        0, named=True
    )
    assert (row["open"], row["high"], row["low"], row["close"]) == (10.0, 40.0, 5.0, 20.0)


@respx.mock
def test_funding_negative_rate_survives_to_parquet(tmp_path: Path) -> None:
    _mock_untested_endpoints()
    respx.get(FUNDING_URL).mock(
        return_value=httpx.Response(200, json=[_funding_row(START_MS, rate=-0.0005)])
    )

    run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    df = pl.read_parquet(dataset_path(tmp_path, "funding", "BTCUSDT"))
    assert df["funding_rate"][0] == -0.0005


@respx.mock
def test_partial_final_kline_bar_is_dropped_then_written_once_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The most recent kline is often an in-progress candle; it must not be
    written until its close time (open_time + 1h) has passed."""
    fixed_now = datetime(2025, 1, 1, 2, 30, tzinfo=UTC)  # 02:30: the 02:00 bar hasn't closed
    monkeypatch.setattr(collect, "_utcnow", lambda: fixed_now)

    _mock_untested_endpoints(exclude=frozenset({PERP_KLINES_URL}))
    respx.get(FUNDING_URL).mock(return_value=httpx.Response(200, json=[_funding_row(START_MS)]))
    closed_bar_time = START_MS  # 00:00, closes 01:00 -- before fixed_now
    partial_bar_time = START_MS + 2 * HOUR_MS  # 02:00, closes 03:00 -- after fixed_now
    perp_route = respx.get(PERP_KLINES_URL)
    perp_route.mock(
        return_value=httpx.Response(
            200,
            json=[
                _kline_row(closed_bar_time, close_price=50.0),
                _kline_row(partial_bar_time, close_price=999.0),  # in-progress
            ],
        )
    )

    run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    perp_path = dataset_path(tmp_path, "klines_perp_1h", "BTCUSDT")
    df = pl.read_parquet(perp_path)
    assert len(df) == 1
    assert _to_ms(df["open_time"][0]) == closed_bar_time

    # time passes; the previously-partial bar is now closed with final values
    later_now = datetime(2025, 1, 1, 4, 0, tzinfo=UTC)
    monkeypatch.setattr(collect, "_utcnow", lambda: later_now)
    perp_route.mock(
        return_value=httpx.Response(
            200, json=[_kline_row(partial_bar_time, close_price=999.0)]
        )
    )

    run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    df2 = pl.read_parquet(perp_path)
    assert len(df2) == 2
    row = df2.filter(pl.col("open_time") == df2["open_time"].max()).row(0, named=True)
    assert row["close"] == 999.0


@respx.mock
def test_collect_funding_info_present_symbol_gets_interval_absent_gets_default(
    tmp_path: Path,
) -> None:
    respx.get(FUNDING_INFO_URL).mock(
        return_value=httpx.Response(200, json=[{"symbol": "ETHUSDT", "fundingIntervalHours": 4}])
    )

    with BinanceClient() as client:
        _collect_funding_info(client, ["BTCUSDT", "ETHUSDT"], tmp_path)

    df = pl.read_parquet(funding_info_path(tmp_path))
    result = dict(zip(df["symbol"].to_list(), df["funding_interval_hours"].to_list(), strict=True))
    assert result == {"BTCUSDT": 8, "ETHUSDT": 4}


@respx.mock
def test_collect_funding_info_upserts_without_destroying_other_symbols(tmp_path: Path) -> None:
    """--symbols is a documented subset flag: a subset re-collect must not wipe
    intervals already recorded for symbols outside the current request."""
    path = funding_info_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {"symbol": "ADAUSDT", "funding_interval_hours": 4},
            {"symbol": "BTCUSDT", "funding_interval_hours": 4},
        ],
        schema=FUNDING_INFO_SCHEMA,
    ).write_parquet(path)
    respx.get(FUNDING_INFO_URL).mock(return_value=httpx.Response(200, json=[]))

    with BinanceClient() as client:
        _collect_funding_info(client, ["BTCUSDT"], tmp_path)

    df = pl.read_parquet(path)
    result = dict(zip(df["symbol"].to_list(), df["funding_interval_hours"].to_list(), strict=True))
    assert result == {"ADAUSDT": 4, "BTCUSDT": 8}


@respx.mock
def test_run_collect_isolates_symbol_failure_and_still_writes_funding_info(
    tmp_path: Path,
) -> None:
    """One symbol's HTTP failure must not abort the run or skip funding_info,
    and must not stop other datasets for that same symbol from being tried."""
    _mock_untested_endpoints()

    def funding_side_effect(request: httpx.Request) -> httpx.Response:
        if request.url.params["symbol"] == "ETHUSDT":
            return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})
        return httpx.Response(200, json=[_funding_row(START_MS)])

    respx.get(FUNDING_URL).mock(side_effect=funding_side_effect)

    run_collect(symbols=["BTCUSDT", "ETHUSDT"], since=SINCE, data_root=tmp_path)

    assert dataset_path(tmp_path, "funding", "BTCUSDT").exists()
    assert not dataset_path(tmp_path, "funding", "ETHUSDT").exists()
    # a different dataset for the same (failing) symbol must still be attempted
    assert dataset_path(tmp_path, "klines_perp_1h", "ETHUSDT").exists()
    info_df = pl.read_parquet(funding_info_path(tmp_path))
    assert set(info_df["symbol"].to_list()) == {"BTCUSDT", "ETHUSDT"}


@respx.mock
def test_run_collect_raises_when_everything_fails(tmp_path: Path) -> None:
    """A cron caller must see a non-zero exit when literally nothing succeeded,
    not a silently "successful" empty run."""
    error_response = httpx.Response(400, json={"code": -1, "msg": "boom"})
    for url in (FUNDING_URL, PERP_KLINES_URL, MARK_KLINES_URL, PREMIUM_KLINES_URL, SPOT_KLINES_URL):
        respx.get(url).mock(return_value=error_response)
    respx.get(FUNDING_INFO_URL).mock(return_value=httpx.Response(200, json=[]))

    with pytest.raises(RuntimeError, match="all 5"):
        run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)


@respx.mock
def test_run_collect_returns_failure_count_on_partial_failure(tmp_path: Path) -> None:
    """A handful of bad symbols/datasets must not raise — just be counted."""
    _mock_untested_endpoints()
    respx.get(FUNDING_URL).mock(return_value=httpx.Response(400, json={"code": -1, "msg": "boom"}))

    failed = run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    assert failed == 1


@respx.mock
def test_funding_non_finite_rate_is_skipped_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-finite funding_rate must never reach parquet: the fetch for that
    (symbol, dataset) is warned about and skipped, not written or crashed."""
    _mock_untested_endpoints()
    respx.get(FUNDING_URL).mock(
        return_value=httpx.Response(200, json=[_funding_row(START_MS, rate=float("nan"))])
    )

    failed = run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    assert failed == 1
    assert not dataset_path(tmp_path, "funding", "BTCUSDT").exists()
    warnings = capsys.readouterr().err
    assert "BTCUSDT" in warnings
    assert "funding" in warnings


@respx.mock
def test_programming_valueerror_propagates_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain ValueError from deeper in the pipeline (e.g. a genuine bug in
    _write_incremental) must NOT be caught by the per-(symbol, dataset)
    isolation the way ParseError and HTTP errors are — only ParseError (a
    ValueError subclass) is skip-with-warning; anything else is a bug."""
    respx.get(FUNDING_URL).mock(return_value=httpx.Response(200, json=[_funding_row(START_MS)]))

    def _boom(*args: object, **kwargs: object) -> None:
        raise ValueError("programming bug, not a data problem")

    monkeypatch.setattr(collect, "_write_incremental", _boom)

    with pytest.raises(ValueError, match="programming bug"):
        run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)


@respx.mock
def test_run_collect_writes_parquet_matching_schemas_exactly(tmp_path: Path) -> None:
    _mock_untested_endpoints()
    respx.get(FUNDING_URL).mock(
        return_value=httpx.Response(
            200, json=[_funding_row(START_MS), _funding_row(START_MS + 1)]
        )
    )

    run_collect(symbols=["BTCUSDT"], since=SINCE, data_root=tmp_path)

    funding_df = pl.read_parquet(dataset_path(tmp_path, "funding", "BTCUSDT"))
    assert funding_df.schema == FUNDING_SCHEMA

    for dataset in ("klines_perp_1h", "klines_mark_1h", "premium_1h", "klines_spot_1h"):
        kline_df = pl.read_parquet(dataset_path(tmp_path, dataset, "BTCUSDT"))
        assert kline_df.schema == KLINE_SCHEMA

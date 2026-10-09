"""Hyperliquid client tests. respx mocks httpx at the transport level — no real network."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from farb.hl.client import FUNDING_HISTORY_PAGE_SIZE, INFO_URL, HLClient

START_MS = 1_700_000_000_000


def _asset_ctxs_response(
    coins: list[str],
    funding: str = "0.0000125",
    mark_px: str = "60000.5",
    open_interest: str = "1234.5",
    oracle_px: str = "60001.0",
) -> list:
    universe = [{"name": coin} for coin in coins]
    ctx = {
        "funding": funding,
        "markPx": mark_px,
        "openInterest": open_interest,
        "oraclePx": oracle_px,
    }
    asset_ctxs = [dict(ctx) for _ in coins]
    return [{"universe": universe}, asset_ctxs]


@respx.mock
def test_asset_ctxs_zips_meta_and_ctx_by_index() -> None:
    """meta.universe[i] and assetCtxs[i] are two parallel arrays with no
    shared key — pairing them by anything other than index would be wrong."""
    btc_ctx = {
        "funding": "0.0000125",
        "markPx": "60000.5",
        "openInterest": "1234.5",
        "oraclePx": "60001.0",
    }
    eth_ctx = {
        "funding": "-0.00003",
        "markPx": "3000.25",
        "openInterest": "50000.0",
        "oraclePx": "3000.1",
    }
    response_body = [
        {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
        [btc_ctx, eth_ctx],
    ]
    respx.post(INFO_URL).mock(return_value=httpx.Response(200, json=response_body))

    with HLClient() as client:
        result = client.asset_ctxs()

    assert set(result) == {"BTC", "ETH"}
    assert result["BTC"].funding_hourly == 0.0000125
    assert result["BTC"].mark_px == 60000.5
    assert result["BTC"].open_interest == 1234.5
    assert result["BTC"].oracle_px == 60001.0
    assert result["ETH"].funding_hourly == -0.00003
    assert result["ETH"].mark_px == 3000.25


@respx.mock
def test_funding_history_full_then_short_batch_paginates_forward_no_dupes() -> None:
    """Full-page (500) then short batch: keep paginating until short; next
    startTime = last event's time + 1, matching Binance's pagination rule."""
    route = respx.post(INFO_URL)
    full_batch = [
        {"time": START_MS + i, "fundingRate": "0.0000125"} for i in range(FUNDING_HISTORY_PAGE_SIZE)
    ]
    short_start = START_MS + FUNDING_HISTORY_PAGE_SIZE
    short_batch = [{"time": short_start + i, "fundingRate": "0.00002"} for i in range(7)]
    route.side_effect = [
        httpx.Response(200, json=full_batch),
        httpx.Response(200, json=short_batch),
    ]

    with HLClient() as client:
        events = client.funding_history("BTC", START_MS)

    assert route.call_count == 2
    assert len(events) == FUNDING_HISTORY_PAGE_SIZE + 7
    assert len({e.time for e in events}) == FUNDING_HISTORY_PAGE_SIZE + 7
    first_body = json.loads(route.calls[0].request.content)
    second_body = json.loads(route.calls[1].request.content)
    assert first_body["startTime"] == START_MS
    assert second_body["startTime"] == short_start


@respx.mock
def test_funding_history_end_ms_bounds_result_and_stops_pagination() -> None:
    """`end_ms` excludes events at/after it from the result AND stops
    pagination once the fetched range reaches it -- a bounded probe must
    not keep paginating all the way to the present just because the
    first page happened to be full.
    """
    route = respx.post(INFO_URL)
    full_batch = [
        {"time": START_MS + i, "fundingRate": "0.0000125"} for i in range(FUNDING_HISTORY_PAGE_SIZE)
    ]
    end_ms = START_MS + 10  # well inside the first (full) batch
    route.side_effect = [httpx.Response(200, json=full_batch)]

    with HLClient() as client:
        events = client.funding_history("BTC", START_MS, end_ms)

    assert route.call_count == 1  # never asked for a second page past end_ms
    assert len(events) == 10
    assert all(e.time.timestamp() * 1000 < end_ms for e in events)


@respx.mock
def test_funding_history_end_ms_none_is_unbounded_as_before() -> None:
    """Omitting `end_ms` (the default) must behave exactly as before --
    backward compatible, paginating all the way to a short/empty batch.
    """
    route = respx.post(INFO_URL)
    full_batch = [
        {"time": START_MS + i, "fundingRate": "0.0000125"} for i in range(FUNDING_HISTORY_PAGE_SIZE)
    ]
    short_start = START_MS + FUNDING_HISTORY_PAGE_SIZE
    short_batch = [{"time": short_start + i, "fundingRate": "0.00002"} for i in range(7)]
    route.side_effect = [
        httpx.Response(200, json=full_batch),
        httpx.Response(200, json=short_batch),
    ]

    with HLClient() as client:
        events = client.funding_history("BTC", START_MS)

    assert route.call_count == 2
    assert len(events) == FUNDING_HISTORY_PAGE_SIZE + 7


@respx.mock
def test_retries_on_500_then_succeeds() -> None:
    """A single 500 must be retried (not raised) and succeed on the next try."""
    route = respx.post(INFO_URL)
    route.side_effect = [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(200, json=_asset_ctxs_response(["BTC"])),
    ]

    with HLClient() as client:
        result = client.asset_ctxs()

    assert route.call_count == 2
    assert "BTC" in result


@respx.mock
def test_asset_ctxs_rejects_non_finite_mark_px() -> None:
    respx.post(INFO_URL).mock(
        return_value=httpx.Response(200, json=_asset_ctxs_response(["BTC"], mark_px="NaN"))
    )

    with HLClient() as client, pytest.raises(ValueError, match="non-finite"):
        client.asset_ctxs()


@respx.mock
def test_book_top_takes_level_zero_of_each_side() -> None:
    response_body = {
        "coin": "BTC",
        "time": START_MS,
        "levels": [
            [{"px": "60000.5", "sz": "1.2"}, {"px": "60000.0", "sz": "0.5"}],  # bids, best first
            [{"px": "60001.0", "sz": "0.8"}, {"px": "60001.5", "sz": "2.0"}],  # asks, best first
        ],
    }
    respx.post(INFO_URL).mock(return_value=httpx.Response(200, json=response_body))

    with HLClient() as client:
        top = client.book_top("BTC")

    assert top.coin == "BTC"
    assert top.best_bid == 60000.5
    assert top.best_ask == 60001.0


@respx.mock
def test_book_top_raises_clearly_on_empty_book() -> None:
    response_body = {"coin": "BTC", "time": START_MS, "levels": [[], []]}
    respx.post(INFO_URL).mock(return_value=httpx.Response(200, json=response_body))

    with HLClient() as client, pytest.raises(ValueError, match="empty"):
        client.book_top("BTC")


@respx.mock
def test_meta_returns_coin_specs_in_universe_order() -> None:
    response_body = {
        "universe": [
            {"name": "BTC", "szDecimals": 5, "maxLeverage": 50},
            {"name": "ETH", "szDecimals": 4, "maxLeverage": 25},
        ]
    }
    respx.post(INFO_URL).mock(return_value=httpx.Response(200, json=response_body))

    with HLClient() as client:
        specs = client.meta()

    assert [s.coin for s in specs] == ["BTC", "ETH"]
    assert specs[0].sz_decimals == 5
    assert specs[0].max_leverage == 50
    assert specs[1].sz_decimals == 4
    assert specs[1].max_leverage == 25


@respx.mock
def test_candle_snapshot_parses_ohlcv_and_defaults_missing_volume() -> None:
    response_body = [
        {
            "t": START_MS,
            "o": "60000.0",
            "h": "60500.0",
            "l": "59800.0",
            "c": "60250.5",
            "v": "12.34",
        },
        # no "v" field at all on this bar -> volume must default to 0.0
        {"t": START_MS + 3_600_000, "o": "60250.5", "h": "60600.0", "l": "60100.0", "c": "60400.0"},
    ]
    respx.post(INFO_URL).mock(return_value=httpx.Response(200, json=response_body))

    with HLClient() as client:
        candles = client.candle_snapshot("BTC", "1h", START_MS, START_MS + 7_200_000)

    assert len(candles) == 2
    assert candles[0].open == 60000.0
    assert candles[0].high == 60500.0
    assert candles[0].low == 59800.0
    assert candles[0].close == 60250.5
    assert candles[0].volume == 12.34
    assert candles[1].volume == 0.0

    body = json.loads(respx.calls[0].request.content)
    assert body["type"] == "candleSnapshot"
    assert body["req"] == {
        "coin": "BTC",
        "interval": "1h",
        "startTime": START_MS,
        "endTime": START_MS + 7_200_000,
    }


@respx.mock
def test_candle_snapshot_rejects_non_finite_close() -> None:
    response_body = [
        {"t": START_MS, "o": "60000.0", "h": "60500.0", "l": "59800.0", "c": "inf", "v": "1.0"}
    ]
    respx.post(INFO_URL).mock(return_value=httpx.Response(200, json=response_body))

    with HLClient() as client, pytest.raises(ValueError, match="non-finite"):
        client.candle_snapshot("BTC", "1h", START_MS, START_MS + 3_600_000)

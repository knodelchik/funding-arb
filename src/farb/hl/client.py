"""Thin REST client over Hyperliquid's public /info endpoint.

No API keys — every method here is a read-only POST to the same public info
endpoint, distinguished only by the "type" field in the JSON body. Mirrors
farb.data.client.BinanceClient's retry/pacing conventions so both venues'
collectors behave the same way operationally.
"""

from __future__ import annotations

import math
import random
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from farb.hl.types import AssetCtx, BookTop, Candle, CoinSpec, FundingEvent

INFO_URL = "https://api.hyperliquid.xyz/info"

MAX_TRIES = 5
BACKOFF_START_SECONDS = 1.0
BACKOFF_JITTER_LOW = 0.5
BACKOFF_JITTER_HIGH = 1.5
MAX_RETRY_AFTER_SECONDS = 60.0
PACING_SECONDS = 0.25

# HL's fundingHistory endpoint takes no page-size param; it caps each
# response at this many events server-side. There's no way to request a
# smaller/larger page — we only ever observe "did we get a full page back".
FUNDING_HISTORY_PAGE_SIZE = 500

_RETRY_STATUS_CODES = {429}


class HLClientError(ValueError):
    """A Hyperliquid response was malformed, empty, or shaped unexpectedly.

    A ValueError subclass (like farb.data.collect.ParseError) so callers
    that isolate on "bad data, not a bug" via `except ValueError` catch this
    too.
    """


def _utcnow() -> datetime:
    """Wall-clock now, as its own function so tests can monkeypatch it."""
    return datetime.now(UTC)


def _from_ms(ms: int) -> datetime:
    """Milliseconds since epoch -> UTC datetime, matching HL's timestamp units."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _finite_float(raw: Any, *, field: str, coin: str) -> float:
    """Parse a Hyperliquid numeric string, rejecting garbage and NaN/Infinity.

    A malformed or non-finite value silently propagating would corrupt the
    signal/funding math downstream (PIT EMA, APR, PnL) without ever raising
    anywhere obvious.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{coin}: unparseable {field} value from Hyperliquid: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{coin}: non-finite {field} value from Hyperliquid: {raw!r}")
    return value


class HLClient:
    """Wraps an httpx.Client with retry-on-throttle and inter-request pacing.

    An httpx.Client can be injected (e.g. one backed by a respx-mocked
    transport) so tests never touch the network; otherwise a plain client
    is created and owned (and closed) by this instance.
    """

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HLClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _post(self, body: dict[str, Any]) -> Any:
        """POST body to /info, retrying on 429/5xx and transport errors.

        Retries use exponential backoff starting at BACKOFF_START_SECONDS
        with jitter, honoring a Retry-After header when the server sends
        one (clamped to MAX_RETRY_AFTER_SECONDS). A short pacing sleep
        follows every successful response.
        """
        last_error: Exception | None = None
        for attempt in range(MAX_TRIES):
            is_last_attempt = attempt == MAX_TRIES - 1
            try:
                response = self._client.post(INFO_URL, json=body)
            except httpx.TransportError as exc:
                last_error = exc
                if is_last_attempt:
                    raise
                time.sleep(self._backoff_delay(attempt, None))
                continue

            if response.status_code in _RETRY_STATUS_CODES or response.status_code >= 500:
                if is_last_attempt:
                    response.raise_for_status()
                time.sleep(self._backoff_delay(attempt, response.headers.get("Retry-After")))
                continue

            response.raise_for_status()
            time.sleep(PACING_SECONDS)
            return response.json()

        # Unreachable: the loop above always raises or returns on its last
        # iteration, but this keeps the return type honest for type checkers.
        raise last_error or RuntimeError("exhausted retries without a response")

    @staticmethod
    def _backoff_delay(attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            try:
                return min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
            except ValueError:
                pass
        base_delay = BACKOFF_START_SECONDS * (2**attempt)
        # Full-ish jitter so many coins backing off together don't retry in lockstep.
        return base_delay * random.uniform(BACKOFF_JITTER_LOW, BACKOFF_JITTER_HIGH)

    def meta(self) -> list[CoinSpec]:
        """POST {"type": "meta"} -> per-coin sizing/leverage specs.

        Used later for order sizing (qty must round to sz_decimals; leverage
        is capped by maxLeverage). Returned as a list in universe order —
        matching the spec's exact ask — rather than keyed by coin like
        asset_ctxs(), since sizing code always needs one coin's own spec,
        not a lookup structure.
        """
        response = self._post({"type": "meta"})
        if not isinstance(response, dict):
            raise HLClientError(f"expected a dict from meta, got {type(response).__name__}")
        universe = response.get("universe")
        if universe is None:
            raise HLClientError(f"meta response missing 'universe': {response!r}")
        return [
            CoinSpec(
                coin=entry["name"],
                sz_decimals=int(entry["szDecimals"]),
                max_leverage=int(entry["maxLeverage"]),
            )
            for entry in universe
        ]

    def candle_snapshot(
        self, coin: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Candle]:
        """POST {"type": "candleSnapshot", "req": {coin, interval, startTime, endTime}}.

        Unlike fundingHistory and l2Book, this endpoint takes an explicit
        end bound, so there is no pagination loop here — one request covers
        the whole [start_ms, end_ms] range for the interval widths this
        client is used with.
        """
        response = self._post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
            }
        )
        if not isinstance(response, list):
            raise HLClientError(
                f"expected a list from candleSnapshot, got {type(response).__name__}"
            )
        return [
            Candle(
                coin=coin,
                open_time=_from_ms(int(row["t"])),
                open=_finite_float(row["o"], field="open", coin=coin),
                high=_finite_float(row["h"], field="high", coin=coin),
                low=_finite_float(row["l"], field="low", coin=coin),
                close=_finite_float(row["c"], field="close", coin=coin),
                volume=(
                    _finite_float(row["v"], field="volume", coin=coin)
                    if row.get("v") is not None
                    else 0.0
                ),
            )
            for row in response
        ]

    def asset_ctxs(self) -> dict[str, AssetCtx]:
        """POST {"type": "metaAndAssetCtxs"} -> {coin: AssetCtx}.

        The response is [meta, assetCtxs]: meta["universe"][i]["name"] pairs
        with assetCtxs[i] purely by INDEX (HL's API shapes this endpoint as
        two parallel arrays, not keyed records) — the zip below is
        load-bearing, not a convenience.
        """
        response = self._post({"type": "metaAndAssetCtxs"})
        if not isinstance(response, list) or len(response) != 2:
            raise HLClientError(
                f"expected [meta, assetCtxs] from metaAndAssetCtxs, got {response!r}"
            )
        meta, asset_ctxs = response
        universe = meta.get("universe") if isinstance(meta, dict) else None
        if universe is None or len(universe) != len(asset_ctxs):
            raise HLClientError(
                f"universe/assetCtxs length mismatch: {len(universe or [])} vs {len(asset_ctxs)}"
            )
        result: dict[str, AssetCtx] = {}
        for entry, ctx in zip(universe, asset_ctxs, strict=True):
            coin = entry["name"]
            result[coin] = AssetCtx(
                coin=coin,
                funding_hourly=_finite_float(ctx["funding"], field="funding", coin=coin),
                mark_px=_finite_float(ctx["markPx"], field="markPx", coin=coin),
                open_interest=_finite_float(ctx["openInterest"], field="openInterest", coin=coin),
                oracle_px=_finite_float(ctx["oraclePx"], field="oraclePx", coin=coin),
            )
        return result

    def funding_history(self, coin: str, start_ms: int) -> list[FundingEvent]:
        """POST {"type": "fundingHistory", "coin": coin, "startTime": start_ms}.

        Forward-paginates like farb.data.client's Binance pagination: next
        startTime = last event's time + 1, stop when the batch is shorter
        than a full page (FUNDING_HISTORY_PAGE_SIZE) or empty.
        """
        events: list[FundingEvent] = []
        next_start = start_ms
        while True:
            batch = self._post({"type": "fundingHistory", "coin": coin, "startTime": next_start})
            if not isinstance(batch, list):
                raise HLClientError(
                    f"expected a list from fundingHistory, got {type(batch).__name__}"
                )
            if not batch:
                break
            for row in batch:
                events.append(
                    FundingEvent(
                        coin=coin,
                        time=_from_ms(int(row["time"])),
                        rate=_finite_float(row["fundingRate"], field="fundingRate", coin=coin),
                    )
                )
            next_start = int(batch[-1]["time"]) + 1
            if len(batch) < FUNDING_HISTORY_PAGE_SIZE:
                break
        return events

    def book_top(self, coin: str) -> BookTop:
        """POST {"type": "l2Book", "coin": coin} -> best bid/ask.

        levels[0] is the bid side, levels[1] the ask side, each sorted best
        price first — level 0 of each side is exactly the top of book. An
        empty book (either side) raises rather than returning a nonsense
        price, since a paper fill against no liquidity is meaningless.
        """
        response = self._post({"type": "l2Book", "coin": coin})
        if not isinstance(response, dict):
            raise HLClientError(f"{coin}: unexpected l2Book response shape: {response!r}")
        levels = response.get("levels")
        if not levels or len(levels) != 2 or not levels[0] or not levels[1]:
            raise HLClientError(f"{coin}: empty or malformed order book from l2Book: {response!r}")
        bids, asks = levels
        time_ms = response.get("time")
        ts = _from_ms(int(time_ms)) if time_ms is not None else _utcnow()
        return BookTop(
            coin=coin,
            best_bid=_finite_float(bids[0]["px"], field="bid px", coin=coin),
            best_ask=_finite_float(asks[0]["px"], field="ask px", coin=coin),
            ts=ts,
        )

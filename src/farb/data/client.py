"""Thin REST client over Binance's public endpoints.

No API keys — only public market-data endpoints are used — so the client
carries no auth, just the resilience (rate-limit backoff, transport-error
retry) and courtesy (inter-request pacing) needed to pull years of history
without getting throttled or banned.
"""

from __future__ import annotations

import random
import time
from typing import Any

import httpx

FAPI_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"

MAX_TRIES = 5
BACKOFF_START_SECONDS = 1.0
BACKOFF_JITTER_LOW = 0.5
BACKOFF_JITTER_HIGH = 1.5
MAX_RETRY_AFTER_SECONDS = 60.0
PACING_SECONDS = 0.15

_RETRY_STATUS_CODES = {429, 418}


class BinanceClient:
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

    def __enter__(self) -> BinanceClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get(self, base: str, path: str, params: dict[str, Any] | None = None) -> list | dict:
        """GET base+path, retrying on 429/418/5xx and transport errors.

        Retries use exponential backoff starting at BACKOFF_START_SECONDS,
        honoring a Retry-After header when the server sends one. A short
        pacing sleep follows every successful response, to stay well under
        Binance's rate limits without doing full request-weight accounting.
        """
        url = f"{base}{path}"
        last_error: Exception | None = None
        for attempt in range(MAX_TRIES):
            is_last_attempt = attempt == MAX_TRIES - 1
            try:
                response = self._client.get(url, params=params)
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

    def get_list(self, base: str, path: str, params: dict[str, Any] | None = None) -> list:
        """Like get(), but asserts the response is a JSON array.

        Every collector endpoint used by farb.data.collect returns an array
        on success; a dict response would mean Binance returned an error
        body that slipped past raise_for_status, or the API shape changed
        underneath us — either way, iterating over it as a list would
        misbehave silently instead of failing loudly here.
        """
        response = self.get(base, path, params)
        assert isinstance(response, list), (
            f"expected a list response from {path}, got {type(response).__name__}"
        )
        return response

    @staticmethod
    def _backoff_delay(attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            try:
                return min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
            except ValueError:
                pass
        base_delay = BACKOFF_START_SECONDS * (2**attempt)
        # Full-ish jitter so many symbols backing off together don't retry in lockstep.
        return base_delay * random.uniform(BACKOFF_JITTER_LOW, BACKOFF_JITTER_HIGH)
